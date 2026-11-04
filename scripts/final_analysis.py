"""Final-confirmation analysis (CPU only, saved samples only).

    python -I scripts/final_analysis.py --base rb16k_base --cands rb16k_p50c rb16k_own_k64_a6 \
        --set practice_select128 --out results/analysis_test/run1

Reads  results/<tag>/<set>/samples.jsonl.gz  (+ summary.json for the run conditions) for the base and every candidate.
Writes <out>/analysis.json and <out>/analysis.md.

What is computed (see reports/preregistration_final.md):
  * accuracy of every run and the difference to the base. Uncertainty: problem-level CLUSTER bootstrap (problems are resampled
    with replacement, all samples of a problem move together; the same resampled problem sets are used for every run, so
    differences are paired). 95 % percentile interval; two-sided bootstrap p-value; Holm correction inside each family.
      family 1 (primary)  : every candidate vs the base                      (always)
      family 2 (secondary): candidate vs candidate, pairs given with --secondary A:B (A - B) or --all-pairs
    The two families are corrected separately (no correction across families).
  * accuracy under a token budget B in {4096, 8192, 16384}: correct AND finish=="stop" AND tokens <= B (same samples).
  * share of samples whose thinking closed, accuracy among them, wrong answers after closing, how the samples cut at the cap
    split into verbatim-loop / other repetition / no repetition, and (unless --no-loops) the period of the loop (find_loop).

Estimand: pooled accuracy = (number of correct samples) / (number of samples) over the resampled problems (ratio estimator).
With one sample per problem this is the plain mean; with unequal sample counts per problem it weights problems by their
sample count. Problems must be the same in every run (error unless --allow-intersection).
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True  # never write __pycache__ next to the project scripts
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import grade as G  # noqa: E402

PERIOD_BINS = [(6, 12), (13, 30), (31, 100), (101, 700)]  # same bins as reports/notes_layer_choice.md 2.3
TOL = 1e-12


# ----------------------------------------------------------------------------- housekeeping
def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def free_gb():
    """Available physical memory in GB (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo)."""
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 2 ** 30
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("inf")


def peak_mem_mb():
    """Peak working set of this process in MB (Windows only; None elsewhere)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)


def wait_for_memory(min_gb, what, poll=30, max_wait=7200):
    """Block (poll every `poll` s) while free memory is below min_gb; give up after max_wait s."""
    t0 = time.time()
    while free_gb() < min_gb:
        if time.time() - t0 > max_wait:
            sys.exit(f"free memory stayed below {min_gb} GB for {max_wait} s; stopped before: {what}")
        print(f"[wait] free memory {free_gb():.1f} GB < {min_gb} GB; waiting before: {what}", file=sys.stderr, flush=True)
        time.sleep(poll)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- statistics
def holm(p):
    """Holm step-down adjusted p-values (returned in the input order)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def boot_p_values(d_hat, d_b):
    """(percentile-inversion p, null-recentred p), both two-sided with the (k+1)/(B+1) rule.
    The first is the p at which the percentile interval would touch 0 (consistent with the reported interval);
    ties at 0 count on both sides, so it is conservative for discrete data."""
    d_b = d_b[~np.isnan(d_b)]
    B = len(d_b)
    le = int((d_b <= TOL).sum())
    ge = int((d_b >= -TOL).sum())
    p_inv = min(1.0, 2.0 * (min(le, ge) + 1) / (B + 1))
    p_rc = (int((np.abs(d_b - d_hat) >= abs(d_hat) - TOL).sum()) + 1) / (B + 1)
    return p_inv, min(1.0, p_rc)


def pct_ci(x, level=0.95):
    x = x[~np.isnan(x)]
    a = (1 - level) / 2
    if len(x) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(x, [100 * a, 100 * (1 - a)])
    return [float(lo), float(hi)]


def cluster_bootstrap(nums, dens, B, seed, chunk=1000):
    """nums/dens: {run: (M x N) arrays}; N = number of problems. Returns {run: (B x M)} bootstrap replicates of
    sum(num)/sum(den) over problems resampled with replacement (multinomial weights, identical for every run)."""
    N = next(iter(nums.values())).shape[1]
    rng = np.random.default_rng(seed)
    pv = np.full(N, 1.0 / N)
    out = {t: np.empty((B, nums[t].shape[0])) for t in nums}
    done = 0
    while done < B:
        c = min(chunk, B - done)
        W = rng.multinomial(N, pv, size=c).astype(np.float64)
        for t in nums:
            num = W @ nums[t].T
            den = W @ dens[t].T
            with np.errstate(divide="ignore", invalid="ignore"):
                out[t][done:done + c] = np.where(den > 0, num / den, np.nan)
        done += c
    return out


# ----------------------------------------------------------------------------- loading
def load_run(results_root, tag, set_name, tokenizer, min_free_gb):
    d = os.path.join(results_root, tag, set_name)
    path = os.path.join(d, "samples.jsonl.gz")
    if not os.path.isfile(path):
        sys.exit(f"missing: {path}")
    find_loop = None
    if tokenizer is not None:
        from analyze_layer_choice import find_loop  # noqa: PLC0415
    recs, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["id"], r.get("k", 0))
            if key in seen:
                sys.exit(f"duplicate sample {key} in {path}")
            seen.add(key)
            text = r.get("text", "")
            if "verbatim_loop_end" in r and "repetitive_end" in r:
                vb, rp = bool(r["verbatim_loop_end"]), bool(r["repetitive_end"])
            else:
                ls = G.loop_signals(text)
                vb, rp = ls["verbatim_loop_end"], ls["repetitive_end"]
            rec = {"id": r["id"], "k": r.get("k", 0), "answer": r.get("answer"), "correct": bool(r["correct"]),
                   "finish": r["finish"], "tokens": int(r["tokens"]), "closed": bool(G.think_closed(text)),
                   "has_pred": bool(r.get("pred")), "cat": "verbatim" if vb else ("other_rep" if rp else "none"),
                   "loop": None}
            if find_loop is not None and rec["finish"] == "length":
                wait_for_memory(min_free_gb, f"loop detection {tag}", poll=15)
                ids = np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)
                rec["loop"] = find_loop(ids)
            recs.append(rec)
    summary = None
    sp = os.path.join(d, "summary.json")
    if os.path.isfile(sp):
        summary = json.load(open(sp, encoding="utf-8"))
    return recs, summary, path


def metric_names(budgets):
    return (["acc"] + [f"budget_{b}" for b in budgets] +
            ["closed", "acc_given_closed", "closed_wrong", "cut", "cut_verbatim", "cut_other_rep", "cut_none"])


def sample_contrib(s, budgets):
    """(num, den) of every metric for one sample, in metric_names order."""
    stop = s["finish"] == "stop"
    cut = s["finish"] == "length"
    c, cl = s["correct"], s["closed"]
    row = [(c, 1)]
    row += [(c and stop and s["tokens"] <= b, 1) for b in budgets]
    row += [(cl, 1), (c and cl, int(cl)), (cl and not c, 1), (cut, 1),
            (cut and s["cat"] == "verbatim", 1), (cut and s["cat"] == "other_rep", 1), (cut and s["cat"] == "none", 1)]
    return row


def build_arrays(recs, ids, budgets):
    idx = {i: j for j, i in enumerate(ids)}
    names = metric_names(budgets)
    NUM = np.zeros((len(names), len(ids)))
    DEN = np.zeros((len(names), len(ids)))
    for s in recs:
        j = idx.get(s["id"])
        if j is None:
            continue
        for m, (n, d) in enumerate(sample_contrib(s, budgets)):
            NUM[m, j] += n
            DEN[m, j] += d
    return NUM, DEN


def raw_counts(recs, ids):
    keep = [s for s in recs if s["id"] in ids]
    n = len(keep)
    cut = [s for s in keep if s["finish"] == "length"]
    closed = [s for s in keep if s["closed"]]
    cw = [s for s in closed if not s["correct"]]
    out = {
        "n_samples": n, "n_correct": sum(s["correct"] for s in keep),
        "n_stop": sum(s["finish"] == "stop" for s in keep), "n_cut": len(cut),
        "n_finish_other": sum(s["finish"] not in ("stop", "length") for s in keep),
        "n_correct_but_cut": sum(s["correct"] and s["finish"] == "length" for s in keep),
        # grade.extract() falls back to the whole text when thinking never closed, so a \boxed{} inside unfinished
        # thinking can be graded correct; these are counted as correct in the primary metric.
        "n_correct_not_closed": sum(s["correct"] and not s["closed"] for s in keep),
        "n_closed": len(closed), "n_closed_correct": len(closed) - len(cw), "n_closed_wrong": len(cw),
        "closed_wrong_answered": sum(s["finish"] == "stop" and s["has_pred"] for s in cw),
        "closed_wrong_no_boxed": sum(s["finish"] == "stop" and not s["has_pred"] for s in cw),
        "closed_wrong_cut_in_answer": sum(s["finish"] == "length" for s in cw),
        "cut_verbatim_loop": sum(s["cat"] == "verbatim" for s in cut),
        "cut_other_repetition": sum(s["cat"] == "other_rep" for s in cut),
        "cut_no_repetition": sum(s["cat"] == "none" for s in cut),
        "mean_tokens": float(np.mean([s["tokens"] for s in keep])) if keep else None,
    }
    return out


def period_bin(p):
    for lo, hi in PERIOD_BINS:
        if lo <= p <= hi:
            return f"{lo}-{hi}"
    return f">{PERIOD_BINS[-1][1]}"


def loop_table(recs, ids):
    cut = [s for s in recs if s["id"] in ids and s["finish"] == "length"]
    labels = [f"{lo}-{hi}" for lo, hi in PERIOD_BINS] + [f">{PERIOD_BINS[-1][1]}", "undetected"]
    counts = {k: 0 for k in labels}
    cross = {c: {k: 0 for k in labels} for c in ("verbatim", "other_rep", "none")}
    periods, onsets = [], []
    for s in cut:
        lb = "undetected" if s["loop"] is None else period_bin(s["loop"][1])
        counts[lb] += 1
        cross[s["cat"]][lb] += 1
        if s["loop"] is not None:
            onsets.append(s["loop"][0])
            periods.append(s["loop"][1])
    out = {"n_cut": len(cut), "by_period_bin": counts, "by_flag_category_x_period_bin": cross,
           "n_detected": len(periods)}
    if periods:
        out.update(period_median=float(np.median(periods)), period_p10_p90=[float(np.percentile(periods, 10)),
                   float(np.percentile(periods, 90))], onset_median=float(np.median(onsets)),
                   onset_p10_p90=[float(np.percentile(onsets, 10)), float(np.percentile(onsets, 90))])
    return out


# ----------------------------------------------------------------------------- formatting
def pc(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"


def pt(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{nd}f}"


def pcci(est, ci):
    return "n/a" if est is None else f"{pc(est)} [{pc(ci[0])}, {pc(ci[1])}]"


def ptci(est, ci):
    return "n/a" if est is None else f"{pt(est)} [{pt(ci[0])}, {pt(ci[1])}]"


def pv(p):
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- main
def main(argv=None):
    _force_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="tag of the base run")
    ap.add_argument("--cands", nargs="+", required=True, help="tags of the candidate runs")
    ap.add_argument("--set", required=True, dest="set_name", help="set name = directory under results/<tag>/")
    ap.add_argument("--out", required=True, help="output directory (analysis.json, analysis.md)")
    ap.add_argument("--results-root", default=os.path.join(ROOT, "results"))
    ap.add_argument("--label", action="append", default=[], metavar="TAG=NAME", help="display name for a tag (repeatable)")
    ap.add_argument("--secondary", nargs="*", default=[], metavar="A:B", help="secondary comparisons A - B (family 2)")
    ap.add_argument("--all-pairs", action="store_true", help="family 2 = all pairs of candidates")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap replicates (default 10000)")
    ap.add_argument("--seed", type=int, default=20261104, help="bootstrap seed (fixed)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--no-loops", action="store_true", help="skip the loop-period analysis (find_loop)")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3.5-0.8B", "tokenizer.json"))
    ap.add_argument("--expect-n", type=int, default=None, help="fail unless the common problem count equals this")
    ap.add_argument("--allow-intersection", action="store_true", help="use problems present in every run if the sets differ")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="wait while free RAM is below this (default 6)")
    a = ap.parse_args(argv)

    tags = [a.base] + list(a.cands)
    if len(set(tags)) != len(tags):
        sys.exit("duplicate tag among --base/--cands")
    labels = {t: t for t in tags}
    for kv in a.label:
        t, _, name = kv.partition("=")
        if t not in labels:
            sys.exit(f"--label for unknown tag {t}")
        labels[t] = name
    pairs = []
    for s in a.secondary:
        x, _, y = s.partition(":")
        if x not in tags or y not in tags or x == y:
            sys.exit(f"bad --secondary {s}")
        pairs.append((x, y))
    if a.all_pairs:
        for i, x in enumerate(a.cands):
            for y in a.cands[i + 1:]:
                if (x, y) not in pairs and (y, x) not in pairs:
                    pairs.append((x, y))
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    warnings = []

    wait_for_memory(a.min_free_gb, "loading")
    tk = None
    if not a.no_loops:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)

    data, summ, paths = {}, {}, {}
    for t in tags:
        data[t], summ[t], paths[t] = load_run(a.results_root, t, a.set_name, tk, a.min_free_gb)

    # --- the problem set must be identical
    idsets = {t: {s["id"] for s in data[t]} for t in tags}
    common = set.intersection(*idsets.values())
    union = set.union(*idsets.values())
    if common != union:
        msg = "problem sets differ: " + ", ".join(f"{t}: {len(idsets[t])}" for t in tags) + f"; common {len(common)}, union {len(union)}"
        if not a.allow_intersection:
            sys.exit(msg + " (use --allow-intersection to analyse the common problems)")
        warnings.append(msg + " -> analysed on the common problems only")
    ids = sorted(common)
    if a.expect_n is not None and len(ids) != a.expect_n:
        sys.exit(f"expected {a.expect_n} problems, found {len(ids)}")
    ans = {}
    for t in tags:
        for s in data[t]:
            if s["id"] in common:
                if ans.setdefault(s["id"], s["answer"]) != s["answer"]:
                    sys.exit(f"gold answer of {s['id']} differs between runs ({t})")
    # --- run conditions
    cond_keys = ("max_tokens", "preset", "thinking")
    conds = {t: {k: (summ[t] or {}).get(k) for k in cond_keys} for t in tags}
    for k in cond_keys:
        if len({json.dumps(conds[t][k]) for t in tags}) > 1:
            warnings.append(f"run condition '{k}' differs between runs: " + ", ".join(f"{t}={conds[t][k]}" for t in tags))
    maxtok = {t: (summ[t] or {}).get("max_tokens") for t in tags}

    names = metric_names(a.budgets)
    NUM, DEN = {}, {}
    for t in tags:
        NUM[t], DEN[t] = build_arrays(data[t], ids, a.budgets)
    point = {t: np.where(DEN[t].sum(1) > 0, NUM[t].sum(1) / np.maximum(DEN[t].sum(1), 1e-300), np.nan) for t in tags}
    n_per = {t: DEN[t][0] for t in tags}  # samples per problem (metric 'acc' has den 1)
    for t in tags:
        if n_per[t].min() < 1:
            sys.exit(f"{t}: some problems have no sample")

    wait_for_memory(a.min_free_gb, "bootstrap")
    boot = cluster_bootstrap(NUM, DEN, a.boot, a.seed)

    # --- per-run results
    runs = {}
    for t in tags:
        d = {"label": labels[t], "samples_file": os.path.relpath(paths[t], a.results_root).replace("\\", "/"),
             "samples_sha256": sha256_of(paths[t]), "summary": summ[t],
             "samples_per_problem": {"min": int(n_per[t].min()), "max": int(n_per[t].max())},
             "counts": raw_counts(data[t], set(ids)), "metrics": {}}
        for m, nm in enumerate(names):
            est = float(point[t][m])
            if nm.startswith("budget_") and maxtok[t] is not None and int(nm[7:]) > maxtok[t]:
                d["metrics"][nm] = None  # budget larger than what this run was allowed to generate
                continue
            d["metrics"][nm] = {"est": None if np.isnan(est) else est, "ci95": pct_ci(boot[t][:, m])}
        if not a.no_loops:
            d["loops"] = loop_table(data[t], set(ids))
        runs[t] = d

    mi = {nm: i for i, nm in enumerate(names)}

    def compare(x, y, metric="acc", m_family=1):
        i = mi[metric]
        d_hat = float(point[x][i] - point[y][i])
        db = boot[x][:, i] - boot[y][:, i]
        p_inv, p_rc = boot_p_values(d_hat, db)
        px, py = NUM[x][0] / n_per[x], NUM[y][0] / n_per[y]  # per-problem accuracy
        out = {"a": x, "b": y, "metric": metric, "diff": d_hat, "ci95": pct_ci(db),
               "ci_bonferroni": pct_ci(db, 1 - 0.05 / m_family), "p_boot": p_inv, "p_boot_recentred": p_rc,
               "acc_a": float(point[x][i]), "acc_b": float(point[y][i]),
               "problems_a_better": int((px > py + TOL).sum()), "problems_b_better": int((py > px + TOL).sum()),
               "problems_tied": int((np.abs(px - py) <= TOL).sum())}
        return out

    fam1 = [compare(t, a.base, "acc", len(a.cands)) for t in a.cands]
    for r, p in zip(fam1, holm([c["p_boot"] for c in fam1])):
        r["p_holm"] = float(p)
    fam2 = [compare(x, y, "acc", max(len(pairs), 1)) for x, y in pairs]
    for r, p in zip(fam2, holm([c["p_boot"] for c in fam2]) if fam2 else []):
        r["p_holm"] = float(p)
    # descriptive differences to the base (not tested): budgets, closed share
    desc = {t: {nm: (lambda c: {"diff": c["diff"], "ci95": c["ci95"]})(compare(t, a.base, nm))
                for nm in ["acc"] + [f"budget_{b}" for b in a.budgets] + ["closed", "closed_wrong", "cut"]
                if runs[t]["metrics"].get(nm) is not None} for t in a.cands}

    prov = {"script": "scripts/final_analysis.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
            "grade_py_sha256": sha256_of(os.path.join(ROOT, "scripts", "grade.py")),
            "python": platform.python_version(), "numpy": np.__version__, "argv": sys.argv[1:] if argv is None else argv}
    if not a.no_loops:
        prov["analyze_layer_choice_py_sha256"] = sha256_of(os.path.join(ROOT, "scripts", "analyze_layer_choice.py"))
        prov["tokenizer_sha256"] = sha256_of(a.tokenizer)
    meta = {"set": a.set_name, "base": a.base, "cands": a.cands, "n_problems": len(ids), "bootstrap_B": a.boot,
            "seed": a.seed, "budgets": a.budgets, "interval": "95% percentile, problem-level cluster bootstrap",
            "p_value": "two-sided; 2*(min(#{d*<=0},#{d*>=0})+1)/(B+1); Holm within each family",
            "estimand": "pooled accuracy = correct samples / samples (ratio estimator over resampled problems)",
            "family1": "primary: each candidate vs base", "family1_m": len(fam1),
            "family2": "secondary: pairs " + (", ".join(f"{x}-{y}" for x, y in pairs) or "(none)"), "family2_m": len(fam2),
            "elapsed_s": round(time.time() - t_start, 1), "peak_working_set_mb": peak_mem_mb(), "warnings": warnings, "provenance": prov}
    result = {"meta": meta, "runs": runs, "family1_primary": fam1, "family2_secondary": fam2, "descriptive_vs_base": desc}
    with open(os.path.join(a.out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    md = render_markdown(result, tags, a, labels, pairs)
    with open(os.path.join(a.out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return result


def render_markdown(R, tags, a, L, pairs):
    meta, runs = R["meta"], R["runs"]
    base = a.base
    o = []
    w = o.append
    w(f"# 最終確認の解析結果（集合 `{meta['set']}`）\n")
    for x in meta["warnings"]:
        w(f"> **警告**: {x}\n")
    w(f"- 問題数 {meta['n_problems']}、ブートストラップ {meta['bootstrap_B']:,} 回（シード {meta['seed']}）、区間は 95% パーセンタイル。")
    w("- 推定量: 正解標本数 / 標本数（問題を再標本化したときの比）。1 回/問なら通常の正解率に一致。")
    w("- 区間と p 値は問題単位のクラスタ・ブートストラップ（同じ再標本化を全候補に使うので、基底との差は対応あり）。")
    w("- 区間は多重性を補正していない（補正は p 値のみ。Holm）。補正後の p 値と区間の食い違いに注意。\n")

    w("## 1. 実行条件")
    w("| 候補 | tag | 標本数 | 標本/問 | 上限 | 設定 | 生成時間(s) | samples.jsonl.gz の sha256（先頭12桁） |")
    w("|---|---|---|---|---|---|---|---|")
    tot = 0
    for t in tags:
        r, s = runs[t], runs[t]["summary"] or {}
        spp = r["samples_per_problem"]
        tot += s.get("gen_seconds") or 0
        w(f"| {L[t]} | `{t}` | {r['counts']['n_samples']} | {spp['min']}" + (f"-{spp['max']}" if spp["max"] != spp["min"] else "")
          + f" | {s.get('max_tokens', 'n/a')} | {s.get('preset', 'n/a')} | {s.get('gen_seconds', 'n/a')} | {r['samples_sha256'][:12]} |")
    w(f"\n評価回数（候補数）: {len(tags)}（基底を含む）。生成時間の合計: {tot} 秒（summary.json の gen_seconds の和）。\n")

    def comp_table(rows, title_note):
        w("| 比較 | 正解率 A | 正解率 B | 差 A−B (pt) [95%区間] | p（ブートストラップ） | p（Holm） | p（中心化） | 問題: A勝 / B勝 / 同 |")
        w("|---|---|---|---|---|---|---|---|")
        for c in rows:
            w(f"| {L[c['a']]} − {L[c['b']]} | {pc(c['acc_a'])} | {pc(c['acc_b'])} | {ptci(c['diff'], c['ci95'])} | {pv(c['p_boot'])} | "
              f"**{pv(c['p_holm'])}** | {pv(c['p_boot_recentred'])} | {c['problems_a_better']} / {c['problems_b_better']} / {c['problems_tied']} |")
        w(title_note)

    w(f"## 2. 主比較（族1）: 各候補 対 基底　— m = {meta['family1_m']}、Holm 補正")
    b = runs[base]["metrics"]["acc"]
    w(f"基底 `{base}` の正解率: {pcci(b['est'], b['ci95'])}（{runs[base]['counts']['n_correct']}/{runs[base]['counts']['n_samples']}）\n")
    comp_table(R["family1_primary"],
               "\np（ブートストラップ）= 2×min(P(d*≤0), P(d*≥0))（区間と整合）、p（中心化）= 参考（差を 0 に中心化した帰無分布）。"
               "Holm は「p（ブートストラップ）」に対して族1 の中だけで適用。\n")
    if R["family2_secondary"]:
        w("## 3. 副比較（族2）: " + "、".join(f"{L[x]} − {L[y]}" for x, y in pairs) + f"　— m = {meta['family2_m']}、族1 とは別に Holm 補正")
        comp_table(R["family2_secondary"], "\n族1 と族2 の間では補正しない。副比較は探索的な位置づけ。\n")
    else:
        w("## 3. 副比較（族2）\n指定なし。\n")

    w("## 4. 予算別の正解率（「正解 かつ finish=stop かつ tokens ≤ B」、同じ標本から）")
    hdr = "| 候補 | " + " | ".join(f"B = {b_:,}" for b_ in meta["budgets"]) + " |"
    sep = "|---|" + "---|" * len(meta["budgets"])
    w(hdr)
    w(sep)
    for t in tags:
        cells = []
        for b_ in meta["budgets"]:
            m = runs[t]["metrics"][f"budget_{b_}"]
            cells.append("n/a（上限超え）" if m is None else pcci(m["est"], m["ci95"]))
        w(f"| {L[t]} | " + " | ".join(cells) + " |")
    w("\n基底との差（pt、記述のみ・検定なし）")
    w(hdr)
    w(sep)
    for t in a.cands:
        cells = []
        for b_ in meta["budgets"]:
            dd = R["descriptive_vs_base"][t].get(f"budget_{b_}")
            cells.append("n/a" if dd is None else ptci(dd["diff"], dd["ci95"]))
        w(f"| {L[t]} | " + " | ".join(cells) + " |")
    cb = [(L[t], runs[t]["counts"]["n_correct_but_cut"], runs[t]["counts"]["n_correct_not_closed"]) for t in tags
          if runs[t]["counts"]["n_correct_but_cut"] or runs[t]["counts"]["n_correct_not_closed"]]
    w("\n" + ("注: 主指標は「採点で正解」の標本をすべて数える。採点（`grade.extract`）は思考が閉じていないと本文全体の最後の `\\boxed{}` を使うため、"
              "finish=length（または思考が閉じていない）のまま正解になる標本があり、最大の B の値が主指標より小さくなる候補がある。"
              "件数（正解かつ finish=length / 正解かつ思考が閉じていない）: " + ", ".join(f"{n} {c} / {u}" for n, c, u in cb) + "。"
              if cb else "注: 正解で finish=length の標本は無い（最大の B の値は主指標と一致する）。") + "\n")

    w("## 5. 思考の終了と誤答")
    w("| 候補 | 思考を閉じた割合 [95%区間] | 閉じた中の正解率 [95%区間] | 閉じた件数 | 閉じて正解 | 閉じた後の誤答 | 内訳: 誤答を提出 | 内訳: boxed なしで終了 | 内訳: 回答部で打切り |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        m, c = runs[t]["metrics"], runs[t]["counts"]
        w(f"| {L[t]} | {pcci(m['closed']['est'], m['closed']['ci95'])} | {pcci(m['acc_given_closed']['est'], m['acc_given_closed']['ci95'])} | "
          f"{c['n_closed']} | {c['n_closed_correct']} | {c['n_closed_wrong']} | {c['closed_wrong_answered']} | "
          f"{c['closed_wrong_no_boxed']} | {c['closed_wrong_cut_in_answer']} |")
    w("\n「閉じた」= `grade.think_closed(text)`。閉じた後の誤答 = 閉じていて不正解の標本（finish=stop で答えが誤り / boxed なしで終了 / 回答部の途中で上限に到達）。\n")

    w("## 6. 上限での打ち切りの内訳（finish=length）")
    w("| 候補 | 打ち切り | 周期あり（完全一致の繰り返し） | その他の繰り返し | 繰り返しなし（まだ思考中など） | 打ち切り割合 [95%区間] |")
    w("|---|---|---|---|---|---|")
    for t in tags:
        c, m = runs[t]["counts"], runs[t]["metrics"]["cut"]
        w(f"| {L[t]} | {c['n_cut']} | {c['cut_verbatim_loop']} | {c['cut_other_repetition']} | {c['cut_no_repetition']} | {pcci(m['est'], m['ci95'])} |")
    w("\n分類は samples.jsonl.gz の `verbatim_loop_end`（末尾が 1〜200 語のブロックの 3 回以上の繰り返し）と `repetitive_end` に従う。\n")

    if not a.no_loops:
        labs = list(next(iter(runs.values()))["loops"]["by_period_bin"].keys())
        w("## 7. 繰り返しの周期別の件数（打ち切られた標本に `find_loop`、周期はトークン数）")
        w("| 候補 | 打ち切り | " + " | ".join(f"周期 {x}" if x != "undetected" else "判定できず" for x in labs) + " | 周期の中央値 [10–90%] | 開始位置の中央値 |")
        w("|---|---|" + "---|" * len(labs) + "---|---|")
        for t in tags:
            lp = runs[t]["loops"]
            med = f"{lp['period_median']:.0f} [{lp['period_p10_p90'][0]:.0f}–{lp['period_p10_p90'][1]:.0f}]" if lp["n_detected"] else "–"
            on = f"{lp['onset_median']:.0f}" if lp["n_detected"] else "–"
            w(f"| {L[t]} | {lp['n_cut']} | " + " | ".join(str(lp["by_period_bin"][x]) for x in labs) + f" | {med} | {on} |")
        w("\n`find_loop`（scripts/analyze_layer_choice.py）は、再トークン化した本文の末尾 3,000 トークンで周期 6〜700、90% 一致、6 周期以上を探す。"
          "2,000 トークン未満の標本と条件を満たさない標本は「判定できず」。\n")
    else:
        w("## 7. 繰り返しの周期別の件数\n`--no-loops` により省略。\n")
    w("---\n再現: `meta.provenance`（スクリプト・入力ファイルの sha256、引数）は `analysis.json` に記録。")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    main()
