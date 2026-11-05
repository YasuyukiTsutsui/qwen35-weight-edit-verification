"""Pre-registered analysis of the sealed blocks C1..C5 (Issue #29; frame: reports/preregistration_final2_frame.md). CPU only, saved samples only.

    python -I scripts/final_analysis2.py --runs TAG [TAG ...] --blocks C1 --primary A:B [--secondary X:Y ...] [--base TAG]
           [--noninf-margin 0.05] --out results/final2_analysis/<name>
    python -I scripts/final_analysis2.py --selftest

Reads results/<tag>/<block>/samples.jsonl.gz (+ summary.json) for every run and block.  Problems and strata come from data/final2/manifest.json.
Differences to scripts/final_analysis.py (which served the first confirmation, Issue #11):
  * ESTIMAND = 0.5 * accuracy(GSM8K problems) + 0.5 * accuracy(MATH problems)  (fixed stratum weights; each stratum accuracy is the
    ratio estimator correct samples / samples).  The raw pooled accuracy is reported as a descriptive number only.
  * BOOTSTRAP = stratified problem-level cluster bootstrap: inside each stratum the problems are resampled with replacement (all samples of a
    problem move together); the same resampled sets are used for every run, so differences are paired.  B = 10,000, seed 20261105 by default.
  * ONE primary comparison (--primary A:B = A - B), two-sided percentile-inversion p-value (no multiplicity correction needed for a single test).
    Everything else (--secondary, --base) is labelled exploratory; --secondary gets Holm inside its own family.
  * NON-INFERIORITY only if --noninf-margin M is given (it must be written in the registration before the run): the 95 % interval of the
    primary difference must lie above -M.
  * Per-stratum and per-block numbers are descriptive.
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import final_analysis as FA  # noqa: E402

STRATA = ("GSM8K", "MATH")
WEIGHTS = {"GSM8K": 0.5, "MATH": 0.5}
COND_KEYS = ("max_tokens", "preset", "thinking", "samples", "seed", "vllm_version", "max_num_seqs")


def stratum_of(pid):
    return "GSM8K" if str(pid).startswith("gsm8k") else "MATH"


def point_estimates(NUM, DEN, strata_idx, weights=WEIGHTS):
    """{run: (M,)} weighted sum over strata of sum(num)/sum(den)."""
    out = {}
    for t in NUM:
        est = np.zeros(NUM[t].shape[0])
        for s, idx in strata_idx.items():
            num, den = NUM[t][:, idx].sum(1), DEN[t][:, idx].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                est += weights[s] * np.where(den > 0, num / den, np.nan)
        out[t] = est
    return out


def strat_bootstrap(NUM, DEN, strata_idx, B, seed, weights=WEIGHTS, chunk=500):
    """Returns ({run: (B, M)} weighted estimate, {run: {stratum: (B, M)}}) over stratified, paired problem resamples."""
    rng = np.random.default_rng(seed)
    M = next(iter(NUM.values())).shape[0]
    tot = {t: np.empty((B, M)) for t in NUM}
    per = {t: {s: np.empty((B, M)) for s in strata_idx} for t in NUM}
    done = 0
    while done < B:
        c = min(chunk, B - done)
        acc = {t: np.zeros((c, M)) for t in NUM}
        for s, idx in strata_idx.items():
            n = len(idx)
            Ws = rng.multinomial(n, np.full(n, 1.0 / n), size=c).astype(np.float64)
            for t in NUM:
                num, den = Ws @ NUM[t][:, idx].T, Ws @ DEN[t][:, idx].T
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = np.where(den > 0, num / den, np.nan)
                per[t][s][done:done + c] = e
                acc[t] += weights[s] * e
        for t in NUM:
            tot[t][done:done + c] = acc[t]
        done += c
    return tot, per


def sha(path):
    return FA.sha256_of(path) if os.path.isfile(path) else None


# ----------------------------------------------------------------------------- self-test
def selftest():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)

    # 1. Holm against a hand computation
    h = FA.holm([0.01, 0.04, 0.03])
    check("holm hand computation", np.allclose(h, [0.03, 0.06, 0.06]), str(h))
    # 2. estimand: fixed weights, not pooled (unequal stratum sizes)
    rng = np.random.default_rng(1)
    nG, nM = 60, 180
    cG, cM = rng.random(nG) < 0.6, rng.random(nM) < 0.3
    num = np.zeros((1, nG + nM)); num[0] = np.concatenate([cG, cM]); den = np.ones_like(num)
    sidx = {"GSM8K": np.arange(nG), "MATH": np.arange(nG, nG + nM)}
    pe = point_estimates({"x": num}, {"x": den}, sidx)["x"][0]
    check("point estimate = 0.5*acc_G + 0.5*acc_M (not the pooled mean)", abs(pe - (0.5 * cG.mean() + 0.5 * cM.mean())) < 1e-12 and abs(pe - num.mean()) > 1e-3,
          f"{pe:.4f} vs pooled {num.mean():.4f}")
    # 3. identical runs -> zero paired difference
    tot, _ = strat_bootstrap({"a": num, "b": num.copy()}, {"a": den, "b": den.copy()}, sidx, 300, 5)
    check("identical runs give an identical bootstrap distribution", np.allclose(tot["a"], tot["b"]))
    # 4. coverage of the 95 % percentile interval for a true difference, many synthetic data sets
    pG = {"a": 0.62, "b": 0.55}
    pM = {"a": 0.33, "b": 0.30}
    true = 0.5 * (pG["a"] - pG["b"]) + 0.5 * (pM["a"] - pM["b"])
    cover, lo_cover, R = 0, 0, 300
    for r in range(R):
        rg = np.random.default_rng(1000 + r)
        uG, uM = rg.random(nG), rg.random(nM)       # shared problem-level draws make the arms correlated, as paired runs are
        A = {k: np.concatenate([(uG < pG[k]), (uM < pM[k])]).astype(float)[None, :] for k in ("a", "b")}
        D = {k: np.ones_like(v) for k, v in A.items()}
        tot, _ = strat_bootstrap(A, D, sidx, 400, 77 + r)
        d = tot["a"][:, 0] - tot["b"][:, 0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        cover += lo <= true <= hi
    check("95 % interval covers the true stratified difference (300 data sets, tolerance 91-99 %)", 0.91 <= cover / R <= 0.99, f"{cover / R:.3f}")
    # 5. bootstrap SE equals the analytic SE of the stratified estimate
    A = {"a": np.concatenate([cG, cM]).astype(float)[None, :]}
    D = {"a": np.ones_like(A["a"])}
    tot, per = strat_bootstrap(A, D, sidx, 4000, 3)
    se_an = np.sqrt(0.25 * cG.var() / nG + 0.25 * cM.var() / nM)
    check("bootstrap SE matches the analytic stratified SE (within 8 %)", abs(tot["a"][:, 0].std() / se_an - 1) < 0.08, f"{tot['a'][:, 0].std():.4f} vs {se_an:.4f}")
    # 6. per-stratum replicates recombine to the total
    check("weighted per-stratum replicates sum to the total", np.allclose(0.5 * per["a"]["GSM8K"] + 0.5 * per["a"]["MATH"], tot["a"]))
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main(argv=None):
    FA._force_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", nargs="+", help="tags of the runs to analyse (every run must exist for every block)")
    ap.add_argument("--blocks", nargs="+", help="block names, e.g. C1")
    ap.add_argument("--primary", help="the ONE primary comparison A:B (A - B)")
    ap.add_argument("--secondary", nargs="*", default=[], help="exploratory comparisons A:B (Holm inside this family)")
    ap.add_argument("--base", help="tag of the unedited model: descriptive differences to it (exploratory)")
    ap.add_argument("--noninf-margin", type=float, default=None, help="non-inferiority margin M (accuracy units, e.g. 0.05); only if registered")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "final2", "manifest.json"))
    ap.add_argument("--exclude-ids", default=None, help="JSON list of problem IDs excluded before any evaluation (recorded in the registration)")
    ap.add_argument("--results-root", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--budgets", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--label", action="append", default=[], metavar="TAG=NAME")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.runs and a.blocks and a.primary and a.out):
        ap.error("--runs, --blocks, --primary and --out are required")
    t0 = time.time()
    labels = {t: t for t in a.runs + ([a.base] if a.base else [])}
    for kv in a.label:
        t, _, nm = kv.partition("=")
        labels[t] = nm
    px, _, py = a.primary.partition(":")
    if px not in a.runs or py not in a.runs or px == py:
        sys.exit("--primary must name two different --runs")
    pairs = []
    for s in a.secondary:
        x, _, y = s.partition(":")
        if x not in a.runs or y not in a.runs or x == y:
            sys.exit(f"bad --secondary {s}")
        pairs.append((x, y))
    tags = list(a.runs) + ([a.base] if a.base and a.base not in a.runs else [])

    man = json.load(open(a.manifest, encoding="utf-8"))
    excl = set(json.load(open(a.exclude_ids, encoding="utf-8"))) if a.exclude_ids else set()
    exp_ids, block_of = [], {}
    for b in a.blocks:
        if b not in man["blocks"]:
            sys.exit(f"unknown block {b}")
        for i in man["blocks"][b]["ids"]:
            if i not in excl:
                exp_ids.append(i)
                block_of[i] = b
    ids = sorted(exp_ids)
    strata_idx = {s: np.array([k for k, i in enumerate(ids) if stratum_of(i) == s]) for s in STRATA}
    warnings = []
    data, summ, paths = {}, {}, {}
    for t in tags:
        recs, ss, pp = [], {}, {}
        for b in a.blocks:
            r, sm, p = FA.load_run(a.results_root, t, b, None, 6.0)
            recs += r
            ss[b], pp[b] = sm, p
        data[t], summ[t], paths[t] = recs, ss, pp
        got = {s["id"] for s in recs}
        if got != set(ids):
            sys.exit(f"{t}: problems differ from the registered blocks (missing {len(set(ids) - got)}, extra {len(got - set(ids))})")
        for b in a.blocks:
            if (ss[b] or {}).get("n_problems") != len(man["blocks"][b]["ids"]) - len([i for i in man["blocks"][b]["ids"] if i in excl]) and not excl:
                warnings.append(f"{t}/{b}: n_problems in summary differs from the manifest")
    for k in COND_KEYS:
        vals = {json.dumps([(summ[t][b] or {}).get(k) for b in a.blocks]) for t in tags}
        if len(vals) > 1:
            warnings.append(f"run condition '{k}' differs between runs: " + "; ".join(f"{t}={[(summ[t][b] or {}).get(k) for b in a.blocks]}" for t in tags))

    names = FA.metric_names(a.budgets)
    NUM, DEN = {}, {}
    for t in tags:
        NUM[t], DEN[t] = FA.build_arrays(data[t], ids, a.budgets)
        if DEN[t][0].min() < 1:
            sys.exit(f"{t}: some problems have no sample")
    point = point_estimates(NUM, DEN, strata_idx)
    boot, per = strat_bootstrap(NUM, DEN, strata_idx, a.boot, a.seed)
    n_per = {t: DEN[t][0] for t in tags}
    mi = {nm: i for i, nm in enumerate(names)}

    def stratum_point(t, s, m):
        idx = strata_idx[s]
        return float(NUM[t][m, idx].sum() / max(DEN[t][m, idx].sum(), 1e-300))

    runs = {}
    for t in tags:
        pooled = float(NUM[t][0].sum() / DEN[t][0].sum())
        d = {"label": labels.get(t, t), "n_samples": int(DEN[t][0].sum()), "pooled_accuracy_descriptive": pooled,
             "metrics": {}, "by_stratum": {s: {"n_problems": int(len(strata_idx[s])), "acc": stratum_point(t, s, 0),
                                                "ci95": FA.pct_ci(per[t][s][:, 0])} for s in STRATA},
             "by_block": {}, "counts": FA.raw_counts(data[t], set(ids)),
             "samples_sha256": {b: sha(paths[t][b]) for b in a.blocks}, "summary": summ[t]}
        for m, nm in enumerate(names):
            if nm.startswith("budget_") and (summ[t][a.blocks[0]] or {}).get("max_tokens") and int(nm[7:]) > summ[t][a.blocks[0]]["max_tokens"]:
                d["metrics"][nm] = None
                continue
            e = float(point[t][m])
            d["metrics"][nm] = {"est": None if np.isnan(e) else e, "ci95": FA.pct_ci(boot[t][:, m])}
        for b in a.blocks:
            bi = {s: np.array([k for k in strata_idx[s] if block_of[ids[k]] == b]) for s in STRATA}
            if all(len(v) for v in bi.values()):
                d["by_block"][b] = float(sum(WEIGHTS[s] * NUM[t][0, bi[s]].sum() / DEN[t][0, bi[s]].sum() for s in STRATA))
        runs[t] = d

    def compare(x, y, metric="acc"):
        i = mi[metric]
        dh = float(point[x][i] - point[y][i])
        db = boot[x][:, i] - boot[y][:, i]
        p_inv, p_rc = FA.boot_p_values(dh, db)
        px_, py_ = NUM[x][0] / n_per[x], NUM[y][0] / n_per[y]
        out = {"a": x, "b": y, "metric": metric, "diff": dh, "ci95": FA.pct_ci(db), "p_boot": p_inv, "p_boot_recentred_reference": p_rc,
               "acc_a": float(point[x][i]), "acc_b": float(point[y][i]),
               "by_stratum": {s: {"diff": stratum_point(x, s, i) - stratum_point(y, s, i), "ci95": FA.pct_ci(per[x][s][:, i] - per[y][s][:, i])} for s in STRATA},
               "problems_a_better": int((px_ > py_ + 1e-12).sum()), "problems_b_better": int((py_ > px_ + 1e-12).sum()),
               "problems_tied": int((np.abs(px_ - py_) <= 1e-12).sum())}
        if a.noninf_margin is not None and metric == "acc":
            out["noninferiority"] = {"margin": a.noninf_margin, "ci_lower": out["ci95"][0], "established": bool(out["ci95"][0] > -a.noninf_margin)}
        return out

    primary = compare(px, py)
    fam2 = [compare(x, y) for x, y in pairs]
    for r, p in zip(fam2, FA.holm([c["p_boot"] for c in fam2]) if fam2 else []):
        r["p_holm"] = float(p)
    desc = {}
    if a.base:
        desc = {t: {nm: (lambda c: {"diff": c["diff"], "ci95": c["ci95"]})(compare(t, a.base, nm))
                    for nm in ["acc"] + [f"budget_{b}" for b in a.budgets] + ["closed", "closed_wrong", "cut"] if runs[t]["metrics"].get(nm) is not None}
                for t in a.runs}
    prov = {"script": "scripts/final_analysis2.py", "script_sha256": sha(os.path.abspath(__file__)),
            "final_analysis_py_sha256": sha(os.path.join(ROOT, "scripts", "final_analysis.py")), "grade_py_sha256": sha(os.path.join(ROOT, "scripts", "grade.py")),
            "manifest_sha256": sha(a.manifest), "python": platform.python_version(), "numpy": np.__version__,
            "argv": sys.argv[1:] if argv is None else argv}
    meta = {"blocks": a.blocks, "n_problems": len(ids), "n_by_stratum": {s: int(len(strata_idx[s])) for s in STRATA}, "excluded_ids": sorted(excl),
            "weights": WEIGHTS, "bootstrap_B": a.boot, "seed": a.seed, "primary": a.primary,
            "interval": "95% percentile, stratified problem-level cluster bootstrap (paired)",
            "p_value": "two-sided, 2*(min(#{d*<=0},#{d*>=0})+1)/(B+1)", "noninf_margin": a.noninf_margin,
            "secondary_family": [f"{x}-{y}" for x, y in pairs], "elapsed_s": round(time.time() - t0, 1), "warnings": warnings, "provenance": prov}
    result = {"meta": meta, "runs": runs, "primary": primary, "secondary_exploratory": fam2, "descriptive_vs_base_exploratory": desc}
    os.makedirs(a.out, exist_ok=True)
    json.dump(result, open(os.path.join(a.out, "analysis.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = render(result, labels, a)
    open(os.path.join(a.out, "analysis.md"), "w", encoding="utf-8").write(md)
    print(md)
    return 0


def render(R, labels, a):
    P = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    PT = lambda x: f"{100 * x:+.1f}"
    CI = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
    m = R["meta"]
    L = [f"# 解析結果（ブロック {', '.join(m['blocks'])}、{m['n_problems']} 問: GSM8K {m['n_by_stratum']['GSM8K']} + MATH {m['n_by_stratum']['MATH']}）", "",
         "推定量 = 0.5 × GSM8K の正答率 + 0.5 × MATH の正答率。区間は層別・問題単位のブートストラップ（95%、対応あり）。", ""]
    pr = R["primary"]
    L += ["## 主比較（1 本）", "", f"{labels.get(pr['a'], pr['a'])} − {labels.get(pr['b'], pr['b'])}: **{PT(pr['diff'])} pt** {CI(pr['ci95'])}、p = {pr['p_boot']:.4f}"
          f"（{labels.get(pr['a'], pr['a'])} {P(pr['acc_a'])}、{labels.get(pr['b'], pr['b'])} {P(pr['acc_b'])}）。",
          f"層別: GSM8K {PT(pr['by_stratum']['GSM8K']['diff'])} {CI(pr['by_stratum']['GSM8K']['ci95'])}、MATH {PT(pr['by_stratum']['MATH']['diff'])} {CI(pr['by_stratum']['MATH']['ci95'])}。",
          f"問題ごとの勝敗: {pr['problems_a_better']} 勝 / {pr['problems_b_better']} 敗 / {pr['problems_tied']} 引き分け。"]
    if "noninferiority" in pr:
        n = pr["noninferiority"]
        L.append(f"非劣性（余白 {100 * n['margin']:.1f} pt）: 区間の下限 {100 * n['ci_lower']:+.1f} pt → **{'成立' if n['established'] else '成立せず'}**。")
    L += ["", "## 各実行の正答率（層別の固定重み）", "", "| 実行 | 正答率 [95%区間] | GSM8K | MATH | 単純な全体平均（参考） | " + " | ".join(f"{b}" for b in m["blocks"]) + " |",
          "|---|---|---|---|---|" + "---|" * len(m["blocks"])]
    for t, d in R["runs"].items():
        e = d["metrics"]["acc"]
        L.append(f"| {d['label']} | {P(e['est'])} [{100 * e['ci95'][0]:.1f}, {100 * e['ci95'][1]:.1f}] | {P(d['by_stratum']['GSM8K']['acc'])} | {P(d['by_stratum']['MATH']['acc'])} | "
                 f"{P(d['pooled_accuracy_descriptive'])} | " + " | ".join(P(d["by_block"].get(b)) for b in m["blocks"]) + " |")
    L += ["", "## 打ち切りと思考の閉じ方（全体、層別の固定重み）", "", "| 実行 | 思考が閉じた割合 | 打ち切り | 打ち切りのうち 同一文の繰り返し | 他の繰り返し | 繰り返しなし |", "|---|---|---|---|---|---|"]
    for t, d in R["runs"].items():
        g = lambda k: P(d["metrics"][k]["est"]) if d["metrics"].get(k) else "n/a"
        L.append(f"| {d['label']} | {g('closed')} | {g('cut')} | {g('cut_verbatim')} | {g('cut_other_rep')} | {g('cut_none')} |")
    if R["secondary_exploratory"]:
        L += ["", "## 副比較（探索的。Holm はこの族の中だけ）", "", "| 比較 | 差 [95%区間] | p | Holm 後 p |", "|---|---|---|---|"]
        for c in R["secondary_exploratory"]:
            L.append(f"| {labels.get(c['a'], c['a'])} − {labels.get(c['b'], c['b'])} | {PT(c['diff'])} {CI(c['ci95'])} | {c['p_boot']:.4f} | {c['p_holm']:.4f} |")
    if R["descriptive_vs_base_exploratory"]:
        L += ["", "## 基底との差（探索的・記述のみ）", "", "| 実行 | 差 [95%区間] |", "|---|---|"]
        for t, d in R["descriptive_vs_base_exploratory"].items():
            L.append(f"| {labels.get(t, t)} | {PT(d['acc']['diff'])} {CI(d['acc']['ci95'])} |")
    if m["warnings"]:
        L += ["", "## 警告", ""] + [f"- {w}" for w in m["warnings"]]
    L += ["", f"解析スクリプト sha256: `{m['provenance']['script_sha256']}`", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
