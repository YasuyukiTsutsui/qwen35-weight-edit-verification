"""Issue #31, steps A-D (all unsealed).  Host only; uses the saved 'correct' flags and predictions.

    python -I scripts/analyze_unsealed.py [A|B|C|D|E|all]      -> results/analysis_unsealed.json

Criteria fixed in Issue #31 before any run:
  A  published edit + budget forcing (4,096) - published edit  >= +3 pt with the lower end of the 95% interval > 0, on BOTH the 128 select and the 256 final_fresh problems.
  B  the non-thinking sampler with the highest accuracy on the select problems (ties: the model card) is used for E1 and step D (scripts/pick_nt_preset.py).
  C  E1 (320 GSM8K + 320 MATH): X' - R with fixed stratum weights 0.5 / 0.5 >= +4 pt and lower end > 0  ->  a sealed-block addendum is worth writing.
     N - R and X' - N are secondary (exploratory), always shown per stratum.   R = published edit, X' = R + budget forcing 4,096, N = chosen non-thinking.
  D  non-thinking maj@8 - pass@1 >= 10 pt (128 select problems)  ->  a consistency-oriented weight change is worth pursuing.
Paired problem-level bootstrap, 10,000 resamples, seed 20261105 (within strata for C).
"""
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, NBOOT = 20261105, 10000
OUT = {}


def load(tag, set_name, fname="samples.jsonl.gz"):
    p = os.path.join(ROOT, "results", tag, set_name, fname)
    if not os.path.exists(p):
        return None
    byp = {}
    for l in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
    return byp


def pv(byp, ids, f=lambda r: float(r["correct"])):
    return np.array([np.mean([f(r) for r in byp[i]]) for i in ids])


def boot(d, rng):
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), [float(x) for x in np.percentile(m, [2.5, 97.5])]


def pct(m, ci):
    return f"{100 * m:+5.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def step_a():
    rng = np.random.default_rng(SEED)
    print("A. published edit + budget forcing - published edit")
    res = {}
    for sn, edit, bf in [("practice_select128", "rb16k_p50c", "bf_pub"), ("final_fresh", "ff16k_p50c", "bf_pub_ff")]:
        e = load(edit, sn)
        ids = sorted(e)
        ev = pv(e, ids)
        for b in (4096, 8192):
            x = load(bf, sn, f"samples_B{b}.jsonl.gz")
            if x is None:
                continue
            m, ci = boot(pv(x, ids) - ev, rng)
            res[f"{sn}_B{b}"] = {"edit": float(ev.mean()), "bf": float(pv(x, ids).mean()), "diff": m, "ci": ci, "met": bool(m >= 0.03 and ci[0] > 0)}
            print(f"  {sn:20s} B={b}: edit {100 * ev.mean():.1f}% -> {100 * pv(x, ids).mean():.1f}%   diff {pct(m, ci)}   {'MET' if res[f'{sn}_B{b}']['met'] else 'not met'}")
    res["criterion_met_B4096_both_sets"] = bool(res.get("practice_select128_B4096", {}).get("met") and res.get("final_fresh_B4096", {}).get("met"))
    print("  criterion (B=4096, both sets):", res["criterion_met_B4096_both_sets"])
    OUT["A"] = res


def step_b():
    print("B. non-thinking sampler sweep (128 select problems x 2 samples, seed 2)")
    res = {}
    for preset, tag in [("T1.0 / presence 2.0 (model card)", "bl_nothink"), ("T0.7 / 0", "bl_nts_T07_p0"), ("T0.7 / 1.5", "bl_nts_T07_p15"),
                        ("T1.0 / 0", "bl_nts_T10_p0"), ("T1.0 / 1.5", "bl_nts_T10_p15")]:
        p = os.path.join(ROOT, "results", tag, "practice_select128", "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        res[preset] = {"tag": tag, "acc": s["acc"], "cut_share": s["cut_share"], "mean_tokens": s["mean_tokens"]}
        print(f"  {preset:34s} {100 * s['acc']:5.1f}%   cut {100 * s['cut_share']:4.1f}%   tokens {s['mean_tokens']:.0f}")
    bp = os.path.join(ROOT, "results", "nt_best_preset.txt")
    res["chosen"] = open(bp).read().strip() if os.path.exists(bp) else None
    print("  chosen:", res["chosen"])
    OUT["B"] = res


def strat_diff(a, b, ids):
    """per-stratum paired differences (arrays) of two {id: samples} dicts"""
    out = {}
    for st, sel in (("GSM8K", [i for i in ids if i.startswith("gsm8k")]), ("MATH", [i for i in ids if not i.startswith("gsm8k")])):
        out[st] = pv(a, sel) - pv(b, sel)
    return out


def step_c():
    rng = np.random.default_rng(SEED)
    R = load("e1_R", "e1")
    if R is None:
        return
    ids = sorted(R)
    idx = {st: rng.integers(0, n, size=(NBOOT, n)) for st, n in (("GSM8K", sum(i.startswith("gsm8k") for i in ids)), ("MATH", sum(not i.startswith("gsm8k") for i in ids)))}

    def fixed_w(d):   # d: {stratum: per-problem differences}
        m = {st: d[st][idx[st]].mean(1) for st in d}
        boots = 0.5 * m["GSM8K"] + 0.5 * m["MATH"]
        return 0.5 * d["GSM8K"].mean() + 0.5 * d["MATH"].mean(), [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    arms = {"R (published edit)": R}
    for b in (4096, 8192):
        x = load("e1_Xp", "e1", f"samples_B{b}.jsonl.gz")
        if x is not None:
            arms[f"X' (R + budget forcing {b})"] = x
    n = load("e1_N", "e1")
    if n is not None:
        arms["N (non-thinking)"] = n
    print(f"C. E1 ({len(ids)} problems; fixed stratum weights 0.5 GSM8K + 0.5 MATH)")
    res = {"arms": {}, "comparisons": {}}
    for name, a in arms.items():
        accs = {st: float(pv(a, sel).mean()) for st, sel in (("GSM8K", [i for i in ids if i.startswith("gsm8k")]), ("MATH", [i for i in ids if not i.startswith("gsm8k")]))}
        fin = float(np.mean([r["finish"] == "stop" for i in ids for r in a[i]]))
        tok = float(np.mean([r["tokens"] for i in ids for r in a[i]]))
        res["arms"][name] = {"acc_gsm8k": accs["GSM8K"], "acc_math": accs["MATH"], "acc_fixed_weight": 0.5 * accs["GSM8K"] + 0.5 * accs["MATH"], "finished": fin, "mean_tokens": tok}
        print(f"  {name:34s} GSM8K {100 * accs['GSM8K']:5.1f}%  MATH {100 * accs['MATH']:5.1f}%  weighted {100 * (accs['GSM8K'] + accs['MATH']) / 2:5.1f}%  finished {100 * fin:4.1f}%  tokens {tok:6.0f}")
    names = list(arms)
    pairs = [(n_, names[0]) for n_ in names[1:]]
    xs = [n_ for n_ in names if n_.startswith("X' (R + budget forcing 4096")]
    nn = [n_ for n_ in names if n_.startswith("N ")]
    if xs and nn:
        pairs.append((xs[0], nn[0]))
    for a_, b_ in pairs:
        d = strat_diff(arms[a_], arms[b_], ids)
        m, ci = fixed_w(d)
        ms = {st: boot(d[st], rng) for st in d}
        res["comparisons"][f"{a_} - {b_}"] = {"diff": m, "ci": ci, "gsm8k": ms["GSM8K"][0], "gsm8k_ci": ms["GSM8K"][1], "math": ms["MATH"][0], "math_ci": ms["MATH"][1]}
        flag = ""
        if a_.startswith("X' (R + budget forcing 4096") and b_.startswith("R "):
            res["criterion_met"] = bool(m >= 0.04 and ci[0] > 0)
            flag = "   <-- PRIMARY: " + ("MET (>= +4 pt, lower > 0)" if res["criterion_met"] else "not met")
        print(f"  {a_} - {b_}: {pct(m, ci)}   GSM8K {pct(*ms['GSM8K'])}   MATH {pct(*ms['MATH'])}{flag}")
    OUT["C"] = res


def step_d():
    rng = np.random.default_rng(SEED)
    K = load("bl_nt_k8", "practice_select128")
    if K is None:
        return
    ids = sorted(K)
    p1 = np.array([np.mean([r["correct"] for r in K[i]]) for i in ids])
    pk = np.array([float(any(r["correct"] for r in K[i])) for i in ids])
    maj = []
    for i in ids:
        c = Counter()
        order = {}
        for r in K[i]:
            if r["pred"] is None:
                continue
            key = " ".join(str(r["pred"]).split())
            c[key] += 1
            order.setdefault(key, len(order))
        if not c:
            maj.append(0.0)
            continue
        top = max(c.values())
        win = min((k for k in c if c[k] == top), key=lambda k: order[k])
        maj.append(float(any(r["correct"] for r in K[i] if r["pred"] is not None and " ".join(str(r["pred"]).split()) == win)))
    maj = np.array(maj)
    k = len(K[ids[0]])
    gap_m, gap_ci = boot(maj - p1, rng)
    res = {"k": k, "pass@1": float(p1.mean()), f"pass@{k}": float(pk.mean()), f"maj@{k}": float(maj.mean()), "gap_maj_minus_pass1": gap_m, "gap_ci": gap_ci,
           "criterion_met": bool(gap_m >= 0.10)}
    print(f"D. non-thinking, {k} samples on {len(ids)} select problems: pass@1 {100 * p1.mean():.1f}%  maj@{k} {100 * maj.mean():.1f}%  pass@{k} {100 * pk.mean():.1f}%")
    print(f"   maj@{k} - pass@1 = {pct(gap_m, gap_ci)}   criterion (>= +10 pt): {'MET' if res['criterion_met'] else 'not met'}   (voting by exact normalised answer string)")
    OUT["D"] = res


def step_e():
    """Descriptive (exploratory, no criterion): complementarity of N and X' on E1; per-problem pass-rate spread and maj@k vs k for the non-thinking k=8 run."""
    rng = np.random.default_rng(SEED)
    res = {}
    R, X, N = load("e1_R", "e1"), load("e1_Xp", "e1", "samples_B4096.jsonl.gz"), load("e1_N", "e1")
    if R is not None and X is not None and N is not None:
        ids = sorted(R)
        g = np.array([i.startswith("gsm8k") for i in ids])
        r, x, n = (pv(a, ids) for a in (R, X, N))
        orc = np.maximum(x, n)
        w = lambda a: float(0.5 * a[g].mean() + 0.5 * a[~g].mean())
        res["E1"] = {"N_weighted": w(n), "Xp_weighted": w(x), "R_weighted": w(r), "oracle_N_or_Xp_weighted": w(orc),
                     "only_N_correct": int(((n == 1) & (x == 0)).sum()), "only_Xp_correct": int(((x == 1) & (n == 0)).sum()), "both": int(((x == 1) & (n == 1)).sum()),
                     "neither": int(((x == 0) & (n == 0)).sum()),
                     "stratum_choice_weighted": float(0.5 * x[g].mean() + 0.5 * n[~g].mean())}
        print(f"E. E1 complementarity: N {100 * w(n):.1f}%  X' {100 * w(x):.1f}%  oracle(N or X') {100 * w(orc):.1f}%  | only N {res['E1']['only_N_correct']}, only X' {res['E1']['only_Xp_correct']}, both {res['E1']['both']}, neither {res['E1']['neither']}")
        print(f"   post-hoc stratum choice (X' on GSM8K, N on MATH): {100 * res['E1']['stratum_choice_weighted']:.1f}%")
    K = load("bl_nt_k8", "practice_select128")
    if K is not None:
        ids = sorted(K)
        k = len(K[ids[0]])
        p = np.array([np.mean([r["correct"] for r in K[i]]) for i in ids])
        res["k8"] = {"problems_never_correct": int((p == 0).sum()), "problems_always_correct": int((p == 1).sum()), "problems_mixed": int(((p > 0) & (p < 1)).sum()),
                     "mean_tokens_per_sample": float(np.mean([r["tokens"] for i in ids for r in K[i]]))}
        print(f"   k8: never correct {res['k8']['problems_never_correct']}, always {res['k8']['problems_always_correct']}, mixed {res['k8']['problems_mixed']} (of {len(ids)});  tokens/sample {res['k8']['mean_tokens_per_sample']:.0f}")

        def maj_of(i, kk):
            c, order = Counter(), {}
            sub = K[i][:kk]
            for r in sub:
                if r["pred"] is None:
                    continue
                key = " ".join(str(r["pred"]).split())
                c[key] += 1
                order.setdefault(key, len(order))
            if not c:
                return 0.0
            top = max(c.values())
            win = min((q for q in c if c[q] == top), key=lambda q: order[q])
            return float(any(r["correct"] for r in sub if r["pred"] is not None and " ".join(str(r["pred"]).split()) == win))
        res["maj_by_k"] = {}
        for kk in (1, 2, 4, 8):
            m = np.mean([maj_of(i, kk) for i in ids])
            res["maj_by_k"][kk] = float(m)
        print("   maj@k:", "  ".join(f"k={kk}: {100 * v:.1f}%" for kk, v in res["maj_by_k"].items()), "(k=1 is the first sample only)")
        gs = [j for j, i in enumerate(ids) if i.startswith("gsm8k")]
        ms = [j for j, i in enumerate(ids) if not i.startswith("gsm8k")]
        mj = np.array([maj_of(i, k) for i in ids])
        res["k8_strata"] = {"GSM8K": {"pass1": float(p[gs].mean()), "maj": float(mj[gs].mean())}, "MATH": {"pass1": float(p[ms].mean()), "maj": float(mj[ms].mean())}}
        print(f"   by stratum: GSM8K pass@1 {100 * p[gs].mean():.1f}% maj@{k} {100 * mj[gs].mean():.1f}%   MATH pass@1 {100 * p[ms].mean():.1f}% maj@{k} {100 * mj[ms].mean():.1f}%")
    OUT["E"] = res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for key, fn in (("A", step_a), ("B", step_b), ("C", step_c), ("D", step_d), ("E", step_e)):
        if which in ("all", key):
            fn()
    json.dump(OUT, open(os.path.join(ROOT, "results", f"analysis_unsealed_{which}.json"), "w"), indent=1)
