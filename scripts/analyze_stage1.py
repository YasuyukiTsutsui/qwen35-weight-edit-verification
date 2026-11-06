"""Issue #32 stage 1: the frozen comparison on E1.  Host only; uses the saved 'correct' flags and predictions.

    python -I scripts/analyze_stage1.py CAND BASE [EXTRA ...]      (tags under results/, set 'e1';  e.g.  e1s_M e1s_Nstar)
    -> results/analysis_stage1_<CAND>.json

Frozen in Issue #32 (comment 4) before any training:
  primary  CAND - BASE on E1 (640 problems, 4 samples each), fixed stratum weights 0.5 GSM8K + 0.5 MATH, problem-level paired bootstrap inside strata,
           10,000 resamples, seed 20261105.
  PASS     difference >= +3.0 pt, lower end of the two-sided 95% interval > 0, and neither stratum worse than -3.0 pt.
  FAIL     upper end of the interval < +3.0 pt (clear).     OTHERWISE: inconclusive (a second seed of CAND is run once).
  EXTRA arms (P, G, a second seed, N at T0.7 ...) are compared with BASE the same way and labelled exploratory.
Descriptive (exploratory): mean tokens, share cut at the cap, accuracy among finished samples, maj@4 - pass@1, share of problems whose 4 answers all agree.
"""
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, NBOOT = 20261105, 10000


def load(tag):
    p = os.path.join(ROOT, "results", tag, "e1", "samples.jsonl.gz")
    byp = {}
    for l in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
    for rs in byp.values():
        rs.sort(key=lambda r: r["k"])
    return byp


def norm(x):
    return " ".join(str(x).split())


def pass1(a, ids):
    return np.array([np.mean([float(r["correct"]) for r in a[i]]) for i in ids])


def vote_ok(rs):
    c = Counter(norm(r["pred"]) for r in rs if r["pred"] is not None)
    if not c:
        return 0.0
    top = max(c.values())
    first = {}
    for r in rs:
        if r["pred"] is not None:
            first.setdefault(norm(r["pred"]), len(first))
    win = min((q for q in c if c[q] == top), key=lambda q: first[q])
    return float(any(r["correct"] for r in rs if r["pred"] is not None and norm(r["pred"]) == win))


def describe(a, ids, strata):
    out = {}
    for st, sel in strata.items():
        rs = [r for i in sel for r in a[i]]
        fin = [r for r in rs if r["finish"] == "stop"]
        out[st] = {"pass1": float(np.mean([r["correct"] for r in rs])), "finished_share": len(fin) / len(rs),
                   "acc_among_finished": float(np.mean([r["correct"] for r in fin])) if fin else float("nan"),
                   "mean_tokens": float(np.mean([r["tokens"] for r in rs])),
                   "maj_k": float(np.mean([vote_ok(a[i]) for i in sel])),
                   "all_agree": float(np.mean([len({norm(r["pred"]) if r["pred"] is not None else None for r in a[i]}) == 1 for i in sel]))}
    w = lambda key: 0.5 * out["GSM8K"][key] + 0.5 * out["MATH"][key]  # noqa: E731
    out["weighted"] = {k: w(k) for k in out["GSM8K"]}
    return out


def main():
    cand, base, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    rng = np.random.default_rng(SEED)
    arms = {t: load(t) for t in [cand, base] + extra}
    ids = sorted(arms[base])
    for t, a in arms.items():
        assert sorted(a) == ids, f"{t}: different problem set"
        assert all(len(a[i]) == len(arms[base][i]) for i in ids), f"{t}: different number of samples"
    strata = {"GSM8K": [i for i in ids if i.startswith("gsm8k")], "MATH": [i for i in ids if not i.startswith("gsm8k")]}
    idx = {st: rng.integers(0, len(sel), size=(NBOOT, len(sel))) for st, sel in strata.items()}
    res = {"problems": len(ids), "samples_per_problem": len(arms[base][ids[0]]), "arms": {}, "comparisons": {}}
    print(f"E1: {len(ids)} problems ({len(strata['GSM8K'])} GSM8K + {len(strata['MATH'])} MATH), {res['samples_per_problem']} samples each; fixed weights 0.5 / 0.5")
    for t, a in arms.items():
        d = describe(a, ids, strata)
        res["arms"][t] = d
        w = d["weighted"]
        print(f"  {t:12s} GSM8K {100 * d['GSM8K']['pass1']:5.1f}%  MATH {100 * d['MATH']['pass1']:5.1f}%  weighted {100 * w['pass1']:5.1f}%  | finished {100 * w['finished_share']:5.1f}%  "
              f"acc|finished {100 * w['acc_among_finished']:5.1f}%  tokens {w['mean_tokens']:6.0f}  | maj@{res['samples_per_problem']} {100 * w['maj_k']:5.1f}%  all-agree {100 * w['all_agree']:5.1f}%")
    for t in [cand] + extra:
        d = {st: pass1(arms[t], sel) - pass1(arms[base], sel) for st, sel in ((s, [i for i in strata[s]]) for s in strata)}
        m = {st: d[st][idx[st]].mean(1) for st in d}
        boots = 0.5 * m["GSM8K"] + 0.5 * m["MATH"]
        est, ci = float(0.5 * d["GSM8K"].mean() + 0.5 * d["MATH"].mean()), [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        per = {}
        for st in d:
            lo, hi = np.percentile(d[st][idx[st]].mean(1), [2.5, 97.5])
            per[st] = {"diff": float(d[st].mean()), "ci": [float(lo), float(hi)]}
        passed = est >= 0.03 and ci[0] > 0 and all(per[st]["diff"] > -0.03 for st in per)
        clear_fail = ci[1] < 0.03
        verdict = "PASS" if passed else ("FAIL (clear)" if clear_fail else "INCONCLUSIVE")
        res["comparisons"][f"{t} - {base}"] = {"diff": est, "ci": ci, "strata": per, "verdict": verdict, "primary": t == cand}
        print(f"  {t} - {base}: {100 * est:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]   GSM8K {100 * per['GSM8K']['diff']:+.1f} [{100 * per['GSM8K']['ci'][0]:+.1f}, {100 * per['GSM8K']['ci'][1]:+.1f}]"
              f"   MATH {100 * per['MATH']['diff']:+.1f} [{100 * per['MATH']['ci'][0]:+.1f}, {100 * per['MATH']['ci'][1]:+.1f}]   -> {verdict}{'' if t == cand else '   (exploratory)'}")
    json.dump(res, open(os.path.join(ROOT, "results", f"analysis_stage1_{cand}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
