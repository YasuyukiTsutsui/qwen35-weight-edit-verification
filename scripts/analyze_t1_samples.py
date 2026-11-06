"""Issue #32 stage 1 (descriptive, exploratory): the 8-sample run on the unsealed training problems T1 -- is the consistency headroom (maj@8 - pass@1) there on problems
that no sampler setting was tuned on?  Host only.

    python -I scripts/analyze_t1_samples.py [DIR ...]      default: results/t1_samples/t1     -> results/analysis_t1_samples.json
Vote = normalised answer string, ties to the first-seen answer (as in analyze_unsealed.py step D).  Bootstrap over problems, 10,000 resamples, seed 20261105.
"""
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, NBOOT = 20261105, 10000
dirs = sys.argv[1:] or ["results/t1_samples/t1"]
byp = {}
for d in dirs:
    for l in gzip.open(os.path.join(ROOT, d, "samples.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
for rs in byp.values():
    rs.sort(key=lambda r: r["k"])
norm = lambda x: " ".join(str(x).split())  # noqa: E731


def vote_ok(rs):
    c = Counter(norm(r["pred"]) for r in rs if r["pred"] is not None)
    if not c:
        return 0.0
    first = {}
    for r in rs:
        if r["pred"] is not None:
            first.setdefault(norm(r["pred"]), len(first))
    top = max(c.values())
    win = min((q for q in c if c[q] == top), key=lambda q: first[q])
    return float(any(r["correct"] for r in rs if r["pred"] is not None and norm(r["pred"]) == win))


ids = sorted(byp)
p1 = np.array([np.mean([float(r["correct"]) for r in byp[i]]) for i in ids])
mj = np.array([vote_ok(byp[i]) for i in ids])
p8 = np.array([float(any(r["correct"] for r in byp[i])) for i in ids])
rng = np.random.default_rng(SEED)
res = {"problems": len(ids), "samples_per_problem": len(byp[ids[0]]), "strata": {}}
for name, sel in (("all", np.ones(len(ids), bool)), ("GSM8K", np.array([i.startswith("gsm8k") for i in ids])), ("MATH", np.array([not i.startswith("gsm8k") for i in ids]))):
    d = (mj - p1)[sel]
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    ci = [float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])]
    allcorrect = float(np.mean(p1[sel] == 1)); allwrong = float(np.mean(p8[sel] == 0))
    res["strata"][name] = {"n": int(sel.sum()), "pass@1": float(p1[sel].mean()), "maj@8": float(mj[sel].mean()), "pass@8": float(p8[sel].mean()), "headroom": float(d.mean()), "headroom_ci": ci,
                           "all_correct": allcorrect, "all_wrong": allwrong, "mixed": 1 - allcorrect - allwrong}
    print(f"{name:6s} n={int(sel.sum()):4d}  pass@1 {100 * p1[sel].mean():5.1f}%  maj@8 {100 * mj[sel].mean():5.1f}%  pass@8 {100 * p8[sel].mean():5.1f}%  maj@8 - pass@1 = {100 * d.mean():+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
          f"   all-correct {100 * allcorrect:.0f}%  all-wrong {100 * allwrong:.0f}%  mixed {100 * (1 - allcorrect - allwrong):.0f}%")
allr = [r for rs in byp.values() for r in rs]
res["cut_share"] = float(np.mean([r["finish"] == "length" for r in allr]))
res["mean_tokens"] = float(np.mean([r["tokens"] for r in allr]))
print(f"cut at the cap {100 * res['cut_share']:.1f}%  mean tokens {res['mean_tokens']:.0f}")
json.dump(res, open(os.path.join(ROOT, "results", "analysis_t1_samples.json"), "w"), indent=1)
