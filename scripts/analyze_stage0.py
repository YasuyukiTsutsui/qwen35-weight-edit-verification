"""Issue #32, stage 0 (weight-free).  How much of the 'majority-vote headroom' (maj@8 - pass@1) of the non-thinking model is left once the sampler is colder?

    python -I scripts/analyze_stage0.py      -> results/analysis_stage0.json, results/stage0_tstar.txt

Rules fixed in Issue #32 before any stage-0 run:
  * candidates: non-thinking, presence 1.5, top_p 1.0, top_k 20, temperature 0.7 (the existing 8-sample run 'bl_nt_k8'), 0.5 and 0.3 (new runs), each
    128 select problems x 8 samples, seed 2;
  * T* = the temperature with the highest pass@1 (mean correct over all 1,024 samples); a tie goes to 0.7 (strict '>' while scanning 0.7, 0.5, 0.3);
  * headroom(T*) = maj@8 - pass@1 at T* (same samples; vote = normalised answer string, ties -> first seen; problem-level bootstrap, 10,000, seed 20261105);
  * headroom(T*) >= 7 pt  -> a training design (stage 1) may be written and must be reviewed again;   < 7 pt -> stop the weight direction for 'consistency'.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_unsealed as U  # noqa: E402

CANDS = [("0.7", "qwen35_nt_T07_p15", "bl_nt_k8"), ("0.5", "qwen35_nt_T05_p15", "bl_s0_T05"), ("0.3", "qwen35_nt_T03_p15", "bl_s0_T03")]
S = "practice_select128"
THRESH = 0.07


def norm(x):
    return " ".join(str(x).split())


def vote_correct(rs):
    c, order = Counter(), {}
    for r in rs:
        if r["pred"] is None:
            continue
        k = norm(r["pred"])
        c[k] += 1
        order.setdefault(k, len(order))
    if not c:
        return 0.0
    top = max(c.values())
    win = min((k for k in c if c[k] == top), key=lambda k: order[k])
    return float(any(r["correct"] for r in rs if r["pred"] is not None and norm(r["pred"]) == win))


def main():
    rng = np.random.default_rng(U.SEED)
    res, best = {}, None
    print("stage 0: non-thinking, presence 1.5, 128 select problems x 8 samples, seed 2")
    for t, preset, tag in CANDS:
        K = U.load(tag, S)
        if K is None:
            print(f"  T={t}: missing ({tag})")
            continue
        ids = sorted(K)
        p1 = np.array([np.mean([r["correct"] for r in K[i]]) for i in ids])
        mj = np.array([vote_correct(K[i]) for i in ids])
        pk = np.array([float(any(r["correct"] for r in K[i])) for i in ids])
        gap, ci = U.boot(mj - p1, rng)
        allr = [r for i in ids for r in K[i]]
        cut = 1 - float(np.mean([r["finish"] == "stop" for r in allr]))
        tok = float(np.mean([r["tokens"] for r in allr]))
        res[t] = {"preset": preset, "tag": tag, "pass@1": float(p1.mean()), "maj@8": float(mj.mean()), "pass@8": float(pk.mean()), "headroom": gap, "headroom_ci": ci,
                  "cut_share": cut, "mean_tokens": tok, "n_samples": len(allr)}
        print(f"  T={t}: pass@1 {100 * p1.mean():5.1f}%  maj@8 {100 * mj.mean():5.1f}%  pass@8 {100 * pk.mean():5.1f}%  headroom {U.pct(gap, ci)}  cut {100 * cut:4.1f}%  tokens {tok:5.0f}  (n={len(allr)})")
        if best is None or p1.mean() > res[best]["pass@1"]:      # strict '>' keeps the earlier one (0.7 first) on ties
            best = t
    if best is None or len(res) < 3:
        print("incomplete")
        return
    h = res[best]["headroom"]
    res["T_star"] = best
    res["preset_star"] = res[best]["preset"]
    res["verdict"] = "PROCEED (design stage 1)" if h >= THRESH else "STOP (weight direction for consistency)"
    print(f"  T* = {best} (preset {res[best]['preset']});  headroom(T*) = {100 * h:.1f} pt  ->  {res['verdict']}   (threshold {100 * THRESH:.0f} pt)")
    json.dump(res, open(os.path.join(ROOT, "results", "analysis_stage0.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "stage0_tstar.txt"), "w").write(res[best]["preset"] + "\n")


if __name__ == "__main__":
    main()
