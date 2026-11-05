"""Non-editing baselines against the centre edit on the same 128 select problems (Issue #30).  Host only (uses the saved 'correct' flags).

    python -I scripts/analyze_baselines.py            -> results/analysis_baselines.json

Reference runs: centre edit  results/vary_c0 (2 samples, seed 2, 43.0%), base results/rb16k_base (1 sample, 13.3%).
Criteria fixed in Issue #30 before any arm was run:  'reaches the edit' = difference to the centre >= -5 pt;  'clearly worse' = upper end of the
95% interval of the difference < 0.  Paired problem-level bootstrap, 10,000 resamples of the 128 problems, seed 20261105.
"""
import gzip
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = "practice_select128"
CENTRE, BASE = "vary_c0", "rb16k_base"
ARMS = [("T10 (T1.0, card)", "bl_t10", None), ("P20 (presence 2.0, card)", "bl_p20", None), ("NT (non-thinking, card)", "bl_nothink", None)] + \
       [(f"BF{b} (base, forced stop)", "bf_base", f"samples_B{b}.jsonl.gz") for b in (2048, 4096, 8192, 12288)] + \
       [(f"BF{b} (edit, forced stop)", "bf_edit", f"samples_B{b}.jsonl.gz") for b in (2048, 4096, 8192, 12288)]


def load(tag, fname=None):
    p = os.path.join(ROOT, "results", tag, SET, fname or "samples.jsonl.gz")
    if not os.path.exists(p):
        return None
    byp = {}
    for l in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
    return byp


def vec(byp, ids, f):
    return np.array([np.mean([f(r) for r in byp[i]]) for i in ids])


def main():
    cen, base = load(CENTRE), load(BASE)
    ids = sorted(cen)
    assert sorted(base) == ids
    rng = np.random.default_rng(20261105)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    corr = lambda r: float(r["correct"])
    ci = lambda x: [float(v) for v in np.percentile(x[idx].mean(1), [2.5, 97.5])]
    cv, bv = vec(cen, ids, corr), vec(base, ids, corr)
    out = {"centre": {"acc": float(cv.mean()), "acc_ci": ci(cv)}, "base": {"acc": float(bv.mean()), "acc_ci": ci(bv)}, "arms": {}}
    print(f"centre {100 * cv.mean():.1f}% {[round(100 * x, 1) for x in ci(cv)]}   base {100 * bv.mean():.1f}% {[round(100 * x, 1) for x in ci(bv)]}")
    print(f"{'arm':30s} {'acc':>6s}  {'[95% CI]':>14s}  {'vs centre':>10s}  {'[95% CI]':>14s}  {'vs base':>8s}  finished  tokens  verdict")
    for name, tag, fname in ARMS:
        v = load(tag, fname)
        if v is None:
            continue
        assert sorted(v) == ids, name
        x = vec(v, ids, corr)
        dc, db = x - cv, x - bv
        dlo, dhi = np.percentile(dc[idx].mean(1), [2.5, 97.5])
        alo, ahi = ci(x)
        fin = float(np.mean([r["finish"] == "stop" for i in ids for r in v[i]]))
        tok = float(np.mean([r["tokens"] for i in ids for r in v[i]]))
        forced = float(np.mean([r.get("forced", False) for i in ids for r in v[i]]))
        verdict = "reaches the edit" if dc.mean() >= -0.05 else ("clearly worse" if dhi < 0 else "between")
        out["arms"][name] = {"tag": tag, "file": fname, "acc": float(x.mean()), "acc_ci": [alo, ahi], "diff_vs_centre": float(dc.mean()),
                             "diff_vs_centre_ci": [float(dlo), float(dhi)], "diff_vs_base": float(db.mean()), "finished_share": fin,
                             "mean_tokens": tok, "forced_share": forced, "verdict": verdict}
        print(f"{name:30s} {100 * x.mean():5.1f}%  [{100 * alo:5.1f},{100 * ahi:5.1f}]  {100 * dc.mean():+9.1f}  [{100 * dlo:+6.1f},{100 * dhi:+6.1f}]  {100 * db.mean():+7.1f}  "
              f"{100 * fin:6.1f}%  {tok:6.0f}  {verdict}")
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
