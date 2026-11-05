"""Paired comparison of the Issue #28 equation variants against the centre edit on the same select128 problems (2 samples each, seed 2).

For every variant: accuracy, paired difference to the centre (vary_c0) with a problem-level cluster bootstrap (10,000 resamples of the 128 problems,
seed 20261105), loop share, closed-thinking share, and the relative weight change rho from the edit log.
Criteria were pre-stated in Issue #28:  improvement = diff >= +5 pt with CI lower bound > 0 (and validation >= centre);
efficiency = diff >= -3 pt with CI lower >= -6 at rho <= 70% of the centre's.

    python -I scripts/analyze_variants_eval.py [tag ...]   ->  results/analysis_variants_eval.json
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTRE = "vary_c0"
VARIANTS = {"eqv_c3_rest": "eq_c3_rest", "eqv_c3_only": "eq_c3_only", "eqv_lda": "eq_lda", "eqv_oblique": "eq_oblique", "eqv_gs": "eq_gs"}


def load(tag):
    p = os.path.join(ROOT, "results", tag, "practice_select128", "samples.jsonl.gz")
    recs = [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]
    byp = {}
    for r in recs:
        byp.setdefault(r["id"], []).append(r)
    return byp


def per_problem(byp, ids, f):
    return np.array([np.mean([f(r) for r in byp[i]]) for i in ids])


def main():
    tags = sys.argv[1:] or [t for t in VARIANTS if os.path.exists(os.path.join(ROOT, "results", t, "practice_select128", "samples.jsonl.gz"))]
    cen = load(CENTRE)
    ids = sorted(cen)
    rng = np.random.default_rng(20261105)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    loop = lambda r: float(r["finish"] == "length" and (r["verbatim_loop_end"] or r["repetitive_end"]))
    corr = lambda r: float(r["correct"])
    cen_c, cen_l = per_problem(cen, ids, corr), per_problem(cen, ids, loop)
    out = {"centre": {"acc": float(cen_c.mean()), "loop_share": float(cen_l.mean())}}
    print(f"centre {CENTRE}: acc {100 * cen_c.mean():.1f}%  loop {100 * cen_l.mean():.1f}%")
    for t in tags:
        v = load(t)
        assert sorted(v) == ids, t
        vc, vl = per_problem(v, ids, corr), per_problem(v, ids, loop)
        d = vc - cen_c
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pval = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        accs = vc[idx].mean(1)
        alo, ahi = np.percentile(accs, [2.5, 97.5])
        closed = float(np.mean([r["finish"] == "stop" for i in ids for r in v[i]]))
        logp = os.path.join(ROOT, "models", "edits", VARIANTS[t], "edit_log.json")
        rho = None
        if os.path.exists(logp):
            lg = json.load(open(logp))
            rho = {k.split("layers.")[1].split(".")[0]: v_["rel_change"] for k, v_ in lg["tensors"].items()}
        cen_log = json.load(open(os.path.join(ROOT, "models", "edits", "bo_k256_a6", "edit_log.json"))) if os.path.exists(os.path.join(ROOT, "models", "edits", "bo_k256_a6", "edit_log.json")) else None
        out[t] = {"acc": float(vc.mean()), "acc_ci": [float(alo), float(ahi)], "loop_share": float(vl.mean()), "finished_share": closed,
                  "diff_vs_centre": float(d.mean()), "diff_ci": [float(lo), float(hi)], "p_two_sided": float(pval), "rho": rho,
                  "n_problems_better": int((d > 0).sum()), "n_problems_worse": int((d < 0).sum())}
        print(f"{t:12s} acc {100 * vc.mean():5.1f}% [{100 * alo:.1f},{100 * ahi:.1f}]  diff {100 * d.mean():+5.1f} [{100 * lo:+.1f},{100 * hi:+.1f}] p={pval:.3f}  "
              f"loop {100 * vl.mean():.1f}%  finished {100 * closed:.1f}%  rho {rho}")
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_variants_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
