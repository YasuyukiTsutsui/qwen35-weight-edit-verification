"""Follow-up to the variation study (Issue #9): two of its results were confounded by edit SIZE.
  randcols  (14.5%, base-like): the same strength alpha on 256 random columns gives only about half the relative change
            (rho 7.3% / 7.6% vs 16.0% / 17.5% for the top-256 columns), so 'wrong columns' and 'too weak' are mixed.
  allcols   (44.9%, = centre): alpha=6 on all 3,584 columns gives rho 26.6% / 30.0%, i.e. a much larger edit for the same accuracy.
Here the SIZE is fixed at the centre's (make_edit.py --rho, per layer 15:0.1598,19:0.1746) and only the columns change:
  randcols_rho_s1 / _s2   256 random columns (seeds 1, 2), rescaled to the centre's rho
  allcols_rho             all columns, rescaled to the centre's rho
Same problems, samples, cap and seed as the variation study (select128 x 2, 16,384, seed 2); compared with the centre re-evaluation c0.

    python scripts/vary_followup.py        # resumable; writes results/vary/followup.json; prints FOLLOWDONE
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_search as B  # noqa: E402
from vary_study import per_problem, run_variant  # noqa: E402

RHO = "15:0.1598,19:0.1746"   # relative change of the centre (alpha 6, top-256 columns); from models/edits/vary_only15 and vary_add23 edit logs
VARIANTS = [
    ("randcols_rho_s1", ["--layers", "15,19", "--random-cols", "--seed", "1", "--rho", RHO]),
    ("randcols_rho_s2", ["--layers", "15,19", "--random-cols", "--seed", "2", "--rho", RHO]),
    ("allcols_rho", ["--layers", "15,19", "--k", "all", "--rho", RHO]),
]


def main():
    ch = json.load(open(os.path.join(B.ROOT, "results", "bo", "final.json")))["chosen"]
    al, k = float(ch["alpha"]), int(ch["k"])
    out_json = os.path.join(B.ROOT, "results", "vary", "followup.json")
    res = {"centre": {"alpha": al, "k": k, "rho": RHO}, "variants": {}}
    for name, extra in VARIANTS:
        s = run_variant(name, extra, al, k, out_json)
        res["variants"][name] = {"acc": s["acc"], "ok": s["ok"], "n": s["total"], "loops": s["cut_with_loop_tail"],
                                 "closed": round(s["think_closed_share"] * s["total"])}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"{name}: acc {s['acc']:.3f} ({s['ok']}/{s['total']}) loops {s['cut_with_loop_tail']}", flush=True)
    rng = np.random.default_rng(0)
    c0 = per_problem("vary_c0")
    ids = sorted(c0)
    base = np.array([np.mean(c0[i]) for i in ids])
    print("\nvariant           acc  closed  loops   diff vs centre re-eval [95% CI, paired by problem]")
    for name, _ in VARIANTS:
        v = res["variants"][name]
        pv = per_problem(f"vary_{name}")
        d = np.array([np.mean(pv[i]) for i in ids]) - base
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        v["diff_vs_c0"] = [float(d.mean()), float(lo), float(hi)]
        print(f"{name:16s} {v['acc']:.3f}  {v['closed'] / v['n']:.3f}  {v['loops']:5d}   {100 * d.mean():+5.1f} pt [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
    json.dump(res, open(out_json, "w"), indent=1)
    print("FOLLOWDONE")


if __name__ == "__main__":
    main()
