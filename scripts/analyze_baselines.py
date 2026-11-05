"""Non-editing baselines against the edited model on the same problems (Issue #30).  Host only (uses the saved 'correct' flags).

    python -I scripts/analyze_baselines.py [practice_select128|final_fresh]      -> results/analysis_baselines_<set>.json

practice_select128 (128 problems): edit = centre edit results/vary_c0 (2 samples, seed 2, 43.0%), base = results/rb16k_base (1 sample).
final_fresh (256 problems, used up for confirmation -> exploratory only): edit = method B results/ff16k_B, base = results/ff16k_base (1 sample each).
Criteria fixed in Issue #30 before any arm was run:  'reaches the edit' = difference to the edit >= -5 pt;  'clearly worse' = upper end of the
95% interval of the difference < 0.  Stacking (stage 2): 'improves' = budget-forced edit minus the edit alone >= +3 pt with the lower end of the
95% interval > 0, required on both problem sets.  Paired problem-level bootstrap, 10,000 resamples, seed 20261105 (all arms share the resamples).
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {
    "practice_select128": dict(edit="vary_c0", base="rb16k_base", refs=[], budgets=(2048, 4096, 8192, 12288), bf_base="bf_base", bf_edit="bf_edit",
                               plain=[("T10 (T1.0, card)", "bl_t10"), ("P20 (presence 2.0, card)", "bl_p20"), ("NT (non-thinking, card)", "bl_nothink"),
                                      ("NT edit (non-thinking, edited)", "bl_nt_edit")]),
    "final_fresh": dict(edit="ff16k_B", base="ff16k_base", refs=[("published edit", "ff16k_p50c"), ("own alpha6", "ff16k_own_a6"), ("method A", "ff16k_A")],
                        budgets=(4096, 8192), bf_base="bf_base_ff", bf_edit="bf_edit_ff",
                        plain=[("NT (non-thinking, card)", "bl_nt_ff"), ("NT edit (non-thinking, edited)", "bl_nt_edit_ff")]),
}


def load(tag, set_name, fname=None):
    p = os.path.join(ROOT, "results", tag, set_name, fname or "samples.jsonl.gz")
    if not os.path.exists(p):
        return None
    byp = {}
    for l in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
    return byp


def vec(byp, ids, f):
    return np.array([np.mean([f(r) for r in byp[i]]) for i in ids])


def pooled():
    """Stage 2 pooled over both problem sets (384 problems, bootstrap stratified by set): the differences that decide the stacking question."""
    sets = ["practice_select128", "final_fresh"]
    rng = np.random.default_rng(20261105)
    data = {}
    for sn in sets:
        c = CFG[sn]
        ids = sorted(load(c["edit"], sn))
        data[sn] = (ids, {k: vec(load(t, sn, f), ids, lambda r: float(r["correct"]))
                          for k, (t, f) in {"edit": (c["edit"], None), "base": (c["base"], None),
                                            "bf_base": (c["bf_base"], "samples_B4096.jsonl.gz"), "bf_edit": (c["bf_edit"], "samples_B4096.jsonl.gz")}.items()})
    idx = {sn: rng.integers(0, len(data[sn][0]), size=(10000, len(data[sn][0]))) for sn in sets}
    def stat(a, b):
        per = [(data[sn][1][a] - data[sn][1][b]) for sn in sets]
        boots = np.concatenate([p[idx[sn]].sum(1)[:, None] for p, sn in zip(per, sets)], axis=1).sum(1) / sum(len(p) for p in per)
        return float(np.concatenate(per).mean()), [float(x) for x in np.percentile(boots, [2.5, 97.5])]
    print("pooled over practice_select128 + final_fresh (384 problems), budget B = 4,096")
    out = {}
    for a, b, label in [("bf_edit", "edit", "budget-forced edit - edit alone"), ("bf_edit", "bf_base", "budget-forced edit - budget-forced base"),
                        ("bf_base", "edit", "budget-forced base - edit alone"), ("edit", "base", "edit alone - base")]:
        m, (lo, hi) = stat(a, b)
        out[label] = {"diff": m, "ci": [lo, hi]}
        print(f"  {label:42s} {100 * m:+5.1f} pt  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_baselines_pooled.json"), "w"), indent=1)


def main():
    set_name = sys.argv[1] if len(sys.argv) > 1 else "practice_select128"
    if set_name == "pooled":
        return pooled()
    c = CFG[set_name]
    edit, base = load(c["edit"], set_name), load(c["base"], set_name)
    ids = sorted(edit)
    assert sorted(base) == ids
    rng = np.random.default_rng(20261105)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    corr = lambda r: float(r["correct"])
    ci = lambda x: [float(v) for v in np.percentile(x[idx].mean(1), [2.5, 97.5])]
    ev, bv = vec(edit, ids, corr), vec(base, ids, corr)
    out = {"set": set_name, "n_problems": len(ids), "edit": {"tag": c["edit"], "acc": float(ev.mean()), "acc_ci": ci(ev)},
           "base": {"tag": c["base"], "acc": float(bv.mean()), "acc_ci": ci(bv)}, "arms": {}}
    f1 = lambda v: [round(100 * x, 1) for x in v]
    print(f"[{set_name}, {len(ids)} problems]  edit {c['edit']} {100 * ev.mean():.1f}% {f1(ci(ev))}   base {c['base']} {100 * bv.mean():.1f}% {f1(ci(bv))}")
    for name, tag in c["refs"]:
        v = load(tag, set_name)
        x = vec(v, ids, corr)
        print(f"  reference {name:16s} {100 * x.mean():5.1f}%")
        out["arms"][name] = {"tag": tag, "acc": float(x.mean())}
    print(f"{'arm':30s} {'acc':>6s}  {'[95% CI]':>14s}  {'vs edit':>8s}  {'[95% CI]':>14s}  {'vs base':>8s}  {'vs BF-base':>10s}  finished  tokens  verdict")
    arms = [(n, t, None) for n, t in c["plain"]]
    arms += [(f"BF{b} base", c["bf_base"], f"samples_B{b}.jsonl.gz") for b in c["budgets"]]
    arms += [(f"BF{b} edit", c["bf_edit"], f"samples_B{b}.jsonl.gz") for b in c["budgets"]]
    vecs = {}
    for name, tag, fname in arms:
        v = load(tag, set_name, fname)
        if v is None:
            continue
        assert sorted(v) == ids, name
        x = vec(v, ids, corr)
        vecs[name] = x
        de, db = x - ev, x - bv
        lo, hi = np.percentile(de[idx].mean(1), [2.5, 97.5])
        alo, ahi = ci(x)
        fin = float(np.mean([r["finish"] == "stop" for i in ids for r in v[i]]))
        tok = float(np.mean([r["tokens"] for i in ids for r in v[i]]))
        verdict = "reaches the edit" if de.mean() >= -0.05 else ("clearly worse" if hi < 0 else "between")
        rec = {"tag": tag, "file": fname, "acc": float(x.mean()), "acc_ci": [alo, ahi], "diff_vs_edit": float(de.mean()), "diff_vs_edit_ci": [float(lo), float(hi)],
               "diff_vs_base": float(db.mean()), "finished_share": fin, "mean_tokens": tok, "verdict_vs_edit": verdict}
        vs_bf = ""
        if name.endswith(" edit") and name.replace(" edit", " base") in vecs:        # stacking: budget-forced edit vs budget-forced base
            d2 = x - vecs[name.replace(" edit", " base")]
            l2, h2 = np.percentile(d2[idx].mean(1), [2.5, 97.5])
            rec["diff_vs_bf_base"], rec["diff_vs_bf_base_ci"] = float(d2.mean()), [float(l2), float(h2)]
            rec["stack_improves_edit"] = bool(de.mean() >= 0.03 and lo > 0)
            vs_bf = f"{100 * d2.mean():+.1f} [{100 * l2:+.1f},{100 * h2:+.1f}]"
        out["arms"][name] = rec
        print(f"{name:30s} {100 * x.mean():5.1f}%  [{100 * alo:5.1f},{100 * ahi:5.1f}]  {100 * de.mean():+7.1f}  [{100 * lo:+6.1f},{100 * hi:+6.1f}]  {100 * db.mean():+7.1f}  "
              f"{vs_bf:>10s}  {100 * fin:6.1f}%  {tok:6.0f}  {verdict}" + ("  STACK IMPROVES" if rec.get("stack_improves_edit") else ""))
    json.dump(out, open(os.path.join(ROOT, "results", f"analysis_baselines_{set_name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
