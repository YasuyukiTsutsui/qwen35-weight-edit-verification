"""Variation study: does making the fixed constants / targets of the edit variable change accuracy?

Centre = the configuration chosen by method B (results/bo/final.json: alpha*, k*; layers 15+19, old direction, top-k columns).
Every variant changes ONE thing relative to the centre and is evaluated exactly like the search objective
(128 select problems x 2 samples, 16,384-token cap), with evaluation seed 2 (the search used seed 0).
The centre is re-evaluated with the same seed (c0), so differences are not inflated by the winner's curse of the search.

  constants   a15x0.5, a15x2, a19x0.5, a19x2      alpha of ONE layer scaled (the other layer stays at alpha*)
  targets     only15, only19                       one layer removed (alpha* on the remaining layer)
              add11, add23                         a third layer added with alpha* (11: full-attention layer; 23: full-attention, large damage in the teacher-forced scan)
              randcols                             k RANDOM columns instead of the top-k by score (same alpha*, same direction)

    python scripts/vary_study.py            # resumable; writes results/vary/summary.json
Edited weights of the variants are deleted after their evaluation (edit_log.json is kept) to save disk.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_search as B  # noqa: E402

ROWS = "data/practice_select128.jsonl"
SEED = 2


def variants(al, k):
    f = lambda x: f"{x:g}"
    return [
        ("c0", None),
        ("a15x0.5", ["--layers", "15,19", "--alpha", f"15:{f(al / 2)},19:{f(al)}"]),
        ("a15x2", ["--layers", "15,19", "--alpha", f"15:{f(al * 2)},19:{f(al)}"]),
        ("a19x0.5", ["--layers", "15,19", "--alpha", f"15:{f(al)},19:{f(al / 2)}"]),
        ("a19x2", ["--layers", "15,19", "--alpha", f"15:{f(al)},19:{f(al * 2)}"]),
        ("only15", ["--layers", "15", "--alpha", f(al)]),
        ("only19", ["--layers", "19", "--alpha", f(al)]),
        ("add11", ["--layers", "11,15,19", "--alpha", f(al)]),
        ("add23", ["--layers", "15,19,23", "--alpha", f(al)]),
        ("randcols", ["--layers", "15,19", "--alpha", f(al), "--random-cols", "--seed", "1"]),
    ]


def run_variant(name, extra, al, k, out_json):
    tag = f"vary_{name}"
    sp = os.path.join(B.ROOT, "results", tag, "practice_select128", "summary.json")
    if os.path.exists(sp):
        return json.load(open(sp))
    log = f"results/logs/{tag}.log"
    if name == "c0":
        edit = f"models/edits/bo_k{k}_a{al:g}"
    else:
        edit = f"models/edits/{tag}"
        if not os.path.exists(os.path.join(B.ROOT, edit, "edit_log.json")):
            rc = B.sh(["python3", "scripts/make_edit.py", "--base", "models/Qwen3.5-0.8B", "--dir", "results/qwen_dir", "--k", str(k),
                       *extra, "--out", edit], log)
            if rc:
                raise RuntimeError(f"make_edit failed: {name}")
    rc = B.sh(["python3", "scripts/eval_math.py", "--model", edit, "--tag", tag, "--rows-file", ROWS, "--samples", "2",
               "--max-tokens", "16384", "--max-num-seqs", "64", "--gpu-mem", "0.7", "--seed", str(SEED)], log)
    if rc or not os.path.exists(sp):
        raise RuntimeError(f"eval failed: {name}")
    if name != "c0":                                    # free disk: keep edit_log.json and small files, drop the weights
        for f in glob.glob(os.path.join(B.ROOT, edit, "*.safetensors")):
            os.remove(f)
    return json.load(open(sp))


def per_problem(tag):
    f = os.path.join(B.ROOT, "results", tag, "practice_select128", "samples.jsonl.gz")
    by = {}
    for line in gzip.open(f, "rt", encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["id"], []).append(1.0 if r["correct"] else 0.0)
    return by


def main():
    ch = json.load(open(os.path.join(B.ROOT, "results", "bo", "final.json")))["chosen"]
    al, k = float(ch["alpha"]), int(ch["k"])
    print(f"centre: alpha*={al:g} k*={k}", flush=True)
    out_json = os.path.join(B.ROOT, "results", "vary", "summary.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    res = {"centre": {"alpha": al, "k": k}, "variants": {}}
    for name, extra in variants(al, k):
        s = run_variant(name, extra, al, k, out_json)
        res["variants"][name] = {"acc": s["acc"], "ok": s["ok"], "n": s["total"], "loops": s["cut_with_loop_tail"],
                                 "closed": round(s["think_closed_share"] * s["total"])}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"{name}: acc {s['acc']:.3f} ({s['ok']}/{s['total']}) loops {s['cut_with_loop_tail']}", flush=True)

    rng = np.random.default_rng(0)
    base = per_problem("vary_c0")
    ids = sorted(base)
    c0 = np.array([np.mean(base[i]) for i in ids])
    print(f"\n{'variant':10s} {'acc':>6s} {'closed':>7s} {'loops':>6s}   diff vs centre re-eval [95% CI, paired by problem]")
    for name, _ in variants(al, k):
        v = res["variants"][name]
        line = f"{name:10s} {v['acc']:6.3f} {v['closed'] / v['n']:7.3f} {v['loops']:6d}"
        if name != "c0":
            pv = per_problem(f"vary_{name}")
            d = np.array([np.mean(pv[i]) for i in ids]) - c0
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            v["diff_vs_c0"] = [float(d.mean()), float(lo), float(hi)]
            line += f"   {100 * d.mean():+5.1f} pt [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
        print(line)
    json.dump(res, open(out_json, "w"), indent=1)
    print("VARYDONE")


if __name__ == "__main__":
    main()
