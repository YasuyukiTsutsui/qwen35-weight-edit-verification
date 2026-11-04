"""Weight-only method W: choose the attention heads to edit from the WEIGHTS alone (scripts/weight_only_probe.py), no activations, no labels,
then check on accuracy (the 128 select problems x 2 samples, 16,384 tokens, seed 0; tag wo_<name>).

 w1_L15h1_zero      remove L15h1 (the strongest weight-only 'copying' head in layer 15: own token in the top-5 written logits for 88% of tokens)
 w2_L15h3_x2        double L15h3 (the strongest 'copy-suppressing' head in layer 15)
 w3_L15h1_zero_h3x2 both
 w4_L15h7_zero      control: L15h7, the head the original edit discussed; the weight-only probe ranks it 30/48 (neutral)

    python scripts/weightonly_study.py          # resumable; writes results/weightonly/summary.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_search as B  # noqa: E402
from extra_study import per_problem  # noqa: E402

ROWS = "data/practice_select128.jsonl"
VARIANTS = [("w1_L15h1_zero", "15:1:0"), ("w2_L15h3_x2", "15:3:2"), ("w3_L15h1_zero_h3x2", "15:1:0,15:3:2"), ("w4_L15h7_zero", "15:7:0")]


def run(name, edits):
    tag = f"wo_{name}"
    sp = os.path.join(B.ROOT, "results", tag, "practice_select128", "summary.json")
    if os.path.exists(sp):
        return json.load(open(sp))
    edit, log = f"models/edits/{name}", f"results/logs/{tag}.log"
    if not os.path.exists(os.path.join(B.ROOT, edit, "edit_log.json")):
        if B.sh(["python3", "scripts/make_head_edit.py", "--base", "models/Qwen3.5-0.8B", "--edits", edits, "--out", edit], log):
            raise RuntimeError(f"make_head_edit failed: {name}")
    rc = B.sh(["python3", "scripts/eval_math.py", "--model", edit, "--tag", tag, "--rows-file", ROWS, "--samples", "2", "--max-tokens", "16384",
               "--max-num-seqs", "64", "--gpu-mem", "0.7", "--seed", "0"], log)
    if rc or not os.path.exists(sp):
        raise RuntimeError(f"eval failed: {name}")
    for f in glob.glob(os.path.join(B.ROOT, edit, "*.safetensors")):
        os.remove(f)
    return json.load(open(sp))


def main():
    res = {}
    out_json = os.path.join(B.ROOT, "results", "weightonly", "summary.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    for name, edits in VARIANTS:
        s = run(name, edits)
        res[name] = {"edits": edits, "acc": s["acc"], "ok": s["ok"], "n": s["total"], "loops": s["cut_with_loop_tail"],
                     "closed": round(s["think_closed_share"] * s["total"])}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"{name}: acc {s['acc']:.3f} ({s['ok']}/{s['total']}) loops {s['cut_with_loop_tail']}", flush=True)
    rng = np.random.default_rng(0)
    res["paired"] = {}
    print("\npaired difference in accuracy (A - B), 128 problems, 95% bootstrap interval")
    for a, b in [(f"wo_{n}", "rb16k_base") for n, _ in VARIANTS] + [("wo_w1_L15h1_zero", "wo_w4_L15h7_zero"), ("wo_w3_L15h1_zero_h3x2", "wo_w1_L15h1_zero")]:
        pa, pb = per_problem(a), per_problem(b)
        d = np.array([np.mean(pa[i]) - np.mean(pb[i]) for i in sorted(pa)])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["paired"][f"{a} - {b}"] = [float(d.mean()), float(lo), float(hi)]
        print(f"  {a:24s} - {b:22s}: {100 * d.mean():+5.1f} pt [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
    json.dump(res, open(out_json, "w"), indent=1)
    print("WODONE")


if __name__ == "__main__":
    main()
