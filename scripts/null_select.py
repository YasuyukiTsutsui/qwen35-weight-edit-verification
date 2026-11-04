"""Null-selection control for method B: run the SAME selection procedure on random directions.

For the chosen (alpha, k) of method B, build n edits whose direction is a random unit vector orthogonal to the real one, with
alpha rescaled per layer so that ||dW||_F equals the real edit's (make_edit.py --random-control --norm-match --seed s),
evaluate each on the search objective (same problems, samples, cap), and report the best one. Its accuracy shows how much of
"best of n" is selection noise rather than a real effect; the best random edit is also carried to the final comparison.

    python scripts/null_select.py --n 5            # reads results/bo/final.json (chosen) unless --alpha/--k are given
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_search as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rows", default="data/practice_select128.jsonl")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--dir", default="results/qwen_dir")
    a = ap.parse_args()
    if a.alpha is None:
        ch = json.load(open(os.path.join(B.ROOT, "results", "bo", "final.json")))["chosen"]
        a.alpha, a.k = ch["alpha"], ch["k"]
    out = os.path.join(B.ROOT, "results", "bo", "null.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["seed"] for r in res}
    for s in range(1, a.n + 1):
        if s in done:
            continue
        name = f"null_k{a.k}_a{a.alpha:g}_s{s}"
        edit = f"models/edits/{name}"
        log = f"results/logs/{name}.log"
        if not os.path.exists(os.path.join(B.ROOT, edit, "edit_log.json")):
            rc = B.sh(["python3", "scripts/make_edit.py", "--base", "models/Qwen3.5-0.8B", "--dir", a.dir, "--layers", "15,19", "--k", str(a.k),
                       "--alpha", f"{a.alpha:g}", "--random-control", "--norm-match", "--seed", str(s), "--out", edit], log)
            if rc:
                raise RuntimeError(f"make_edit failed: {name}")
        tag = f"{name}"
        rc = B.sh(["python3", "scripts/eval_math.py", "--model", edit, "--tag", tag, "--rows-file", a.rows, "--samples", str(a.samples),
                   "--max-tokens", "16384", "--max-num-seqs", "64", "--gpu-mem", "0.7"], log)
        sp = os.path.join(B.ROOT, "results", tag, os.path.splitext(os.path.basename(a.rows))[0], "summary.json")
        if rc or not os.path.exists(sp):
            raise RuntimeError(f"eval failed: {name}")
        sm = json.load(open(sp))
        res.append({"seed": s, "name": name, "ok": sm["ok"], "n_seq": sm["total"], "acc": sm["acc"], "loops": sm["cut_with_loop_tail"]})
        json.dump(res, open(out, "w"), indent=1)
        print(f"{name}: acc {sm['acc']:.3f} ({sm['ok']}/{sm['total']}), loops {sm['cut_with_loop_tail']}", flush=True)
    best = max(res, key=lambda r: r["acc"])
    print("NULL best of", len(res), json.dumps(best))


if __name__ == "__main__":
    main()
