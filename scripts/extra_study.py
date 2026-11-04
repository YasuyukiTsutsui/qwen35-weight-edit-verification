"""Extra studies approved by the user (Issue #14 layer follow-ups, Issue #13 data-efficiency of the direction).

All evaluated like the search objective: the 128 select problems x 2 samples, 16,384-token cap, seed 0, tag ext_<name>.
Edited weights are deleted after the evaluation (edit_log.json is kept).

 #14  x_L10_14_r06       layers 10+14, matched-pair direction, relative change 6% (the screen used 12%)      -> is 12% simply too strong?
      x_L11_r12          layer 11 alone (a full-attention layer), 12%                                       -> layer type
      x_L10_14_rand_r12  layers 10+14, norm-matched random orthogonal direction, 12%                        -> is 25.8% from the direction?
 #13  x_n4_L15_19_r12    layers 15+19, direction built from 4 loops (instead of 64), 12%                    -> how little data is enough?
      x_n16_L15_19_r12   same with 16 loops
    (the 64-loop reference is scr_t1_mp_L15_19_r12, 38.3%)

    python scripts/extra_study.py        # resumable; writes results/extra/summary.json
"""
import glob
import gzip
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bo_search as B  # noqa: E402

ROWS = "data/practice_select128.jsonl"
MP = "results/qwen_dir_mp"


def alpha_for(direction, layers, rho, k=64):
    r = subprocess.run([sys.executable, "-I", "scripts/rho_alpha.py", "--dir", direction, "--layers", layers, "--k", str(k), "--rho", str(rho)],
                       cwd=B.ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def build_direction(n):
    out = f"{MP}_n{n}"
    if os.path.exists(os.path.join(B.ROOT, out, "direction.json")):
        return out
    log = f"results/logs/ext_dir_n{n}.log"
    rc = B.sh(["python3", "scripts/find_direction_mp.py", "--model", "models/Qwen3.5-0.8B", "--samples", "results/qwen_base/practice_find/samples.jsonl.gz",
               "--old-dir", "results/qwen_dir", "--pub", "models/Qwen3.5-0.8B-p50c", "--out", out, "--train", str(n), "--test", "0", "--good", "0",
               "--skip-scan"], log)
    if rc or not os.path.exists(os.path.join(B.ROOT, out, "direction.json")):
        raise RuntimeError(f"direction build failed: n={n}")
    return out


def plans():
    yield "x_L10_14_r06", lambda: ["--dir", MP, "--layers", "10,14", "--alpha", alpha_for(MP, "10,14", 0.06)]
    yield "x_L11_r12", lambda: ["--dir", MP, "--layers", "11", "--alpha", alpha_for(MP, "11", 0.12)]
    yield "x_L10_14_rand_r12", lambda: ["--dir", MP, "--layers", "10,14", "--alpha", alpha_for(MP, "10,14", 0.12),
                                         "--random-control", "--norm-match", "--seed", "1"]
    for n in (16, 4):
        yield f"x_n{n}_L15_19_r12", (lambda n=n: (lambda d: ["--dir", d, "--layers", "15,19", "--alpha", alpha_for(d, "15,19", 0.12)])(build_direction(n)))


def run(name, args_fn):
    tag = f"ext_{name}"
    sp = os.path.join(B.ROOT, "results", tag, "practice_select128", "summary.json")
    if os.path.exists(sp):
        return json.load(open(sp))
    edit, log = f"models/edits/{name}", f"results/logs/{tag}.log"
    if not os.path.exists(os.path.join(B.ROOT, edit, "edit_log.json")):
        rc = B.sh(["python3", "scripts/make_edit.py", "--base", "models/Qwen3.5-0.8B", "--k", "64", *args_fn(), "--out", edit], log)
        if rc:
            raise RuntimeError(f"make_edit failed: {name}")
    rc = B.sh(["python3", "scripts/eval_math.py", "--model", edit, "--tag", tag, "--rows-file", ROWS, "--samples", "2", "--max-tokens", "16384",
               "--max-num-seqs", "64", "--gpu-mem", "0.7", "--seed", "0"], log)
    if rc or not os.path.exists(sp):
        raise RuntimeError(f"eval failed: {name}")
    for f in glob.glob(os.path.join(B.ROOT, edit, "*.safetensors")):
        os.remove(f)
    return json.load(open(sp))


def per_problem(tag):
    by = {}
    for line in gzip.open(os.path.join(B.ROOT, "results", tag, "practice_select128", "samples.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["id"], []).append(1.0 if r["correct"] else 0.0)
    return by


def main():
    res = {}
    out_json = os.path.join(B.ROOT, "results", "extra", "summary.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    for name, fn in plans():
        s = run(name, fn)
        res[name] = {"acc": s["acc"], "ok": s["ok"], "n": s["total"], "loops": s["cut_with_loop_tail"], "closed": round(s["think_closed_share"] * s["total"])}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"{name}: acc {s['acc']:.3f} ({s['ok']}/{s['total']}) loops {s['cut_with_loop_tail']}", flush=True)

    # paired comparisons (128 problems, bootstrap over problems)
    rng = np.random.default_rng(0)
    pairs = [("ext_x_L10_14_r06", "scr_t1_L10_14_r12"), ("ext_x_L10_14_r06", "rb16k_base"),
             ("ext_x_L11_r12", "scr_t1_L10_r12"), ("ext_x_L11_r12", "scr_t1_L14_r12"), ("ext_x_L11_r12", "rb16k_base"),
             ("ext_x_L10_14_rand_r12", "scr_t1_L10_14_r12"), ("ext_x_L10_14_rand_r12", "rb16k_base"),
             ("ext_x_n16_L15_19_r12", "scr_t1_mp_L15_19_r12"), ("ext_x_n4_L15_19_r12", "scr_t1_mp_L15_19_r12"),
             ("ext_x_n16_L15_19_r12", "rb16k_base"), ("ext_x_n4_L15_19_r12", "rb16k_base")]
    res["paired"] = {}
    print("\npaired difference in accuracy (A - B), 128 problems, 95% bootstrap interval")
    for a, b in pairs:
        pa, pb = per_problem(a), per_problem(b)
        ids = sorted(pa)
        d = np.array([np.mean(pa[i]) - np.mean(pb[i]) for i in ids])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["paired"][f"{a} - {b}"] = [float(d.mean()), float(lo), float(hi)]
        print(f"  {a:26s} - {b:24s}: {100 * d.mean():+5.1f} pt [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")

    # data efficiency of the direction: agreement with the 64-loop direction and overlap of the chosen columns
    import torch
    ref_u = torch.load(os.path.join(B.ROOT, MP, "directions.pt")).double()
    ref_i = json.load(open(os.path.join(B.ROOT, MP, "direction.json")))
    res["direction_vs_n64"] = {}
    print("\ndirection built from n loops vs the 64-loop direction")
    for n in (4, 16):
        d = f"{MP}_n{n}"
        u = torch.load(os.path.join(B.ROOT, d, "directions.pt")).double()
        info = json.load(open(os.path.join(B.ROOT, d, "direction.json")))
        row = {}
        for L in (15, 19):
            cos = float(u[L] @ ref_u[L] / (u[L].norm() * ref_u[L].norm()))
            ov = len(set(info["layers"][L]["top_neurons"][:64]) & set(ref_i["layers"][L]["top_neurons"][:64]))
            row[L] = {"cos": cos, "top64_overlap": ov}
            print(f"  n={n:2d} layer {L}: cos {cos:.3f}, top-64 column overlap {ov}/64")
        res["direction_vs_n64"][n] = row
    json.dump(res, open(out_json, "w"), indent=1)
    print("EXTRADONE")


if __name__ == "__main__":
    main()
