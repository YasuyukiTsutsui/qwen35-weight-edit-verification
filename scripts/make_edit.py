"""Build an edited checkpoint: for each chosen layer L, on the top-k neurons S (by find_direction's score),
    W_down[:, S] <- W_down[:, S] - alpha * u_L (u_L^T W_down[:, S])
(k = 0 or 'all': every column). Everything else is copied unchanged.

    python scripts/make_edit.py --base models/Qwen3.5-0.8B --dir results/qwen_dir --layers 15,19 --k 64 --alpha 6 \
        --out models/edits/qwen_L15-19_k64_a6
Optional --random-control: replace u_L with a random unit vector orthogonal to u_L (norm-matched negative control).
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dir", required=True, help="find_direction output (directions.pt, direction.json)")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--k", default="64")
    ap.add_argument("--alpha", default="1.0", help="one value for all layers, or per layer like 15:6,19:12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--norm-match", action="store_true",
                    help="with --random-control: rescale alpha per layer so that ||dW||_F equals the real direction's")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random control direction")
    ap.add_argument("--random-cols", action="store_true",
                    help="control for the column choice: edit k RANDOM columns (seeded) instead of the top-k by score")
    ap.add_argument("--rho", default=None,
                    help="set alpha per layer so that the relative change ||dW||_F/||W||_F equals this (one value, or per layer like 15:0.16,19:0.175); "
                         "overrides --alpha. Used to compare column choices / column counts at the SAME edit size")
    a = ap.parse_args()
    U = torch.load(os.path.join(a.dir, "directions.pt")).double()
    info = json.load(open(os.path.join(a.dir, "direction.json")))
    layers = [int(x) for x in a.layers.split(",")]
    if ":" in a.alpha:
        alphas = {int(k): float(v) for k, v in (kv.split(":") for kv in a.alpha.split(","))}
    else:
        alphas = {L: float(a.alpha) for L in layers}

    rhos = None
    if a.rho is not None:
        rhos = ({int(k): float(v) for k, v in (kv.split(":") for kv in a.rho.split(","))} if ":" in a.rho else {L: float(a.rho) for L in layers})
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.base):
        p = os.path.join(a.base, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copyfile(p, os.path.join(a.out, f))
    shards = sorted(f for f in os.listdir(a.base) if f.endswith(".safetensors"))
    log = {"base": a.base, "dir": a.dir, "layers": layers, "k": a.k, "alpha": alphas, "seed": a.seed, "norm_match": a.norm_match,
           "random_control": a.random_control, "random_cols": a.random_cols, "rho": a.rho, "tensors": {}}
    g = torch.Generator().manual_seed(a.seed)
    for sh in shards:
        sd = load_file(os.path.join(a.base, sh))
        for L in layers:
            pat = re.compile(rf"(language_model|text_model)?.*\blayers\.{L}\.mlp\.down_proj\.weight$")
            keys = [k for k in sd if pat.search(k) and not re.search(r"vision|audio|visual|mtp", k)]
            if not keys:
                continue
            assert len(keys) == 1, keys
            key = keys[0]
            W = sd[key].double()
            u = U[L]
            u_real = u
            if a.random_control:
                r = torch.randn(u.shape, generator=g, dtype=torch.float64)
                r -= (r @ u) * u
                u = r / r.norm()
            if a.k in ("0", "all"):
                S = torch.arange(W.shape[1])
            elif a.random_cols:
                S = torch.randperm(W.shape[1], generator=g)[:int(a.k)]
            else:
                S = torch.tensor(info["layers"][L]["top_neurons"][:int(a.k)])
            WS = W[:, S]
            al = alphas[L]
            if rhos is not None:
                al = rhos[L] * float(W.norm()) / float((u @ WS).norm())  # ||dW||_F = alpha * ||W_S^T u||
            if a.random_control and a.norm_match:
                al = al * float((u_real @ WS).norm() / (u @ WS).norm())
            W[:, S] = WS - al * torch.outer(u, u @ WS)
            new = W.to(sd[key].dtype)
            log["tensors"][key] = {"cols": len(S), "alpha_used": al, "rel_change": float((new.double() - sd[key].double()).norm()
                                                                       / sd[key].double().norm())}
            sd[key] = new.contiguous()
        save_file(sd, os.path.join(a.out, sh), metadata={"format": "pt"})
    assert len(log["tensors"]) == len(layers), log["tensors"]
    json.dump(log, open(os.path.join(a.out, "edit_log.json"), "w"), indent=1)
    print(json.dumps(log["tensors"]))


if __name__ == "__main__":
    main()
