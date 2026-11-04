"""Edit strength alpha that gives a chosen relative change rho = ||dW||_F / ||W||_F per layer (CPU only).

For  dW = -alpha * u (u^T W[:,S])  with a unit vector u:  ||dW||_F = alpha * ||W[:,S]^T u||,  so  alpha = rho * ||W||_F / ||W[:,S]^T u||.
Prints the value in make_edit.py's per-layer format, e.g.  10:7.92,14:7.68

    python -I scripts/rho_alpha.py --dir results/qwen_dir_mp --layers 10,14 --k 64 --rho 0.12
"""
import argparse
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_down(base, L):
    pat = re.compile(rf"layers\.{L}\.mlp\.down_proj\.weight$")
    for f in sorted(os.listdir(base)):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(base, f))
            ks = [k for k in sd if pat.search(k) and not re.search(r"vision|audio|visual|mtp", k)]
            if ks:
                return sd[ks[0]].double()
    raise KeyError(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "Qwen3.5-0.8B"))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--rho", type=float, required=True)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    U = torch.load(os.path.join(d, "directions.pt")).double()
    info = json.load(open(os.path.join(d, "direction.json")))
    out = []
    for L in [int(x) for x in a.layers.split(",")]:
        W = load_down(a.base, L)
        u = U[L] / U[L].norm()
        S = torch.tensor(info["layers"][L]["top_neurons"][:a.k])
        out.append(f"{L}:{a.rho * float(W.norm()) / float((W[:, S].T @ u).norm()):.4g}")
    print(",".join(out))


if __name__ == "__main__":
    sys.exit(main())
