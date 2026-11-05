"""Is the 'few residual coordinates' concentration of the edit direction really ONE coordinate (CPU only, saved directions)?

For each saved direction file (layers x 1024 residual coordinates) we report, per layer:
  - share of u's squared norm on coordinate 0 and on the next 15 largest coordinates (ranks 2-16),
  - the same ranks-2-16 share for random unit vectors (Monte Carlo, 2000 draws) as the reference,
  - cosine between u and u with coordinate 0 zeroed and renormalised (what a row-restricted edit would keep).
Because the edit  dW = -alpha u (u^T W[:,S])  is rank 1, the share of ||dW||_F^2 that sits in a set R of rows equals the share of u^2 in R.

    python -I scripts/analyze_coord0.py   ->  results/analysis_coord0.json
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
rnd = rng.standard_normal((2000, 1024))
rnd = rnd ** 2 / (rnd ** 2).sum(1, keepdims=True)
rs = -np.sort(-rnd, axis=1)
ref = {"random_top1": float(rs[:, 0].mean()), "random_ranks2_16": float(rs[:, 1:16].sum(1).mean())}
out = {"reference": ref, "files": {}}
for name in ("qwen_dir", "qwen_dir_mp"):
    d = torch.load(os.path.join(ROOT, "results", name, "directions.pt"), map_location="cpu").float().numpy()
    res = {}
    for L in (0, 1, 15, 19, 23):
        u = d[L] / np.linalg.norm(d[L])
        e = u ** 2
        order = np.argsort(-e)
        v = u.copy(); v[0] = 0
        res[str(L)] = {"top_coord": int(order[0]), "coord0_share": float(e[0]), "ranks2_16_share_excluding_coord0": float(np.sort(e[1:])[::-1][:15].sum()),
                       "cos_u_vs_u_without_coord0": float(u @ v / np.linalg.norm(v)), "cos_u_vs_coord0_only": float(abs(u[0]))}
    out["files"][name] = res
json.dump(out, open(os.path.join(ROOT, "results", "analysis_coord0.json"), "w"), indent=1)
print("random: top-1 share %.4f, ranks 2-16 share %.3f" % (ref["random_top1"], ref["random_ranks2_16"]))
for n, r in out["files"].items():
    for L, x in r.items():
        print(f"{n} L{L:>2}: top coord {x['top_coord']:>4}  coord0 share {x['coord0_share']:.3f}  ranks2-16 {x['ranks2_16_share_excluding_coord0']:.3f}  cos(u, u w/o coord0) {x['cos_u_vs_u_without_coord0']:.3f}")
