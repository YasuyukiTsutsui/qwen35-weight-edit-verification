"""How concentrated is the edit direction u over the 1024 residual-stream coordinates (the ROW side of down_proj)?  CPU only.

The edit is dW = -alpha * u (u^T W[:, S]): u is dense over rows, the sparsity is on the columns S (MLP neurons).  If u were spread evenly, "which rows"
would be meaningless; if it sits on a few coordinates, a row-restricted edit becomes a natural variant (Issue "row-sparse edit").
Reference: random unit vectors in 1024 dimensions.   python -I scripts/analyze_direction_concentration.py  -> results/analysis_direction_concentration.json
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
out = {}
for d in ("qwen_dir", "qwen_dir_mp"):
    U = torch.load(os.path.join(ROOT, "results", d, "directions.pt")).double().numpy()
    for L in (15, 19):
        u = U[L] / np.linalg.norm(U[L])
        e = np.sort(u ** 2)[::-1]
        rv = rng.standard_normal((200, u.size))
        rv /= np.linalg.norm(rv, axis=1, keepdims=True)
        rs = np.sort(rv ** 2, axis=1)[:, ::-1]
        out[f"{d}/L{L}"] = {"energy_top16": float(e[:16].sum()), "energy_top64": float(e[:64].sum()), "energy_top256": float(e[:256].sum()),
                            "participation_ratio": float(1 / np.sum(u ** 4)), "top_coords": np.argsort(-u ** 2)[:8].tolist(),
                            "random_energy_top16": float(rs[:, :16].sum(1).mean()), "random_energy_top64": float(rs[:, :64].sum(1).mean()),
                            "random_energy_top256": float(rs[:, :256].sum(1).mean()), "random_participation_ratio": float(np.mean(1 / np.sum(rv ** 4, axis=1)))}
        print(d, L, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[f"{d}/L{L}"].items()})
json.dump(out, open(os.path.join(ROOT, "results", "analysis_direction_concentration.json"), "w"), indent=1)
