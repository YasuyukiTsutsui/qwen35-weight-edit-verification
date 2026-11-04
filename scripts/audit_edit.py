"""Audit an edited Qwen checkpoint against the base: what exactly changed, and how it compares with the published p50c edit.

    python scripts/audit_edit.py <edited dir> [<edited dir> ...]
Prints per edited tensor: changed columns, rank/strength of the change on those columns, relative change,
cosine of the removed direction with the published one, overlap of the edited columns with the published ones.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_published_edit import bf16_to_f32, read_bf16_bits  # noqa: E402

F = "model.safetensors-00001-of-00001.safetensors"
BASE = "models/Qwen3.5-0.8B"
PUB = json.load(open("results/published_p50c_direction.json"))


def audit(edir):
    b = read_bf16_bits(os.path.join(BASE, F))
    e = read_bf16_bits(os.path.join(edir, F))
    changed = [k for k in b if not np.array_equal(b[k][1], e[k][1])]
    print(f"\n== {edir}: tensors changed = {len(changed)}  ({'; '.join(k.split('layers.')[-1] for k in changed)})")
    tot = 0
    for n in changed:
        L = int(n.split("layers.")[1].split(".")[0])
        W = bf16_to_f32(b[n][1].view(np.uint16)).reshape(b[n][2]).astype(np.float64)
        E = bf16_to_f32(e[n][1].view(np.uint16)).reshape(e[n][2]).astype(np.float64)
        D = E - W
        S = np.where(np.abs(D).sum(0) > 0)[0]
        tot += D.size and int((D != 0).sum())
        U, s, _ = np.linalg.svd(D[:, S], full_matrices=False)
        u = U[:, 0]
        proj = u @ W[:, S]
        alpha = abs((u @ D[:, S]) @ proj / (proj @ proj))
        if (u @ np.array(PUB[str(L)]["u"])) < 0:
            u = -u
        pub_u, pub_cols = np.array(PUB[str(L)]["u"]), set(PUB[str(L)]["cols"]) if str(L) in PUB else set()
        print(f"  layer {L}: matrix {W.shape}, edited columns = {len(S)} of {W.shape[1]}, "
              f"entries differing after bf16 rounding = {(D != 0).sum():,}")
        print(f"    rank-1 check: s1/s2 = {s[0] / s[1]:.0f};  strength alpha = {alpha:.2f};  "
              f"|dW|/|W| (whole matrix) = {np.linalg.norm(D) / np.linalg.norm(W):.3f}")
        print(f"    vs published: cos(u, u_pub) = {float(u @ pub_u):+.3f}; shared columns = {len(set(S.tolist()) & pub_cols)}/{len(S)}")
    print(f"  total weights changed: {tot:,} ({tot / 0.8e9 * 100:.4f}% of ~0.8B)")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        audit(d)
