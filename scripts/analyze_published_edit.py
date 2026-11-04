"""Recover the direction u and strength alpha of the published p50c edit (W' = W - alpha u u^T W on columns S)."""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from apply_published_edit import read_bf16_bits, bf16_to_f32
base = read_bf16_bits("models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
edit = read_bf16_bits("models/Qwen3.5-0.8B-p50c/model.safetensors-00001-of-00001.safetensors")
out = {}
for L in (15, 19):
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    W = bf16_to_f32(base[n][1].view(np.uint16)).reshape(base[n][2]).astype(np.float64)
    E = bf16_to_f32(edit[n][1].view(np.uint16)).reshape(edit[n][2]).astype(np.float64)
    D = E - W
    S = np.where(np.abs(D).sum(0) > 0)[0]
    U, s, Vt = np.linalg.svd(D[:, S], full_matrices=False)
    u = U[:, 0]
    proj = u @ W[:, S]                      # u^T W_S
    alpha = -(u @ D[:, S]) @ proj / (proj @ proj)
    if alpha < 0: u, alpha = -u, -alpha     # sign convention irrelevant; keep alpha>0
    # sign of u: make the base MLP column contributions mostly positive along u is undefined; leave as is
    resid = np.linalg.norm(D[:, S] + alpha * np.outer(u, u @ W[:, S])) / np.linalg.norm(D[:, S])
    print(f"L{L}: {len(S)} cols, sv ratio s1/s2={s[0]/s[1]:.1f}, alpha={alpha:.3f}, rel.resid={resid:.3f}, "
          f"|uW_S|/|W_S| = {np.linalg.norm(proj)/np.linalg.norm(W[:,S]):.3f}")
    out[L] = {"cols": S.tolist(), "u": u.tolist(), "alpha": float(alpha)}
c = np.dot(out[15]["u"], out[19]["u"])
print("cos(u15,u19) =", round(float(c), 3))
json.dump(out, open("results/published_p50c_direction.json", "w"))
