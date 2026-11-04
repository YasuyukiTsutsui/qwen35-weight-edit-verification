"""Weight-only probe (CPU, no activations, no data, no labels): which attention heads of the full-attention layers COPY tokens?

For head h of layer L (GQA: 8 query heads share 2 key/value heads) the value-output map is  OV_h = W_O[:, h-slice] @ W_V[kv(h)-slice, :].
If the head attends to a position holding token t, its direct write to the output logits is (final-norm gain) * OV_h * (LN gain) * E[t],
read out with the tied embedding E. We take 3,000 frequent tokens (ids [lo, lo+3000)) and form the 3000 x 3000 score matrix
S[t, t'] = E[t'] . (g_final * OV_h(g_ln * E[t]/rms)). A head "copies" when S[t, t] is high among row t (a standard OV-circuit test),
and "suppresses copies" when S[t, t] is low. This ignores everything between the head and the logits (later layers, the softmax
attention pattern itself), so it is a weight-only SCREEN, not a proof of a mechanism.

    python -I scripts/weight_only_probe.py [--gain on|off]       writes results/weight_probe/heads.json
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "Qwen3.5-0.8B")
FULL = (3, 7, 11, 15, 19, 23)
NH, NKV, HD = 8, 2, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--gain", choices=["on", "off"], default="on")
    a = ap.parse_args()
    torch.set_num_threads(2)
    wm = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))["weight_map"]
    P = "model.language_model."

    def get(name, rows=None):
        with safe_open(os.path.join(BASE, wm[name]), "pt") as f:
            sl = f.get_slice(name)
            return (sl[rows[0]:rows[1]] if rows else sl[:]).float()

    E = get(P + "embed_tokens.weight", (a.lo, a.lo + a.n))                       # (n, 1024)
    gf = get(P + "norm.weight")
    print(f"final-norm gain: mean {gf.mean():+.3f}, std {gf.std():.3f}  (zero-centred if the mean is near 0)", flush=True)
    zero_centred = abs(float(gf.mean())) < 0.5
    adj = (lambda w: 1.0 + w) if zero_centred else (lambda w: w)
    gF = adj(gf) if a.gain == "on" else torch.ones_like(gf)
    out = {"gain": a.gain, "zero_centred": zero_centred, "tokens": [a.lo, a.lo + a.n], "heads": []}
    for L in FULL:
        gl = get(P + f"layers.{L}.input_layernorm.weight")
        gL = adj(gl) if a.gain == "on" else torch.ones_like(gl)
        x = E / E.pow(2).mean(-1, keepdim=True).sqrt() * gL
        Wv = get(P + f"layers.{L}.self_attn.v_proj.weight")                       # (512, 1024)
        Wo = get(P + f"layers.{L}.self_attn.o_proj.weight")                       # (1024, 2048)
        for h in range(NH):
            g = h // (NH // NKV)
            ov_out = (x @ Wv[g * HD:(g + 1) * HD].T) @ Wo[:, h * HD:(h + 1) * HD].T   # (n, 1024)
            S = (ov_out * gF) @ E.T                                                   # (n, n): S[t, t'] write of source token t to logit of t'
            diag = S.diagonal()
            mu, sd = S.mean(1), S.std(1)
            z = ((diag - mu) / sd).mean().item()
            rank = (S > diag[:, None]).sum(1)                                          # number of tokens scoring above the copy target
            top5 = (rank < 5).float().mean().item()
            bot5 = (rank >= a.n - 5).float().mean().item()
            out["heads"].append({"layer": L, "head": h, "copy_z": z, "top5": top5, "bottom5": bot5, "ov_norm": float(ov_out.norm() / x.norm())})
    H = out["heads"]
    for key, rev, title in [("top5", True, "copying (own token among the top-5 written logits)"), ("bottom5", True, "copy-suppressing (own token among the bottom-5)")]:
        srt = sorted(H, key=lambda r: -r[key])
        print(f"\n== top 8 heads by {title}")
        for r in srt[:8]:
            print(f"  L{r['layer']:2d}h{r['head']}  {key} {r[key]:.3f}  copy_z {r['copy_z']:+.2f}")
    for L, h in [(15, 7), (19, 5)]:
        r = next(r for r in H if r["layer"] == L and r["head"] == h)
        rk_c = 1 + sum(x["top5"] > r["top5"] for x in H)
        rk_s = 1 + sum(x["bottom5"] > r["bottom5"] for x in H)
        print(f"\nL{L}h{h} (the head the original edit discussed): top5 {r['top5']:.3f} (rank {rk_c}/{len(H)}), bottom5 {r['bottom5']:.3f} (rank {rk_s}/{len(H)}), copy_z {r['copy_z']:+.2f}")
    print("\nper layer: mean top5 / max top5 / mean bottom5 / max bottom5")
    for L in FULL:
        hs = [r for r in H if r["layer"] == L]
        print(f"  L{L:2d}: {sum(r['top5'] for r in hs) / len(hs):.3f} / {max(r['top5'] for r in hs):.3f} / {sum(r['bottom5'] for r in hs) / len(hs):.3f} / {max(r['bottom5'] for r in hs):.3f}")
    os.makedirs(os.path.join(ROOT, "results", "weight_probe"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "weight_probe", f"heads_gain_{a.gain}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
