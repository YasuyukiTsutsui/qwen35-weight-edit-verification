"""Edit variants of Issue #28 (equation changes from the literature), all at the SAME relative change rho per layer as the centre edit
(models/edits/bo_k256_a6: alpha 6, k 256, layers 15+19, direction results/qwen_dir).

    --variant c3_rest   centre dW with the coordinate-0 row zeroed, rescaled to keep rho            (no statistics needed)
              c3_only   centre dW restricted to the coordinate-0 row (rho is lower; reported)
              lda       dW = -beta u v^T,  v = (C_P + lam I)^-1 (mu_L - mu_P)   (all 3584 columns; needs results/stats)
              oblique   W[:,S] -= alpha u (r^T W[:,S]),  r = Sigma_o^-1 u / (u^T Sigma_o^-1 u)  (S = centre's top-256 columns; needs stats)
              gs        centre edit with u replaced by u - (u.h)h normalised, h = mean productive MLP output  (needs stats)
    python scripts/make_variant_edit.py --variant lda --out models/edits/eq_lda
"""
import argparse
import json
import os
import re
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = (15, 19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["c3_rest", "c3_only", "lda", "oblique", "gs"])
    ap.add_argument("--base", default="models/Qwen3.5-0.8B")
    ap.add_argument("--centre", default="models/edits/bo_k256_a6")
    ap.add_argument("--dir", default="results/qwen_dir")
    ap.add_argument("--stats", default="results/stats")
    ap.add_argument("--lam", type=float, default=0.01, help="ridge for lda, relative to tr(C_P)/3584")
    ap.add_argument("--rho-scale", type=float, default=1.0, help="multiply the centre's relative change rho by this factor (dose test; lda, oblique, gs only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    U = torch.load(os.path.join(a.dir, "directions.pt")).double().numpy()
    info = json.load(open(os.path.join(a.dir, "direction.json")))
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.base):
        p = os.path.join(a.base, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copyfile(p, os.path.join(a.out, f))
    log = {"variant": a.variant, "centre": a.centre, "lam": a.lam, "rho_scale": a.rho_scale, "tensors": {}}
    for sh in sorted(f for f in os.listdir(a.base) if f.endswith(".safetensors")):
        sd = load_file(os.path.join(a.base, sh))
        cen = load_file(os.path.join(a.centre, sh))
        for L in LAYERS:
            keys = [k for k in sd if re.search(rf"layers\.{L}\.mlp\.down_proj\.weight$", k) and not re.search(r"vision|audio|visual|mtp", k)]
            if not keys:
                continue
            key = keys[0]
            W = sd[key].double().numpy()
            Wc = cen[key].double().numpy()
            dWc = Wc - W
            rho_centre = np.linalg.norm(dWc) / np.linalg.norm(W)   # the centre's relative change on this layer
            rho = rho_centre * a.rho_scale                         # target relative change of the variant
            u = U[L] / np.linalg.norm(U[L])
            S = np.array(info["layers"][L]["top_neurons"][:256])
            extra = {}
            if a.variant in ("c3_rest", "c3_only"):
                dW = np.zeros_like(dWc)
                if a.variant == "c3_only":
                    dW[0] = dWc[0]
                else:
                    dW[1:] = dWc[1:]
                    dW *= np.linalg.norm(dWc) / np.linalg.norm(dW)
            else:
                st = torch.load(os.path.join(a.stats, f"stats_L{L}.pt"))
                rep, prod = st["rep"], st["prod"]
                if a.variant == "lda":
                    muL, muP = (rep["sa"] / rep["n"]).double().numpy(), (prod["sa"] / prod["n"]).double().numpy()
                    CP = (prod["saa"] / prod["n"]).double().numpy()
                    lam = a.lam * np.trace(CP) / CP.shape[0]
                    v = np.linalg.solve(CP + lam * np.eye(CP.shape[0]), muL - muP)
                    beta = rho * np.linalg.norm(W) / np.linalg.norm(v)
                    dW = -beta * np.outer(u, v)
                    extra = {"beta": float(beta), "cos_v_with_uTW": float(v @ (u @ W) / (np.linalg.norm(v) * np.linalg.norm(u @ W)))}
                elif a.variant == "oblique":
                    So = (rep["soo"] + prod["soo"]).double().numpy() / (rep["n"] + prod["n"])
                    r = np.linalg.solve(So + 1e-6 * np.trace(So) / So.shape[0] * np.eye(So.shape[0]), u)
                    r = r / (u @ r)
                    row = r @ W[:, S]
                    al = rho * np.linalg.norm(W) / np.linalg.norm(row)
                    dW = np.zeros_like(W)
                    dW[:, S] = -al * np.outer(u, row)
                    extra = {"alpha_used": float(al), "cos_r_u": float(r @ u / np.linalg.norm(r))}
                else:  # gs
                    h = (prod["so"] / prod["n"]).double().numpy()
                    h = h / np.linalg.norm(h)
                    ug = u - (u @ h) * h
                    ug = ug / np.linalg.norm(ug)
                    row = ug @ W[:, S]
                    al = rho * np.linalg.norm(W) / np.linalg.norm(row)
                    dW = np.zeros_like(W)
                    dW[:, S] = -al * np.outer(ug, row)
                    extra = {"alpha_used": float(al), "cos_u_ug": float(u @ ug)}
            new = torch.tensor(W + dW).to(sd[key].dtype)
            log["tensors"][key] = {"centre_rel_change": float(rho_centre), "rel_change": float((new.double().numpy() - W).__pow__(2).sum() ** 0.5 / np.linalg.norm(W)), **extra}
            sd[key] = new.contiguous()
        save_file(sd, os.path.join(a.out, sh), metadata={"format": "pt"})
    assert len(log["tensors"]) == len(LAYERS), log["tensors"]
    json.dump(log, open(os.path.join(a.out, "edit_log.json"), "w"), indent=1)
    print(json.dumps(log["tensors"]))


if __name__ == "__main__":
    main()
