"""CPU pre-checks for the equation variants of Issue #28, from results/stats/stats_L{15,19}.pt (scripts/collect_stats.py) and the weights.

The current edit on layer L is  dW = -alpha u (u^T W[:,S]),  i.e. it subtracts  alpha * (u^T W_j) * a_j  from the u-component of the output for every
chosen column j (a = down_proj input). Its 'read vector' is  v_cur[j] = alpha u^T W_j  for j in S, 0 elsewhere (3584-d). For any read vector v the edit
changes the u-component of the layer output by  -v^T a. Statistics (uncentered, from teacher-forced base-model contexts):
    mu_L, mu_P : mean of a on loop / productive contexts;  C_P : second moment of a on productive contexts;  dmu = mu_L - mu_P.
Per unit weight change ||v|| the edit
    removes  contrast = v^T dmu / ||v||  of the loop-vs-productive gap along u,
    but disturbs productive text by  damage = sqrt(v^T C_P v) / ||v||,   and  Fisher = (v^T dmu)^2 / (v^T C_P v) = contrast^2 / damage^2.
Compared: v_cur (top-64 / top-256 contrast columns), v_all = u^T W (all columns), v_lda = (C_P + lam I)^-1 dmu (ROME / LDA read vector),
random columns. Also: spectrum of C_P (how much null space is there for an AlphaEdit-type projection), oblique projector r = Sigma_o^-1 u / (u^T Sigma_o^-1 u),
cos(u, mean productive output), and the shift of u^T o under the current edit in units of the standard deviation of u^T o.

    python -I scripts/analyze_variants_cpu.py   ->  results/analysis_variants_cpu.json
"""
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "Qwen3.5-0.8B")


def read_down(L):
    for f in sorted(os.listdir(BASE)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(BASE, f), "pt") as sf:
                for k in sf.keys():
                    if re.search(rf"layers\.{L}\.mlp\.down_proj\.weight$", k) and not re.search(r"vision|audio|visual|mtp", k):
                        return sf.get_tensor(k).double().numpy()
    raise KeyError(L)


def main():
    U = torch.load(os.path.join(ROOT, "results", "qwen_dir", "directions.pt")).double().numpy()
    info = json.load(open(os.path.join(ROOT, "results", "qwen_dir", "direction.json")))
    out = {}
    for L in (15, 19):
        st = torch.load(os.path.join(ROOT, "results", "stats", f"stats_L{L}.pt"))
        W = read_down(L)
        u = U[L] / np.linalg.norm(U[L])
        rep, prod = st["rep"], st["prod"]
        nL, nP = rep["n"], prod["n"]
        muL, muP = (rep["sa"] / nL).double().numpy(), (prod["sa"] / nP).double().numpy()
        CP = (prod["saa"] / nP).double().numpy()
        dmu = muL - muP
        wu = u @ W                                            # (3584,)  u^T W_j
        top = info["layers"][L]["top_neurons"]
        res = {"n_tokens": {"rep": nL, "prod": nP}}

        def metrics(v):
            nv = np.linalg.norm(v)
            q = float(v @ CP @ v)
            return {"contrast_per_unit": float(v @ dmu / nv), "damage_per_unit": float(np.sqrt(q) / nv), "fisher": float((v @ dmu) ** 2 / q)}

        cands = {"top64": np.where(np.isin(np.arange(3584), top[:64]), wu, 0.0), "top256": np.where(np.isin(np.arange(3584), top[:256]), wu, 0.0),
                 "all_columns": wu.copy()}
        rng = np.random.default_rng(0)
        cands["random256"] = np.where(np.isin(np.arange(3584), rng.choice(3584, 256, replace=False)), wu, 0.0)
        for lam_rel in (1e-3, 1e-2, 1e-1):
            lam = lam_rel * np.trace(CP) / 3584
            cands[f"lda_lam{lam_rel:g}"] = np.linalg.solve(CP + lam * np.eye(3584), dmu)
        res["read_vectors"] = {k: metrics(v) for k, v in cands.items()}
        base_f = res["read_vectors"]["top256"]["fisher"]
        res["fisher_gain_over_top256"] = {k: m["fisher"] / base_f for k, m in res["read_vectors"].items()}
        res["cos_lda_vs_current"] = {k: float(cands["lda_lam0.01"] @ v / (np.linalg.norm(cands["lda_lam0.01"]) * np.linalg.norm(v)))
                                     for k, v in cands.items() if k.startswith(("top", "all"))}
        # spectrum of C_P: null-space availability for an AlphaEdit-type projection
        ev = np.linalg.eigvalsh(CP)[::-1]
        res["C_P_spectrum"] = {"trace": float(ev.sum()), "eig_over_max": {f"n_eig_below_{t:g}_of_max": int((ev < t * ev[0]).sum()) for t in (1e-2, 1e-3, 1e-4)},
                               "effective_rank_90pct": int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1), "top5_share": float(ev[:5].sum() / ev.sum())}
        # oblique projector and mean-orthogonalised direction (output space)
        So = (rep["soo"] + prod["soo"]).double().numpy() / (nL + nP)
        mo = (rep["so"] + prod["so"]).double().numpy() / (nL + nP)
        hP = (prod["so"] / nP).double().numpy()
        r = np.linalg.solve(So + 1e-6 * np.trace(So) / 1024 * np.eye(1024), u)
        r = r / (u @ r)
        hh = hP / np.linalg.norm(hP)
        ug = u - (u @ hh) * hh
        res["oblique"] = {"cos_r_u": float(r @ u / np.linalg.norm(r)), "cos_u_meanProdOutput": float(u @ hh), "cos_u_vs_gramschmidt_u": float(u @ ug / np.linalg.norm(ug)),
                          "coord0_share_u": float(u[0] ** 2), "coord0_share_gramschmidt_u": float(ug[0] ** 2 / (ug @ ug))}
        # strength calibration: shift of u^T o produced by the centre edit (alpha=6) on the loop mean, relative to the std of u^T o
        gap = float(u @ (rep["so"] / nL).double().numpy() - u @ hP)
        sd = float(np.sqrt(max(u @ So @ u - (u @ mo) ** 2, 1e-12)))
        v6 = 6.0 * cands["top256"]
        res["strength"] = {"gap_loop_minus_prod_along_u": gap, "sd_u_o": sd, "gap_in_sd": gap / sd,
                           "alpha6_top256_shift_on_loop_mean": float(-v6 @ muL), "alpha6_top256_shift_on_prod_mean": float(-v6 @ muP)}
        out[L] = res
        print(f"L{L}:", json.dumps({k: res[k] for k in ("fisher_gain_over_top256", "oblique", "strength")}, indent=1), flush=True)
        print("  spectrum:", res["C_P_spectrum"], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_variants_cpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
