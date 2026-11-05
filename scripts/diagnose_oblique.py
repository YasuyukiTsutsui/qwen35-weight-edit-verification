"""Why did the oblique edit (Issue #28) fall below the unedited model? Compare the read vectors of the centre, LDA and oblique edits on the
statistics of results/stats: how much each shifts the loop-vs-productive gap along the written direction (contrast), the damage on productive text.
Read vector of an edit  dW = -u v^T  is v (3584-d). Positive 'contrast' = removes more of the u-component on looping contexts than on productive ones.
    python -I scripts/diagnose_oblique.py  ->  results/diagnose_oblique.json
"""
import json, os, re
import numpy as np, torch
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
U = torch.load(os.path.join(ROOT, "results", "qwen_dir", "directions.pt")).double().numpy()
info = json.load(open(os.path.join(ROOT, "results", "qwen_dir", "direction.json")))
out = {}
for L in (15, 19):
    st = torch.load(os.path.join(ROOT, "results", "stats", f"stats_L{L}.pt")); W = read_down(L); u = U[L] / np.linalg.norm(U[L])
    rep, prod = st["rep"], st["prod"]; muL, muP = (rep["sa"] / rep["n"]).double().numpy(), (prod["sa"] / prod["n"]).double().numpy()
    CP = (prod["saa"] / prod["n"]).double().numpy(); dmu = muL - muP
    S = np.array(info["layers"][L]["top_neurons"][:256]); mask = np.zeros(3584, bool); mask[S] = True
    So = (rep["soo"] + prod["soo"]).double().numpy() / (rep["n"] + prod["n"])
    r = np.linalg.solve(So + 1e-6 * np.trace(So) / 1024 * np.eye(1024), u); r /= (u @ r)
    v = {"centre": np.where(mask, u @ W, 0.0), "oblique": np.where(mask, r @ W, 0.0), "lda": np.linalg.solve(CP + 0.01 * np.trace(CP) / 3584 * np.eye(3584), dmu)}
    res = {}
    for k, x in v.items():
        n = np.linalg.norm(x); res[k] = {"contrast_per_unit": float(x @ dmu / n), "damage_per_unit": float(np.sqrt(x @ CP @ x) / n),
                                          "mean_shift_loop_per_unit": float(x @ muL / n), "mean_shift_prod_per_unit": float(x @ muP / n)}
    res["cos_oblique_vs_centre_readvec"] = float(v["oblique"] @ v["centre"] / (np.linalg.norm(v["oblique"]) * np.linalg.norm(v["centre"])))
    out[L] = res; print(L, json.dumps(res, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "diagnose_oblique.json"), "w"), indent=1)
