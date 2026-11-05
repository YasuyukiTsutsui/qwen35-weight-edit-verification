"""Gradient probe (Issue #27 reviewer's P0, and the gradient part of N2 / Issue #22).  Run inside the GPU container.

Question: does the first-order policy-gradient signal on the down_proj weights of layers 15 and 19 carry reproducible information
(a) about WHERE the edit direction u should go, and (b) about WHICH columns matter?  If it does not, a reward-only optimiser
(method A started from a random direction) cannot find a direction, and a 4.5-hour training run would be wasted.

It computes the RLOO policy-gradient matrix  G_L = dLoss/dW_L  (1024 x 3584, fp32) at the INITIAL edited model (models/edits/bo_k64_a5,
the policy that produced the saved samples), on two disjoint halves of the mixed-reward problems of results/pg_r0/practice_find160.
Same loss as scripts/train_rank1_pg.py:  Loss = - sum_i A_i sum_t log pi_sampler(y_it) / (n_seq * C),  A_i = leave-one-out advantage.

Reported per layer (all with a random baseline in the output):
  1. split-half reproducibility of the top-3 left/right singular vectors of G;
  2. the share of |u*|^2 inside the top-3 left singular subspace of G, for u* in {own direction, own multi-prompt direction, published direction
     (recovered from the weight difference p50c - base, sign fixed against the own direction)};
  3. the tangent gradient of the rank-1 edit's u (t = -(I - u0 u0^T) G[:,S] v at the initial edit) and its cosine with the tangent part of each u*,
     and between the two halves;
  4. column scores s_j = u0^T G[:,j] (positive = removing u0 from column j lowers the loss to first order): overlap of the top-64/256 sets between
     halves, and with the contrast-based set used so far.
Caveats (also in the Issue): first order, off-policy only through the initial policy, teacher-forced proxies have mispredicted before.

    bash scripts/drun.sh python3 scripts/grad_probe.py --data results/pg_r0/practice_find160 --out results/grad_probe
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rank1_pg import load_data, seq_logp  # noqa: E402

LAYERS = (15, 19)


def read_down(model_dir, L):
    for f in sorted(os.listdir(model_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, f), "pt") as sf:
            for k in sf.keys():
                if re.search(rf"layers\.{L}\.mlp\.down_proj\.weight$", k) and not re.search(r"vision|audio|visual|mtp", k):
                    return sf.get_tensor(k).double()
    raise KeyError(L)


def unit(x):
    return x / x.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/edits/bo_k64_a5")
    ap.add_argument("--base", default="models/Qwen3.5-0.8B")
    ap.add_argument("--pub", default="models/Qwen3.5-0.8B-p50c")
    ap.add_argument("--dir", default="results/qwen_dir")
    ap.add_argument("--dir-mp", default="results/qwen_dir_mp")
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-half", type=int, default=8, help="mixed-reward problems per half")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--const", type=float, default=8192.0)
    ap.add_argument("--out", default="results/grad_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from find_direction import load_model
    model, dec, _ = load_model(a.model)
    for q in model.parameters():
        q.requires_grad_(False)
    layers = dec.layers
    Ws = {L: layers[L].mlp.down_proj.weight for L in LAYERS}
    for L in LAYERS:
        Ws[L].requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lm_head = model.lm_head

    data = load_data(a.data)
    groups = {}
    for d in data:
        groups.setdefault(d["id"], []).append(d)
    mixed = [g for g in groups.values() if len(g) >= 2 and 0 < sum(x["reward"] for x in g) < len(g)]
    random.Random(0).shuffle(mixed)
    print(f"problems {len(groups)}, mixed {len(mixed)}; using {2 * a.per_half}", flush=True)
    halves = [mixed[:a.per_half], mixed[a.per_half:2 * a.per_half]]
    G = {}
    t0 = time.time()
    for h, batch in enumerate(halves):
        nseq = sum(len(g) for g in batch)
        acc = {L: torch.zeros_like(Ws[L], dtype=torch.float32) for L in LAYERS}
        for g in batch:
            R = torch.tensor([x["reward"] for x in g])
            for i, rec in enumerate(g):
                adv = float(R[i] - (R.sum() - R[i]) / (len(g) - 1))
                lp = seq_logp(dec, lm_head, rec, a.max_len)
                ok = torch.isfinite(lp)
                (-adv * lp[ok].sum() / (nseq * a.const)).backward()
                for L in LAYERS:
                    acc[L] += Ws[L].grad.float()
                    Ws[L].grad = None
            print(f"half {h}: done problem, {time.time() - t0:.0f}s", flush=True)
        G[h] = acc
    del model
    torch.cuda.empty_cache()

    # ---- analysis (fp32 on the GPU)
    dev = "cuda"
    U0 = torch.load(os.path.join(a.dir, "directions.pt")).float()
    UM = torch.load(os.path.join(a.dir_mp, "directions.pt")).float()
    info = json.load(open(os.path.join(a.dir, "direction.json")))
    res = {}
    cols_out = {}
    g = torch.Generator().manual_seed(0)
    for L in LAYERS:
        Wb, We, Wp = read_down(a.base, L), read_down(a.model, L), read_down(a.pub, L)
        u0 = unit(U0[L].double())
        um = unit(UM[L].double())
        dW = (Wp - Wb)
        uu, ss, vv = torch.linalg.svd(dW, full_matrices=False)
        up = unit(uu[:, 0])
        if float(up @ u0) < 0:
            up = -up
        S = torch.tensor(info["layers"][L]["top_neurons"][:256])
        # initial edit's v: dW_edit = -u0 v^T on the columns S (checked by its rank-1 fit)
        v = -(u0 @ (We - Wb))                                  # (3584,)
        refs = {"own": u0, "own_mp": um, "published": up}
        r = {"cos_between_refs": {"own~published": float(u0 @ up), "own~own_mp": float(u0 @ um), "own_mp~published": float(um @ up)},
             "v_norm_rel": float(v.norm() / Wb.norm())}
        Gs = {h: G[h][L].to(dev) for h in (0, 1)}
        sv = {}
        for h in (0, 1):
            Uh, Sh, Vh = torch.linalg.svd(Gs[h], full_matrices=False)
            sv[h] = (Uh[:, :3], Vh[:3].T, Sh[:3])
        r["top_singular_values"] = {h: [float(x) for x in sv[h][2]] for h in (0, 1)}
        r["split_half_abs_cos_left"] = [abs(float(sv[0][0][:, i] @ sv[1][0][:, i])) for i in range(3)]
        r["split_half_abs_cos_right"] = [abs(float(sv[0][1][:, i] @ sv[1][1][:, i])) for i in range(3)]
        # random baseline for a 3-dim subspace share in 1024 dims: 3/1024
        r["subspace_share_top3_left"] = {n: {h: float(((sv[h][0].T @ x.float().to(dev)) ** 2).sum()) for h in (0, 1)} for n, x in refs.items()}
        r["subspace_share_random_baseline"] = 3 / 1024
        # tangent gradient of u at the initial edit
        vv_ = v.float().to(dev)
        t = {}
        for h in (0, 1):
            raw = -(Gs[h].float() @ vv_)                       # d loss / d u (up to the unit-norm projection)
            raw = raw - (raw @ u0.float().to(dev)) * u0.float().to(dev)
            t[h] = raw
        r["tangent_cos_between_halves"] = float(unit(t[0]) @ unit(t[1]))
        r["tangent_cos_with_target_tangent"] = {}
        for n, x in refs.items():
            xt = x.float().to(dev)
            xt = xt - (xt @ u0.float().to(dev)) * u0.float().to(dev)
            r["tangent_cos_with_target_tangent"][n] = {h: float(unit(-t[h]) @ unit(xt)) if float(xt.norm()) > 1e-6 else None for h in (0, 1)}
        # sign convention note: loss decreases along -dL/du, so the descent direction is -raw; raw already is dL/du (t = -(G v) is -dL/du*(-1)).
        # column scores
        sc = {h: (u0.float().to(dev) @ Gs[h].float()).cpu() for h in (0, 1)}   # s_j = u0^T G[:,j]; ">0" = subtracting u0 from col j lowers the loss
        # careful: d loss / d W_j = G[:,j]; subtracting c u0 from column j changes loss by -c u0^T G_j, so the beneficial columns have u0^T G_j > 0
        cur = set(info["layers"][L]["top_neurons"][:256])
        r["columns"] = {}
        for K in (64, 256):
            top = {h: set(torch.topk(sc[h], K).indices.tolist()) for h in (0, 1)}
            tops = set(torch.topk(sc[0] + sc[1], K).indices.tolist())
            r["columns"][K] = {"overlap_between_halves": len(top[0] & top[1]) / K, "overlap_with_contrast_set": len(tops & set(info["layers"][L]["top_neurons"][:K])) / K,
                               "random_overlap": K / 3584}
        pos = (sc[0] + sc[1]).clamp(min=0)
        r["share_positive_score_in_contrast_top256"] = float(pos[list(cur)].sum() / pos.sum())
        res[L] = r
        cols_out[L] = {"score_sum": (sc[0] + sc[1]), "score_h0": sc[0], "score_h1": sc[1]}
        print(json.dumps({L: r}, indent=1), flush=True)
    json.dump(res, open(os.path.join(a.out, "grad_probe.json"), "w"), indent=1)
    torch.save(cols_out, os.path.join(a.out, "colscores.pt"))
    print("GRADPROBEDONE", flush=True)


if __name__ == "__main__":
    main()
