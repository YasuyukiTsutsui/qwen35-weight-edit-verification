"""Method A: policy-gradient training of a rank-1 edit  dW_L = -u_L v_L^T  (layers 15 and 19) with AdamW.

Reward = 1 if the sampled answer is correct, else 0 (accuracy is the objective). Samples come from the CURRENT policy
(vLLM, `eval_math.py --save-tokens 1`). Loss = RLOO policy gradient:
    A_i = r_i - mean_{j != i in the same problem} r_j ;   L = - sum_i A_i * sum_t log pi_sampler(y_it) / (n_seq * C)
where pi_sampler is the SAMPLING distribution (presence penalty 1.5 -> temperature 0.6 -> top-k 20 -> top-p 0.95), so the
gradient matches what the generator actually sampled from; C is a constant (no per-length normalisation). Problems whose
samples all get the same reward have A = 0 and are skipped.
Parameters (fp32): u_L (renormalised to unit norm in the forward pass) and v_L over a fixed set of columns S (top-256 by
the direction finder's score); init = the own edit (top-64 columns, --alpha-init, default 5 = models/edits/bo_k64_a5, whose
relative change 0.098/0.117 is inside the budget):  v_j = alpha * (u . W_j) for the top-64, 0 elsewhere. The samples must
come from the model with this same initial edit, so that the first update is (nearly) on-policy.
Budget: ||v|| / ||W||_F is projected to <= --rho-max after every step. Gradient clip 1.0. No weight decay toward 0.
Each step uses a different batch of problems, all with samples from the initial policy (no importance ratio; the mean
token log-prob drift against the saved vLLM log-probs is logged as a diagnostic).

Modes:  check  (sampler log-probs vs vLLM's saved ones, hook vs generator; exits non-zero if too many targets fall outside
                the sampling support)
        bench  (forward+backward throughput / memory on real sequences, with per-layer activation checkpointing)
        train  (--steps steps, each on --problems-per-step problems; exports the edited checkpoint; saves u, v after each step)
    python3 scripts/train_rank1_pg.py --mode check --data results/pg_r0/practice_find
Review of this file: reports/review_method_a.md (A-1 ... A-11).
"""
import argparse
import gzip
import json
import math
import os
import re
import shutil
import sys
import time

import torch
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEMP, TOPK, TOPP, PRES = 0.6, 20, 0.95, 1.5
LAYERS = (15, 19)


def sampler_logprobs(h, lm_head, y, chunk=256):
    """h: (L, dm) hidden states whose next token is y (L,). Log-prob of y under presence-penalty/temperature/top-k/top-p."""
    L = y.shape[0]
    U, inv = torch.unique(y, return_inverse=True)
    first = torch.full((len(U),), L, device=y.device, dtype=torch.long).scatter_reduce(
        0, inv, torch.arange(L, device=y.device), reduce="amin")
    outs = []

    def fn(hc, yc, tpos):
        z = lm_head(hc).float()
        pen = (first[None, :] < tpos[:, None]).float() * PRES
        z = z.index_copy(1, U, z[:, U] - pen)
        vals, idx = (z / TEMP).topk(TOPK, dim=-1)
        with torch.no_grad():                       # the support (top-p set) carries no gradient
            p = torch.softmax(vals, -1)
            keep = (p.cumsum(-1) - p) < TOPP
        lp = torch.log_softmax(vals.masked_fill(~keep, float("-inf")), -1)
        # y must be inside the top-k AND the top-p set. A logsumexp over an all -inf row has a NaN backward (review A-1),
        # so pick the single finite entry with where() and return a constant -inf for rows where y is outside the support.
        hit = (idx == yc[:, None]) & keep
        val = torch.where(hit, lp, torch.zeros_like(lp)).sum(-1)
        return torch.where(hit.any(-1), val, torch.full_like(val, float("-inf")))

    for c0 in range(0, L, chunk):
        c1 = min(L, c0 + chunk)
        tpos = torch.arange(c0, c1, device=y.device)
        outs.append(torch.utils.checkpoint.checkpoint(fn, h[c0:c1], y[c0:c1], tpos, use_reentrant=False))
    return torch.cat(outs)


class Rank1:
    def __init__(self, layers, dirpath, k_mask=256, k_init=64, alpha_init=5.0):
        U = torch.load(os.path.join(dirpath, "directions.pt")).float()
        info = json.load(open(os.path.join(dirpath, "direction.json")))
        self.p = {}
        for L in LAYERS:
            W = layers[L].mlp.down_proj.weight
            S = torch.tensor(info["layers"][L]["top_neurons"][:k_mask], device="cuda")
            u = U[L].cuda().clone()
            u = u / u.norm()
            v = torch.zeros(len(S), device="cuda")
            v[:k_init] = alpha_init * (u.double() @ W[:, S[:k_init]].double()).float()
            self.p[L] = {"S": S, "u": torch.nn.Parameter(u), "v": torch.nn.Parameter(v), "Wnorm": float(W.double().norm())}
        self.hooks = []

    def params(self):
        return [q for L in LAYERS for q in (self.p[L]["u"], self.p[L]["v"])]

    def attach(self, layers):
        for L in LAYERS:
            st = self.p[L]

            def hook(m, inp, out, st=st):
                u = st["u"] / st["u"].norm()
                return out - ((inp[0][..., st["S"]].float() @ st["v"])[..., None] * u).to(out.dtype)
            self.hooks.append(layers[L].mlp.down_proj.register_forward_hook(hook))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def project(self, rho_max):
        info = {}
        for L in LAYERS:
            st = self.p[L]
            st["u"].div_(st["u"].norm())
            rel = float(st["v"].norm()) / st["Wnorm"]
            if rel > rho_max:
                st["v"].mul_(rho_max / rel)
                rel = rho_max
            info[L] = rel
        return info

    @torch.no_grad()
    def export(self, base_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(base_dir):
            p = os.path.join(base_dir, f)
            if os.path.isfile(p) and not f.endswith(".safetensors"):
                shutil.copyfile(p, os.path.join(out_dir, f))
        log = {}
        for sh in sorted(f for f in os.listdir(base_dir) if f.endswith(".safetensors")):
            sd = load_file(os.path.join(base_dir, sh))
            for L in LAYERS:
                ks = [k for k in sd if re.search(rf"layers\.{L}\.mlp\.down_proj\.weight$", k) and not re.search(r"vision|audio|visual|mtp", k)]
                if not ks:
                    continue
                st = self.p[L]
                W = sd[ks[0]].double()
                u = (st["u"] / st["u"].norm()).double().cpu()
                W[:, st["S"].cpu()] -= torch.outer(u, st["v"].double().cpu())
                new = W.to(sd[ks[0]].dtype)
                log[ks[0]] = {"rel_change": float((new.double() - sd[ks[0]].double()).norm() / sd[ks[0]].double().norm())}
                sd[ks[0]] = new.contiguous()
            save_file(sd, os.path.join(out_dir, sh), metadata={"format": "pt"})
        assert len(log) == len(LAYERS), f"edited {list(log)} (expected layers {LAYERS})"      # review A-7
        json.dump(log, open(os.path.join(out_dir, "edit_log.json"), "w"), indent=1)
        return log

    def state(self):
        return {L: {"u": self.p[L]["u"].detach().cpu(), "v": self.p[L]["v"].detach().cpu(), "S": self.p[L]["S"].cpu()} for L in LAYERS}


def load_data(path):
    toks = [json.loads(l) for l in gzip.open(os.path.join(path, "tokens.jsonl.gz"), "rt", encoding="utf-8")]
    samp = [json.loads(l) for l in gzip.open(os.path.join(path, "samples.jsonl.gz"), "rt", encoding="utf-8")]
    assert len(toks) == len(samp)
    out = []
    for t, s in zip(toks, samp):
        assert t["id"] == s["id"] and t["k"] == s["k"]
        out.append({**t, "reward": float(s["correct"]), "finish": s["finish"]})
    return out


def seq_logp(dec, lm_head, rec, max_len):
    ids = rec["prompt_ids"] + rec["token_ids"][:max_len]
    n_resp = len(ids) - len(rec["prompt_ids"])
    x = torch.tensor([ids], device="cuda")
    h = dec(input_ids=x, use_cache=False).last_hidden_state[0]
    P = len(rec["prompt_ids"])
    y = x[0, P:]
    return sampler_logprobs(h[P - 1:P - 1 + n_resp], lm_head, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "bench", "train"], required=True)
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--dir", default="results/qwen_dir")
    ap.add_argument("--data", required=True, help="dir with samples.jsonl.gz + tokens.jsonl.gz (eval_math --save-tokens 1)")
    ap.add_argument("--max-len", type=int, default=16384, help="response tokens kept; the samples were generated with a 16,384 cap, "
                    "and loops are at the END of a sequence, so truncating below the cap drops the part that matters (review A-4)")
    ap.add_argument("--n", type=int, default=4, help="check/bench: number of sequences")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--problems-per-step", type=int, default=12)
    ap.add_argument("--min-samples", type=int, default=8, help="warn if a problem has fewer samples")
    ap.add_argument("--lr-v", type=float, default=1e-2, help="with Adam each element moves by about lr per step; v's elements are ~0.25 rms")
    ap.add_argument("--lr-u", type=float, default=5e-4, help="u's elements are ~0.03 rms")
    ap.add_argument("--alpha-init", type=float, default=5.0, help="initial edit strength; must match the model the samples came from")
    ap.add_argument("--rho-max", type=float, default=0.12)
    ap.add_argument("--const", type=float, default=8192.0)
    ap.add_argument("--no-ckpt", action="store_true", help="disable per-layer activation checkpointing (bench/train)")
    ap.add_argument("--out", default="models/edits/pg_r1")
    a = ap.parse_args()

    from find_direction import load_model
    model, dec, _ = load_model(a.model)
    for q in model.parameters():
        q.requires_grad_(False)
    layers = dec.layers
    r1 = Rank1(layers, a.dir, alpha_init=a.alpha_init)
    print("initial relative change (before projection):", {L: round(float(r1.p[L]['v'].norm()) / r1.p[L]['Wnorm'], 4) for L in LAYERS}, flush=True)
    r1.attach(layers)
    if a.mode != "check" and not a.no_ckpt:        # review A-2: recompute each decoder layer in the backward pass
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()                           # checkpointing is only active in training mode; there is no dropout in this model
            print("activation checkpointing: on", flush=True)
        except Exception as e:                      # noqa: BLE001
            print("WARNING: could not enable activation checkpointing:", e, flush=True)
    data = load_data(a.data)
    lm_head = model.lm_head
    print(f"{len(data)} sequences, mean reward {sum(d['reward'] for d in data) / len(data):.3f}", flush=True)

    if a.mode == "check":
        # sequences of different lengths: the shortest, a middle one and the longest ones (review A-8)
        srt = sorted(data, key=lambda d: len(d["token_ids"]))
        pick = [srt[0], srt[len(srt) // 2]] + srt[::-1][:max(a.n - 2, 1)]
        bad = False
        for rec in pick:
            with torch.no_grad():
                lp = seq_logp(dec, lm_head, rec, a.max_len).cpu()
            ref = torch.tensor(rec["logprobs"][:len(lp)])
            ok = torch.isfinite(lp)
            d = (lp[ok] - ref[ok])
            fin = float(ok.float().mean())
            print(f"len {len(lp)}: finite {fin:.4f}  mean|diff| {d.abs().mean():.4f}  mean diff {d.mean():+.4f}  "
                  f"max|diff| {d.abs().max():.3f}  sum new {lp[ok].sum():.1f} vs vLLM {ref[ok].sum():.1f}", flush=True)
            bad |= fin < 0.99 or float(d.abs().mean()) > 0.1
        if bad:
            print("CHECK FAILED: too many targets outside the sampling support (finite < 0.99) or mean|diff| > 0.1", flush=True)
            sys.exit(2)
        print("CHECK OK", flush=True)
        return

    if a.mode == "bench":
        recs = sorted(data, key=lambda d: -len(d["token_ids"]))
        recs = [r for r in recs if len(r["token_ids"]) >= 4000][:a.n] or recs[:a.n]
        torch.cuda.reset_peak_memory_stats()
        for rec in recs:
            t0 = time.time()
            lp = seq_logp(dec, lm_head, rec, a.max_len)
            loss = -lp[torch.isfinite(lp)].sum() / a.const
            loss.backward()
            torch.cuda.synchronize()
            n = min(len(rec["token_ids"]), a.max_len)
            print(f"{n} tokens: {time.time() - t0:.1f}s ({n / (time.time() - t0):.0f} tok/s), peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                  f"|grad v15| {float(r1.p[15]['v'].grad.norm()):.3e}", flush=True)
        return

    # ---- train
    groups = {}
    for d in data:
        groups.setdefault(d["id"], []).append(d)
    mixed = [g for g in groups.values() if len(g) >= 2 and 0 < sum(x["reward"] for x in g) < len(g)]
    print(f"problems {len(groups)}, with mixed rewards {len(mixed)}", flush=True)
    short = [i for i, g in groups.items() if len(g) < a.min_samples]
    if short:
        print(f"WARNING: {len(short)} problems have fewer than {a.min_samples} samples (the leave-one-out baseline is noisy)", flush=True)
    need = a.steps * a.problems_per_step
    if len(mixed) < need:                           # review A-7: never "train" on nothing and export the initial edit as trained
        sys.exit(f"not enough mixed-reward problems: {len(mixed)} < steps x problems-per-step = {need} "
                 f"(needs >= 2 samples per problem and at least one right and one wrong answer)")
    os.makedirs(a.out, exist_ok=True)
    opt = torch.optim.AdamW([{"params": [r1.p[L]["v"] for L in LAYERS], "lr": a.lr_v},
                             {"params": [r1.p[L]["u"] for L in LAYERS], "lr": a.lr_u}], betas=(0.9, 0.99), weight_decay=0.0)
    import random
    rnd = random.Random(0)
    rnd.shuffle(mixed)
    log = []
    print("relative change at the start (after projection):", r1.project(a.rho_max), flush=True)
    for step in range(a.steps):
        batch = mixed[step * a.problems_per_step:(step + 1) * a.problems_per_step]
        opt.zero_grad()
        nseq = sum(len(g) for g in batch)
        tot, drift, n_tok, n_fin = 0.0, [], 0, 0
        for g in batch:
            R = torch.tensor([x["reward"] for x in g])
            for i, rec in enumerate(g):
                adv = float(R[i] - (R.sum() - R[i]) / (len(g) - 1))
                lp = seq_logp(dec, lm_head, rec, a.max_len)
                ok = torch.isfinite(lp)
                n_tok += len(lp)
                n_fin += int(ok.sum())
                loss = -adv * lp[ok].sum() / (nseq * a.const)
                loss.backward()
                tot += float(loss)
                ref = torch.tensor(rec["logprobs"][:len(lp)], device=lp.device)
                drift.append(float((lp[ok] - ref[ok]).mean()))
        gn = torch.nn.utils.clip_grad_norm_(r1.params(), 1.0)
        entry = {"step": step, "loss": tot, "grad_norm": float(gn), "finite_token_share": n_fin / max(n_tok, 1),
                 "mean_token_logp_drift": sum(drift) / len(drift)}
        if not math.isfinite(float(gn)):            # review A-1: never step on a NaN gradient
            entry["skipped"] = "non-finite gradient"
            log.append(entry)
            print(json.dumps(entry), flush=True)
            continue
        opt.step()
        entry["rel_change"] = r1.project(a.rho_max)
        log.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save(r1.state(), os.path.join(a.out, "ckpt.pt"))     # review A-7: a crash must not lose the work
        json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    if not any("skipped" not in e for e in log):
        sys.exit("all steps were skipped (non-finite gradients); nothing exported")
    r1.detach()
    print(json.dumps(r1.export(a.model, a.out)))


if __name__ == "__main__":
    main()
