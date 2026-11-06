"""Issue #32, stage 1: supervised fine-tuning of Qwen3.5-0.8B on the model's own (selected) non-thinking answers, with a self-written LoRA.

    bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode bench --data results/t1_samples/t1 --select results/t1_sft/M.json --targets all
    bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode train --data ... --select ... --targets all --out models/sft/M
    bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode check-export --out models/sft/M

Design notes (from the two independent reviews in Issue #32):
  * No peft: a LoRA is a forward hook  out + s * ((x A^T) B^T)  on a frozen nn.Linear (A, B in fp32; B starts at 0; s = alpha / r).
  * Training sequences are exactly what vLLM generated: prompt_ids (the non-thinking chat template, ending in <think>\\n\\n</think>\\n\\n) +
    token_ids (including the final <|im_end|>).  Loss = token-level cross-entropy over the response tokens only, at temperature 1, no penalty.
  * Per-layer activation checkpointing; the vocabulary projection is done in chunks (a 248k-vocab logit matrix for 8k tokens would not fit).
  * Export = patch the ORIGINAL safetensors shards in place (W + s B A in fp32, rounded to bf16, original key names), copy tokenizer/config files.
  * Safety: the job saves every 15 min and exits with code 75 (resume with --resume) when the stop file appears, when other programs' GPU memory
    grows by more than --grow-mib for 3 consecutive steps, when the GPU temperature reaches --temp-max, or on CUDA OOM.  Non-finite gradients skip the
    step (abort after 5 in a row).
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXIT_STOP = 75
DEV = os.environ.get("WSP_DEVICE", "cuda")      # "cpu" only for the CPU test of the code path (scripts/test_lora_trainer_cpu.sh)
if DEV == "cpu":
    torch.backends.mkldnn.enabled = False            # oneDNN has no bf16 backward on some CPUs (avx2_vnni_2); the plain kernels are slow but exist
SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")


class LoRA:
    def __init__(self, layer_idx, suffix, lin, r, alpha):
        self.layer, self.suffix, self.lin = layer_idx, suffix, lin
        self.A = nn.Parameter(torch.empty(r, lin.in_features, device=DEV, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=DEV, dtype=torch.float32))
        self.s = alpha / r
        self.handle = lin.register_forward_hook(self.hook)

    def hook(self, m, inp, out):
        return out + (((inp[0].float() @ self.A.t()) @ self.B.t()) * self.s).to(out.dtype)

    def delta(self):                                  # (out, in) fp32
        return (self.B.detach() @ self.A.detach()) * self.s

    def remove(self):
        self.handle.remove()


def attach_lora(layers, targets, r, alpha):
    mods = []
    for i, layer in enumerate(layers):
        for name, m in layer.named_modules():
            if not isinstance(m, nn.Linear) or name not in SUFFIXES:
                continue
            if targets == "l15_19_down" and not (i in (15, 19) and name == "mlp.down_proj"):
                continue
            mods.append(LoRA(i, name, m, r, alpha))
    return mods


def lora_params(mods):
    return [p for m in mods for p in (m.A, m.B)]


def ce_sum(h, lm_head, y, chunk=512):
    """sum of -log p(y | h) over the tokens, in chunks so that the logits of at most `chunk` tokens exist at a time"""
    def fn(hc, yc):
        return F.cross_entropy(lm_head(hc).float(), yc, reduction="sum")
    tot = 0.0
    for c0 in range(0, y.shape[0], chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(fn, h[c0:c0 + chunk], y[c0:c0 + chunk], use_reentrant=False)
    return tot


def seq_hidden(dec, rec):
    ids = rec["prompt_ids"] + rec["token_ids"]
    x = torch.tensor([ids], device=DEV)
    h = dec(input_ids=x, use_cache=False).last_hidden_state[0]
    P = len(rec["prompt_ids"])
    return h[P - 1:P - 1 + len(rec["token_ids"])], x[0, P:]


def load_records(data_dir, select):
    import gzip
    keep = {(i, k) for i, k in json.load(open(select))["pairs"]}
    toks = {}
    for l in gzip.open(os.path.join(data_dir, "tokens.jsonl.gz"), "rt", encoding="utf-8"):
        t = json.loads(l)
        if (t["id"], t["k"]) in keep:
            toks[(t["id"], t["k"])] = {"id": t["id"], "k": t["k"], "prompt_ids": t["prompt_ids"], "token_ids": t["token_ids"]}
    missing = keep - set(toks)
    assert not missing, f"{len(missing)} selected samples have no saved tokens"
    return [toks[p] for p in sorted(toks)]


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:  # noqa: BLE001
        return float("nan")


class Guard:
    def __init__(self, stop_file, grow_mib, temp_max):
        self.stop_file, self.grow, self.tmax = stop_file, grow_mib, temp_max
        self.base, self.hits = None, 0

    def other_mib(self):
        if DEV != "cuda":
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**20 - torch.cuda.memory_reserved() / 2**20

    def arm(self):
        self.base = self.other_mib()

    def check(self, step):
        if os.path.exists(self.stop_file):
            return "stop file"
        if self.base is not None:
            grown = self.other_mib() - self.base
            self.hits = self.hits + 1 if grown > self.grow else 0
            if self.hits >= 3:
                return f"other programs' GPU memory grew by {grown:.0f} MiB"
        if step % 5 == 0:
            t = gpu_temp()
            if t >= self.tmax:
                return f"GPU temperature {t:.0f} C"
        return None


def lr_at(step, total, lr, warm=5, floor=0.1):
    if step < warm:
        return lr * (step + 1) / warm
    p = (step - warm) / max(total - warm, 1)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def export(mods, base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(base_dir):
        p = os.path.join(base_dir, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copyfile(p, os.path.join(out_dir, f))
    by = {(m.layer, m.suffix): m for m in mods}
    done, rel = set(), []
    for sh in sorted(f for f in os.listdir(base_dir) if f.endswith(".safetensors")):
        sd = load_file(os.path.join(base_dir, sh))
        for key in list(sd):
            mt = re.search(r"layers\.(\d+)\.(.+)\.weight$", key)
            if not mt or re.search(r"vision|audio|visual|mtp", key):
                continue
            tag = (int(mt.group(1)), mt.group(2))
            if tag not in by:
                continue
            assert tag not in done, f"duplicate key for {tag}"
            W = sd[key]
            d = by[tag].delta().cpu()
            new = (W.float() + d).to(W.dtype)
            rel.append({"key": key, "rel_fp32": float(d.norm() / W.float().norm()),
                        "rel_bf16": float((new.float() - W.float()).norm() / W.float().norm()), "bf16_vs_fp32": float(((new.float() - W.float()) - d).norm() / max(float(d.norm()), 1e-12))})
            sd[key] = new.contiguous()
            done.add(tag)
        save_file(sd, os.path.join(out_dir, sh), metadata={"format": "pt"})
    assert done == set(by), f"modules not merged: {sorted(set(by) - done)[:5]} ..."
    json.dump(rel, open(os.path.join(out_dir, "merge_log.json"), "w"), indent=1)
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "train", "check-export"], required=True)
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--data", help="dir with tokens.jsonl.gz (eval_baselines.py --save-tokens 1)")
    ap.add_argument("--select", help="json {'pairs': [[id, k], ...]} (scripts/make_sft_sets.py)")
    ap.add_argument("--targets", choices=["all", "l15_19_down"], default="all")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=3, help="bench: number of sequences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/sft/run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=float, default=900.0, help="seconds between checkpoints")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--grow-mib", type=float, default=1500.0)
    ap.add_argument("--temp-max", type=float, default=85.0)
    a = ap.parse_args()
    stop_file = a.stop_file or os.path.join("results", "STOP_train")

    from find_direction import load_model
    if a.mode == "check-export":
        model, dec, _ = load_model(a.out)
        ref = json.load(open(os.path.join(a.out, "export_ref.json")))
        worst = 0.0
        for r in ref["seqs"]:
            with torch.no_grad():
                h, y = seq_hidden(dec, r)
                lp = -F.cross_entropy(model.lm_head(h).float(), y, reduction="none").cpu()
            d = (lp - torch.tensor(r["logp_hooked"])).abs()
            worst = max(worst, float(d.mean()))
            print(f"len {len(lp)}: mean|logp(exported) - logp(base+hooks)| = {float(d.mean()):.4f}  max {float(d.max()):.3f}", flush=True)
        print("CHECK-EXPORT", "OK" if worst < 0.05 else "FAILED (mean difference >= 0.05)", flush=True)
        sys.exit(0 if worst < 0.05 else 2)

    model, dec, _ = load_model(a.model)
    for q in model.parameters():
        q.requires_grad_(False)
    mods = attach_lora(dec.layers, a.targets, a.rank, a.alpha)
    params = lora_params(mods)
    print(f"LoRA modules {len(mods)}, trainable parameters {sum(p.numel() for p in params) / 1e6:.2f} M (rank {a.rank}, alpha {a.alpha}, targets {a.targets})", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                                  # activation checkpointing is only active in training mode; the model has no dropout
    recs = load_records(a.data, a.select)
    assert all(len(r["token_ids"]) <= a.max_len for r in recs), "a selected sample is longer than --max-len (truncating would drop the ending)"
    ntok = sum(len(r["token_ids"]) for r in recs)
    last = {}
    for r in recs:
        last[r["token_ids"][-1]] = last.get(r["token_ids"][-1], 0) + 1
    print(f"{len(recs)} sequences, {ntok} response tokens (mean {ntok / len(recs):.0f}); last-token histogram {dict(sorted(last.items(), key=lambda kv: -kv[1])[:4])}", flush=True)
    lm_head = model.lm_head

    if a.mode == "bench":
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for r in sorted(recs, key=lambda r: -len(r["token_ids"]))[:a.n]:
            t0 = time.time()
            h, y = seq_hidden(dec, r)
            loss = ce_sum(h, lm_head, y) / len(y)
            loss.backward()
            if DEV == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"{len(y)} response tokens: {dt:.1f}s ({len(y) / dt:.0f} tok/s), peak mem {(torch.cuda.max_memory_allocated() if DEV == "cuda" else 0) / 2**30:.1f} GiB, loss/token {float(loss):.3f}", flush=True)
        return

    # ---- train
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    order = []
    for _ in range(a.epochs):
        idx = list(range(len(recs)))
        rnd.shuffle(idx)
        order += idx
    batches = [order[i:i + a.batch_seqs] for i in range(0, len(order), a.batch_seqs)]
    total = len(batches)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    ck = os.path.join(a.out, "ckpt.pt")
    step0, log = 0, []
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        for p, v in zip(params, st["params"]):
            p.data.copy_(v)
        opt.load_state_dict(st["opt"])
        step0, log = st["step"], st["log"]
        print(f"resumed at step {step0} of {total}", flush=True)
    guard = Guard(stop_file, a.grow_mib, a.temp_max)
    bad_in_row, last_save = 0, time.time()

    def save(step):
        torch.save({"params": [p.detach() for p in params], "opt": opt.state_dict(), "step": step, "log": log}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)

    for step in range(step0, total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr)
        batch = [recs[i] for i in batches[step]]
        nt = sum(len(r["token_ids"]) for r in batch)
        opt.zero_grad(set_to_none=True)
        t0, tot = time.time(), 0.0
        try:
            for r in batch:
                h, y = seq_hidden(dec, r)
                s = ce_sum(h, lm_head, y)
                (s / nt).backward()
                tot += float(s)
        except torch.cuda.OutOfMemoryError:
            save(step)
            print("CUDA OOM: checkpoint saved; exiting with code", EXIT_STOP, flush=True)
            sys.exit(EXIT_STOP)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        entry = {"step": step, "loss_per_token": tot / nt, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "tokens": nt, "sec": round(time.time() - t0, 1)}
        if not math.isfinite(gn) or not math.isfinite(tot):
            entry["skipped"] = "non-finite"
            bad_in_row += 1
            if bad_in_row >= 5:
                save(step)
                sys.exit("5 non-finite steps in a row: stopped")
        else:
            bad_in_row = 0
            opt.step()
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if step == step0:
            guard.arm()
        why = guard.check(step)
        if why or time.time() - last_save > a.save_every:
            save(step + 1)
            last_save = time.time()
        if why:
            print(f"STOP: {why}; checkpoint saved at step {step + 1}; exiting with code {EXIT_STOP}", flush=True)
            sys.exit(EXIT_STOP)
    save(total)
    # reference log-probs of the model WITH the hooks (to check the merged export against)
    model.eval()
    ref = []
    with torch.no_grad():
        for r in sorted(recs, key=lambda r: len(r["token_ids"]))[:: max(len(recs) // 3, 1)][:3]:
            r = {"prompt_ids": r["prompt_ids"], "token_ids": r["token_ids"][:2048]}
            h, y = seq_hidden(dec, r)
            lp = -F.cross_entropy(lm_head(h).float(), y, reduction="none")
            ref.append({**r, "logp_hooked": [float(x) for x in lp.cpu()]})
    rel = export(mods, a.model, a.out)
    json.dump({"seqs": ref, "data": a.data, "select": a.select}, open(os.path.join(a.out, "export_ref.json"), "w"))
    json.dump({"args": vars(a), "steps": total, "sequences": len(recs), "tokens": ntok, "final_loss_per_token": log[-1]["loss_per_token"]}, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print("exported to", a.out, "| mean rel. change (bf16)", sum(x["rel_bf16"] for x in rel) / len(rel), "| mean bf16-vs-fp32 error", sum(x["bf16_vs_fp32"] for x in rel) / len(rel), flush=True)


if __name__ == "__main__":
    main()
