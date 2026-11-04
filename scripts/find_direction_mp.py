"""Matched-pair repetition direction + teacher-forced layer scan (forward passes only; no generation, no training).

Why: the first direction (find_direction.py) contrasts the END of 16k-token loops with the middle of other answers, so
position and context length are confounded with "being in a loop". Here the contrast is between the SAME tokens seen the
first time (the loop unit's first occurrence) and seen again as a repeat (units 3-5 of the loop):
    d_L = mean over loops of ( mean_t out_L[repeat tokens] - mean_t out_L[first-occurrence tokens] ).

Sequences are truncated to  prompt + response[s-768 : s+5p]  (s = start of the periodic region, p = period) so they are
short. Directions come from the --train loops; the teacher-forced scan runs on disjoint --test loops and on correct
finished answers:  for an edit  W[:,S] -= alpha u (u^T W[:,S])  with a fixed relative change rho = ||dW||_F/||W||_F,
dR = mean change of the log-prob of the actual next token inside repeats (want < 0), dF / dP / dG = the same on
first occurrences, pre-loop reasoning and correct answers (want ~ 0).

    python3 scripts/find_direction_mp.py --model models/Qwen3.5-0.8B --samples results/qwen_base/practice_find/samples.jsonl.gz \
        --old-dir results/qwen_dir --pub models/Qwen3.5-0.8B-p50c --out results/qwen_dir_mp
"""
import argparse
import gzip
import json
import os
import random
import re
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade as G  # noqa: E402
from eval_math import INSTRUCTION  # noqa: E402
from find_direction import load_model  # noqa: E402


def find_loop(a, min_p=6, max_p=700, tail=3000):
    """a: np.array of response token ids. Returns (s, p): s = first index of the periodic region, p = period; or None."""
    n = len(a)
    if n < 2000:
        return None
    T = a[-tail:]
    p = None
    for q in range(min_p, min(max_p, len(T) // 4) + 1):
        if (T[q:] == T[:-q]).mean() >= 0.9:
            p = q
            break
    if p is None:
        return None
    eq = a[p:] == a[:-p]
    good = 0
    for b in range(1, len(eq) // p + 1):
        blk = eq[len(eq) - b * p: len(eq) - (b - 1) * p]
        if blk.mean() >= 0.9:
            good = b
        else:
            break
    if good < 6:
        return None
    i0 = n - good * p - p          # approx. index of the first occurrence's start (eq[i] compares a[i+p] with a[i])
    if i0 < 0:
        return None
    while i0 > 0 and eq[i0 - 1]:   # token-level refinement: extend back while the sequence is still periodic
        i0 -= 1
    return i0 + p, p               # s = first index of the first REPEAT; the first occurrence is a[s-p : s]


def load_down(model_dir, L):
    pat = re.compile(rf"layers\.{L}\.mlp\.down_proj\.weight$")
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(model_dir, f))
            ks = [k for k in sd if pat.search(k) and not re.search(r"vision|audio|visual|mtp", k)]
            if ks:
                return sd[ks[0]].double()
    raise KeyError(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--pub", required=True, help="published-edit model dir (to recover its u and columns)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--good", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.08)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--pre", type=int, default=768)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [json.loads(l) for l in gzip.open(a.samples, "rt", encoding="utf-8")]
    problems = {json.loads(l)["id"]: json.loads(l) for l in open("data/practice_find.jsonl", encoding="utf-8")}
    model, dec, _ = load_model(a.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    layers = dec.layers
    nL = len(layers)
    rnd = random.Random(0)

    def prompt_ids(rec):
        p = problems[rec["id"]]
        s = tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return tok(s, add_special_tokens=False)["input_ids"]

    loops_all = [r for r in recs if r["finish"] == "length" and (r["verbatim_loop_end"] or r["repetitive_end"])]
    rnd.shuffle(loops_all)
    seqs = []                                                   # dicts: ids, wins
    for r in loops_all:
        if len(seqs) >= a.train + a.test:
            break
        rid = np.asarray(tok(r["text"], add_special_tokens=False)["input_ids"])
        fl = find_loop(rid)
        if fl is None:
            continue
        s, p = fl
        lo = max(0, s - a.pre)
        pid = prompt_ids(r)
        ids = pid + rid[lo: s + 5 * p].tolist()
        o = len(pid) - lo
        seqs.append({"ids": ids, "p": p, "s": s, "kind": "loop",
                     "wins": {"F": (o + s - p, o + s), "R": (o + s + 2 * p, o + s + 5 * p),
                              "P": (len(pid) + 32, o + s - p)}})
    print(f"usable loops: {len(seqs)} (period median {int(np.median([q['p'] for q in seqs]))})", flush=True)
    train, test = seqs[:a.train], seqs[a.train:a.train + a.test]
    goods = []
    gl = [r for r in recs if r["finish"] == "stop" and r["correct"] and G.think_closed(r["text"])]
    rnd.shuffle(gl)
    for r in gl[:a.good]:
        think = G.split_think(r["text"])[0]
        tid = tok(think, add_special_tokens=False)["input_ids"][:1280]
        if len(tid) < 600:
            continue
        pid = prompt_ids(r)
        goods.append({"ids": pid + tid, "kind": "good", "wins": {"G": (len(pid) + 256, len(pid) + len(tid))}})
    print(f"train loops {len(train)}, test loops {len(test)}, good {len(goods)}", flush=True)

    # ---- direction from matched pairs (train loops)
    dm, dff = layers[0].mlp.down_proj.out_features, layers[0].mlp.down_proj.in_features
    dsum = torch.zeros(nL, dm, dtype=torch.float64, device="cuda")
    aR = torch.zeros(nL, dff, dtype=torch.float64, device="cuda")
    aF = torch.zeros(nL, dff, dtype=torch.float64, device="cuda")
    cur = {}
    hs = [layers[i].mlp.down_proj.register_forward_hook(
        (lambda i: lambda m, inp, out: cur.__setitem__(i, (inp[0][0], out[0])))(i)) for i in range(nL)]
    for q in train:
        cur.clear()
        with torch.inference_mode():
            dec(input_ids=torch.tensor([q["ids"]], device="cuda"), use_cache=False)
        (r0, r1), (f0, f1) = q["wins"]["R"], q["wins"]["F"]
        for i in range(nL):
            act, out = cur[i]
            dsum[i] += out[r0:r1].double().mean(0) - out[f0:f1].double().mean(0)
            aR[i] += act[r0:r1].double().mean(0)
            aF[i] += act[f0:f1].double().mean(0)
    for h in hs:
        h.remove()
    N = len(train)
    U_new, top_new, cos_old = [], [], []
    U_old = torch.load(os.path.join(a.old_dir, "directions.pt")).double().cuda()
    info_old = json.load(open(os.path.join(a.old_dir, "direction.json")))
    res = {"layers": [], "n_train": N}
    for i in range(nL):
        d = dsum[i] / N
        u = d / d.norm()
        W = layers[i].mlp.down_proj.weight.double()
        c = (u @ W) * ((aR[i] - aF[i]) / N)
        top = torch.argsort(c, descending=True)
        old_top = set(info_old["layers"][i]["top_neurons"][:a.k])
        res["layers"].append({"layer": i, "contrast_norm": float(d.norm()), "cos_old_u": float(u @ U_old[i]),
                              "overlap_top_old": len(old_top & set(top[:a.k].tolist())),
                              "share_topk": float(c[top[:a.k]].sum() / c.sum()), "top_neurons": top[:256].tolist()})
        U_new.append(u.float().cpu())
        top_new.append(top[:a.k].cpu())
    torch.save(torch.stack(U_new), os.path.join(a.out, "directions.pt"))
    json.dump(res, open(os.path.join(a.out, "direction.json"), "w"))

    # ---- published edit: recover u and columns for layers 15, 19
    pub = {}
    for L in (15, 19):
        D = (load_down(a.model, L) - load_down(a.pub, L)).cuda()
        S = torch.nonzero(D.abs().sum(0) > 0)[:, 0]
        Uu, Ss, _ = torch.linalg.svd(D[:, S], full_matrices=False)
        u = Uu[:, 0]
        if float(u @ U_new[L].double().cuda()) < 0:
            u = -u
        W = layers[L].mlp.down_proj.weight.double()
        alpha = float(D[:, S].norm() / (u @ W[:, S]).norm())
        pub[L] = (u, S, alpha)
        print(f"published L{L}: {len(S)} columns, alpha={alpha:.3f}, cos(u_pub, u_new)={float(u @ U_new[L].double().cuda()):.3f}, "
              f"cos(u_pub, u_old)={float(u @ U_old[L]):.3f}", flush=True)

    # ---- teacher-forced scan
    @torch.no_grad()
    def lp(q):
        x = torch.tensor([q["ids"]], device="cuda")
        h = dec(input_ids=x, use_cache=False).last_hidden_state[0]
        out = {}
        for name, (w0, w1) in q["wins"].items():
            parts = []
            for c0 in range(w0, w1, 512):
                c1 = min(w1, c0 + 512)
                lg = model.lm_head(h[c0 - 1:c1 - 1]).float()
                parts.append(torch.log_softmax(lg, -1).gather(1, x[0, c0:c1, None])[:, 0])
            out[name] = torch.cat(parts).cpu()
        return out

    evalset = test + goods
    base = [lp(q) for q in evalset]
    print("base log-probs cached", flush=True)

    def run_cfg(L, u, S, rho):
        W = layers[L].mlp.down_proj.weight
        Sidx = S.cuda()
        WSu = (W[:, Sidx].double().T @ u).float()
        alpha = rho * float(W.double().norm()) / float(WSu.double().norm())
        uf = u.float()

        def hook(m, inp, out):
            dlt = (inp[0][..., Sidx].float() @ WSu)[..., None] * uf
            return out - (alpha * dlt).to(out.dtype)
        h = layers[L].mlp.down_proj.register_forward_hook(hook)
        agg = {}
        for q, b in zip(evalset, base):
            e = lp(q)
            for name in e:
                agg.setdefault(name, []).append((e[name] - b[name]).mean().item())
        h.remove()
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        off = np.mean([abs(m[k]) for k in ("F", "P", "G") if k in m])
        return {"alpha": alpha, **{"d" + k: m[k] for k in m}, "selectivity": -m["R"] / max(off, 1e-3)}

    rows = []
    for L in range(nL):
        r = run_cfg(L, U_new[L].double().cuda(), top_new[L], a.rho)
        rows.append({"layer": L, "source": "matched", "rho": a.rho, **r})
        print(f"L{L:2d} matched rho={a.rho:.2f} alpha={r['alpha']:.2f} dR={r['dR']:+.4f} dF={r['dF']:+.4f} dP={r['dP']:+.4f} "
              f"dG={r['dG']:+.4f} sel={r['selectivity']:.1f}", flush=True)
    g = torch.Generator().manual_seed(0)
    for L in (15, 19):
        old_S = torch.tensor(info_old["layers"][L]["top_neurons"][:a.k])
        rr = torch.randn(U_new[L].shape, generator=g, dtype=torch.float64).cuda()
        un = U_new[L].double().cuda()
        rr = rr - (rr @ un) * un
        rr = rr / rr.norm()
        srcs = {"old": (U_old[L], old_S), "published": (pub[L][0], pub[L][1].cpu()),
                "matched": (U_new[L].double().cuda(), top_new[L]), "random_orth": (rr, top_new[L]),
                "matched_dirOnly_oldcols": (U_new[L].double().cuda(), old_S)}
        for name, (u, S) in srcs.items():
            for rho in (0.04, 0.08, 0.12):
                r = run_cfg(L, u, S, rho)
                rows.append({"layer": L, "source": name, "rho": rho, **r})
                print(f"L{L:2d} {name:24s} rho={rho:.2f} alpha={r['alpha']:.2f} dR={r['dR']:+.4f} dF={r['dF']:+.4f} dP={r['dP']:+.4f} "
                      f"dG={r['dG']:+.4f} sel={r['selectivity']:.1f}", flush=True)
    json.dump({"rows": rows, "n_test_loops": len(test), "n_good": len(goods)}, open(os.path.join(a.out, "tf_scan.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
