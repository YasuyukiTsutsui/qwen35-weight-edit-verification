"""Second-order statistics for the equation variants of Issue #28 (ROME/LDA read vector, oblique projector). Run in the GPU container.

Same contexts as scripts/find_direction.py (the 'rep' = last span tokens of looping answers; 'prod' = tokens [skip, skip+span) of the same
answers plus the middle of the thinking of correct finished answers; teacher-forced on the BASE model), for layers 15 and 19 only.
For each layer and group we accumulate, over all tokens of the group:
    a  = down_proj INPUT (3584-d):   sum a, sum a a^T
    o  = down_proj OUTPUT (1024-d):  sum o, sum o o^T
All uncentered (the layer has no bias).  Output: results/stats/stats_L{15,19}.pt  (float32 tensors, counts).

    bash scripts/drun.sh python3 scripts/collect_stats.py --samples results/qwen_base/practice_find/samples.jsonl.gz --out results/stats
"""
import argparse
import gzip
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade as G  # noqa: E402
from eval_math import INSTRUCTION  # noqa: E402
from find_direction import load_model  # noqa: E402

LAYERS = (15, 19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="results/stats")
    ap.add_argument("--span", type=int, default=1024)
    ap.add_argument("--skip", type=int, default=256)
    ap.add_argument("--max-per-group", type=int, default=96)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in gzip.open(a.samples, "rt", encoding="utf-8")]
    problems = {json.loads(l)["id"]: json.loads(l) for l in open("data/practice_find.jsonl", encoding="utf-8")}
    loops = [r for r in recs if r["finish"] == "length" and (r["verbatim_loop_end"] or r["repetitive_end"])]
    good = [r for r in recs if r["finish"] == "stop" and r["correct"] and G.think_closed(r["text"])]
    rnd = random.Random(0)                      # same shuffles as find_direction.py -> same sequences
    rnd.shuffle(loops)
    rnd.shuffle(good)
    loops, good = loops[:a.max_per_group], good[:a.max_per_group]
    print(f"looping answers: {len(loops)}, correct finished: {len(good)}", flush=True)

    model, dec, _ = load_model(a.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    layers = dec.layers
    dm, dff = layers[0].mlp.down_proj.out_features, layers[0].mlp.down_proj.in_features
    acc = {(g, L): {"sa": torch.zeros(dff, dtype=torch.float64, device="cuda"), "saa": torch.zeros(dff, dff, dtype=torch.float64, device="cuda"),
                    "so": torch.zeros(dm, dtype=torch.float64, device="cuda"), "soo": torch.zeros(dm, dm, dtype=torch.float64, device="cuda"), "n": 0}
           for g in ("rep", "prod") for L in LAYERS}
    cur = {}

    def hook(i):
        def f(mod, inp, out):
            cur[i] = (inp[0][0].float(), out[0].float())
        return f
    hs = [layers[i].mlp.down_proj.register_forward_hook(hook(i)) for i in LAYERS]

    def run(rec, spans):
        p = problems[rec["id"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        rid = tok(rec["text"], add_special_tokens=False)["input_ids"]
        end = max(e for _, _, e in spans)
        ids = torch.tensor([pid + rid[:end]], device="cuda")
        cur.clear()
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        for g, s, e in spans:
            s, e = len(pid) + s, len(pid) + min(e, len(rid))
            if e <= s:
                continue
            for L in LAYERS:
                act, out = cur[L]
                x, y = act[s:e], out[s:e]
                d = acc[(g, L)]
                d["sa"] += x.double().sum(0)
                d["saa"] += (x.T @ x).double()
                d["so"] += y.double().sum(0)
                d["soo"] += (y.T @ y).double()
                d["n"] += e - s

    for k, r in enumerate(loops):
        n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        run(r, [("prod", a.skip, a.skip + a.span), ("rep", n - a.span, n)])
        if k % 10 == 0:
            print(f"loop {k}/{len(loops)}", flush=True)
    for k, r in enumerate(good):
        think = G.split_think(r["text"])[0]
        n = len(tok(think, add_special_tokens=False)["input_ids"])
        mid = max(0, n // 2 - a.span // 2)
        run(r, [("prod", mid, min(n, mid + a.span))])
    for h in hs:
        h.remove()
    for L in LAYERS:
        torch.save({g: {k: (v.float().cpu() if torch.is_tensor(v) else v) for k, v in acc[(g, L)].items()} for g in ("rep", "prod")},
                   os.path.join(a.out, f"stats_L{L}.pt"))
    print("STATSDONE", {g: acc[(g, 15)]["n"] for g in ("rep", "prod")}, flush=True)


if __name__ == "__main__":
    main()
