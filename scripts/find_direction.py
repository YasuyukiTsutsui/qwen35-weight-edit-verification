"""Find a "repetition" direction in MLP outputs by contrasting looping vs productive reasoning (teacher-forced).

Run in the GPU container after a base-model generation pass on the find split:
    python scripts/find_direction.py --model models/Qwen3.5-0.8B --samples results/qwen_base/practice_find/samples.jsonl.gz \
        --out results/qwen_dir

Groups of response tokens (all teacher-forced with the full prefix as context):
  rep  : the last --span tokens of answers cut at max tokens whose tail is a loop (grade.loop_signals)
  prod : tokens [--skip, --skip+span) of the same looping answers (matched problems) and the thinking of answers that
         finished correctly
For every layer L we accumulate the mean MLP output (residual write, d_model) and the mean MLP hidden activation (input
of down_proj, d_ff) per group. Direction u_L = normalize(mean_rep - mean_prod of the MLP output). Neuron score
c_j = (u_L . W_down[:, j]) * (mean_rep a_j - mean_prod a_j): how much neuron j's write along u_L grows under looping.
No gradients, no fitting: only means of activations.
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


def load_model(path):
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    model = cls.from_pretrained(path, dtype=getattr(torch, os.environ.get("WSP_DTYPE", "bfloat16")), device_map=os.environ.get("WSP_DEVICE", "cuda")).eval()
    for p in ("model.language_model", "model.text_model", "language_model.model", "model"):
        node = model
        for part in p.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers"):
            return model, node, p
    raise RuntimeError("no decoder layers found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--span", type=int, default=1024)
    ap.add_argument("--skip", type=int, default=256)
    ap.add_argument("--max-per-group", type=int, default=96)
    ap.add_argument("--thinking", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [json.loads(l) for l in gzip.open(a.samples, "rt", encoding="utf-8")]
    problems = {json.loads(l)["id"]: json.loads(l) for f in ("data/practice_find.jsonl",)
                for l in open(f, encoding="utf-8")}
    loops = [r for r in recs if r["finish"] == "length" and (r["verbatim_loop_end"] or r["repetitive_end"])]
    good = [r for r in recs if r["finish"] == "stop" and r["correct"] and G.think_closed(r["text"])]
    rnd = random.Random(0)
    rnd.shuffle(loops)
    rnd.shuffle(good)
    loops, good = loops[:a.max_per_group], good[:a.max_per_group]
    print(f"looping answers: {len(loops)}, correct finished: {len(good)} (of {len(recs)})", flush=True)

    model, dec, path = load_model(a.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    layers = dec.layers
    nL = len(layers)
    dm = layers[0].mlp.down_proj.out_features
    dff = layers[0].mlp.down_proj.in_features
    acc = {g: {"out": torch.zeros(nL, dm, dtype=torch.float64, device="cuda"),
               "act": torch.zeros(nL, dff, dtype=torch.float64, device="cuda"), "n": 0} for g in ("rep", "prod")}
    cur = {}

    def hook(i):
        def f(mod, inp, out):
            cur[i] = (inp[0][0].float(), out[0].float())   # (T, dff), (T, dm)
        return f
    hs = [layers[i].mlp.down_proj.register_forward_hook(hook(i)) for i in range(nL)]

    def run(rec, spans):
        """spans: list of (group, start, end) in response-token coordinates."""
        p = problems[rec["id"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=bool(a.thinking))
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        rid = tok(rec["text"], add_special_tokens=False)["input_ids"]
        end = max(e for _, _, e in spans)
        ids = torch.tensor([pid + rid[:end]], device="cuda")
        cur.clear()
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, **({"logits_to_keep": 1}))
        for g, s, e in spans:
            s, e = len(pid) + s, len(pid) + min(e, len(rid))
            if e <= s:
                continue
            for i in range(nL):
                act, out = cur[i]
                acc[g]["act"][i] += act[s:e].double().sum(0)
                acc[g]["out"][i] += out[s:e].double().sum(0)
            acc[g]["n"] += e - s

    for k, r in enumerate(loops):
        n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        run(r, [("prod", a.skip, a.skip + a.span), ("rep", n - a.span, n)])
        if k % 10 == 0:
            print(f"loop {k}/{len(loops)} ({n} tokens)", flush=True)
    for k, r in enumerate(good):
        think = G.split_think(r["text"])[0]
        n = len(tok(think, add_special_tokens=False)["input_ids"])
        mid = max(0, n // 2 - a.span // 2)
        run(r, [("prod", mid, min(n, mid + a.span))])
    for h in hs:
        h.remove()

    res = {"model": a.model, "path": path, "n_tokens": {g: acc[g]["n"] for g in acc},
           "n_loops": len(loops), "n_good": len(good), "layers": []}
    U = []
    for i in range(nL):
        mo = {g: acc[g]["out"][i] / acc[g]["n"] for g in acc}
        ma = {g: acc[g]["act"][i] / acc[g]["n"] for g in acc}
        d = mo["rep"] - mo["prod"]
        u = d / d.norm()
        W = layers[i].mlp.down_proj.weight.double()      # (dm, dff)
        wu = u @ W                                        # u . W[:, j]
        c = wu * (ma["rep"] - ma["prod"])
        top = torch.argsort(c, descending=True)
        # share of the (rep - prod) write along u carried by the top-64 neurons
        share64 = float(c[top[:64]].sum() / c.sum())
        res["layers"].append({"layer": i, "contrast_norm": float(d.norm()),
                              "rel_contrast": float(d.norm() / (0.5 * (mo["rep"].norm() + mo["prod"].norm()))),
                              "share_top64": share64, "top_neurons": top[:256].tolist(),
                              "neuron_scores_top": c[top[:256]].tolist()})
        U.append(u.float().cpu())
    torch.save(torch.stack(U), os.path.join(a.out, "directions.pt"))
    json.dump(res, open(os.path.join(a.out, "direction.json"), "w"))
    for L in res["layers"]:
        print(f"L{L['layer']:2d} |d|={L['contrast_norm']:.3f} rel={L['rel_contrast']:.3f} top64share={L['share_top64']:.2f}")


if __name__ == "__main__":
    main()
