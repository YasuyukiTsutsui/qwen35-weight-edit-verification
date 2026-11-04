"""DecisionBench evaluation of Jev-Omni (akhilaaa3/Jev-Omni), text-only, one request per question.

Re-implements the public loader's inference (jev_omni.py: prompt format, last hidden state of the Gemma text backbone
-> normalised -> 256-way linear head, softmax over the first n options) with the backbone optionally loaded in 8-bit
(the bf16 model is ~24 GB; the GPU has 16 GB).

    python3 scripts/eval_jev.py --model models/jev-omni --tag jev_base --subset medium [--fmt v1] [--load 8bit]

Typed question -> (question text, options): not documented by the release; --fmt selects our mapping.
Output: results/<tag>/decisionbench_<subset>/{summary.json, per_question.jsonl, hidden.pt}
"""
import argparse
import json
import os
import time

import torch


def jev_prompt(state, question, options):
    choices = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(options))
    return (f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)}).\n"
            "Output a single number and nothing else.")


def typed_to_options(q, fmt):
    """returns (question text, option strings, option values)."""
    t, ins, cr = q["type"], q["instructions"], q["criteria"]
    if t == "noul":
        vals = [True, False]
        if fmt == "v1":     # criteria inside the options
            opts = [f"Yes. {cr['true']}", f"No. {cr['false']}"]
        else:               # v2: criteria in the question, bare options
            ins = f"{ins}\nYes means: {cr['true']}\nNo means: {cr['false']}"
            opts = ["Yes", "No"]
        return ins, opts, vals
    if t == "choice":
        vals = list(cr)
        if fmt == "v1":
            opts = [f"{k}: {v}" for k, v in cr.items()]
        else:
            ins = ins + "\n" + "\n".join(f"{k} means: {v}" for k, v in cr.items())
            opts = vals
        return ins, opts, vals
    if t == "score":
        vals = list(range(len(cr)))
        if fmt == "v1":
            opts = [f"{i}: {c}" for i, c in enumerate(cr)]
        else:
            ins = ins + "\n" + "\n".join(f"{i} means: {c}" for i, c in enumerate(cr))
            opts = [str(i) for i in vals]
        return ins, opts, vals
    raise ValueError(t)


class Head256(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, hidden))
        self.register_buffer("sd", torch.ones(1, hidden))
        self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

    def forward(self, features, n):
        z = self.linear((features.float() - self.mu) / self.sd)
        return z[:, :n]


def load(path, mode):
    import transformers
    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if mode in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        skip = ["lm_head", "vision", "audio", "embed_vision", "embed_audio"]
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip) if mode == "8bit" else
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=skip))
    kw["low_cpu_mem_usage"] = True
    model = cls.from_pretrained(path, **kw).eval()
    dec = model.model.language_model
    dc = json.load(open(os.path.join(path, "decision_config.json")))
    head = Head256(dc["hidden_size"]).cuda().eval()
    head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cuda", weights_only=True))
    return model, dec, head, AutoProcessor.from_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/jev-omni")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", default="medium")
    ap.add_argument("--fmt", default="v1")
    ap.add_argument("--load", default="8bit")
    ap.add_argument("--limit", type=int, default=0, help="first N states only")
    ap.add_argument("--gpu-fraction", type=float, default=0.8, help="hard cap on this process's GPU memory (fraction)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)   # leave headroom for the display / other processes
    out = os.path.join("results", a.tag, f"decisionbench_{a.subset}_{a.fmt}")
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(f"data/decision-bench/data/{a.subset}.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    model, dec, head, proc = load(a.model, a.load)
    cap = {}
    dec.register_forward_hook(lambda m, i, o: cap.__setitem__(
        "h", (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])[:, -1].float()))
    extra = {"logits_to_keep": 1}
    recs, H = [], []
    part = os.path.join(out, "partial.pt")
    if os.path.exists(part):          # resume after an interruption
        d = torch.load(part)
        recs, H = d["recs"], d["H"]
        print(f"resumed with {len(recs)} questions", flush=True)
    done = {(x["state"], x["key"]) for x in recs}
    t0 = time.time()
    for r in rows:
        qs, ans = json.loads(r["questions"]), json.loads(r["answers"])
        for key, q in qs.items():
            if (r["id"], key) in done:
                continue
            qtext, opts, vals = typed_to_options(q, a.fmt)
            msg = [{"role": "user", "content": [{"type": "text", "text": jev_prompt(r["state"], qtext, opts)}]}]
            inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**inp, use_cache=False, **extra)
                p = head(cap["h"], len(opts))[0].softmax(-1).float().cpu()
            best = int(p.argmax())
            gold = vals.index(ans[key]) if ans[key] in vals else None
            recs.append({"state": r["id"], "family": r["family"], "key": key, "type": q["type"], "n_opts": len(opts),
                         "gold": gold, "pred": best, "correct": best == gold, "probs": p.tolist(),
                         "tokens": int(inp["input_ids"].shape[1])})
            H.append(cap["h"][0].cpu())
        torch.save({"recs": recs, "H": H}, part + ".tmp")
        os.replace(part + ".tmp", part)
        print(f"{len(recs)} questions, acc so far {sum(x['correct'] for x in recs) / len(recs):.3f}", flush=True)
    # metrics: micro, state-macro, by type, ECE (10 bins)
    micro = sum(x["correct"] for x in recs) / len(recs)
    by_state = {}
    for x in recs:
        by_state.setdefault(x["state"], []).append(x["correct"])
    macro = sum(sum(v) / len(v) for v in by_state.values()) / len(by_state)
    by_type = {t: sum(x["correct"] for x in recs if x["type"] == t) / max(1, sum(x["type"] == t for x in recs))
               for t in ("noul", "choice", "score")}
    conf = torch.tensor([max(x["probs"]) for x in recs])
    corr = torch.tensor([float(x["correct"]) for x in recs])
    ece = 0.0
    for b in range(10):
        m = (conf > b / 10) & (conf <= (b + 1) / 10)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - corr[m].mean()).abs()
    summ = {"tag": a.tag, "subset": a.subset, "fmt": a.fmt, "load": a.load, "questions": len(recs),
            "micro_acc": micro, "state_macro_acc": macro, "by_type": by_type, "ece": float(ece),
            "seconds": round(time.time() - t0)}
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    torch.save(torch.stack(H), os.path.join(out, "hidden.pt"))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
