"""Math evaluation with vLLM (run inside the Docker image; see scripts/drun.sh).

    python scripts/eval_math.py --model models/Qwen3.5-0.8B --tag qwen_base --set math500 --n 100 --samples 2

Protocol defaults follow the blog's Experiment 8 (Qwen3.5-0.8B, thinking on): user turn = problem + "\\n" +
"Please reason step by step, and put your final answer within \\boxed{}.", temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, max 32,768 new tokens. Problems: a fixed seeded subset (same for every model).
Output: results/<tag>/<set>/{summary.json, samples.jsonl.gz}
"""
import argparse
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade as G  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PRESETS = {
    # Qwen3.5 model card, thinking mode (as in the blog)
    "qwen35_think": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5),
    # Gemma model card sampling
    "gemma": dict(temperature=1.0, top_p=0.95, top_k=64),
}


def load_rows(name, n, seed=0):
    rows = [json.loads(l) for l in open(os.path.join("data", f"{name}.jsonl"), encoding="utf-8")]
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def run_environment(a):
    """Conditions of the run that were not recorded before 2026-11-05 (Issue #29): seed, batch limits, vLLM version, GPU."""
    env = {"seed": a.seed, "max_num_seqs": a.max_num_seqs, "gpu_mem": a.gpu_mem}
    try:
        import vllm
        env["vllm_version"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0)
        env["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="math500")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset-seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--preset", default="qwen35_think")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--keep-special", type=int, default=0, help="keep special tokens in the text (Gemma 4 channel markers)")
    ap.add_argument("--rows-file", default=None, help="jsonl of rows to use instead of --set/--n")
    ap.add_argument("--save-tokens", type=int, default=0,
                    help="also write tokens.jsonl.gz: output token ids and the log-prob of each sampled token under the "
                         "processed (temperature / top-k / top-p / penalty) distribution, for policy-gradient training")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (vary it to get fresh samples)")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    rows = ([json.loads(l) for l in open(a.rows_file, encoding="utf-8")] if a.rows_file
            else load_rows(a.set, a.n, a.offset_seed))
    out = os.path.join("results", a.tag, os.path.splitext(os.path.basename(a.rows_file))[0] if a.rows_file else a.set)
    os.makedirs(out, exist_ok=True)
    extra = {"logprobs_mode": "processed_logprobs"} if a.save_tokens else {}
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, seed=a.seed,
              max_model_len=a.max_tokens + 2048, max_num_seqs=a.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0}, **extra)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=bool(a.thinking))
               for r in rows]
    sp = SamplingParams(n=a.samples, max_tokens=a.max_tokens, seed=a.seed, skip_special_tokens=not a.keep_special,
                        **({"logprobs": 0} if a.save_tokens else {}), **PRESETS[a.preset])
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    tot = ok = cut = loops_cut = closed = 0
    toks = []
    tf = gzip.open(os.path.join(out, "tokens.jsonl.gz"), "wt", encoding="utf-8") if a.save_tokens else None
    with gzip.open(os.path.join(out, "samples.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            for k, s in enumerate(o.outputs):
                if tf is not None:
                    ids = list(s.token_ids)
                    lps = [float(s.logprobs[i][t].logprob) for i, t in enumerate(ids)]
                    tf.write(json.dumps({"id": r["id"], "k": k, "prompt_ids": list(o.prompt_token_ids),
                                         "token_ids": ids, "logprobs": lps}) + "\n")
                pred = G.extract(s.text)
                good = G.is_correct(pred, r["answer"])
                ls = G.loop_signals(s.text)
                is_cut = s.finish_reason == "length"
                tot += 1
                ok += good
                cut += is_cut
                loops_cut += is_cut and (ls["verbatim_loop_end"] or ls["repetitive_end"])
                closed += G.think_closed(s.text)
                toks.append(len(s.token_ids))
                f.write(json.dumps({"id": r["id"], "k": k, "answer": r["answer"], "pred": pred, "correct": good,
                                    "finish": s.finish_reason, "tokens": len(s.token_ids), **ls,
                                    "text": s.text},
                                   ensure_ascii=False) + "\n")
    if tf is not None:
        tf.close()
    summ = {"model": a.model, "tag": a.tag, "set": a.rows_file or a.set, "n_problems": len(rows),
            "samples": a.samples, "max_tokens": a.max_tokens, "preset": a.preset, "thinking": a.thinking,
            "acc": ok / tot, "ok": ok, "total": tot, "cut_share": cut / tot, "cut_with_loop_tail": loops_cut,
            "think_closed_share": closed / tot, "mean_tokens": sum(toks) / len(toks), "gen_seconds": round(gen_s),
            **run_environment(a)}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
