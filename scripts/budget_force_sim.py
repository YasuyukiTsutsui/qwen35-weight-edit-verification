"""Budget forcing, simulated on saved samples (Issue #30).  Run inside the container (needs vLLM and math-verify):

    bash scripts/drun.sh python3 scripts/budget_force_sim.py --model models/Qwen3.5-0.8B --source rb16k_base --tag bf_base

For every saved sample of --source (a normal run: thinking on, 16,384 cap) and every budget B:
  * the thinking closed within the first B tokens  -> the saved result is kept (the same prefix would have been produced by a run capped at B);
  * the sample ended before B tokens                -> kept;
  * otherwise the first B tokens are kept, the thinking is closed with a fixed sentence ("Considering the limited time by the user, I have to
    give the solution based on the thinking directly now.</think>") and the model writes the answer, at most 16,384 - B - (sentence) new tokens,
    so that the total never exceeds the normal run's cap.
The first B tokens of a 16,384-token run are distributed exactly like the output of a B-capped run, so this equals running the two-stage
procedure from scratch, except that the text is re-tokenised and the presence penalty counts only the answer tokens.
Output: results/<tag>/<set>/{summary.json, samples_B<B>.jsonl.gz}.   The model is the one that writes the answer (use the edited model with the edit's own samples).
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_math as E  # noqa: E402
import grade as G  # noqa: E402

FORCED = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="tag of the saved normal run (results/<source>/<set>/samples.jsonl.gz)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows-file", default="data/practice_select128.jsonl")
    ap.add_argument("--budgets", default="2048,4096,8192,12288")
    ap.add_argument("--total-cap", type=int, default=16384)
    ap.add_argument("--preset", default="qwen35_think")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    set_name = os.path.splitext(os.path.basename(a.rows_file))[0]

    from vllm import LLM, SamplingParams
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.rows_file, encoding="utf-8"))}
    src = [json.loads(l) for l in gzip.open(os.path.join("results", a.source, set_name, "samples.jsonl.gz"), "rt", encoding="utf-8")]
    out = os.path.join("results", a.tag, set_name)
    os.makedirs(out, exist_ok=True)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, seed=a.seed,
              max_model_len=a.total_cap + 2048, max_num_seqs=a.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    tok = llm.get_tokenizer()
    close_id = tok.convert_tokens_to_ids("</think>")
    forced_ids = tok.encode(FORCED, add_special_tokens=False)
    prompt_ids = {i: tok.encode(tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + E.INSTRUCTION}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)
                  for i, r in rows.items()}
    ids_of = [tok.encode(s["text"], add_special_tokens=False) for s in src]
    base_sp = E.PRESETS[a.preset]

    jobs = []      # (budget, source index)
    for b in budgets:
        for n, s in enumerate(src):
            if close_id in ids_of[n][:b] or s["tokens"] <= b:
                continue
            jobs.append((b, n))
    prompts = [{"prompt_token_ids": prompt_ids[src[n]["id"]] + ids_of[n][:b] + forced_ids} for b, n in jobs]
    sps = [SamplingParams(n=1, max_tokens=a.total_cap - b - len(forced_ids), seed=a.seed, **base_sp) for b, _ in jobs]
    print(f"forced continuations: {len(jobs)} (of {len(budgets) * len(src)} sample-budget pairs)", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sps) if prompts else []
    gen_s = time.time() - t0
    cont = {(b, n): o.outputs[0] for (b, n), o in zip(jobs, outs)}

    summ = {"model": a.model, "source": a.source, "tag": a.tag, "set": a.rows_file, "preset": a.preset, "seed": a.seed, "total_cap": a.total_cap,
            "forced_sentence": FORCED, "n_source_samples": len(src), "gen_seconds": round(gen_s), "budgets": {}, **E.run_environment(a)}
    for b in budgets:
        ok = forced = 0
        toks = []
        with gzip.open(os.path.join(out, f"samples_B{b}.jsonl.gz"), "wt", encoding="utf-8") as f:
            for n, s in enumerate(src):
                if (b, n) in cont:
                    o = cont[(b, n)]
                    full = tok.decode(ids_of[n][:b], skip_special_tokens=True) + FORCED + o.text
                    pred = G.extract(full)
                    good = G.is_correct(pred, rows[s["id"]]["answer"])
                    rec = {"id": s["id"], "k": s["k"], "answer": s["answer"], "pred": pred, "correct": good, "finish": o.finish_reason,
                           "tokens": b + len(forced_ids) + len(o.token_ids), "forced": True, "text": full}
                    forced += 1
                else:
                    good = s["correct"]
                    rec = {"id": s["id"], "k": s["k"], "answer": s["answer"], "pred": s["pred"], "correct": good, "finish": s["finish"],
                           "tokens": s["tokens"], "forced": False, "text": s["text"]}
                ok += bool(good)
                toks.append(rec["tokens"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summ["budgets"][str(b)] = {"acc": ok / len(src), "ok": ok, "forced_share": forced / len(src), "mean_tokens": sum(toks) / len(toks)}
        print(f"B={b}: acc {100 * ok / len(src):.1f}%  forced {100 * forced / len(src):.0f}%  mean tokens {sum(toks) / len(toks):.0f}", flush=True)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ["budgets"]))


if __name__ == "__main__":
    main()
