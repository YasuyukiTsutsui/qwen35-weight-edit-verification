"""Pre-run check for scripts/budget_force_sim.py (Issue #30): the think-end token, the forced sentence, the chat-template tail, and how many
saved samples each budget would force.  CPU only, no model weights.   bash scripts/drun.sh python3 scripts/check_bf_tokens.py rb16k_base
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from budget_force_sim import FORCED  # noqa: E402

tok = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
cid = tok.convert_tokens_to_ids("</think>")
f = tok.encode(FORCED, add_special_tokens=False)
print("close id", cid, "| forced sentence tokens", len(f), "| contains the close token", cid in f)
p = tok.apply_chat_template([{"role": "user", "content": "What is 1+1?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=True)
print("template tail", repr(p[-60:]))
recs = [json.loads(l) for l in gzip.open(f"results/{sys.argv[1]}/practice_select128/samples.jsonl.gz", "rt", encoding="utf-8")]
print(len(recs), "samples; finish:", {k: sum(r["finish"] == k for r in recs) for k in ("stop", "length")})
ids = [tok.encode(r["text"], add_special_tokens=False) for r in recs]
dev = [abs(len(i) - r["tokens"]) for i, r in zip(ids, recs)]
print("re-tokenised length vs saved length: max abs deviation", max(dev), "mean", round(float(np.mean(dev)), 2))
print("close token present in", sum(cid in i for i in ids), "samples")
for B in (2048, 4096, 8192, 12288):
    kept = [(cid in i[:B]) or r["tokens"] <= B for i, r in zip(ids, recs)]
    kc = sum(r["correct"] for k, r in zip(kept, recs) if k)
    print(f"B={B}: kept {sum(kept)}, forced {len(recs) - sum(kept)}, kept and correct {kc} -> floor {100 * kc / len(recs):.1f}%")
