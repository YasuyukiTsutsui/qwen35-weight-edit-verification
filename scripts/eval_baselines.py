"""Non-editing baselines of Issue #30: eval_math.py (unchanged, its hash is fixed in the pre-registration frame) with three more sampler presets.

    python scripts/eval_baselines.py --model models/Qwen3.5-0.8B --tag bl_t10 --preset qwen35_think_t10 --rows-file data/practice_select128.jsonl ...

Presets (Qwen3.5-0.8B model card, text tasks, https://huggingface.co/Qwen/Qwen3.5-0.8B, read 2026-11-06):
  qwen35_think_t10  thinking, temperature 1.0, top_p 0.95, top_k 20, presence 1.5   (the card's thinking-mode values; the blog used 0.6)
  qwen35_think_p20  thinking, the blog's sampler but presence penalty 2.0           (the card: "adjust presence_penalty between 0 and 2 to reduce endless repetitions")
  qwen35_nothink    non-thinking (use with --thinking 0), temperature 1.0, top_p 1.0, top_k 20, presence 2.0 (the card's non-thinking values)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_math as E  # noqa: E402

E.PRESETS.update({
    "qwen35_think_t10": dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
    "qwen35_think_p20": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=2.0),
    "qwen35_nothink": dict(temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0),
})

if __name__ == "__main__":
    E.main()
