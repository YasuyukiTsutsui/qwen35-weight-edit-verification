#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for m in "qwen_base:models/Qwen3.5-0.8B" "qwen_p50c:models/Qwen3.5-0.8B-p50c" "qwen_own:models/edits/qwen_own" "qwen_rand:models/edits/qwen_rand"; do
  tag=${m%%:*}; path=${m#*:}
  bash scripts/drun.sh python3 scripts/eval_math.py --model $path --tag quick_$tag --set gsm8k_test --n 64 --samples 1 --max-tokens 8192 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/quick_$tag.log 2>&1
  grep SUMMARY results/logs/quick_$tag.log || echo "FAILED $tag"
done
echo ALLDONE
