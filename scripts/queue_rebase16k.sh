#!/usr/bin/env bash
# Re-baseline on the practice "select" 128 problems with a 16,384-token cap (the original edit's evaluation used 32,768).
cd "$(dirname "$0")/.."
ev() { # tag path
  bash scripts/drun.sh python3 scripts/eval_math.py --model $2 --tag rb16k_$1 --rows-file data/practice_select128.jsonl --samples 1 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/rb16k_$1.log 2>&1
  grep -a SUMMARY results/logs/rb16k_$1.log | cut -c1-700 || echo "FAILED $1"
}
ev base models/Qwen3.5-0.8B
ev p50c models/Qwen3.5-0.8B-p50c
ev own_k64_a6 models/edits/qwen_own
ev own_k64_a12 models/edits/own_k64_a12
echo ALLDONE
