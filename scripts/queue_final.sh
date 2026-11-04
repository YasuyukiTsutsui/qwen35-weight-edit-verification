#!/usr/bin/env bash
cd "$(dirname "$0")/.."
ev() { # tag path set
  bash scripts/drun.sh python3 scripts/eval_math.py --model $2 --tag final_$1 --set $3 --n 64 --samples 1 --max-tokens 8192 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/final_$1_$3.log 2>&1
  grep -a SUMMARY results/logs/final_$1_$3.log | cut -c1-500 || echo "FAILED $1 $3"
}
ev own_k64_a12 models/edits/own_k64_a12 gsm8k_test
for s in math500; do
  ev base models/Qwen3.5-0.8B $s
  ev p50c models/Qwen3.5-0.8B-p50c $s
  ev own_k64_a12 models/edits/own_k64_a12 $s
done
echo ALLDONE
