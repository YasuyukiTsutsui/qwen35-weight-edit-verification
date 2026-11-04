#!/usr/bin/env bash
# Phase 1: reproduction (base vs published p50c) + base generations for direction finding.
set -u
cd "$(dirname "$0")/.."
run() { local name=$1; shift; echo "== $name $(date)"; bash scripts/drun.sh python3 scripts/eval_math.py "$@" > results/logs/$name.log 2>&1; grep SUMMARY results/logs/$name.log || echo "FAILED $name"; }
run qwen_base_find   --model models/Qwen3.5-0.8B      --tag qwen_base --rows-file data/practice_find.jsonl --samples 1 --max-tokens 16384
run qwen_base_m500   --model models/Qwen3.5-0.8B      --tag qwen_base --set math500    --n 100 --samples 2
run qwen_p50c_m500   --model models/Qwen3.5-0.8B-p50c --tag qwen_p50c --set math500    --n 100 --samples 2
run qwen_base_gsm    --model models/Qwen3.5-0.8B      --tag qwen_base --set gsm8k_test --n 100 --samples 2
run qwen_p50c_gsm    --model models/Qwen3.5-0.8B-p50c --tag qwen_p50c --set gsm8k_test --n 100 --samples 2
echo "== done $(date)"
