#!/usr/bin/env bash
# 1) alpha 1..4 on the practice select split  2) Jev-Omni DecisionBench (resumable)
cd "$(dirname "$0")/.."
for a in 1 2 3 4; do
  [ -f results/sweep_own_k64_a$a/practice_select128/summary.json ] && continue
  bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/own_k64_a$a --tag sweep_own_k64_a$a --rows-file data/practice_select128.jsonl --samples 1 --max-tokens 8192 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/sweep_own_k64_a$a.log 2>&1
  grep -a SUMMARY results/logs/sweep_own_k64_a$a.log | cut -c1-600 || echo "FAILED a$a"
done
echo ALPHADONE
for cfg in "medium v1" "medium v2" "hard v1" "hard v2"; do
  set -- $cfg
  [ -f results/jev_base/decisionbench_$1_$2/summary.json ] && continue
  bash scripts/drun.sh python3 scripts/eval_jev.py --model models/jev-omni --tag jev_base --subset $1 --fmt $2 --load 8bit >> results/logs/jev_base_$1_$2.log 2>&1
  grep -a SUMMARY results/logs/jev_base_$1_$2.log | cut -c1-600 || echo "FAILED jev $1 $2"
done
echo ALLDONE
