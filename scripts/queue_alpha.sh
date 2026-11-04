#!/usr/bin/env bash
# wait for the running Jev-Omni container, then evaluate alpha=1..4 (k=64, layers 15+19) on the practice select split
cd "$(dirname "$0")/.."
docker wait e477b476dafd > /dev/null 2>&1
for a in 1 2 3 4; do
  bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/own_k64_a$a --tag sweep_own_k64_a$a --rows-file data/practice_select128.jsonl --samples 1 --max-tokens 8192 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/sweep_own_k64_a$a.log 2>&1
  grep -a SUMMARY results/logs/sweep_own_k64_a$a.log | cut -c1-600 || echo "FAILED a$a"
done
echo ALLDONE
