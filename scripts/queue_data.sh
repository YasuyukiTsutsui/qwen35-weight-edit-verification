#!/usr/bin/env bash
# After the re-baseline queue: sample the own edit (k=64, alpha=12) on the practice "find" problems (disjoint from select128),
# 2 samples each at a 16,384-token cap. Correct answers become the positives for the AdamW-trained rank-1 edit.
cd "$(dirname "$0")/.."
while ! grep -q ALLDONE results/logs/queue_rebase16k.log 2>/dev/null; do sleep 30; done
bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/own_k64_a12 --tag data_own_a12 --rows-file data/practice_find.jsonl --samples 2 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/data_own_a12.log 2>&1
grep -a SUMMARY results/logs/data_own_a12.log | cut -c1-700 || echo "FAILED data_own_a12"
echo DATADONE
