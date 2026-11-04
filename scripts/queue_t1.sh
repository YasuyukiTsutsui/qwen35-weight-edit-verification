#!/usr/bin/env bash
# T1: matched-pair direction + teacher-forced layer scan. Waits for the 16k re-baseline queue to finish (one GPU job at a time).
cd "$(dirname "$0")/.."
while ! grep -q ALLDONE results/logs/queue_rebase16k.log 2>/dev/null; do sleep 30; done
bash scripts/drun.sh python3 scripts/find_direction_mp.py --model models/Qwen3.5-0.8B --samples results/qwen_base/practice_find/samples.jsonl.gz \
  --old-dir results/qwen_dir --pub models/Qwen3.5-0.8B-p50c --out results/qwen_dir_mp > results/logs/t1.log 2>&1
tail -n 5 results/logs/t1.log | cut -c1-300
echo T1DONE
