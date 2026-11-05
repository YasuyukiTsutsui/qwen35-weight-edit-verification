#!/usr/bin/env bash
# Issue #30: non-editing baselines, one after another (GPU is shared with the user's own use, so the preflight comes first).
#   bash scripts/queue_baselines.sh
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
BASE=models/Qwen3.5-0.8B
COMMON="--rows-file data/practice_select128.jsonl --samples 2 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 2"
wait_clear() {
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 30; continue; fi
    ok=0; [ $((i % 15)) -eq 1 ] && echo "[$(date +%T)] waiting for free RAM / GPU (poll $i)"; sleep 120
  done
  return 1
}
# run <tag> <done-file> <command...>   (restarts from scratch up to 3 times when a run ends abnormally)
run() {
  local tag="$1" donef="$2"; shift 2
  if [ -f "$donef" ]; then echo "[$(date +%T)] $tag already done"; return 0; fi
  for attempt in 1 2 3; do
    wait_clear || { echo "[$(date +%T)] gave up waiting at $tag"; exit 1; }
    echo "[$(date +%T)] $tag attempt $attempt"
    rm -rf results/$tag
    "$@" >> results/logs/$tag.log 2>&1
    if [ -f "$donef" ]; then echo "[$(date +%T)] $tag finished"; return 0; fi
    echo "[$(date +%T)] $tag attempt $attempt ended abnormally"; sleep 90
  done
  echo "[$(date +%T)] $tag FAILED"; exit 1
}
echo "[$(date +%T)] queue_baselines start"
run bf_base results/bf_base/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/budget_force_sim.py --model $BASE --source rb16k_base --tag bf_base
run bl_nothink results/bl_nothink/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_nothink --preset qwen35_nothink --thinking 0 $COMMON
run bl_t10 results/bl_t10/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_t10 --preset qwen35_think_t10 $COMMON
run bl_p20 results/bl_p20/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_p20 --preset qwen35_think_p20 $COMMON
echo BASELINESDONE
