#!/usr/bin/env bash
# Issue #30, stage 3: non-thinking mode on the final_fresh problems, and with the edited model (starts after queue_baselines2.sh has finished)
#   bash scripts/queue_baselines3.sh
# (stage 1 was: non-editing baselines, one after another (GPU is shared with the user's own use, so the preflight comes first).
#   bash scripts/queue_baselines.sh)
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
until grep -q BASELINES2DONE results/logs/queue_baselines2.log 2>/dev/null; do sleep 30; done
sleep 45
EDIT=models/edits/bo_k256_a6
FF="--rows-file data/final_fresh.jsonl --samples 1 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 0 --preset qwen35_nothink --thinking 0"
echo "[$(date +%T)] queue_baselines3 start"
run bl_nt_ff results/bl_nt_ff/final_fresh/summary.json     bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_nt_ff $FF
run bl_nt_edit_ff results/bl_nt_edit_ff/final_fresh/summary.json     bash scripts/drun.sh python3 scripts/eval_baselines.py --model $EDIT --tag bl_nt_edit_ff $FF
run bl_nt_edit results/bl_nt_edit/practice_select128/summary.json     bash scripts/drun.sh python3 scripts/eval_baselines.py --model $EDIT --tag bl_nt_edit --preset qwen35_nothink --thinking 0 $COMMON
echo BASELINES3DONE
