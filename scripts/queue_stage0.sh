#!/usr/bin/env bash
# Issue #32 stage 0 (weight-free): colder non-thinking samplers, 8 samples on the 128 select problems; then, if T* is not 0.7, N(T*) on E1.
#   bash scripts/queue_stage0.sh   (the GPU is shared with the user's own use: the preflight comes first)
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
BASE=models/Qwen3.5-0.8B
wait_clear() {
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 30; continue; fi
    ok=0; [ $((i % 15)) -eq 1 ] && echo "[$(date +%T)] waiting for free RAM / GPU (poll $i)"; sleep 120
  done
  return 1
}
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
echo "[$(date +%T)] queue_stage0 start"
for T in 05 03; do
  run bl_s0_T$T results/bl_s0_T$T/practice_select128/summary.json \
      bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_s0_T$T --preset qwen35_nt_T${T}_p15 --rows-file data/practice_select128.jsonl --samples 8 \
          --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 2 --thinking 0
done
python -I scripts/analyze_stage0.py >> results/logs/queue_stage0.log 2>&1
STAR=$(cat results/stage0_tstar.txt 2>/dev/null)
echo "[$(date +%T)] T* preset: $STAR"
if [ -n "$STAR" ] && [ "$STAR" != "qwen35_nt_T07_p15" ]; then
  run e1_Nstar results/e1_Nstar/e1/summary.json \
      bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag e1_Nstar --preset $STAR --thinking 0 --rows-file data/e1.jsonl --samples 1 \
          --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 102
fi
echo STAGE0DONE
