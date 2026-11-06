#!/usr/bin/env bash
# Issue #32 stage 1, step 2: 8 non-thinking samples (T* = 0.3, presence 1.5) for each of the 768 unsealed training problems T1, with token ids saved.
#   bash scripts/queue_stage1_sample.sh   (the GPU is shared with the user's own use: the preflight comes first)
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
echo "[$(date +%T)] queue_stage1_sample start (4 chunks: the single 768-problem run held every token log-prob in RAM and was killed at 99% by the 12 GB container cap, twice)"
PARTS=""
for p in 0 1 2 3; do
  run t1_samples_p$p results/t1_samples_p$p/t1_p$p/summary.json       bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag t1_samples_p$p --preset qwen35_nt_T03_p15 --thinking 0 --rows-file data/t1_p$p.jsonl --samples 8           --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 3 --save-tokens 1
  PARTS="$PARTS results/t1_samples_p$p/t1_p$p"
done
python -I scripts/merge_sample_parts.py $PARTS --out results/t1_samples/t1 || { echo "[$(date +%T)] merge FAILED"; exit 1; }
echo STAGE1SAMPLEDONE
