#!/usr/bin/env bash
# Evaluate edit variants of Issue #28 one at a time, exactly like the centre re-evaluation (vary_c0): select128 x 2 samples, 16,384 cap, seed 2.
#   bash scripts/queue_eq.sh <wait-file> <wait-pattern> tag:model [tag:model ...]
# Waits until <wait-pattern> appears in <wait-file> (e.g. the log of the job that holds the GPU), then until the preflight passes twice in a row
# (the user's own use of the PC comes first), then evaluates each variant; a run that ended abnormally is restarted from scratch (up to 4 times).
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
WF="$1"; WP="$2"; shift 2
until grep -q "$WP" "$WF" 2>/dev/null; do sleep 30; done
sleep 45
wait_clear() {
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 30; continue; fi
    ok=0; [ $((i % 15)) -eq 1 ] && echo "[$(date +%T)] waiting for free RAM / GPU (poll $i)"; sleep 120
  done
  return 1
}
echo "[$(date +%T)] queue_eq start: $*"
for pair in "$@"; do
  tag="${pair%%:*}"; model="${pair#*:}"
  if [ -f results/$tag/practice_select128/summary.json ]; then echo "[$(date +%T)] $tag already done"; continue; fi
  ok=0
  for attempt in 1 2 3 4; do
    wait_clear || { echo "[$(date +%T)] gave up waiting at $tag"; exit 1; }
    echo "[$(date +%T)] $tag attempt $attempt"
    rm -rf results/$tag
    bash scripts/drun.sh python3 scripts/eval_math.py --model "$model" --tag "$tag" --rows-file data/practice_select128.jsonl --samples 2 \
      --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 2 >> results/logs/$tag.log 2>&1
    if [ -f results/$tag/practice_select128/summary.json ]; then ok=1; break; fi
    echo "[$(date +%T)] $tag attempt $attempt ended abnormally"; sleep 90
  done
  [ $ok -eq 1 ] || { echo "[$(date +%T)] $tag FAILED"; exit 1; }
  echo "[$(date +%T)] $tag finished"
done
echo EQDONE
