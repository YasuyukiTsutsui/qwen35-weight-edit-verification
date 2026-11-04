#!/usr/bin/env bash
# The rest of the GPU chain, PATIENT version (replaces queue_train_a.sh).
# Why: on 2026-11-05 the norm-matched column follow-up (#9) could not start because the user was playing a game (host RAM free 4-6 GB,
# GPU 4-5 GB used); the preflight check (>= 8 GB free RAM, GPU used <= 2500 MiB) refused 5 times and the stage failed. The user's own use of
# the PC has priority and the preflight is the protection against memory crashes, so this queue WAITS (polls every 2 minutes, up to 12 h per
# stage) instead of failing.
#   1. vary_followup.py   (resumable)  -> FOLLOWDONE2
#   2. method A training (Issue #10), then evaluation on the select and validation problems -> ATRAINDONE
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500

wait_clear() {  # wait until the preflight would pass (twice in a row, 60 s apart); give up after 12 h
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then
      ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 60; continue
    fi
    ok=0
    [ $((i % 15)) -eq 1 ] && echo "[$(date +%T)] waiting for free RAM / GPU (poll $i): $(python scripts/monitor.py --check $MINRAM $MAXGPU 2>&1 | tail -n 1 | cut -c1-160)"
    sleep 120
  done
  return 1
}

patient() {  # patient LOG CMD...: wait for resources, run; if it failed because the preflight refused, wait and run again
  local log="$1"; shift
  local i marker
  marker=$(mktemp)
  for i in $(seq 1 12); do
    wait_clear || { echo "gave up waiting for resources"; return 1; }
    touch "$marker"
    "$@" >> "$log" 2>&1 && { rm -f "$marker"; return 0; }
    # preflight refusal anywhere in the logs written since the marker?  (the failing job may log it in a sub-log)
    if find results/logs -name '*.log' -newer "$marker" -print0 2>/dev/null | xargs -0 -r tail -q -c 600 2>/dev/null | grep -q "preflight failed"; then
      echo "[$(date +%T)] preflight refused during the run (attempt $i); waiting"; continue
    fi
    rm -f "$marker"; return 1
  done
  rm -f "$marker"; return 1
}

echo "[$(date +%T)] queue_rest start"
patient results/logs/vary_followup.log python scripts/vary_followup.py && echo "FOLLOWDONE2" || echo "FOLLOWUP FAILED (continuing to method A)"
tail -n 12 results/logs/vary_followup.log | cut -c1-240

patient results/logs/pg_train.log bash scripts/drun.sh python3 scripts/train_rank1_pg.py --mode train --data results/pg_r0/practice_find160 --out models/edits/pg_r1 \
  || { echo "A-TRAIN FAILED"; tail -n 8 results/logs/pg_train.log | cut -c1-240; exit 1; }
tail -n 12 results/logs/pg_train.log | cut -c1-240
patient results/logs/pg_eval_select.log bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/pg_r1 --tag pg_r1_s0 --rows-file data/practice_select128.jsonl --samples 2 \
  --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 0 || { echo "A-EVAL FAILED (select)"; exit 1; }
patient results/logs/pg_eval_val.log bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/pg_r1 --tag pg_r1_val_s1 --rows-file data/practice_select_rest128.jsonl --samples 2 \
  --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 1 || { echo "A-EVAL FAILED (validation)"; exit 1; }
echo ATRAINDONE
