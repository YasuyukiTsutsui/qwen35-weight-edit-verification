#!/usr/bin/env bash
# Method A training (Issue #10), after everything else in queue_resume.sh and queue_followup.sh. The check (sampler log-probs vs vLLM, mixed-reward
# count 105 >= 96 needed) and the bench (16,384 tokens: 25 s forward+backward, 6.0 GiB peak) passed on 2026-11-05.
#   1. train: 8 steps x 12 problems x 8 samples from the initial policy (results/pg_r0/practice_find160), AdamW, ~30 min per step
#   2. evaluate the exported edit on the select problems (same protocol as method B) and on the validation problems
cd "$(dirname "$0")/.."
retry() {  # run in the GPU container; retry when the preflight check refuses to start
  local log="$1"; shift
  for i in 1 2 3 4 5 6; do
    bash scripts/drun.sh "$@" >> "$log" 2>&1 && return 0
    tail -c 300 "$log" | grep -q "preflight failed" || return 1
    sleep 90
  done
  return 1
}
while ! grep -q "FOLLOWDONE" results/logs/queue_followup.out 2>/dev/null; do sleep 120; done
sleep 120
retry results/logs/pg_train.log python3 scripts/train_rank1_pg.py --mode train --data results/pg_r0/practice_find160 --out models/edits/pg_r1 || { echo "A-TRAIN FAILED"; tail -n 8 results/logs/pg_train.log | cut -c1-240; exit 1; }
tail -n 12 results/logs/pg_train.log | cut -c1-240
retry results/logs/pg_eval_select.log python3 scripts/eval_math.py --model models/edits/pg_r1 --tag pg_r1_s0 --rows-file data/practice_select128.jsonl --samples 2 \
  --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 0 || { echo "A-EVAL FAILED (select)"; exit 1; }
retry results/logs/pg_eval_val.log python3 scripts/eval_math.py --model models/edits/pg_r1 --tag pg_r1_val_s1 --rows-file data/practice_select_rest128.jsonl --samples 2 \
  --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 1 || { echo "A-EVAL FAILED (validation)"; exit 1; }
echo ATRAINDONE
