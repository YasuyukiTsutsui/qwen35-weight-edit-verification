#!/usr/bin/env bash
# Method A preparation (after the variation study): on-policy samples from the initial edit, then the feasibility checks.
#   1. 160 training problems x 8 samples from models/edits/bo_k64_a5 (alpha 5, top-64, layers 15+19) with --save-tokens 1
#   2. train_rank1_pg.py --mode check  (sampler log-probs vs vLLM's saved ones, on sequences of different lengths)
#   3. train_rank1_pg.py --mode bench  (memory / speed of the backward pass at 16,384 tokens)
# Training itself is NOT started here: its settings depend on what check / bench show.
cd "$(dirname "$0")/.."
retry() {  # run in the GPU container; retry when the preflight check refuses to start
  local log="$1"; shift
  for i in 1 2 3 4 5; do
    bash scripts/drun.sh "$@" >> "$log" 2>&1 && return 0
    tail -c 300 "$log" | grep -q "preflight failed" || return 1
    sleep 90
  done
  return 1
}
while ! grep -q "VARYDONE\|VARY FAILED\|FAILED" results/logs/queue_vary.out results/logs/vary_run.log 2>/dev/null; do sleep 60; done
sleep 60
python -I - <<'EOF'
import json
rows = [l for l in open("data/practice_find.jsonl", encoding="utf-8")][:160]
open("data/practice_find160.jsonl", "w", encoding="utf-8", newline="").writelines(rows)
print("practice_find160:", len(rows))
EOF
if [ ! -f models/edits/bo_k64_a5/edit_log.json ]; then
  retry results/logs/pg_prep.log python3 scripts/make_edit.py --base models/Qwen3.5-0.8B --dir results/qwen_dir --layers 15,19 --k 64 --alpha 5 --out models/edits/bo_k64_a5 || { echo "A-PREP FAILED (edit)"; exit 1; }
fi
retry results/logs/pg_gen.log python3 scripts/eval_math.py --model models/edits/bo_k64_a5 --tag pg_r0 --rows-file data/practice_find160.jsonl \
  --samples 8 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 0 --save-tokens 1 || { echo "A-PREP FAILED (generation)"; exit 1; }
retry results/logs/pg_check.log python3 scripts/train_rank1_pg.py --mode check --data results/pg_r0/practice_find160 --n 4 || echo "CHECK exited non-zero (see results/logs/pg_check.log)"
retry results/logs/pg_bench.log python3 scripts/train_rank1_pg.py --mode bench --data results/pg_r0/practice_find160 --n 2 || echo "BENCH exited non-zero (see results/logs/pg_bench.log)"
tail -n 6 results/logs/pg_check.log | cut -c1-240
tail -n 4 results/logs/pg_bench.log | cut -c1-240
echo ADONE
