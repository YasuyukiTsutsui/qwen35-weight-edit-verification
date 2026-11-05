#!/usr/bin/env bash
# Final confirmation (Issue #11), run exactly as fixed in reports/preregistration_final.md (§2, §4, §10, §11):
#   256 fresh problems (data/final_fresh.jsonl), 1 sample/problem, 16,384-token cap, seed 0, tag ff16k_<name>.
# One candidate at a time; nothing else may use the GPU meanwhile. Each candidate is evaluated ONCE; a run that ends abnormally
# (preflight refusal, container stop, wrong sample count) is restarted from scratch with the same settings and the restart is logged.
# No analysis of results is done here (§11: no peeking). The patient wait follows queue_rest.sh (the user's own use of the PC comes first).
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500

wait_clear() {
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then
      ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 60; continue
    fi
    ok=0
    [ $((i % 15)) -eq 1 ] && echo "[$(date +%T)] waiting for free RAM / GPU (poll $i)"
    sleep 120
  done
  return 1
}

# name  model-path
CANDS=(
  "base models/Qwen3.5-0.8B"
  "p50c models/Qwen3.5-0.8B-p50c"
  "own_a6 models/edits/qwen_own"
  "mp_L15_19 models/edits/t1_mp_L15_19_r12"
  "B models/edits/bo_k256_a6"
  "A models/edits/pg_r1"
  "null models/edits/null_k256_a6_s1"
  "LF models/edits/bo_k256_a5.5"
)

ok_summary() {  # the finished run must have exactly 256 samples and every finish in {stop, length}
  python -I - "$1" <<'EOF'
import gzip, json, sys
tag = sys.argv[1]
n = 0; bad = 0
for l in gzip.open(f"results/{tag}/final_fresh/samples.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(l); n += 1; bad += r["finish"] not in ("stop", "length")
sys.exit(0 if n == 256 and bad == 0 else 1)
EOF
}

echo "[$(date +%T)] queue_final16k start"
for c in "${CANDS[@]}"; do
  set -- $c; name=$1; model=$2; tag=ff16k_$name; log=results/logs/${tag}.log
  if [ -f results/$tag/final_fresh/summary.json ] && ok_summary $tag; then echo "[$(date +%T)] $tag already done"; continue; fi
  done_flag=0
  for attempt in 1 2 3 4 5 6; do
    wait_clear || { echo "[$(date +%T)] gave up waiting for resources at $tag"; exit 1; }
    echo "[$(date +%T)] $tag attempt $attempt"
    rm -rf results/$tag
    bash scripts/drun.sh python3 scripts/eval_math.py --model "$model" --tag "$tag" --rows-file data/final_fresh.jsonl --samples 1 \
      --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 >> "$log" 2>&1
    if [ -f results/$tag/final_fresh/summary.json ] && ok_summary $tag; then done_flag=1; break; fi
    echo "[$(date +%T)] $tag attempt $attempt ended abnormally (restart from scratch, same settings)"
    sleep 120
  done
  [ $done_flag -eq 1 ] || { echo "[$(date +%T)] $tag FAILED after 6 attempts"; exit 1; }
  echo "[$(date +%T)] $tag finished"
done
echo FINALEVALDONE
