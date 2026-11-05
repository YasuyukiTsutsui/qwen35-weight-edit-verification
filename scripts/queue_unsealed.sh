#!/usr/bin/env bash
# Issue #31: unsealed confirmation steps A-D.   bash scripts/queue_unsealed.sh   (the GPU is shared with the user's own use: the preflight comes first)
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
BASE=models/Qwen3.5-0.8B; PUB=models/Qwen3.5-0.8B-p50c
SEL="--rows-file data/practice_select128.jsonl --samples 2 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 2 --thinking 0"
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
echo "[$(date +%T)] queue_unsealed start"
# A. published edit + budget forcing on the two saved sets
run bf_pub results/bf_pub/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/budget_force_sim.py --model $PUB --source rb16k_p50c --tag bf_pub --budgets 4096,8192
run bf_pub_ff results/bf_pub_ff/final_fresh/summary.json \
    bash scripts/drun.sh python3 scripts/budget_force_sim.py --model $PUB --source ff16k_p50c --tag bf_pub_ff --rows-file data/final_fresh.jsonl --budgets 4096,8192
# B. non-thinking sampler sweep (select128 x 2, seed 2)
for cfg in T07_p0 T07_p15 T10_p0 T10_p15; do
  run bl_nts_$cfg results/bl_nts_$cfg/practice_select128/summary.json \
      bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_nts_$cfg --preset qwen35_nt_$cfg $SEL
done
python -I scripts/pick_nt_preset.py >> results/logs/queue_unsealed.log 2>&1
NTP=$(cat results/nt_best_preset.txt)
echo "[$(date +%T)] chosen non-thinking preset: $NTP"
# C. E1: R (published edit, thinking), X' (budget forcing on R's own samples), N (chosen non-thinking)
E1="--rows-file data/e1.jsonl --samples 1 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7"
run e1_R results/e1_R/e1/summary.json \
    bash scripts/drun.sh python3 scripts/eval_math.py --model $PUB --tag e1_R $E1 --seed 101
run e1_Xp results/e1_Xp/e1/summary.json \
    bash scripts/drun.sh python3 scripts/budget_force_sim.py --model $PUB --source e1_R --tag e1_Xp --rows-file data/e1.jsonl --budgets 4096,8192
run e1_N results/e1_N/e1/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag e1_N --preset $NTP --thinking 0 $E1 --seed 102
# D. majority vote of the chosen non-thinking setting: 8 samples on select128
run bl_nt_k8 results/bl_nt_k8/practice_select128/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag bl_nt_k8 --preset $NTP --rows-file data/practice_select128.jsonl --samples 8 \
        --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 2 --thinking 0
echo UNSEALEDDONE
