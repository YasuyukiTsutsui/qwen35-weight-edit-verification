#!/usr/bin/env bash
# Issue #32 stage 1, second training seed of M (Issue #32 comment 8): train M2 (seed 1), check-export, evaluate on E1 with the identical protocol (640 x 4, seed 202).
#   nohup bash scripts/queue_stage1_seed2.sh >> results/logs/queue_stage1_seed2.log 2>&1 &
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
PRESET=qwen35_nt_T03_p15
say() { echo "[$(date +%T)] $*"; }
die() { say "ABORT: $*"; exit 1; }
wait_clear() {
  local i ok=0
  for i in $(seq 1 360); do
    if python scripts/monitor.py --check $MINRAM $MAXGPU > /dev/null 2>&1; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; sleep 30; continue; fi
    ok=0; [ $((i % 15)) -eq 1 ] && say "waiting for free RAM / GPU (poll $i)"; sleep 120
  done
  return 1
}
run() {
  local tag="$1" donef="$2"; shift 2
  if [ -f "$donef" ]; then say "$tag already done"; return 0; fi
  for attempt in 1 2 3; do
    wait_clear || die "gave up waiting at $tag"
    say "$tag attempt $attempt"
    rm -rf results/$tag
    "$@" >> results/logs/$tag.log 2>&1
    if [ -f "$donef" ]; then say "$tag finished"; return 0; fi
    say "$tag attempt $attempt ended abnormally"; sleep 90
  done
  die "$tag FAILED"
}
say "queue_stage1_seed2 start"
n=0
while [ ! -f models/sft/M2/train_summary.json ]; do
  [ $n -lt 60 ] || die "too many resumes"
  wait_clear || die "gave up before training"
  say "train M2 (attempt $((n + 1)))"
  bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode train --data results/t1_samples/t1 --select results/t1_sft/M.json --targets all --out models/sft/M2 --seed 1 --resume >> results/logs/stage1_train_M2.log 2>&1
  rc=$?
  if [ $rc -eq 75 ]; then say "train exited 75 (saved; will resume)"; n=$((n + 1)); sleep 120; continue; fi
  [ $rc -eq 0 ] || die "train exited with code $rc (see results/logs/stage1_train_M2.log)"
done
wait_clear || die "gave up before check-export"
bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode check-export --out models/sft/M2 > results/logs/stage1_check_M2.log 2>&1
worst=$(grep -oE "= [0-9.]+  max" results/logs/stage1_check_M2.log | awk '{print $2}' | sort -n | tail -1)
[ -n "$worst" ] || die "check-export of M2 produced no numbers"
say "check-export models/sft/M2: worst mean |dlogp| = $worst"
python -I -c "import sys; sys.exit(0 if float('$worst') < 0.15 else 1)" || die "check-export of M2: difference $worst >= 0.15 (bug level)"
mkdir -p results/stage1_M2 && cp models/sft/M2/train_log.json models/sft/M2/merge_log.json models/sft/M2/train_summary.json results/logs/stage1_check_M2.log results/stage1_M2/
E1="--rows-file data/e1.jsonl --samples 4 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 202 --thinking 0 --preset $PRESET"
run e1s_M2 results/e1s_M2/e1/summary.json bash scripts/drun.sh python3 scripts/eval_baselines.py --model models/sft/M2 --tag e1s_M2 $E1
echo STAGE1SEED2DONE
