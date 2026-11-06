#!/usr/bin/env bash
# Issue #32 stage 1, steps 3-6: SFT-sample selection -> trainer bench -> smoke test of train/export/vLLM load -> train M -> check-export -> evaluate M and the base N* on E1.
#   nohup bash scripts/queue_stage1_train.sh >> results/logs/queue_stage1_train.log 2>&1 &
# Gates (any failure stops the queue; nothing is retried blindly):  bench exit 0 and peak memory <= 14 GiB; smoke check-export OK; smoke vLLM load OK; check-export of M OK.
# The user's own PC use has priority: the preflight (free RAM >= 8 GB, other programs' GPU <= 2500 MiB) comes first; the trainer itself stops (exit 75) when
# other programs' GPU memory grows, when the GPU is hot, or when results/STOP_train exists, and is resumed from its checkpoint.
cd "$(dirname "$0")/.."
MINRAM=8; MAXGPU=2500
BASE=models/Qwen3.5-0.8B
PRESET=qwen35_nt_T03_p15
SAMPLE_LOG=results/logs/queue_stage1_sample.log
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
# one GPU job with up to 3 attempts (vLLM start-up failures); $1 tag, $2 file that proves it finished, rest: the command
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
# trainer with resume on exit code 75
train() {
  local out="$1" sel="$2" logf="$3" n=0 rc
  while [ $n -lt 60 ]; do
    wait_clear || die "gave up waiting before training"
    say "train $out (attempt $((n + 1)))"
    bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode train --data results/t1_samples/t1 --select "$sel" --targets all --out "$out" --resume >> "$logf" 2>&1
    rc=$?
    [ $rc -eq 0 ] && return 0
    if [ $rc -eq 75 ]; then say "train exited 75 (saved; will resume)"; n=$((n + 1)); sleep 120; continue; fi
    die "train exited with code $rc (see $logf)"
  done
  die "too many resumes"
}

say "queue_stage1_train start"
until grep -q STAGE1SAMPLEDONE $SAMPLE_LOG 2>/dev/null; do
  grep -qE "FAILED|gave up" $SAMPLE_LOG 2>/dev/null && die "sampling queue failed"
  sleep 60
done
say "sampling finished"

# --- sample selection (host only, no GPU)
python -I scripts/make_sft_sets.py > results/logs/make_sft_sets.log 2>&1 || die "make_sft_sets failed"
tail -4 results/logs/make_sft_sets.log
NM=$(python -I -c "import json;print(len(json.load(open('results/t1_sft/M.json'))['pairs']))")
say "M has $NM samples"
[ "$NM" -ge 300 ] || die "M has only $NM samples (< 300): review by hand"

# --- bench: 3 longest M sequences
mkdir -p results/logs models/sft
if [ ! -f results/logs/stage1_bench.done ]; then
  wait_clear || die "gave up before bench"
  bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode bench --data results/t1_samples/t1 --select results/t1_sft/M.json --targets all --n 3 > results/logs/stage1_bench.log 2>&1 \
    || die "bench failed (results/logs/stage1_bench.log)"
  grep -E "LoRA modules|sequences|peak mem" results/logs/stage1_bench.log
  PEAK=$(grep -oE "peak mem [0-9.]+" results/logs/stage1_bench.log | awk '{print $3}' | sort -n | tail -1)
  python -I -c "import sys; sys.exit(0 if float('${PEAK:-99}') <= 14 else 1)" || die "bench peak memory ${PEAK} GiB > 14"
  touch results/logs/stage1_bench.done
fi

# --- smoke test: 2 steps of training on 16 M samples, export, check-export, vLLM load of the exported model
python -I - <<'EOF'
import json
m = json.load(open("results/t1_sft/M.json"))
json.dump({"rule": "smoke", "pairs": m["pairs"][:16]}, open("results/t1_sft/smoke.json", "w"))
EOF
if [ ! -f models/sft/smoke/train_summary.json ]; then
  rm -rf models/sft/smoke
  train models/sft/smoke results/t1_sft/smoke.json results/logs/stage1_smoke_train.log
fi
wait_clear || die "gave up before smoke check-export"
bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode check-export --out models/sft/smoke > results/logs/stage1_smoke_check.log 2>&1 || die "smoke check-export failed"
tail -2 results/logs/stage1_smoke_check.log
head -8 data/t1_p0.jsonl > data/_smoke_rows.jsonl
run smoke_load results/smoke_load/_smoke_rows/summary.json \
    bash scripts/drun.sh python3 scripts/eval_baselines.py --model models/sft/smoke --tag smoke_load --preset $PRESET --thinking 0 --rows-file data/_smoke_rows.jsonl --samples 2 \
        --max-tokens 2048 --max-num-seqs 16 --gpu-mem 0.7 --seed 5
rm -rf models/sft/smoke data/_smoke_rows.jsonl results/smoke_load
say "smoke test passed"

# --- the real run
if [ ! -f models/sft/M/train_summary.json ]; then
  train models/sft/M results/t1_sft/M.json results/logs/stage1_train_M.log
fi
wait_clear || die "gave up before check-export"
bash scripts/drun.sh python3 scripts/train_lora_sft.py --mode check-export --out models/sft/M > results/logs/stage1_check_M.log 2>&1 || die "check-export of M failed (see results/logs/stage1_check_M.log)"
tail -2 results/logs/stage1_check_M.log

# --- evaluation on E1: 640 problems x 4 samples, non-thinking T0.3 / presence 1.5, seed 202; M and the base N* with the identical protocol
E1="--rows-file data/e1.jsonl --samples 4 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 --seed 202 --thinking 0 --preset $PRESET"
run e1s_M results/e1s_M/e1/summary.json bash scripts/drun.sh python3 scripts/eval_baselines.py --model models/sft/M --tag e1s_M $E1
run e1s_Nstar results/e1s_Nstar/e1/summary.json bash scripts/drun.sh python3 scripts/eval_baselines.py --model $BASE --tag e1s_Nstar $E1
echo STAGE1TRAINDONE
