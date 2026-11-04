#!/usr/bin/env bash
# Screen the T1-derived edits (matched-pair direction) at 16,384 tokens on the 128 select problems x 2 samples, then run method B.
cd "$(dirname "$0")/.."
mk() { # name dir layers alpha
  bash scripts/drun.sh python3 scripts/make_edit.py --base models/Qwen3.5-0.8B --dir $2 --layers $3 --k 64 --alpha $4 --out models/edits/$1 > results/logs/edit_$1.log 2>&1 || { echo "make_edit FAILED $1"; exit 1; }
}
ev() { # name
  bash scripts/drun.sh python3 scripts/eval_math.py --model models/edits/$1 --tag scr_$1 --rows-file data/practice_select128.jsonl --samples 2 --max-tokens 16384 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/scr_$1.log 2>&1
  grep -a SUMMARY results/logs/scr_$1.log | cut -c1-520 || echo "FAILED $1"
}
D=results/qwen_dir_mp
mk t1_L10_r12 $D 10 10:7.92;            ev t1_L10_r12
mk t1_L14_r12 $D 14 14:7.68;            ev t1_L14_r12
mk t1_L10_14_r12 $D 10,14 10:7.92,14:7.68; ev t1_L10_14_r12
mk t1_mp_L15_19_r12 $D 15,19 15:6.32,19:5.24; ev t1_mp_L15_19_r12
echo SCREENDONE
bash scripts/queue_b.sh
