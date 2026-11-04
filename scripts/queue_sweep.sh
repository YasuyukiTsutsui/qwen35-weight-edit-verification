#!/usr/bin/env bash
# Sweep (k columns, alpha) of the own edit on the practice "select" split; references: base and published p50c.
cd "$(dirname "$0")/.."
ev() { # tag path
  bash scripts/drun.sh python3 scripts/eval_math.py --model $2 --tag sweep_$1 --rows-file data/practice_select128.jsonl --samples 1 --max-tokens 8192 --max-num-seqs 64 --gpu-mem 0.7 > results/logs/sweep_$1.log 2>&1
  grep -a SUMMARY results/logs/sweep_$1.log | cut -c1-600 || echo "FAILED $1"
}
for cfg in "64 12" "128 6" "128 12" "256 6"; do
  set -- $cfg; k=$1; a=$2; name=own_k${k}_a${a}
  bash scripts/drun.sh python3 scripts/make_edit.py --base models/Qwen3.5-0.8B --dir results/qwen_dir --layers 15,19 --k $k --alpha $a --out models/edits/$name > results/logs/edit_$name.log 2>&1
done
ev base models/Qwen3.5-0.8B
ev p50c models/Qwen3.5-0.8B-p50c
ev own_k64_a6 models/edits/qwen_own
for cfg in "64 12" "128 6" "128 12" "256 6"; do set -- $cfg; ev own_k$1_a$2 models/edits/own_k$1_a$2; done
echo ALLDONE
