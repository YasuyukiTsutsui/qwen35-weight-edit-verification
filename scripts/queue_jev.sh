#!/usr/bin/env bash
cd "$(dirname "$0")/.."
ev() { # subset fmt
  bash scripts/drun.sh python3 scripts/eval_jev.py --model models/jev-omni --tag jev_base --subset $1 --fmt $2 --load 8bit > results/logs/jev_base_$1_$2.log 2>&1
  grep -a SUMMARY results/logs/jev_base_$1_$2.log | cut -c1-600 || echo "FAILED $1 $2"
}
ev medium v1
ev medium v2
ev hard v1
ev hard v2
echo ALLDONE
