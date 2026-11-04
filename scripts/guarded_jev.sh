#!/usr/bin/env bash
# Run Jev-Omni (4-bit) on DecisionBench with a watchdog: kill the container if host free RAM < 1.5 GB or GPU > 85 C.
# usage: bash scripts/guarded_jev.sh <subset> <fmt>
cd "$(dirname "$0")/.."
SUB=$1; FMT=$2
bash scripts/drun.sh python3 scripts/eval_jev.py --model models/jev-omni --tag jev_4bit --subset $SUB --fmt $FMT --load 4bit >> results/logs/jev_4bit_${SUB}_${FMT}.log 2>&1 &
RUNPID=$!
while kill -0 $RUNPID 2>/dev/null; do
  sleep 10
  L=$(tail -1 results/logs/monitor.csv)
  FREE=$(echo "$L" | cut -d, -f3); TEMP=$(echo "$L" | cut -d, -f6)
  if python -I -c "import sys; sys.exit(0 if (float('$FREE')<1.5 or float('$TEMP')>85) else 1)" 2>/dev/null; then
    echo "WATCHDOG: free RAM $FREE GB / GPU $TEMP C -> killing containers" | tee -a results/logs/watchdog.log
    docker ps -q --filter ancestor=wsp-vllm:latest | xargs -r docker kill
    break
  fi
done
wait $RUNPID 2>/dev/null
grep -a -E "SUMMARY" results/logs/jev_4bit_${SUB}_${FMT}.log | cut -c1-700 || echo "NO SUMMARY (stopped or failed)"
