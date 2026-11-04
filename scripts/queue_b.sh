#!/usr/bin/env bash
# After T1: method B (noisy-EI search over alpha, k on ALL 128 select problems x 2 samples, 16,384-token cap),
# then the null-selection control (5 norm-matched orthogonal random directions through the same procedure).
cd "$(dirname "$0")/.."
while ! grep -q "^DONE" results/logs/t1.log 2>/dev/null; do
  if grep -q "Traceback" results/logs/t1.log 2>/dev/null; then echo "T1 FAILED - not starting B"; exit 1; fi
  sleep 30
done
sleep 60   # let the T1 container exit
python scripts/bo_search.py --run --n-init 8 --n-bo 8 > results/logs/bo_run.log 2>&1 || { echo "BO FAILED"; tail -n 5 results/logs/bo_run.log; exit 1; }
tail -n 3 results/logs/bo_run.log
python scripts/null_select.py --n 5 > results/logs/null_select.log 2>&1 || { echo "NULL FAILED"; tail -n 5 results/logs/null_select.log; exit 1; }
tail -n 2 results/logs/null_select.log
echo BDONE
