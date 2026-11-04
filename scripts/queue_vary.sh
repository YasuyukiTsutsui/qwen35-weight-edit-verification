#!/usr/bin/env bash
# After method B and the null-selection control: the variation study (constants and targets made variable, one at a time).
cd "$(dirname "$0")/.."
while ! grep -q "NULL best of" results/logs/null_select.log 2>/dev/null; do
  if grep -q "Traceback" results/logs/null_select.log 2>/dev/null; then echo "NULL FAILED - not starting the variation study"; exit 1; fi
  if grep -q "Traceback" results/logs/bo_run.log 2>/dev/null; then echo "BO FAILED - not starting the variation study"; exit 1; fi
  sleep 60
done
sleep 90   # let the last container exit
python scripts/vary_study.py > results/logs/vary_run.log 2>&1 || { echo "VARY FAILED"; tail -n 8 results/logs/vary_run.log; exit 1; }
tail -n 16 results/logs/vary_run.log
echo VARYDONE
