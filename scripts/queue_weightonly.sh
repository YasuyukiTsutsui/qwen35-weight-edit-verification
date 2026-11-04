#!/usr/bin/env bash
# After the extra studies: weight-only method W (head edits chosen from the weights alone). See scripts/weightonly_study.py.
cd "$(dirname "$0")/.."
while ! grep -q "EXTRADONE\|EXTRA FAILED" results/logs/queue_extra.out 2>/dev/null; do sleep 60; done
sleep 60
python scripts/weightonly_study.py > results/logs/weightonly_run.log 2>&1 || { echo "WO FAILED"; tail -n 8 results/logs/weightonly_run.log; exit 1; }
tail -n 14 results/logs/weightonly_run.log
echo WODONE
