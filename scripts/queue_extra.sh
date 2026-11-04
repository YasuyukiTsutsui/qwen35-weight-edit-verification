#!/usr/bin/env bash
# After the variation study and the method-A preparation: the extra studies approved in Issue #20
# (#14 layer follow-ups, #13 data-efficiency of the direction). See scripts/extra_study.py.
cd "$(dirname "$0")/.."
while ! grep -q "ADONE\|A-PREP FAILED" results/logs/queue_a.out 2>/dev/null; do sleep 60; done
sleep 60
python scripts/extra_study.py > results/logs/extra_run.log 2>&1 || { echo "EXTRA FAILED"; tail -n 8 results/logs/extra_run.log; exit 1; }
tail -n 24 results/logs/extra_run.log
echo EXTRADONE
