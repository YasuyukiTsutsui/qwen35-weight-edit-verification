#!/usr/bin/env bash
# After queue_resume.sh (variation / extra / weight-only studies): the norm-matched column controls of Issue #9 (scripts/vary_followup.py).
cd "$(dirname "$0")/.."
while ! grep -q "RESUMEDONE" results/logs/queue_resume.out 2>/dev/null; do sleep 120; done
sleep 90
python scripts/vary_followup.py > results/logs/vary_followup.log 2>&1 || { echo "FOLLOWUP FAILED"; tail -n 8 results/logs/vary_followup.log; }
tail -n 8 results/logs/vary_followup.log
echo FOLLOWDONE
