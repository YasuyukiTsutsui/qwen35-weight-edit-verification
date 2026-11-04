#!/usr/bin/env bash
# Re-chain the GPU queue after the host slept (2026-11-04 13:37-20:04) and method B died at point 14 ("eval failed", no log output),
# which made the old chain skip ahead to the method-A preparation. Order from here on:
#   (method-A preparation, already running) -> method B (resumes from results/bo/state.json) -> null control -> variation study
#   -> extra studies -> weight-only study.   Everything is resumable; B is retried up to 3 times.
cd "$(dirname "$0")/.."
while ! grep -q "ADONE\|A-PREP FAILED" results/logs/queue_a.out 2>/dev/null; do sleep 60; done
sleep 90
for i in 1 2 3; do
  python scripts/bo_search.py --run --n-init 8 --n-bo 8 >> results/logs/bo_run.log 2>&1 && break
  echo "B attempt $i failed"; sleep 180
done
[ -f results/bo/final.json ] || { echo "RESUME: B FAILED"; exit 1; }
echo BDONE2
python scripts/null_select.py --n 5 > results/logs/null_select.log 2>&1 || { echo "NULL FAILED"; tail -n 5 results/logs/null_select.log; }
sleep 60
python scripts/vary_study.py > results/logs/vary_run.log 2>&1 || { echo "VARY FAILED"; tail -n 8 results/logs/vary_run.log; }
sleep 60
python scripts/extra_study.py > results/logs/extra_run.log 2>&1 || { echo "EXTRA FAILED"; tail -n 8 results/logs/extra_run.log; }
sleep 60
python scripts/weightonly_study.py > results/logs/weightonly_run.log 2>&1 || { echo "WO FAILED"; tail -n 8 results/logs/weightonly_run.log; }
echo RESUMEDONE
