#!/usr/bin/env bash
# Overnight orchestrator. Runs unattended; nohup'd and independent of any session.
#
#   1. wait for the first matrix's SR runs (speed 1.0) to finish
#   2. evaluate that matrix end to end and write its summary
#   3. hand off to the speed-[0.6,0.75] matrix
#
# Step 2 exists because nothing else evaluates the first matrix's SR arms, and they are the
# only data for RQ1 and RQ3 at speed 1.0. Doing it before step 3 rather than concurrently
# keeps the CPU free for it -- evaluation takes ~30 min against the matrix's ~12 h.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/week4/launch
note () { echo "[$(date '+%m-%d %H:%M:%S')] [overnight] $*" >> "$LOG/progress.log"; }

note "orchestrator started; waiting for first-matrix SR runs"
while pgrep -f "train_flat --arm ppo_sr" > /dev/null 2>&1; do sleep 60; done
note "first matrix training complete: $(ls models/week4/ppo_sr_*/final_model.zip 2>/dev/null | wc -l)/6 SR models present"

note "evaluating first matrix (speed 1.0), 200 episodes x 3 seeds"
for attempt in 1 2; do
  if .venv/bin/python - >> "$LOG/eval_matrix1.log" 2>&1 <<'PYEND'
import os
os.chdir(os.environ.get("PWD", "."))
import experiments.week4 as w4
w4.run_evaluation(["ppo_fixed", "ppo_random", "ppo_sr_fixed", "ppo_sr_random"],
                  [0, 1, 2], n_eval_episodes=200)
w4.run_evaluation(["ppo_fixed", "ppo_sr_fixed"], [0, 1, 2],
                  n_eval_episodes=200, zero_shot=True)
w4.build_result_table()
w4.build_paired_comparisons([0, 1, 2])
PYEND
  then note "first-matrix evaluation done"; break
  else note "first-matrix evaluation attempt $attempt failed"; fi
done

.venv/bin/python week4_summary.py results/week4 > results/week4/SUMMARY.txt 2>&1
note "first-matrix summary -> results/week4/SUMMARY.txt"

note "handing off to the speed-[0.6,0.75] matrix (budget ${BUDGET:-5000000})"
exec ./run_week4_matrix_speed.sh
