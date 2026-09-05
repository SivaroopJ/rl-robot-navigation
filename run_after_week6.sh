#!/usr/bin/env bash
# Runs after the Week 6 (4M) matrix exits, in this order:
#   1. Experiment 4 re-pilot at the corrected reward  (~1.3h, 4 arms x 3M)
#   2. The Week 7 matrix                              (12 runs x 10M)
# The re-pilot goes first because it is short and answers an open question; the Week 7 matrix
# is a ~21h grind whose start slipping by that much costs ~6% of its wall clock.
set -u
cd "$(dirname "$0")"
WAIT_PID=${WAIT_PID:-1047087}
LOG=results/week7/launch; mkdir -p "$LOG" results/experiment4_repilot/launch
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/chain.log"; }

note "waiting on Week 6 matrix (pid $WAIT_PID)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
note "Week 6 exited; draining stragglers"
sleep 120
while pgrep -f "training.train_flat.*_v3" >/dev/null; do sleep 60; done

note "STAGE 1: Experiment 4 re-pilot (collision -0.01, ent_coef 0, gamma 0.99)"
ROOT=results/experiment4_repilot TAG=_repilot BUDGET=3000000 ./run_exp4_pilot.sh >> "$LOG/chain.log" 2>&1
note "STAGE 1 done (exit $?)"

note "STAGE 2: Week 7 matrix -- 12 runs x 10M, root results/week7, tag _v4"
ROOT=results/week7 TAG=_v4 BUDGET=10000000 ./run_week6_matrix.sh >> "$LOG/chain.log" 2>&1
note "STAGE 2 done (exit $?)"
