#!/usr/bin/env bash
# Waits for the running Week 6 matrix (4M) to exit, then launches the same 12 runs at 10M.
# 10M is the budget the probe justified: PPO gains +0.06 success from 5M->10M, which is 8x
# the SR-vs-PPO effect size (-0.008) the Week 5 matrix failed to resolve.
set -u
cd "$(dirname "$0")"
WAIT_PID=${WAIT_PID:-1047087}
LOG=results/week7/launch; mkdir -p "$LOG"
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/chain.log"; }

note "waiting on Week 6 matrix (pid $WAIT_PID)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
note "Week 6 exited; waiting 120s for stragglers"
sleep 120
while pgrep -f "training.train_flat.*_v3" >/dev/null; do sleep 60; done

note "launching Week 7 matrix: 12 runs x 10M, root results/week7, tag _v4"
ROOT=results/week7 TAG=_v4 BUDGET=10000000 ./run_week6_matrix.sh >> "$LOG/chain.log" 2>&1
note "Week 7 chain finished (exit $?)"
