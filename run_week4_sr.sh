#!/usr/bin/env bash
# Re-run of the six PPO+SR arms after the rollout-batch-size fix.
# The six PPO runs from the first launch are unaffected and are NOT repeated.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=${STEPS:-3000000}
SEEDS=${SEEDS:-"0 1 2"}
EPS_FIXED=0.7792714676940609
EPS_RANDOM=0.9705424417947079
LOG=results/week4/launch
run () {
  echo "[$(date +%H:%M:%S)] START $1 seed $2" >> $LOG/progress.log
  .venv/bin/python -m training.train_flat --arm "$1" --seed "$2" \
      --timesteps "$STEPS" --epsilon "$3" > "$LOG/$1_seed$2.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $1 seed $2 (exit $?)" >> $LOG/progress.log
}
echo "=== SR RE-RUN after rollout fix, started $(date) ===" >> $LOG/progress.log
{ for s in $SEEDS; do run ppo_sr_fixed  "$s" $EPS_FIXED;  done; } & A=$!
{ for s in $SEEDS; do run ppo_sr_random "$s" $EPS_RANDOM; done; } & B=$!
wait $A $B
echo "=== SR RE-RUN COMPLETE $(date) ===" >> $LOG/progress.log
