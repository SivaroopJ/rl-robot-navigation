#!/usr/bin/env bash
# Week 4 full training: 4 arms x 3 seeds, 3M steps each, in two balanced parallel streams.
#
# epsilon is per-ENVIRONMENT, calibrated from the rho distribution under an untrained
# policy (results/week4/calibration/). It is an SR-only hyperparameter, so it cannot
# advantage or disadvantage the PPO arms; the fixed and randomized environments get their
# own value because rho's scale depends on the environment.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=${STEPS:-3000000}
SEEDS=${SEEDS:-"0 1 2"}
EPS_FIXED=0.7792714676940609
EPS_RANDOM=0.9705424417947079
LOG=results/week4/launch

run () {   # run <arm> <seed> <epsilon-or-empty>
  local arm=$1 seed=$2 eps=${3:-}
  local extra=""; [ -n "$eps" ] && extra="--epsilon $eps"
  echo "[$(date +%H:%M:%S)] START $arm seed $seed" >> $LOG/progress.log
  .venv/bin/python -m training.train_flat --arm "$arm" --seed "$seed" \
      --timesteps "$STEPS" $extra > "$LOG/${arm}_seed${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $arm seed $seed (exit $?)" >> $LOG/progress.log
}

stream_a () { for s in $SEEDS; do run ppo_fixed  "$s"; done
              for s in $SEEDS; do run ppo_sr_fixed "$s" $EPS_FIXED; done; }
stream_b () { for s in $SEEDS; do run ppo_random "$s"; done
              for s in $SEEDS; do run ppo_sr_random "$s" $EPS_RANDOM; done; }

mkdir -p $LOG
echo "=== Week 4 training started $(date) : ${STEPS} steps, seeds ${SEEDS} ===" > $LOG/progress.log
stream_a & A=$!
stream_b & B=$!
wait $A $B
echo "=== ALL TRAINING COMPLETE $(date) ===" >> $LOG/progress.log
