#!/usr/bin/env bash
# Finish the remaining four PPO+SR runs at 4-way concurrency.
#
# Measured motivation: two concurrent jobs were using only 6.2 of 16 cores, because PPO
# alternates an 8-worker rollout burst (~28% of the time) with a SINGLE-THREADED update
# phase (~72%), during which seven of a job's eight workers idle. A job therefore averages
# ~3 cores, not 8. Concurrency is result-neutral: each run is independently seeded and
# wall-clock speed does not enter the computation.
#
# The two seed-0 jobs are already running and are left alone. Seed 1 starts immediately to
# reach 4-way; seed 2 backfills each slot as its seed-0 job exits.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=3000000
EPS_FIXED=0.7792714676940609
EPS_RANDOM=0.9705424417947079
LOG=results/week4/launch

run () {
  echo "[$(date +%H:%M:%S)] START $1 seed $2" >> $LOG/progress.log
  .venv/bin/python -m training.train_flat --arm "$1" --seed "$2" \
      --timesteps $STEPS --epsilon "$3" > "$LOG/$1_seed$2.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $1 seed $2 (exit $?)" >> $LOG/progress.log
}

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 20; done; }

echo "[$(date +%H:%M:%S)] === switching to 4-way concurrency ===" >> $LOG/progress.log
run ppo_sr_fixed  1 $EPS_FIXED  & P1=$!
run ppo_sr_random 1 $EPS_RANDOM & P2=$!
# backfill seed 2 into each slot as the corresponding seed-0 job finishes
( wait_pid ${SEED0_FIXED};  run ppo_sr_fixed  2 $EPS_FIXED  ) & P3=$!
( wait_pid ${SEED0_RANDOM}; run ppo_sr_random 2 $EPS_RANDOM ) & P4=$!
wait $P1 $P2 $P3 $P4
echo "[$(date +%H:%M:%S)] === ALL SR RUNS COMPLETE ===" >> $LOG/progress.log
