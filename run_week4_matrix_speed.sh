#!/usr/bin/env bash
# Second Week 4 matrix: 4 arms x 3 seeds, obstacle speed sampled from [0.6, 0.75].
#
# Built to run UNATTENDED. It is nohup'd and independent of any interactive session; it
# retries a failed run rather than leaving a hole in the matrix, guards against the shared
# filesystem filling up, and writes its own summary at the end so results are readable
# without reconstructing them from logs.
#
# WHY 0.6-0.75. Measured sweep (results/week4/obstacle_speed_sweep.json), policy held fixed:
# above 1.0 the task stops measuring the policy (at 1.5, collision 0.844 and obstacles
# outrun the agent, so many failures are unavoidable); below ~0.5 the curve flattens
# (0.408 at 0.25 vs 0.389 at 0.50, inside the seed spread) and obstacles stop being a
# hazard at all. 0.6-0.75 keeps success responsive to speed while leaving a real evasion
# margin (agent diagonal 1.414 vs obstacle 0.75).
#
# The first matrix is untouched: --tag separates model/log paths, results go to a new root.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

BUDGET=${BUDGET:-5000000}
SEEDS=${SEEDS:-"0 1 2"}
LOW=0.6
HIGH=0.75
MID=0.675
TAG="_spd06_075"
ROOT=results/week4_speed06_075
LOG=$ROOT/launch
MIN_FREE_GB=5
mkdir -p "$LOG" "$ROOT/calibration"

note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG/progress.log"; }
free_gb () { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

disk_ok () {
  local f
  f=$(free_gb)
  if [ "${f:-0}" -lt "$MIN_FREE_GB" ]; then
    note "ABORT: only ${f}G free (need ${MIN_FREE_GB}G). Not starting more runs."
    return 1
  fi
  return 0
}

# The orchestrator already waited for the first matrix; guard anyway in case this script
# is ever run directly.
while pgrep -f "train_flat --arm ppo_sr" > /dev/null 2>&1; do sleep 60; done
note "starting speed matrix. Budget $BUDGET, speed [$LOW,$HIGH], $(free_gb)G free"

calib () {
  local name=$1
  shift
  local attempt
  for attempt in 1 2; do
    if .venv/bin/python -m sibling_rivalry.calibrate --n-pairs 128 --n-obstacles 6 \
        --obstacle-speed-range $LOW $HIGH --seed 0 "$@" \
        --out "$ROOT/calibration/epsilon_${name}.json" > "$LOG/calib_${name}.log" 2>&1; then
      return 0
    fi
    note "calibration ($name) attempt $attempt failed; retrying"
  done
  return 1
}
note "calibrating epsilon (rho scale is environment dependent, so speed 1.0 values do not carry over)"
calib randomized --randomize || note "WARNING: randomized calibration failed"
calib fixed || note "WARNING: fixed calibration failed"

read_eps () {
  .venv/bin/python -c "
import json
try:
    print(json.load(open('$1'))['recommended_epsilon'])
except Exception:
    print('')
" 2>/dev/null
}
EPS_R=$(read_eps "$ROOT/calibration/epsilon_randomized.json")
EPS_F=$(read_eps "$ROOT/calibration/epsilon_fixed.json")
if [ -z "$EPS_R" ]; then EPS_R=SKIP; fi
if [ -z "$EPS_F" ]; then EPS_F=SKIP; fi
if [ "$EPS_R" = "SKIP" ] || [ "$EPS_F" = "SKIP" ]; then
  note "WARNING: missing calibrated epsilon. SR arms with no epsilon are SKIPPED rather"
  note "         than trained on a guessed value; PPO arms proceed normally."
fi
note "epsilon: fixed=$EPS_F randomized=$EPS_R"

run () {
  local arm=$1 seed=$2 eps=${3:-} extra="" attempt
  if [ "$eps" = "SKIP" ]; then
    note "SKIP $arm seed $seed (no calibrated epsilon)"
    return 0
  fi
  if [ "$eps" != "-" ]; then extra="--epsilon $eps"; fi
  for attempt in 1 2; do
    disk_ok || return 1
    note "START $arm seed $seed (attempt $attempt)"
    if .venv/bin/python -m training.train_flat --arm "$arm" --seed "$seed" \
         --timesteps "$BUDGET" --obstacle-speed-range $LOW $HIGH \
         --tag "$TAG" --results-root "$ROOT" $extra \
         > "$LOG/${arm}_seed${seed}.log" 2>&1; then
      note "DONE  $arm seed $seed"
      return 0
    fi
    note "FAILED $arm seed $seed attempt $attempt: $(tail -3 "$LOG/${arm}_seed${seed}.log" | tr '\n' ' ' | cut -c1-200)"
  done
  note "GIVING UP on $arm seed $seed after 2 attempts"
  return 1
}

q1 () { for s in $SEEDS; do run ppo_fixed     "$s" "-";     done; }
q2 () { for s in $SEEDS; do run ppo_random    "$s" "-";     done; }
q3 () { for s in $SEEDS; do run ppo_sr_fixed  "$s" "$EPS_F"; done; }
q4 () { for s in $SEEDS; do run ppo_sr_random "$s" "$EPS_R"; done; }
q1 &
q2 &
q3 &
q4 &
wait
note "=== TRAINING COMPLETE ($(ls models/week4/*${TAG}/final_model.zip 2>/dev/null | wc -l)/12 models) ==="

note "evaluating all arms at fixed speed $MID (range midpoint, so arms compare at one speed)"
SEEDLIST=$(echo $SEEDS | tr ' ' ',')
.venv/bin/python - >> "$LOG/eval.log" 2>&1 <<PYEND
import os
os.chdir("$PWD")
import experiments.week4 as w4
w4.RESULTS_ROOT = "$ROOT"
w4.model_path_for = lambda arm, seed: os.path.join(
    "models", "week4", arm + "_seed" + str(seed) + "$TAG", "final_model.zip")
w4.run_evaluation(["ppo_fixed", "ppo_random", "ppo_sr_fixed", "ppo_sr_random"],
                  [$SEEDLIST], n_eval_episodes=200, obstacle_speed=$MID)
w4.run_evaluation(["ppo_fixed", "ppo_sr_fixed"], [$SEEDLIST],
                  n_eval_episodes=200, obstacle_speed=$MID, zero_shot=True)
w4.build_result_table(os.path.join("$ROOT", "result_table.json"))
w4.build_paired_comparisons([$SEEDLIST], os.path.join("$ROOT", "paired_comparisons.json"))
PYEND
note "evaluation done"

.venv/bin/python week4_summary.py "$ROOT" > "$ROOT/SUMMARY.txt" 2>&1
note "=== ALL DONE. Summary written to $ROOT/SUMMARY.txt ==="
