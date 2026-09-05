#!/usr/bin/env bash
# Week 5 pilot: one seed per arm, short budget, 4-way concurrency.
# Purpose is a GATE, not a result: confirm the stalling is gone and nothing destabilised
# before committing 12 runs. Also the timing measurement for sizing the full matrix.
set -u
cd "$(dirname "$0")"

ROOT=results/week5_pilot
TAG=_w5pilot
BUDGET=${BUDGET:-1500000}
LOG=$ROOT/launch
mkdir -p "$LOG"

EPS_F=$(.venv/bin/python -c "import json;print(json.load(open('$ROOT/calibration/epsilon_fixed.json'))['recommended_epsilon'])")
EPS_R=$(.venv/bin/python -c "import json;print(json.load(open('$ROOT/calibration/epsilon_randomized.json'))['recommended_epsilon'])")

note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

run () {
  local arm=$1 eps=$2
  local extra=""
  [ "$eps" != "-" ] && extra="--epsilon $eps"
  note "START $arm"
  PYTHONPATH=. .venv/bin/python -m training.train_flat \
    --arm "$arm" --seed 0 --timesteps "$BUDGET" \
    --obstacle-speed-range 0.6 0.75 \
    --tag "$TAG" --results-root "$ROOT" $extra \
    > "$LOG/${arm}_seed0.log" 2>&1
  note "DONE  $arm (exit $?)"
}

note "=== Week 5 pilot: $BUDGET steps, seed 0, eps_fixed=$EPS_F eps_random=$EPS_R ==="
run ppo_fixed     "-"     &
run ppo_random    "-"     &
run ppo_sr_fixed  "$EPS_F" &
run ppo_sr_random "$EPS_R" &
wait
note "=== PILOT TRAINING COMPLETE ==="

note "evaluating pilot at speed 0.675"
.venv/bin/python - >> "$LOG/eval.log" 2>&1 <<PYEND
import os
os.chdir("$PWD")
import experiments.week4 as w4
w4.RESULTS_ROOT = "$ROOT"
w4.model_path_for = lambda arm, seed: os.path.join(
    "models", "week4", arm + "_seed" + str(seed) + "$TAG", "final_model.zip")
w4.run_evaluation(["ppo_fixed","ppo_random","ppo_sr_fixed","ppo_sr_random"],
                  [0], n_eval_episodes=200, obstacle_speed=0.675)
w4.build_result_table(os.path.join("$ROOT","result_table.json"))
PYEND
note "=== PILOT DONE ==="
