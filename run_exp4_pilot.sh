#!/usr/bin/env bash
# Experiment 4 pilot. A GATE, not a result: is the maze solvable at all, and does SR
# separate from PPO the way the paper reports? If PPO+SR does not beat PPO here, the
# reward scale or MAX_STEP is wrong and the 12-run matrix would be worthless.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=${ROOT:-results/experiment4_pilot}; TAG=${TAG:-_pilot}
LOG=$ROOT/launch; mkdir -p "$LOG"
BUDGET=${BUDGET:-600000}
E_SR=$(.venv/bin/python -c "import json;print(json.load(open('results/experiment4/calibration/epsilon_ppo_sr.json'))['recommended_epsilon'])")
E_SRL=$(.venv/bin/python -c "import json;print(json.load(open('results/experiment4/calibration/epsilon_ppo_sr_lidar.json'))['recommended_epsilon'])")
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

run () {
  local arm=$1 eps=$2 extra=""
  [ "$eps" != "-" ] && extra="--epsilon $eps"
  note "START $arm"
  PYTHONPATH=. .venv/bin/python -m training.train_exp4 --arm "$arm" --seed 0 \
    --timesteps "$BUDGET" --tag "$TAG" --results-root "$ROOT" $extra \
    > "$LOG/${arm}.log" 2>&1
  note "DONE  $arm (exit $?)"
}
note "=== Experiment 4 pilot [$ROOT tag $TAG]: $BUDGET steps, seed 0, eps_sr=$E_SR eps_sr_lidar=$E_SRL ==="
run ppo          "-"      &
run ppo_lidar    "-"      &
run ppo_sr       "$E_SR"  &
run ppo_sr_lidar "$E_SRL" &
wait
note "=== PILOT COMPLETE ==="
