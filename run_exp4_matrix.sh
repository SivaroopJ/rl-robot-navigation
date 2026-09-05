#!/usr/bin/env bash
# Experiment 4 matrix: 4 arms x 3 seeds. Queues are MIXED because the LiDAR arms run at
# ~740 fps against ~1030 for the others; an arm-per-queue split would leave two queues idle.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=${ROOT:-results/experiment4}; TAG=${TAG:-_v1}; BUDGET=${BUDGET:-5000000}
LOG=$ROOT/launch; mkdir -p "$LOG"
E_SR=$(.venv/bin/python -c "import json;print(json.load(open('results/experiment4/calibration/epsilon_ppo_sr.json'))['recommended_epsilon'])")
E_SRL=$(.venv/bin/python -c "import json;print(json.load(open('results/experiment4/calibration/epsilon_ppo_sr_lidar.json'))['recommended_epsilon'])")
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

run () {
  local arm=$1 seed=$2 eps=$3 extra=""
  [ "$eps" != "-" ] && extra="--epsilon $eps"
  local free=$(df -P . | awk 'NR==2{print int($4/1048576)}')
  if [ "$free" -lt 10 ]; then note "ABORT: only ${free}G free"; return 1; fi
  note "START $arm seed $seed"
  PYTHONPATH=. .venv/bin/python -m training.train_exp4 --arm "$arm" --seed "$seed" \
    --timesteps "$BUDGET" --tag "$TAG" --results-root "$ROOT" $extra \
    > "$LOG/${arm}_seed${seed}.log" 2>&1 || {
      note "RETRY $arm seed $seed"
      PYTHONPATH=. .venv/bin/python -m training.train_exp4 --arm "$arm" --seed "$seed" \
        --timesteps "$BUDGET" --tag "$TAG" --results-root "$ROOT" $extra \
        >> "$LOG/${arm}_seed${seed}.log" 2>&1 || note "GIVING UP on $arm seed $seed"; }
  note "DONE  $arm seed $seed"
}

note "=== Experiment 4 matrix: $BUDGET steps/run, eps_sr=$E_SR eps_sr_lidar=$E_SRL ==="
q1 () { run ppo_sr_lidar 0 "$E_SRL"; run ppo_sr_lidar 1 "$E_SRL"; run ppo 0 "-"; }
q2 () { run ppo_sr_lidar 2 "$E_SRL"; run ppo_lidar   0 "-";       run ppo 1 "-"; }
q3 () { run ppo_lidar    1 "-";      run ppo_lidar   2 "-";       run ppo 2 "-"; }
q4 () { run ppo_sr       0 "$E_SR";  run ppo_sr      1 "$E_SR";   run ppo_sr 2 "$E_SR"; }
q1 & q2 & q3 & q4 &
wait
note "=== EXP4 TRAINING COMPLETE ($(ls models/experiment4/*${TAG}/final_model.zip 2>/dev/null|wc -l)/12) ==="
