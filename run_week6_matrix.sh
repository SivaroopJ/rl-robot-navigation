#!/usr/bin/env bash
# Experiments 1-3 re-run at the budget the probe measured. Identical to the Week 5 matrix
# in every respect except the step budget, so week5 -> week6 isolates budget alone.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=${ROOT:-results/week6}; TAG=${TAG:-_v3}; BUDGET=${BUDGET:-10000000}
LOG=$ROOT/launch; mkdir -p "$LOG" "$ROOT/calibration"
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

note "=== Week 6 matrix: $BUDGET steps/run, speed [0.6,0.75] ==="
note "calibrating epsilon (same env as Week 5, so this should reproduce it; recorded anyway)"
for mode in fixed randomized; do
  flag=""; [ "$mode" = "randomized" ] && flag="--randomize"
  PYTHONPATH=. .venv/bin/python -m sibling_rivalry.calibrate $flag --n-pairs 128 \
    --obstacle-speed-range 0.6 0.75 --out "$ROOT/calibration/epsilon_${mode}.json" \
    > "$LOG/calib_${mode}.log" 2>&1
done
EPS_F=$(.venv/bin/python -c "import json;print(json.load(open('$ROOT/calibration/epsilon_fixed.json'))['recommended_epsilon'])")
EPS_R=$(.venv/bin/python -c "import json;print(json.load(open('$ROOT/calibration/epsilon_randomized.json'))['recommended_epsilon'])")
note "epsilon: fixed=$EPS_F randomized=$EPS_R"

run () {
  local arm=$1 seed=$2 eps=$3 extra=""
  [ "$eps" != "-" ] && extra="--epsilon $eps"
  local free=$(df -P . | awk 'NR==2{print int($4/1048576)}')
  if [ "$free" -lt 10 ]; then note "ABORT: only ${free}G free"; return 1; fi
  note "START $arm seed $seed"
  PYTHONPATH=. .venv/bin/python -m training.train_flat --arm "$arm" --seed "$seed" \
    --timesteps "$BUDGET" --obstacle-speed-range 0.6 0.75 --tag "$TAG" \
    --results-root "$ROOT" $extra > "$LOG/${arm}_seed${seed}.log" 2>&1 || {
      note "RETRY $arm seed $seed"
      PYTHONPATH=. .venv/bin/python -m training.train_flat --arm "$arm" --seed "$seed" \
        --timesteps "$BUDGET" --obstacle-speed-range 0.6 0.75 --tag "$TAG" \
        --results-root "$ROOT" $extra >> "$LOG/${arm}_seed${seed}.log" 2>&1 || \
        note "GIVING UP on $arm seed $seed"; }
  note "DONE  $arm seed $seed"
}

# Mixed queues: SR runs ~1.7x slower than PPO, so an arm-per-queue split idles two queues.
q1 () { run ppo_sr_fixed  0 "$EPS_F"; run ppo_sr_fixed  1 "$EPS_F"; run ppo_fixed  0 "-"; }
q2 () { run ppo_sr_fixed  2 "$EPS_F"; run ppo_sr_random 0 "$EPS_R"; run ppo_fixed  1 "-"; }
q3 () { run ppo_sr_random 1 "$EPS_R"; run ppo_sr_random 2 "$EPS_R"; run ppo_fixed  2 "-"; }
q4 () { run ppo_random    0 "-";      run ppo_random    1 "-";      run ppo_random 2 "-"; }
q1 & q2 & q3 & q4 &
wait
note "=== TRAINING COMPLETE ($(ls models/week4/*${TAG}/final_model.zip 2>/dev/null|wc -l)/12) ==="

note "evaluating all arms at speed 0.675"
.venv/bin/python - >> "$LOG/eval.log" 2>&1 <<PYEND
import os
os.chdir("$PWD")
import experiments.week4 as w4
w4.RESULTS_ROOT = "$ROOT"
w4.model_path_for = lambda arm, seed: os.path.join(
    "models","week4", arm + "_seed" + str(seed) + "$TAG", "final_model.zip")
w4.run_evaluation(["ppo_fixed","ppo_random","ppo_sr_fixed","ppo_sr_random"],
                  [0,1,2], n_eval_episodes=200, obstacle_speed=0.675)
w4.run_evaluation(["ppo_fixed","ppo_sr_fixed"], [0,1,2],
                  n_eval_episodes=200, obstacle_speed=0.675, zero_shot=True)
w4.build_result_table(os.path.join("$ROOT","result_table.json"))
w4.build_paired_comparisons([0,1,2], os.path.join("$ROOT","paired_comparisons.json"))
PYEND
.venv/bin/python week4_summary.py "$ROOT" > "$ROOT/SUMMARY.txt" 2>&1
note "=== ALL DONE -> $ROOT/SUMMARY.txt ==="
