#!/usr/bin/env bash
# Overnight chain, unattended:
#   A  wait for the Experiment 4 pilot (already running)
#   B  GATE on the pilot -> Experiment 4 matrix (12 runs) if it passes, skip if not
#      + concurrently, a 10M single-run PROBE for Experiments 1-3
#   C  evaluate probe checkpoints -> choose the Exp 1-3 budget from the measured curve
#   D  launch the 12-run Experiment 1-3 matrix at that budget, then evaluate
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/overnight; mkdir -p "$LOG"
note () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/chain.log"; }
free_gb () { df -P . | awk 'NR==2{print int($4/1048576)}'; }

# ---------------------------------------------------------------- A: wait for pilot
note "=== chain start ==="
while ! grep -q "PILOT COMPLETE" results/experiment4_pilot/launch/progress.log 2>/dev/null; do
  if ! pgrep -f "run_exp4_pilot.sh" >/dev/null && \
     ! pgrep -f "train_exp4 --arm" >/dev/null; then
    note "pilot processes vanished without completing; continuing to the gate anyway"; break
  fi
  sleep 60
done
note "A: pilot finished"

# ---------------------------------------------------------------- B: gate + probe
.venv/bin/python - > "$LOG/gate.json" 2> "$LOG/gate.log" <<'PYEND'
"""Gate: did SR separate from PPO on the maze? Evaluated deterministically, not from the
noisy training callback. Skipping the matrix when SR has not separated is the point of
having a pilot at all -- launching 12 runs on a task no arm can solve measures nothing."""
import json, os, numpy as np
os.chdir(os.environ.get("PWD", "."))
from stable_baselines3 import PPO
from robot_env.point_maze import PointMazeEnv

def score(path, sr, use_lidar, n=100):
    if not os.path.exists(path):
        return None
    if sr:
        from sibling_rivalry.ppo_sr import SiblingRivalryPPO
        model = SiblingRivalryPPO.load(path, device="cpu")
        pad = int(model.policy.anti_goal_dim)
    else:
        model, pad = PPO.load(path, device="cpu"), 0
    env = PointMazeEnv(use_lidar=use_lidar)
    S, C = [], []
    for i in range(n):
        obs, _ = env.reset(seed=200_000 + i)
        for _ in range(env.MAX_STEPS):
            mo = obs if pad == 0 else np.concatenate([obs, np.zeros(pad, obs.dtype)])
            a, _ = model.predict(mo, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        S.append(info["success"]); C.append(info["collision_count"])
    env.close()
    return {"success": float(np.mean(S)), "collisions": float(np.mean(C))}

out = {}
for arm, sr, lid in (("ppo", False, False), ("ppo_lidar", False, True),
                     ("ppo_sr", True, False), ("ppo_sr_lidar", True, True)):
    out[arm] = score(f"models/experiment4/{arm}_seed0_pilot/final_model.zip", sr, lid)

sr_best = max((out[a] or {"success": 0})["success"] for a in ("ppo_sr", "ppo_sr_lidar"))
ppo_best = max((out[a] or {"success": 0})["success"] for a in ("ppo", "ppo_lidar"))
out["_gate"] = {
    "sr_best": sr_best, "ppo_best": ppo_best,
    "rule": "pass if SR >= 0.15 and SR > PPO + 0.10",
    "pass": bool(sr_best >= 0.15 and sr_best > ppo_best + 0.10),
}
print(json.dumps(out, indent=2))
PYEND
GATE=$(.venv/bin/python -c "import json;print(json.load(open('$LOG/gate.json'))['_gate']['pass'])" 2>/dev/null || echo False)
SRB=$(.venv/bin/python -c "import json;print(round(json.load(open('$LOG/gate.json'))['_gate']['sr_best'],3))" 2>/dev/null || echo NA)
PPB=$(.venv/bin/python -c "import json;print(round(json.load(open('$LOG/gate.json'))['_gate']['ppo_best'],3))" 2>/dev/null || echo NA)
note "B: Exp4 gate -> pass=$GATE  (SR best $SRB vs PPO best $PPB)"

# probe runs concurrently with whatever Exp 4 does next
note "B: launching Exp 1-3 budget probe (ppo_random seed 0, 10M)"
PYTHONPATH=. nohup .venv/bin/python -m training.train_flat --arm ppo_random --seed 0 \
  --timesteps 10000000 --obstacle-speed-range 0.6 0.75 --tag _probe10M \
  --results-root results/probe10M > "$LOG/probe.log" 2>&1 &
PROBE_PID=$!

if [ "$GATE" = "True" ]; then
  note "B: Exp4 gate PASSED -> launching the 12-run Experiment 4 matrix at 5M"
  BUDGET=5000000 ./run_exp4_matrix.sh >> "$LOG/exp4_matrix.log" 2>&1
  note "B: Experiment 4 matrix finished"
else
  note "B: Exp4 gate FAILED -> matrix SKIPPED (see $LOG/gate.json). Not spending 6h on a"
  note "   task no arm solved. Experiment 4 needs a design change, not more compute."
fi

note "C: waiting for the budget probe"
wait $PROBE_PID 2>/dev/null
note "C: probe finished; evaluating its checkpoints"
.venv/bin/python - > "$LOG/budget.json" 2>> "$LOG/chain.log" <<'PYEND'
"""Choose the Exp 1-3 budget from the probe's own curve rather than by guesswork.

Rule: the smallest checkpoint B at which the NEXT 2M of training adds less than 0.02
success. If it is still climbing faster than that at 10M, take 10M -- the probe's maximum
is then the honest answer and the limitation gets recorded rather than hidden."""
import glob, json, os, re
import numpy as np
os.chdir(os.environ.get("PWD", "."))
from evaluation.evaluate_week4 import evaluate_model

files = glob.glob("models/week4/ppo_random_seed0_probe10M/ppo_*_steps.zip")
avail = sorted((int(re.search(r"_(\d+)_steps", f).group(1)), f) for f in files)
curve = {}
for target in (2e6, 4e6, 6e6, 8e6, 10e6):
    if not avail: break
    step, path = min(avail, key=lambda sf: abs(sf[0] - target))
    rec = evaluate_model(path, sr=False, randomize=True, n_dynamic_obstacles=6,
                         obstacle_speed=0.675, n_eval_episodes=150, deterministic=True)
    eps = rec["episodes"]
    curve[step] = float(np.mean([e["success"] for e in eps]))

steps = sorted(curve)
budget = steps[-1] if steps else 5_000_000
for i in range(len(steps) - 1):
    gain = curve[steps[i + 1]] - curve[steps[i]]
    span = (steps[i + 1] - steps[i]) / 1e6
    if gain / max(span, 1e-9) < 0.02:
        budget = steps[i]
        break
budget = int(min(max(budget, 3_000_000), 10_000_000))
print(json.dumps({"curve": curve, "chosen_budget": budget,
                  "rule": "first checkpoint whose next 2M adds < 0.02 success/M; else 10M"},
                 indent=2))
PYEND
BUD=$(.venv/bin/python -c "import json;print(json.load(open('$LOG/budget.json'))['chosen_budget'])" 2>/dev/null || echo 10000000)
note "C: chosen Exp 1-3 budget = $BUD steps"

# ---------------------------------------------------------------- D: Exp 1-3 matrix
if [ "$(free_gb)" -lt 10 ]; then note "D: ABORT, only $(free_gb)G free"; exit 1; fi
note "D: launching the 12-run Experiment 1-3 matrix at $BUD steps -> results/week6"
BUDGET=$BUD ROOT=results/week6 TAG=_v3 ./run_week6_matrix.sh >> "$LOG/week6.log" 2>&1
note "=== CHAIN COMPLETE ==="
