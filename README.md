# rl-robot-navigation

Autonomous navigation of a mobile robot in a dynamic 2D environment using **Proximal Policy Optimization (PPO)** — a deep reinforcement learning algorithm. The agent learns to reach a target position while avoiding both static and moving obstacles, without any predefined rules or map of the environment.

The robot learns a reactive navigation policy entirely through trial and error in a custom simulation, with no prior knowledge of the environment dynamics.

---

## Research focus

This project investigates the performance of PPO under varying environmental conditions rather than comparing different reinforcement learning algorithms. The four experiments isolate the effect of individual design choices:

| Experiment | Variable studied |
|---|---|
| E1 | Obstacle density (N = 3, 6, 10) |
| E2 | Obstacle speed (0.5, 1.0, 1.5) |
| E3 | Generalization to unseen configurations |
| E4 | Reward shaping vs no reward shaping (ablation) |

The internal comparisons—particularly Experiment E4 and the progression from v1 to v4—serve as the primary ablation study. A comparison with other reinforcement learning algorithms (e.g., SAC or TD3) would require equivalent training budgets, network architectures, and curriculum designs to ensure a fair evaluation, and is therefore left for future work.

---

## Environment

The environment is built from scratch as a custom Gymnasium environment. It's a 2D space where the robot has to find its way to a target while dealing with both walls and moving obstacles. To sense its surroundings, the agent uses a simulated LIDAR with 24 rays that measure the distance to the nearest obstacle in each direction. It also receives the velocities of the 3 nearest dynamic obstacles, so it has some sense of where they're heading and not just how close they are.

Training runs with PPO through Stable-Baselines3, with 8 environments running in parallel to speed things up. The experiments cover a few different scenarios — changing how many obstacles there are, how fast they move, and testing whether the agent can handle layouts it has never seen before. Results are visualized through learning curves and trajectory plots.

Each episode starts with a randomized agent position and target, so the agent can't just memorize specific scenarios.

| Element | Description |
|---|---|
| Agent | Continuous 2D velocity control |
| Target | Randomly placed each episode |
| Static obstacles | Fixed walls and objects |
| Dynamic obstacles | N entities moving on random trajectories |

**Observation space:**

```
s = [dx, dy, vx, vy, l1...l24, ovx1, ovy1, ovx2, ovy2, ovx3, ovy3]
```

- `dx, dy` — relative position of the target (normalized by world size)
- `vx, vy` — current agent velocity (normalized by max speed)
- `l1...l24` — LIDAR readings at 15° intervals (normalized by lidar range)
- `ovx1, ovy1...ovx3, ovy3` — velocity of the 3 nearest dynamic obstacles (normalized; zero-padded if fewer than 3 are present)

Total observation size: **34** (up from 12 in v1.0-baseline)

**Action space:** continuous vector `[vx, vy] ∈ [−1, 1]²`

**Reward function:**

```
+10.0        reaching the target
−10.0        collision with any obstacle (episode ends)
+0.3·Δd      progress toward the target (reward shaping)
−0.02        penalty per timestep (encourages efficiency)
−f(d)        proximity penalty, active when closer than danger_zone_radius
−0.16·‖Δa‖²  action smoothing penalty (penalizes large changes in action between consecutive steps)
```
The proximity penalty `f(d)` scales smoothly from `0` at `danger_zone_radius` to `-proximity_penalty_scale` at contact distance. The idea is to discourage the agent from getting too close to obstacles even when it doesn't actually collide. The action smoothing penalty discourages sudden, erratic movement by penalizing the squared difference between consecutive actions, encouraging the agent to follow smoother and more stable trajectories. Both reward-shaping terms (progress + proximity) are only active when `use_reward_shaping=True`. Action smoothing is controlled separately via `action_smoothing_scale` in `config.json`.


## Installation

```bash
git clone https://github.com/PetraMicanovic/rl-robot-navigation.git
cd rl-robot-navigation
pip install -r requirements.txt
```

**Requirements:** 
- Python 3.10+, see `requirements.txt` for full list.
---

## Running

Set `MODE` at the top of `main.py` and run:

```bash
python main.py
```

| MODE | What it does |
|---|---|
| `"curriculum"` | Only trains the curriculum model, no experiments |
| `"experiments"` | Runs E1–E4, each trains its own model from scratch |
| `"all"` | Trains curriculum model, then E1-E4 evaluate that same curriculum model |

---

## Releases

### [v1.0-baseline](https://github.com/PetraMicanovic/rl-robot-navigation/releases/tag/v1.0-baseline)

Initial working version of the environment and training pipeline.

- 8-ray LIDAR observation space: `[dx, dy, vx, vy, l1...l8]` (12 dimensions)
- Basic reward function: goal, collision, progress shaping, step penalty
- Experiments E1–E4 covering obstacle density, speed, generalization, and reward shaping

---

### [v2.0-obs-reward](https://github.com/PetraMicanovic/rl-robot-navigation/releases/tag/v2.0-obs-reward)

Major upgrade to observation space and reward function.

**Observation space** expanded from 12 to 34 dimensions:
- LIDAR resolution increased from 8 to **24 rays** (45° → 15° angular resolution)
- Added velocities of the **3 nearest dynamic obstacles** `[vx1, vy1, vx2, vy2, vx3, vy3]` — gives the agent awareness of obstacle movement direction, not just proximity

**Reward function** extended with a proximity penalty:
- Smooth penalty scaling from `0` at `danger_zone_radius` to `-proximity_penalty_scale` at contact distance

---

### [v3.0-curriculum](https://github.com/PetraMicanovic/rl-robot-navigation/releases/tag/v3.0-curriculum)

Replaces fixed two-phase training with a multi-stage curriculum that gradually increases difficulty.

**Curriculum stages** (default):

| Stage | Obstacles | Speed | Timesteps | Threshold |
|---|---|---|---|---|
| 1 | 0 | 1.0 | 500 000 | 70% |
| 2 | 3 | 0.5 | 600 000 | 55% |
| 3 | 3 | 1.0 | 600 000 | 50% |
| 4 | 6 | 1.0 | 800 000 | 45% |
| 5 | 10 | 1.0 | 1 000 000 | — |

Each stage starts from the best model saved in the previous stage. The threshold shows the success rate the agent needs to reach before advancing — if not met, the agent still advances but the log will note it.

**Training entry points** (`train.py`):
- `train()` — single fixed-config run, same as v1/v2
- `train_curriculum()` — full curriculum pipeline, saves a canonical model alias at `models/ppo_robot_nav_curriculum_shaping.zip`

Experiments E1–E4 support two modes: evaluating the curriculum model directly (`train_models=False`) or running per-experiment training (`train_models=True`), controlled by `MODE` in `main.py`.

---

### v4.0-action-smoothing ← current

Two changes compared to v3.0-curriculum:

```json
"action_smoothing_scale": 0.16
```

```python
lr = linear_schedule(3e-4)
```

The action smoothing penalty discourages erratic movement by penalizing large action changes between consecutive steps. Several values were tested; `0.16` gave the best results. The linear learning rate schedule decays the LR from 3e-4 to 0 over training, allowing larger updates early on and more conservative updates as the policy stabilizes.

The curriculum is the same 5-stage progression as v3 (N=0 -> N=3 -> N=3 -> N=6 -> N=10), unchanged. Total training is approximately 3 900 000 timesteps.

These two changes together gave consistent gains of ~+20pp over v3 across all experiments, with collision rates dropping from 40–60% down to 15–45% depending on the configuration.

See [`results/RESULTS.md`](results/RESULTS.md) for the full evaluation.

---