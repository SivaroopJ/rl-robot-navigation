# Experiment 2 — PPO trained on smoothly randomized dynamic obstacles

**Status: implementation complete, smoke test complete, full training NOT run.**

## Question

Can PPO learn effectively when the dynamic obstacles have smooth stochastic trajectories?
This is Week 4's RQ2, and it is a prerequisite for RQ3: if PPO cannot learn here at all,
comparing it against PPO+SR measures nothing.

**This arm is TRAINED on randomized obstacles.** It is not a fixed-trained model tested on
randomized ones — that is the separate zero-shot sub-experiment, documented in
`ZERO_SHOT_ANALYSIS.md` and never merged into these rows.

## The randomized obstacle model

Each obstacle carries `(x, y, theta, v, omega)` and is integrated as

```
omega <- (1 - lambda*dt) * omega + sigma * sqrt(dt) * epsilon,   epsilon ~ N(0, 1)
omega <- clip(omega, -omega_max, +omega_max)
theta <- theta + omega * dt
x, y  <- x + v*cos(theta)*dt,  y + v*sin(theta)*dt
```

The Ornstein-Uhlenbeck term is the point. Redrawing `theta` each step would be
unpredictable but physically absurd — the obstacle would jitter rather than travel, and
would present no learnable structure. OU gives a heading that wanders smoothly: bounded
turn rate at every step, but genuinely unpredictable over an episode.

**Only the direction process is randomized.** Speed is drawn once per episode from
`speed_range` (default `[1.0, 1.0]`, i.e. the original fixed speed) and held. Obstacle
count, radius and the arena are identical to Experiment 1.

| parameter | value | meaning |
|---|---|---|
| `max_turn_rate` | 0.7 rad/s | turning radius 1.43 units in a 10-unit arena |
| `angular_velocity_decay` | 0.8 | correlation time ≈ 12 steps |
| `angular_noise_sigma` | 0.35 | stationary `omega` std 0.28, about 40% of `omega_max` |
| `boundary_margin` | 2.2 | exceeds the turning radius; leaves a 5.6-unit interior |
| `boundary_steer_gain` | 4.0 | |
| `steer_full_error` | pi/4 | steer saturates past 45 deg of heading error |
| `steer_full_depth` | 0.35 | full authority 35% into the margin |
| `randomize_initial_heading` | true | biased inward when spawned inside the margin |
| `trajectory_seed` | null | trajectories derive from the episode seed |

### Boundary handling

An elastic reflection is not used: it would reintroduce the discontinuity the OU process
exists to remove. Within `boundary_margin` a corrective term is added to `omega` **before**
the clip, so the bounded-turn-rate guarantee covers boundary handling too. The obstacle
curves away over several steps. A per-axis position projection remains as a backstop; it is
a *graze and slide* that leaves the heading untouched, not a bounce.

### Verified properties

Measured over 380,000 obstacle-steps with obstacles spawned flush against the walls
(`tests/test_week4_env.py`):

* `|omega| <= omega_max` at every step;
* max `|d theta|` per step = **0.0700 rad (4.01 deg)** = exactly `omega_max * dt`;
* speed constant to 1e-6;
* obstacles **always inside the valid region** after every update;
* wall contact on **0.74%** of obstacle-steps, projection depth at most one step of travel;
* `omega` lag-1 autocorrelation **0.98** — correlated in time, not white noise;
* arena coverage 0.86 of cells, so obstacles still traverse rather than loop locally.

Figures: `results/week4/trajectory_examples/obstacle_trajectories.png` shows the original
deterministic path beside three randomized ones, with the `omega` trace and the measured
`max |d theta|` under each. See `trajectory_examples.md`.

## Reproducibility

Randomized trajectories derive entirely from the environment's `np_random`, seeded by
`reset(seed=...)`. Given the same seed the maze, obstacle initial states and the whole
trajectory are reproduced exactly; different seeds diverge. Both are asserted by tests.

## Training and evaluation

Identical to Experiment 1 in every respect except `randomize = true`: same PPO
hyperparameters, architecture, budget, `n_envs`, episode length cap, reward, observation
and start/goal distribution.

Evaluation uses seeds `1_000_000 ..`, **disjoint from every seed training consumes**, so
the reported numbers are on unseen trajectory realizations. Recorded per episode: success,
collision (split into wall / static / dynamic), timeout, episode length, path length,
return, final distance, SPL, terminal position, collision position and the start-goal
distance band.

## Commands

```bash
python -m sibling_rivalry.calibrate --randomize --n-obstacles 6 --n-pairs 64 \
    --out results/week4/experiment2/sr_calibration.json
python -m experiments.week4 train    --arms ppo_random --seeds 0 1 2 --timesteps 2000000
python -m experiments.week4 evaluate --arms ppo_random --seeds 0 1 2 --episodes 200
```
