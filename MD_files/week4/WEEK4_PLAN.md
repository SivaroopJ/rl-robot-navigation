# Week 4 Plan — Sibling Rivalry under smooth stochastic dynamic obstacles

**Branch:** `Week_4`  ·  **Base commit:** `9e307cf` (branch `main`)

Step 1 of the Week 4 brief required inspecting the repository and verifying, rather than
assuming, that the code matches the brief's description. It does not, in several
load-bearing ways. This document records what was found, what was decided, and why.

---

## 1. What the brief assumed, and what is actually here

| Brief assumes | Verified reality |
|---|---|
| "an existing PPO + Sibling Rivalry implementation" | **Does not exist in this repository.** `grep -ri "sibling\|rivalry\|antigoal"` returns nothing across all code, docs and 131 commits of history. A validated implementation exists in the sibling project `~/projects/ppo-nav` (`ppo_sr/`), written against MuJoCo PointMaze. |
| "2D continuous **maze**" | **There is no maze.** An open 10x10 arena with five isolated static rectangles occupying 13% of the area, fully connected, no corridors and no dead ends. No maze generation code exists. |
| "Do NOT add LiDAR — that is Experiment 4" | **LiDAR is already present**: 24 rays are 24 of the 34 observation dimensions (`robot_nav_env.py:132-134`). There is nothing to add and nothing was removed. |
| "the existing **Euclidean** reward"; "no Micanovic proximity reward" | The reward already contains a proximity penalty (`danger_zone_radius 0.8`, `proximity_penalty_scale 0.3`) and an action-smoothing penalty (`0.16`), both gated behind `use_reward_shaping`. It is not a pure Euclidean reward. It was left **exactly as it is**. |
| obstacles follow "fixed/deterministic trajectories" | Obstacles draw a **uniform random heading at every reset**, then travel at constant velocity with a **hard elastic reflection** off the world boundary. They pass through the static rectangles and through each other. The trajectory is a random-heading billiard path, deterministic only given the seed. |
| `e1..e4` are the described baseline | They are unrelated: an obstacle-density sweep, a speed sweep, an unseen-configuration generalization study and a reward-shaping ablation, all evaluating one curriculum-trained model. Week 4 writes to `results/week4/experiment{1,2,3}/` to avoid the collision. |

## 2. The measurement that drove the design

Episodes on `main` last **4-6 steps**. Measured across all thirteen committed eval files;
for `e1 N=6`: minimum 1 step, maximum 17, mean 5.1, and **zero timeouts out of 200**.
`max_steps = 500` is never approached.

There are two independent causes:

1. The action is a **direct displacement**, not a force or an integrated velocity:
   `position += action * max_speed` moves the agent up to 1.0 per axis and 1.41 diagonally
   in a 10-unit world.
2. Start and goal were both sampled uniformly with only a 2.0-unit minimum, giving a mean
   separation of **5.21** and **31.7% of pairs closer than 4.0 units**.

The agent therefore points at the goal and arrives in about five hops. This defeats Week 4
twice over:

* **The obstacle model cannot express itself.** An Ornstein-Uhlenbeck process is defined by
  its correlation time. Over five steps it never leaves its initial transient, so a
  "randomized" obstacle is indistinguishable from the constant-heading one it replaced, and
  Experiment 2 would be training on Experiment 1's environment.
* **Sibling Rivalry cannot engage.** Two siblings diverge only through five steps of action
  noise, so `rho` is tiny, `rho < epsilon` is essentially always true, and the acceptance
  rule never binds. SR degenerates into PPO with a negligible terminal term.

There was also a third concern, raised and accepted rather than solved: SR exists to escape
**local optima** in distance-based shaping, and an open arena has few. This is confirmed by
the project's own prior results in `ppo-nav/docs/ppo_sr_bc_baseline.md`, where SR ties
Euclidean shaping on easy (0.995 / 0.995) and medium (0.968 / 0.955) mazes and separates
only on the decoy-corridor hard maze (0.792 vs 0.050).

## 3. Decisions taken

Agreed with the project owner before any code was written:

1. **Work in this repository** and implement SR here, reusing `ppo-nav`'s validated logic
   rather than re-deriving it.
2. **Three seeds per arm.** The existing results are single-seed, and the effects in
   question are a few percentage points.
3. **Implementation and smoke tests this session**; full training is handed over.
4. **Fix the horizon with an explicit `dt`**, leaving arena geometry alone.
5. **Flat training** at one configuration rather than the five-stage curriculum.
6. **Keep the open arena**, but replace start/goal sampling with a stratified scheme that
   covers every region of the map and spans a controlled range of separations.

**Accepted risk, stated up front:** with the geometry unchanged, if moving obstacles alone
do not create transient local optima, the honest outcome is SR ≈ PPO. That is a clean and
interpretable null result, not a failed experiment — but it is a null result about *this
arena*, not about Sibling Rivalry in general, and `WEEK4_RESULTS.md` will say so.

## 4. What changed in the environment

### 4.1 An explicit timestep

`dt = 0.1` multiplies **both** the agent's and the obstacles' displacement, so the
obstacle-to-agent speed ratio, the radii, the obstacle count and the arena are all
preserved and only the temporal resolution changes. `dt = 1.0` reproduces the original
environment exactly, and a test pins that bit-for-bit against `main`.

A side benefit: the original 1.0-unit step could jump an agent straight through a 0.6-unit
contact zone, since collision is tested only at the step endpoint. At `dt = 0.1` that
tunnelling is largely gone.

### 4.2 Stratified start/goal sampling

Implemented in `robot_env/start_goal.py`. Two strata drawn jointly:

* **Region.** The free span is cut into a 4x4 lattice of 2.20-unit cells. A region is
  usable when more than 25% of its area is free; **15 of 16 qualify** on the shipped
  layout (the excluded one is 16% free, mostly filled by the lower-left bar).
* **Distance.** A band is drawn uniformly from `[[4.0, 5.5], [5.5, 7.0], [7.0, 8.5],
  [8.5, 10.5]]` and the goal is rejection-sampled to land inside it.

Naively drawing the start region uniformly and hoping the band was reachable failed badly —
a mid-map start cannot be 10 units from anything — producing a 25% fallback rate and bands
of 42/31/20/8%. The sampler instead precomputes which region *pairs* can serve each band,
and two passes of **iterative proportional fitting** rebalance the within-band weights so
the marginal over regions comes out level for starts and for goals. Both strata then hold
at once instead of trading off.

Measured over 6000 draws:

| | original uniform | Week 4 stratified |
|---|---|---|
| mean separation | 5.21 | **6.87** |
| median | 4.92 | 6.92 |
| minimum | 2.00 | 4.00 |
| fraction under 4.0 | **31.7%** | **0%** |
| band shares | — | 25.7 / 25.4 / 24.4 / 24.6 % |
| fallback rate | — | **0.00%** |

Every usable region receives at least half its uniform share as both start and goal
(asserted by `tests/test_week4_start_goal.py`).

**Consequence, recorded honestly:** this changes the task distribution, so Week 4's
Experiment 1 is a *new* baseline and is **not** numerically comparable to the committed
`e1..e4` numbers. That was already true for other reasons (dt, flat training, SR's widened
critic). No existing result file was modified.

### 4.3 The task is now genuinely a navigation problem

Two measurements, both new:

* A greedy straight-line policy — previously near-optimal — now **collides in about 90% of
  episodes**, overwhelmingly with static obstacles.
* Over 400 sampled pairs, A\* on the static map finds **0 unreachable pairs**, with a
  detour ratio (shortest path / straight line) of **mean 1.175, median 1.143, p90 1.309,
  max 2.42**; 17.5% of pairs require more than a 25% detour.

For comparison, `ppo-nav` measured detour ratios of 1.08 (easy), 1.19 (medium) and 1.17
(hard). The Week 4 arena now sits in the same range, which is the regime where Euclidean
shaping has something for SR to escape.

### 4.4 Smooth stochastic obstacle motion

`robot_env/dynamic_obstacles.py` holds two interchangeable models. `DeterministicMotion` is
the original behaviour, lifted verbatim, and is what Experiment 1 uses.
`SmoothStochasticMotion` implements the brief's model, per obstacle:

```
omega <- (1 - lambda*dt) * omega + sigma * sqrt(dt) * N(0,1)
omega <- clip(omega, -omega_max, +omega_max)
theta <- theta + omega * dt
x, y  <- x + v*cos(theta)*dt,  y + v*sin(theta)*dt
```

Speed is drawn once per episode and held; **only the direction process is randomized**.
Obstacle count, radius, speed and the arena are untouched.

Final parameters, chosen by measurement rather than assumption (see §5):

| parameter | value | rationale |
|---|---|---|
| `max_turn_rate` | 0.7 rad/s | turning radius `v/omega` = **1.43 units** in a 10-unit arena |
| `angular_velocity_decay` | 0.8 | correlation time `1/lambda` ≈ 12 steps |
| `angular_noise_sigma` | 0.35 | stationary `omega` std = `sigma/sqrt(2*lambda)` = 0.28 |
| `boundary_margin` | 2.2 | exceeds the turning radius, leaving a 5.6-unit interior |
| `boundary_steer_gain` | 4.0 | |
| `steer_full_error` | pi/4 | the steer saturates once heading is >45 deg from safety |
| `steer_full_depth` | 0.35 | full authority 35% into the margin, most of it still in hand |

## 5. Boundary handling, and a correction worth recording

The brief forbids an elastic reflection at the boundary because it reintroduces exactly the
discontinuity the OU process removes. The obstacle is instead steered back toward the
interior, with a position projection retained as a backstop.

The first implementation made the steer proportional to the heading error over the whole
margin. That is far too gentle: at margin entry the correction is near zero, so the
*effective* margin was much smaller than the configured one, and compensating by enlarging
the margin to 3.0 left only a 4-unit interior — every obstacle then spent its life turning,
looping tightly and covering just **55%** of the arena, against essentially 100% for the
original billiard motion. That would have been a much larger behavioural change than
"deterministic trajectory → stochastic trajectory", which is the only change the brief
sanctions.

The steer now saturates on both axes (beyond 45 degrees of heading error, and 35% into the
margin), which allows a smaller margin and restores traversal.

The remaining correction is to what is being asserted. Requiring the projection to *never*
fire is not the right test, and chasing it was optimising a proxy. The brief prohibits
*instant velocity reversal* and *sudden trajectory discontinuity*. The projection is
per-axis, so an obstacle that reaches a wall keeps its tangential motion and **slides**,
and its heading is not modified at all — it is a graze, not a bounce, and the steer curves
it away over the following steps. What matters is that contact is rare and shallow, and
that the heading stays continuous. Measured over 380,000 obstacle-steps with obstacles
spawned flush against the walls:

* max `|d theta|` per step = **0.0700 rad (4.01 deg)**, exactly the `omega_max * dt` bound;
* obstacles are **always inside the valid region** after every update;
* wall contact on **0.74%** of obstacle-steps, projection depth at most one step of travel;
* arena coverage **0.86** of cells, against 0.55 for the over-margined configuration.

`tests/test_week4_env.py` asserts these properties directly.

## 6. Sibling Rivalry

Implemented in `sibling_rivalry/`, ported from `ppo-nav/ppo_sr/`:

* `select.py` — `relabel_and_select`, Trott et al. Algorithm 1 lines 5-13: mutual
  relabeling, `tau_f` always retained, `tau_c` retained only when `rho < epsilon` or
  `d_c < delta`.
* `reward.py` — the terminal payout `min(0, -d(s_T, g) + d(s_T, gbar))`, **added to** the
  environment's reward on the final transition only. The per-step reward is untouched, so
  "keep the reward exactly the same" and "do not change SR" hold simultaneously. This is
  ppo-nav's `composite_sr` pattern, which applies directly because its `CompositeTerms` is
  literally this environment's reward.
* `policy.py` — the asymmetric critic `V(s, g, gbar)`. The critic reads 36 inputs, the
  **actor reads the unmodified 34**, so the SR actor is architecturally identical to the
  PPO actor and the arms differ by Sibling Rivalry alone. Had the anti-goal gone into the
  shared observation, the comparison would have confounded SR with a wider network.
* `ppo_sr.py` — `SiblingRivalryPPO`, overriding `collect_rollouts` to gather complete
  paired episodes. Stock PPO cuts episodes at a fixed slice boundary, which SR cannot use:
  the anti-goal *is* a terminal state, and the acceptance rule keeps or discards whole
  episodes.

**Sibling pairing.** Sub-envs `2i` and `2i+1` are reset with the *same* seed, so start,
goal, obstacle spawns, initial headings and the entire OU noise stream are byte-identical
and the siblings diverge only through sampled actions. This is the design question the
brief's structure leaves open, and it is settled the paper-faithful way: had the siblings
seen different obstacle trajectories, `rho` would mix policy variance with environment
variance and `epsilon` would be calibrated against the wrong quantity. `_assert_pairs_match`
checks it every rollout. `VecEnv.seed()` could not be used, because it deliberately hands
sub-env `i` the seed `base + i`.

**`epsilon` must be calibrated, not guessed.** `rho` is a world-unit distance whose scale
depends on the arena, the horizon and the policy's stochasticity. In the smoke run, an
untrained policy gave `rho` mean 1.11 and median 0.82; a placeholder `epsilon = 2.0`
accepted 94% of pairs, which is very nearly the "accept everything" degenerate case that
turns SR into PPO. `sibling_rivalry/calibrate.py` measures the distribution and reports a
percentile. `delta` is fixed to the environment's own `target_radius` (0.6) so SR's success
set and the environment's termination test cannot disagree.

## 7. Experiment structure

| Arm | Trained on | SR | Output |
|---|---|---|---|
| `ppo_fixed` | original obstacles | off | `results/week4/experiment1/ppo/` |
| `ppo_sr_fixed` | original obstacles | on | `results/week4/experiment1/ppo_sr/` |
| `ppo_random` | randomized obstacles | off | `results/week4/experiment2/ppo_randomized/` |
| `ppo_sr_random` | randomized obstacles | on | `results/week4/experiment3/ppo_sr_randomized/` |

The **zero-shot sub-experiment** evaluates the two Experiment 1 models on randomized
obstacles without retraining, into `results/week4/zero_shot/`. It answers a different
question from Experiments 2 and 3 and is reported separately, never merged into the same
table rows without a `zero_shot` label.

Evaluation uses seeds from `1_000_000`, disjoint from every seed training consumes, so
Experiments 2 and 3 are tested on trajectory realizations never trained on. Every arm sees
the identical seed block, making the comparison paired episode by episode.

## 8. Open items

* `sr.epsilon` in `config.json` is deliberately `null`. It must be set from a calibration
  run before full training; the trainer refuses to start an SR arm without it.
* The training timestep budget is a calibration output, not a guess. Episodes are now
  roughly an order of magnitude longer, so a given step budget buys far fewer episodes.
* SPL uses A\* over the static map as its denominator (`evaluation/shortest_path.py`),
  because the straight-line stand-in would overstate SPL by ~17.5% on average and would
  flatter a policy most on exactly the pairs where navigation was hardest. This is an
  evaluation-time metric only — it never enters a reward or an observation, so the brief's
  exclusion of geodesic distance is respected.
