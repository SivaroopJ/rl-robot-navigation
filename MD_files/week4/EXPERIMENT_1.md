# Experiment 1 — PPO vs PPO + Sibling Rivalry on the original obstacle motion

**Status: implementation complete, smoke test complete, full training NOT run.**
No result in this document is a measurement of a trained policy.

## Question

Does Sibling Rivalry improve navigation in the environment's original dynamic-obstacle
regime? This is Week 4's RQ1 and the control against which Experiments 2 and 3 are read.

## What "the original environment" means here

The brief describes Experiment 1's obstacles as following "fixed/deterministic
trajectories". Inspection showed otherwise, and the description is worth stating precisely
because it is the baseline everything else is compared against:

* each obstacle draws a **uniform random heading at every reset**;
* it then travels in a **straight line at constant speed**;
* at a world boundary its velocity component is **reflected elastically** — a discontinuous
  heading flip;
* obstacles pass **through** the static rectangles and **through each other**.

So the motion is a random-heading billiard path: piecewise linear, deterministic given the
seed. `DeterministicMotion` in `robot_env/dynamic_obstacles.py` is a verbatim lift of the
original `_bounce_obstacles` logic, and `tests/test_week4_env.py::
test_legacy_path_is_bit_identical_to_main` replays the original file out of git and asserts
identical observations, rewards and termination flags across obstacle counts {0, 3, 6, 10}
and six seeds.

## Configuration

Both arms share everything except Sibling Rivalry.

| | value |
|---|---|
| obstacles | 6, radius 0.3, speed 1.0, `randomize = false` |
| arena | 10 x 10, five static rectangles, unchanged |
| `dt` | 0.1 |
| start/goal | stratified, 4x4 regions, bands `[4.0,5.5] [5.5,7.0] [7.0,8.5] [8.5,10.5]` |
| observation | the existing 34-dim vector (`dx, dy, vx, vy`, 24 LiDAR rays, 3 nearest obstacle velocities) — unchanged, and no LiDAR added |
| reward | unchanged: +10 goal, -10 collision, +0.3·Δd, -0.02/step, proximity penalty, -0.16‖Δa‖² |
| PPO | SB3 2.9.0, `n_envs 8`, `n_steps 2048`, `batch_size 64`, `n_epochs 10`, `gamma 0.99`, `ent_coef 0.01`, `lr 3e-4`, `net_arch [256, 256]` |
| SR | `delta = 0.6` (the env's own `target_radius`), `epsilon` from calibration, `use_anti_goal = true` |
| seeds | 0, 1, 2 |
| evaluation | 200 episodes, seeds `1_000_000 .. 1_000_199`, deterministic policy |

The two arms differ in exactly two places: the SR arm's critic reads two extra inputs (the
anti-goal), and its rollouts are collected as paired complete episodes and filtered by the
acceptance rule. **The actor is architecturally identical in both arms.**

## Why this is a new baseline rather than the committed `e1` numbers

Week 4's Experiment 1 is *not* numerically comparable to `results/e1/`. Three things
changed relative to the committed runs, all deliberately and all agreed in advance:

1. `dt = 0.1` — without it, episodes last 4-6 steps and neither SR nor the Experiment 2/3
   obstacle model can express itself at all.
2. Stratified start/goal sampling — mean separation 5.21 → 6.87, and no episodes under 4.0.
3. Flat training instead of the five-stage curriculum, so the two arms cannot receive
   different effective budgets through the curriculum's retry-on-threshold logic.

Setting `dt = 1.0` and `start_goal.sampling = "uniform"` recovers the original environment
bit-for-bit, so the change is reversible and pinned by a test — but the trained numbers
belong to the new task distribution and are labelled as such. **No file under `results/e1`
… `results/e4` was modified.**

## Why the headline number is far below `results/e1`, and why that is not a regression

The committed `e1 N=6` figure is **75.5%**. Week 4's `ppo_fixed` seed 0 scores **38.3%**
(60 deterministic episodes, preliminary — the full run is 200 episodes across three seeds).
The gap is the task, not the policy, and the per-band breakdown shows it directly:

| start-goal band | success | note |
|---|---|---|
| 4.0 – 5.5 units | **0.71** | where essentially all of last week's episodes lived |
| 5.5 – 7.0 | 0.53 | |
| 7.0 – 8.5 | 0.21 | |
| 8.5 – 10.5 | 0.08 | |

Last week's separations averaged **5.21** with a median of 4.92 — squarely inside the first
band — and **31.7% of pairs were closer than 4.0 units**, i.e. easier than anything the
Week 4 sampler now produces. On the band that is actually comparable, this policy scores
**0.71 against the old 0.755**. The aggregate falls to 0.38 because three quarters of
episodes now sit at separations the old distribution never generated.

Three compounding changes, all deliberate and all recorded in `WEEK4_PLAN.md §3`:

| | committed `e1` | Week 4 Experiment 1 |
|---|---|---|
| `dt` | none (implicitly 1.0) | 0.1 |
| episode length | ~5 steps (max 17, zero timeouts) | 52–77 steps |
| separation | mean 5.21, 31.7% under 4.0 | mean 6.87, none under 4.0 |
| greedy straight-line policy | near-optimal | ~90% collision rate |
| training | 3.9M, five-stage curriculum | 3.0M, flat |

The old numbers were largely measuring a five-step dash at close range, which is precisely
why neither Sibling Rivalry nor a temporally-correlated obstacle model could have been
evaluated on that task at all.

**Where the failures are.** Measured on the same checkpoint: wall collisions **0**, static
rectangles **8**, dynamic obstacles **29**. Failures are 78% dynamic-obstacle collisions,
which is the failure mode Week 4 exists to study. This also refutes a hypothesis raised
during the run — that the agent might be driving into walls to collect the −10 early, since
a collision and a timeout pay the same. It is not doing that.

**The absolute number is a floor.** Training had not converged at 3M steps; see
`WEEK4_RESULTS.md §20.3` for the learning curve and the residual +0.056/1M trend. The budget
is held constant across all twelve runs so the arms stay mutually comparable, and a
higher-budget re-run is deferred rather than mixed into this matrix.

## Reading the outcome

RQ1 is answered by the paired comparison in
`results/week4/paired_comparisons.json → experiment1_fixed`, which matches PPO and PPO+SR
episode by episode on identical seeds, and reports McNemar counts (`only_a_success`,
`only_b_success`) rather than only the marginal rates — with matched episodes those counts,
not the difference of means, are what a significance test uses.

A null result here is informative and should not be presented as a failure: it is the
control that tells us whether any Experiment 2/3 difference is attributable to the
stochastic obstacles rather than to SR in general. The prior expectation, from this
project's own PointMaze results, is that SR ties Euclidean shaping wherever the geometry
lacks a strong local optimum.

## Commands

```bash
# calibrate epsilon on this environment first (the trainer refuses to run without it)
python -m sibling_rivalry.calibrate --n-obstacles 6 --n-pairs 64 \
    --out results/week4/experiment1/sr_calibration.json

python -m experiments.week4 train    --arms ppo_fixed ppo_sr_fixed --seeds 0 1 2 \
    --timesteps 2000000 --epsilon <calibrated>
python -m experiments.week4 evaluate --arms ppo_fixed ppo_sr_fixed --seeds 0 1 2 --episodes 200
```
