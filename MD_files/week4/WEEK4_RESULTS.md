# Week 4 Results — Sibling Rivalry under smooth stochastic dynamic obstacles

> **STATUS**
>
> | stage | state |
> |---|---|
> | Implementation | **complete** |
> | Smoke tests | **complete** — 51 tests pass; full pipeline runs end to end |
> | Full training | **NOT RUN** |
> | Full evaluation | **NOT RUN** |
>
> Every table below marked *pending* contains **no numbers**, because none have been
> measured. The measured numbers in this document describe the **environment** and the
> **implementation**, not any trained policy. Nothing here is projected, estimated or
> inferred from a short run.
>
> Branch `Week_4`, base commit `9e307cf`.

---

## 1. Motivation

Sibling Rivalry (Trott et al., NeurIPS 2019) exists to escape the local optima that
distance-based reward shaping creates in goal-reaching tasks. Every published use of it —
including this project's own earlier work — studies it against **static** geometry, where
the trap is a fixed feature of the map.

Week 4 asks whether the same machinery helps when the trap is *transient*: created and
dissolved by obstacles that move. That is a genuinely new question, and it is the reason
the obstacle motion model matters as much as the algorithm.

## 2. Research questions

* **RQ1** — Does SR improve PPO on the original deterministic obstacle motion?
* **RQ2** — Can PPO learn effectively when obstacles move on smooth stochastic trajectories?
* **RQ3** — Does PPO+SR outperform PPO when **both are trained** on that stochastic
  environment? *(primary)*
* **RQ4** — Does an SR model trained on deterministic motion transfer better to stochastic
  motion than ordinary PPO? *(supplementary)*

The answers are to be read off the experiments, not assumed. §20 states in advance what
would make each answer trustworthy.

## 3. Environment

A 10 × 10 continuous arena — **not a maze**, contrary to the brief's description. Five
static rectangles occupy 13% of the area; the space is fully connected with no corridors or
dead ends. The agent is a disc of radius 0.3 commanding a velocity in `[-1, 1]²`.

Observation: the existing **34-dimensional** vector — `dx, dy, vx, vy`, **24 LiDAR rays**,
and the velocities of the 3 nearest dynamic obstacles. The brief asked that no LiDAR be
added; LiDAR was already 24 of the 34 dimensions, so nothing was added and nothing removed.

Reward, unchanged from `main`: `+10` goal, `−10` collision, `+0.3·Δd` progress, `−0.02` per
step, a proximity penalty, and `−0.16‖Δa‖²` action smoothing. The brief described this as
"the Euclidean reward" and asked for no proximity term; the proximity and smoothing terms
were already present, and were left exactly as they are. **No reward constant was changed**,
and a test asserts the `reward` and `environment` sections of `config.json` still match
`main` byte for byte.

## 4. The original dynamic-obstacle behaviour

The brief calls these "fixed/deterministic trajectories". Verified, they are not: each
obstacle draws a **uniform random heading at every reset**, travels in a straight line at
constant speed, and **reflects elastically** off the world boundary — a discontinuous
heading flip. Obstacles pass through the static rectangles and through each other.

`DeterministicMotion` is a verbatim lift of that logic, and a test replays the original file
out of git and asserts bit-identical observations, rewards and termination flags across
obstacle counts {0, 3, 6, 10} and six seeds.

## 5. The randomized dynamic-obstacle model

Per obstacle, state `(x, y, θ, v, ω)`:

```
ω ← (1 − λ·dt)·ω + σ·√dt·N(0,1)
ω ← clip(ω, −ω_max, +ω_max)
θ ← θ + ω·dt
x, y ← x + v·cos θ·dt,  y + v·sin θ·dt
```

Only the **direction** process is randomized. Speed is drawn once per episode and held;
obstacle count, radius and the arena are unchanged.

| parameter | value | meaning |
|---|---|---|
| `max_turn_rate` | 0.7 rad/s | turning radius 1.43 units |
| `angular_velocity_decay` | 0.8 | correlation time ≈ 12 steps |
| `angular_noise_sigma` | 0.35 | stationary `ω` std 0.28 |
| `boundary_margin` | 2.2 | exceeds the turning radius |
| `boundary_steer_gain` | 4.0 | |
| `steer_full_error` | π/4 | steer saturates past 45° of heading error |
| `steer_full_depth` | 0.35 | full authority 35% into the margin |

Boundary handling is a gradual inward steer applied **before** the turn-rate clip, never a
velocity reversal. A per-axis position projection remains as a backstop; because it is
per-axis, an obstacle reaching a wall keeps its tangential motion and slides, with its
heading untouched — a graze, not a bounce.

## 6. Why the randomization is smooth — measured

380,000 obstacle-steps, 25 seeds × {3, 6, 10} obstacles × 800 steps, obstacles spawned flush
against the walls:

| property | measured |
|---|---|
| max per-step heading change | **0.0700 rad = 4.01°**, exactly the `ω_max·dt` bound |
| `\|ω\|` | never exceeds `ω_max` |
| speed | constant to 1e-6 |
| position | always inside the valid region after every update |
| wall contact | **0.74%** of obstacle-steps, depth ≤ one step of travel |
| `ω` lag-1 autocorrelation | **0.98** — correlated, not white noise |
| arena coverage | **0.86** of cells, against ~1.0 for the original motion |

The autocorrelation is the load-bearing number. A bounded turn rate alone does not make
motion smooth *and* unpredictable — an uncorrelated process with tiny steps would satisfy
the bound and wander nowhere. See `trajectory_examples.md`.

## 7-10. Experiments

Full configurations in `EXPERIMENT_1.md`, `EXPERIMENT_2.md`, `EXPERIMENT_3.md` and
`ZERO_SHOT_ANALYSIS.md`. In summary:

| Arm | Trained on | SR | Output |
|---|---|---|---|
| `ppo_fixed` | original obstacles | off | `results/week4/experiment1/ppo/` |
| `ppo_sr_fixed` | original obstacles | on | `results/week4/experiment1/ppo_sr/` |
| `ppo_random` | randomized obstacles | off | `results/week4/experiment2/ppo_randomized/` |
| `ppo_sr_random` | randomized obstacles | on | `results/week4/experiment3/ppo_sr_randomized/` |

Zero-shot evaluates the two Experiment 1 checkpoints on randomized obstacles **without
retraining**, and is reported separately throughout — Question A (training under
stochasticity) and Question B (transfer) are never merged.

## 11. Training configuration

PPO is stock Stable-Baselines3 2.9.0, identical across all four arms: `n_envs 8`,
`n_steps 2048`, `batch_size 64`, `n_epochs 10`, `gamma 0.99`, `ent_coef 0.01`, `lr 3e-4`,
`net_arch [256, 256]`, seeds 0/1/2.

Two things differ on the SR arms, and both are Sibling Rivalry:

1. The critic is anti-goal-conditioned — `V(s, g, ḡ)` reads 36 inputs. **The actor reads
   the same 34 inputs in both arms and has identical widths**, so the comparison is not
   confounded with network capacity. Conditioning the *policy* is impossible in principle:
   the anti-goal is a property of a sibling pair and does not exist at action time.
2. Rollouts are paired complete episodes filtered by the acceptance rule (`τ_f` always
   retained; `τ_c` only when `ρ < ε` or `d_c < δ`).

`δ = 0.6`, the environment's own `target_radius`, so SR's success set and the environment's
termination test cannot disagree. **`ε` is calibrated, not guessed** — see §19.

Neither arm was tuned separately, and `num_timesteps` counts every environment step
including those in discarded episodes, so both arms spend the same environment budget.

## 12. Evaluation methodology

200 episodes per arm per seed, deterministic policy, seeds `1_000_000 … 1_000_199` —
**disjoint from every seed training consumes**, so Experiments 2 and 3 are tested on
trajectory realizations never trained on. Every arm sees the identical block, so the
comparison is paired episode by episode and scenario difficulty cancels.

Recorded per episode: success, collision **split into wall / static / dynamic**, timeout,
episode length, path length, return, final distance, SPL, terminal position, collision
position, and the start–goal distance band.

SPL uses A\* over the static map as its denominator, not the straight line. The straight
line would overstate SPL by ~17.5% on average (§18) and would flatter a policy most on
exactly the pairs where navigation was hardest. This is an evaluation-time metric only; it
never enters a reward or an observation, so the brief's exclusion of geodesic distance in
the *learning* signal is respected.

## 13-14. Success-rate and collision-rate results

**PENDING — full training not run.** The result table is generated by
`python -m experiments.week4 table` into `results/week4/result_table.json`, in this shape,
with the two zero-shot rows explicitly labelled:

| Model | Trained on | Evaluated on | Zero-shot | Success | Collision |
|---|---|---|---|---|---|
| PPO | fixed | fixed | no | — | — |
| PPO + SR | fixed | fixed | no | — | — |
| PPO | randomized | randomized | no | — | — |
| PPO + SR | randomized | randomized | no | — | — |
| PPO (fixed-trained) | fixed | randomized | **yes** | — | — |
| PPO + SR (fixed-trained) | fixed | randomized | **yes** | — | — |

The **primary metric is collision rate**, with success rate the other major one. Both are
also broken down by start–goal distance band, which the stratified sampler makes possible:
"does SR help on the long crossings specifically?" is answerable rather than averaged into
the middle.

## 15. Learning curves

**PENDING.** `results/week4/plots/learning_curves.png` — episode return, episode length and
success rate against environment steps, mean ± 1 std across three seeds, identical axes,
PPO overlaid with PPO+SR.

Two deliberate departures from the existing plotting code. It hardcodes `ylim(-11, -2)` on
every curve, which would silently clip SR's anti-goal-penalised returns off the bottom of
its own plot; nothing here sets a fixed y-limit. And it plots single runs; every Week 4
curve carries a seed band, because a few-point difference is not interpretable without one.

Each run now writes its own Monitor CSV under `logs/week4/<arm>_seed<n>/`. The repository's
existing behaviour appended every run to one shared `logs/monitor.monitor.csv`, from which
no per-arm curve could be recovered.

## 16. Terminal-state distributions

**PENDING.** `results/week4/terminal_heatmaps/`. Spatial density over the terminal states of
**all** evaluation episodes — successes, collisions and timeouts alike — for PPO and PPO+SR
on the same map, the same evaluation seeds and a **shared colour normalisation**.

This is the plot that speaks directly to Sibling Rivalry's motivation: if SR is escaping a
local optimum, its terminal states should be less concentrated away from the goal than
PPO's. It is generated separately from, and must not be read as, the collision heatmap.

## 17. Collision heatmaps

**PENDING.** `results/week4/collision_heatmaps/`. Locations of failures only, split by
collision type. Where the environment is dangerous — a different question from where
policies end up.

## 18. Representative trajectories

**PENDING** for policy trajectories, which require trained models; the five required cases
(PPO succeeds, PPO+SR succeeds, PPO collides, PPO+SR avoids, both fail) are selected by a
stated programmatic rule over the paired evaluation episodes, not hand-picked, and the rule
is recorded alongside the figures.

**COMPLETE** for obstacle trajectories: `results/week4/trajectory_examples/` shows the
original deterministic path beside three randomized ones with their `ω` traces, and
`results/week4/start_goal_coverage/` shows the region and distance coverage.

## 19. Statistical analysis

Two levels, because they answer different things.

**Across seeds** — mean ± std over three seeds per arm. The episode-level spread is *not*
used as the error bar: evaluation episodes are identical across arms by construction, so
their spread describes scenario difficulty and says nothing about the policy.

**Within seed, paired** — `results/week4/paired_comparisons.json` reports, per seed, the
McNemar counts `only_a_success`, `only_b_success`, `both_success`, `neither_success`. With
matched episodes those discordant counts, not the difference of marginal rates, are what a
significance test on a binary outcome actually uses.

**SR diagnostics.** Logged every iteration under `sr/`: `rho_mean`, `rho_p50`, `rho_p90`,
`accept_rate_closer`, `n_retained`, `epsilon`. These exist because the most likely
uninteresting outcome is that SR silently degenerates into PPO — which happens when `ρ`
collapses or `ε` is high enough to accept everything. In the smoke run an **untrained**
policy gave `ρ` mean 1.11 and median 0.82, and a placeholder `ε = 2.0` accepted **94%** of
pairs, very nearly the degenerate case. **Any reported null for RQ3 must be accompanied by
an `accept_rate_closer` well away from both 0 and 1**, or it is a statement about the
hyperparameter rather than about Sibling Rivalry.

## 20. Limitations

1. **No maze, and therefore few static local optima.** This is the most important
   limitation. SR exists to escape traps in distance shaping, and an open arena has few. The
   project's own PointMaze results show SR tying Euclidean shaping on easy (0.995/0.995) and
   medium (0.968/0.955) maps and separating only on the decoy-corridor hard map
   (0.792/0.050). Week 4 deliberately keeps the arena and tests the narrower hypothesis that
   **moving** obstacles create *transient* traps. If they do not, the honest reading of a
   null RQ3 is "dynamic obstacles alone do not create the conditions SR addresses **in this
   arena**" — not "Sibling Rivalry does not work".

2. **The task distribution changed, so Week 4 Experiment 1 is a new baseline** and is not
   numerically comparable to the committed `e1…e4`. `dt` and stratified start/goal sampling
   were both necessary — at 4–6 step episodes neither SR nor the OU obstacle model can
   express itself at all — but they are changes, and setting `dt = 1.0` with
   `sampling = "uniform"` recovers the original environment bit-for-bit.

3. **The 3M-step budget is a known ceiling, not a converged result.** Measured on
   `ppo_fixed` seed 0, training-rollout success by 0.5M-step window:

   | steps | 0–0.5M | 0.5–1M | 1–1.5M | 1.5–2M | 2–2.5M | 2.5–3M |
   |---|---|---|---|---|---|---|
   | success | 0.003 | 0.174 | 0.290 | 0.335 | 0.338 | 0.363 |

   Learning decelerates sharply but does not stop: a linear fit over the final million
   steps still gives **+0.056 success per 1M steps**. Extrapolating, roughly 45–50% looks
   reachable at 6M. 3M was chosen to keep the twelve-run matrix inside about nine hours of
   wall clock, and the budget is **held constant across all twelve runs** — raising it
   mid-matrix would have made the completed runs incomparable to the rest, which costs more
   than the extra points are worth. Every arm is therefore under-trained by the same amount,
   so the PPO-vs-SR comparison remains fair; the absolute numbers are a floor.

   The first 0.5M steps are a transient local optimum worth recording: reward improves while
   success stays at 0, because the agent learns to end episodes quickly. A collision pays
   −10 and exhausting the 500-step limit pays −0.02 × 500 = −10, so nothing in the reward
   penalises dying early. PPO escapes it on its own by ~0.7M steps. This is not a wall-
   seeking behaviour — measured wall collisions are **zero** (§14).

4. **Three seeds is the minimum, not a comfortable number** for differences of a few points.

5. **Obstacles pass through static rectangles and through each other.** Inherited from the
   original environment and deliberately not changed, but it means "dynamic obstacle" here
   means a moving disc unconstrained by the map.

6. **Collision detection is not swept.** It is tested at the step endpoint only. `dt = 0.1`
   largely removes the tunnelling this allowed at `dt = 1.0`, but does not eliminate it.

7. **One arena, one obstacle count, one speed.** No sweep over obstacle density or speed
   under randomization.

8. **`ε` is calibrated under an untrained policy**, which is the regime where a bad value
   does the most damage, but `ρ` shifts as the policy sharpens. A fixed `ε` is the paper's
   design and ppo-nav's, and is kept — but the logged `accept_rate_closer` should be checked
   for drift toward 0 or 1 over training.

9. **Flat training departs from the baseline's curriculum**, chosen so the arms cannot
   receive different effective budgets through the curriculum's retry logic, at the cost of
   comparability with the existing runs.

## 21. Conclusions

**PENDING — full training has not been run, and no conclusion is drawn in advance.**

What can be stated now, from measurement:

* The repository did **not** contain a Sibling Rivalry implementation. One was built here,
  porting the validated selection rule and terminal reward from `ppo-nav/ppo_sr/`.
* The environment as committed **could not have answered** any of RQ1–RQ4: at 4–6 step
  episodes an OU process never leaves its initial transient, and SR's siblings never diverge
  enough for its acceptance rule to bind.
* With `dt = 0.1` and stratified start/goal pairs, episodes run ~70+ steps, mean separation
  is 6.87 units, no pair is closer than 4.0, and the greedy straight-line policy — formerly
  near-optimal — now **collides in ~90% of episodes**. Every sampled pair remains reachable
  (0 of 400 unreachable), with a detour ratio of mean 1.175 and max 2.42, comparable to
  `ppo-nav`'s medium and hard mazes. The task is now a navigation problem rather than a
  five-step dash.
* The randomized obstacle motion is verifiably stochastic **and** smooth: bounded turn rate
  at every step, lag-1 autocorrelation 0.98, no velocity reversals, and arena coverage
  comparable to the original motion.

RQ1–RQ4 remain open pending the full runs. The commands are in §7–10 of each experiment
document and in the handover summary.
