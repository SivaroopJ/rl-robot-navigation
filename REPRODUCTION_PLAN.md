# REPRODUCTION_PLAN.md

Faithful reproduction of the `rl-robot-navigation` (Micanovic) PPO + LiDAR continuous 2D
navigation results.

**Everything below was read out of the source code**, not from the README. Where the README
or `results/RESULTS.md` disagrees with the code, the code is treated as the source of truth
and the disagreement is flagged explicitly.

Repository state used: commit `9e307cf` ("Add scope and limitations section to evaluation
report"), branch `main`, clean tree.

**No file under `robot_env/`, `training/`, `evaluation/`, `experiments/`, `visualization/`,
or `config.json` is modified by this reproduction.** All reproduction output is written to
`reproduction_results/`.

---

## 0. Documentation vs. code — discrepancies found during inspection

These were found by reading the implementation and must be understood before interpreting
any result. None of them are "fixed"; the reproduction runs the code exactly as written.

| # | Claim | Where | Reality in code |
|---|---|---|---|
| D1 | Curriculum stage 4 = 800 000 steps, total ≈ 3.5M | `README.md` v3.0 table | `training/train.py:DEFAULT_CURRICULUM_STAGES` stage 4 = **1 200 000**. Nominal total = **3 900 000**, which is what the README v4 text ("approximately 3 900 000") actually states. The v3 table in the README is stale. |
| D2 | "Action smoothing is controlled separately via `action_smoothing_scale`" | `README.md` | The action-smoothing term sits **inside** the `if self.use_reward_shaping:` block (`robot_nav_env.py`, `step()`). It is *scaled* separately but *gated* by reward shaping. The no-shaping variant therefore receives **no action-smoothing penalty at all**. |
| D3 | "The most likely cause [of no-shaping reaching 28.5%] is the action smoothing penalty" | `results/RESULTS.md`, E4 discussion | Cannot be true, by D2. The no-shaping agent's reward is exactly `-0.02` per step, `+10` on goal, `-10` on collision — no smoothing term is ever applied to it. |
| D4 | Reported E1/E2/E3/E4 numbers | `results/RESULTS.md` tables | **Do not match the committed `results/*/eval_*.json` files.** e.g. RESULTS.md says N=6 → 71.5%; `results/e1/eval_obs6_spd1.0.json` says 72.0%. N=10 → RESULTS.md 62.0% vs JSON 67.0%. The JSONs are newer than the prose. Both sets are recorded in `REPRODUCTION_RESULTS.md`; the **JSON files are used as the primary reference** since they are machine-written artifacts of an actual run. |
| D5 | Learning rate = `linear_schedule(3e-4)` | `README.md` v4, `RESULTS.md` | Only for **loaded** models. `training/agent.py:build_ppo_agent()` passes the raw float `config["training"]["learning_rate"]` (3e-4, **constant**). The linear schedule is applied only in `PPO.load(..., custom_objects={"learning_rate": linear_schedule(...)})`. So curriculum **stage 1 trains at constant 3e-4**; stages 2–5 use the linear schedule. |
| D6 | `ent_coef = 0.01` | `config.json:training` | Curriculum training overrides it to `config["curriculum"]["ent_coef"] = **0.05**` for every stage (fresh build and every load). The 0.01 value is only used by the standalone `train()` path, which the default `MODE="all"` never calls. |
| D7 | No saved checkpoints ship with the repo | — | `models/` and `logs/` are `.gitignore`d and absent. **The trained policy cannot be downloaded; it must be retrained from scratch.** This is the single largest source of reproduction variance. |
| D8 | End-of-stage models are saved as `..._{shaping_tag}.zip` | `train_curriculum()` print + `os.path.exists(stage_model_path + ".zip")` | **The `.zip` is never appended.** The path contains `spd1.0`, so `pathlib.Path(path).suffix == ".0_shaping"` and SB3's `save()` concludes the file is already extensioned. Confirmed on disk: `models/ppo_robot_nav_curriculum_stage1_obs0_spd1.0_shaping` exists with no extension, while the log line claims `...shaping.zip`. Harmless in practice **only because** `EvalCallback` writes `best_model.zip` (suffix `""`, so `.zip` *is* appended) and `train_curriculum()` prefers that file for both the next stage and the final alias. The end-of-stage fallback path is dead code. |

---

## 1. Environment

`robot_env/robot_nav_env.py :: RobotNavEnv`, a hand-written `gymnasium.Env`. No MuJoCo, no
Gazebo, no ROS. Pure NumPy + optional pygame for rendering only.

### World

| Property | Value | Source |
|---|---|---|
| World | 10.0 × 10.0 continuous square, origin at bottom-left | `config.json:environment.world_size = 10.0` |
| Agent representation | Disc, radius 0.3, **no orientation / no heading state** | `agent_radius = 0.3` |
| Target radius | 0.6 | `target_radius = 0.6` |
| Dynamic obstacle radius | 0.3 | `obstacle_radius = 0.3` |
| Agent max speed | 1.0 **world units per timestep** | `max_speed = 1.0` |
| Episode horizon | 500 steps (truncation) | `max_steps = 500` |

### Dynamics and timestep

There is **no explicit `dt`**. The integration in `step()` is:

```python
action           = np.clip(action, -1.0, 1.0)
self.agent_velocity = action * self.MAX_SPEED          # MAX_SPEED = 1.0
new_position     = self.agent_position + self.agent_velocity   # dt is implicitly 1.0
```

So the action is a **direct per-step displacement**. Consequences that must be preserved:

- The agent is **fully holonomic and memoryless in velocity**: commanded velocity is applied
  instantly, there is no mass, no acceleration limit, no inertia.
- The action space is a **square, not a disc**: the diagonal command `[1, 1]` moves the agent
  `sqrt(2) ≈ 1.414` units in one step. Speed is *not* norm-limited to `MAX_SPEED`.
- Because a step displaces the agent up to 1.414 units while the agent radius is 0.3 and the
  obstacle radius 0.3, collision detection is **discrete, not swept** — tunnelling through
  thin static obstacles and past dynamic obstacles is possible and is part of the original
  dynamics.
- Episodes are consequently very short. The mean reported episode length is ~4.5–5.7 steps,
  which is consistent: the goal is ≥2.0 away and the agent covers ~1–1.4 units/step.

After the move, the position is clamped: `np.clip(new_position, 0.3, 9.7)`.

### Start distribution

`_random_free_position(exclude=None, min_dist=1.0)`, rejection sampling, up to 200 attempts:

- `wall_margin = max(AGENT_RADIUS * 2, 0.5) = 0.6`
- candidate `~ Uniform([0.6, 9.4]^2)`
- rejected if within `min_dist` of any excluded position
- rejected if inside any static rectangle **inflated by `wall_margin`** on each axis
  (`|cx - x| < hw + 0.6 and |cy - y| < hh + 0.6`)
- fallback after 200 failures: the arena centre `(5.0, 5.0)`

Agent: `_random_free_position()` (no exclusions). Agent velocity initialised to `[0, 0]`.

### Goal distribution

`_random_free_position(exclude=[agent_position], min_dist=2.0)` — the goal is at least
**2.0** units from the agent's start.

### Static obstacles

Five fixed axis-aligned rectangles, `(center_x, center_y, half_width, half_height)`, from
`config.json:environment.static_obstacles`. **Identical every episode** — they are not
randomised:

| # | cx | cy | half-w | half-h |
|---|---|---|---|---|
| 1 | 2.0 | 2.0 | 0.5 | 1.5 |
| 2 | 7.0 | 3.0 | 1.0 | 0.5 |
| 3 | 4.5 | 7.0 | 0.5 | 1.0 |
| 4 | 2.0 | 7.5 | 1.5 | 0.5 |
| 5 | 7.5 | 7.0 | 0.5 | 1.5 |

### Dynamic (moving) obstacles

- Count `N` = `n_dynamic_obstacles`, overridden per experiment / curriculum stage.
- Shape: circle, radius 0.3.
- Placement at reset: `_random_free_position(exclude=[agent, target], min_dist=1.5)`.
  Note: obstacles are **not** de-conflicted against *each other* — two obstacles may spawn
  overlapping.
- Initial heading: `theta ~ Uniform[0, 2*pi)`, independent per obstacle.
- Velocity: `(cos theta, sin theta) * episode_speed`, **constant magnitude for the whole
  episode**; obstacles never accelerate or re-randomise heading.
- Motion: `obstacle_positions += obstacle_velocities` every step (again `dt = 1`).
- Wall handling `_bounce_obstacles()`: elastic reflection at `[0.3, 9.7]` per axis, with the
  position clamped back onto the boundary and that velocity component negated.
- Obstacles **pass straight through static rectangles and through each other**. Only world
  walls bounce them.
- Speed semantics: `obstacle_speed = 1.0` means obstacles move at exactly the agent's max
  axis speed.
- An `obstacle_speed_range` mode exists (per-episode uniform sampling, with observation
  normalisation anchored at the range midpoint). **The default curriculum and all four
  experiments use fixed speeds**, so this path is never exercised in the main reproduction.

### Collision handling

Checked every step, after both the agent and the obstacles have moved. Three sources:

1. **Wall** — evaluated on the *pre-clamp* position:
   `new_position[i] < 0.3 or new_position[i] > 9.7` for either axis.
2. **Dynamic obstacle** — `||agent - obstacle|| < AGENT_RADIUS + OBSTACLE_RADIUS = 0.6`.
3. **Static rectangle** — distance from the agent centre to the closest point on the
   rectangle `< AGENT_RADIUS = 0.3`.

Termination precedence in `step()`:

```python
goal_reached = current_distance < TARGET_RADIUS      # 0.6
if goal_reached:                       reward += +10.0 ; terminated = True
elif wall_collision or _check_collision(): reward += -10.0 ; terminated = True
truncated = step_count >= MAX_STEPS    # 500
```

**Reaching the goal takes precedence over a simultaneous collision.**

---

## 2. Observation

`spaces.Box(low=-1.0, high=1.0, shape=(34,), dtype=np.float32)`, built by
`_get_observation()` and **clipped to [-1, 1] as the final operation**.

Dimensionality: `4 + N_LIDAR_RAYS + N_OBSTACLE_VELOCITIES * 2 = 4 + 24 + 6 = **34**`.

| Idx | Feature | Formula | Normaliser | Effective range |
|---|---|---|---|---|
| 0–1 | `dx, dy` — goal relative position | `(target_position - agent_position) / WORLD_SIZE` | 10.0 | ~[-0.88, 0.88] |
| 2–3 | `vx, vy` — agent velocity | `agent_velocity / MAX_SPEED` | 1.0 | [-1, 1] |
| 4–27 | `l1..l24` — LiDAR | `_cast_lidar_rays() / LIDAR_RANGE` | 5.0 | [0, 1] |
| 28–33 | `ovx1,ovy1,ovx2,ovy2,ovx3,ovy3` | `obstacle_velocities[nearest] / self.obstacle_speed` | `obstacle_speed` | [-1, 1] |

Key properties that must **not** be changed:

- **Goal representation** is a *relative Cartesian offset in the world frame*, normalised by
  world size. It is **not** distance+bearing, and it is **not** agent-relative in
  orientation (the agent has no orientation).
- **Velocity representation** is the velocity *actually applied on the previous step*
  (`= clip(a_{t-1}) * 1.0`), i.e. the previous action. It is zero at reset.
- **Moving-obstacle information** is the velocity of the **3 nearest** dynamic obstacles by
  Euclidean distance to the agent (`np.argsort(distances)[:3]`). **Positions are not given**
  — only velocities; position information reaches the agent only through the LiDAR.
  Zero-padded when `N < 3`, and the whole block is zero when `N == 0` or
  `obstacle_speed == 0`.
- Obstacle velocities are normalised by `self.obstacle_speed`, the *configured* speed, so at
  a fixed speed each component is exactly `cos/sin` of the obstacle heading — magnitude
  information is normalised away, only direction survives.
- The final `np.clip(observation, -1.0, 1.0)` is what keeps the vector inside the declared
  Box.

### LiDAR

| Property | Value |
|---|---|
| Ray count | **24** (`n_lidar_rays`) |
| Ray angles | `np.linspace(0, 2*pi, 24, endpoint=False)` = 0°, 15°, 30°, …, 345° |
| Frame | **World frame, fixed.** Ray 0 always points along +x. There is no agent heading, so the LiDAR never rotates with the agent. |
| Origin | Agent centre (rays are *not* offset by the agent radius) |
| Max range | **5.0** (`lidar_range`) |
| Normalisation | `/ 5.0` → [0, 1] |
| Default / no-hit value | `LIDAR_RANGE` (→ 1.0 normalised) |

Each ray takes the **minimum** over three intersection tests:

1. `_ray_vs_walls` — parametric intersection with the four world boundaries, computed
   against the **agent-radius-inset** boundary (`AGENT_RADIUS` and `WORLD_SIZE -
   AGENT_RADIUS`), smallest positive `t`.
2. `_ray_vs_rect` — slab / AABB method against each of the 5 static rectangles. Valid when
   `t_enter <= t_exit and t_exit > 0`; uses `t_enter` if positive, else `t_exit` (i.e. if the
   ray origin is inside the rectangle it returns the exit distance). Clipped to
   `[0, LIDAR_RANGE]`.
3. `_ray_vs_circle` — analytic quadratic against each dynamic obstacle circle (radius 0.3),
   taking the nearest positive root. Clipped to `[0, LIDAR_RANGE]`.

Cost: `24 * (2 + 5 + N)` intersection tests per step, in a pure-Python double loop. This is
the throughput bottleneck (see §7).

---

## 3. Action

**Verified directly from the code — the holonomic continuous `[vx, vy]` assumption is correct.**

| Property | Value |
|---|---|
| Space | `spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)` |
| Dimensionality | 2 |
| Bounds | `[-1, 1]` per component (a **square**, not a disc) |
| Meaning | `a = [vx, vy]`, a normalised world-frame velocity command. `a[0]` = +x velocity, `a[1]` = +y velocity. No rotation, no differential drive, no acceleration command. |
| Conversion to movement | `agent_velocity = clip(a, -1, 1) * MAX_SPEED (= 1.0)`; `agent_position += agent_velocity`; then `clip(position, 0.3, 9.7)`. |
| Max displacement / step | 1.0 axis-aligned, **1.414 diagonal** (not norm-limited) |

The policy is a Gaussian over this 2-D space; SB3 clips the sampled action into the box, and
`step()` clips again defensively.

---

## 4. PPO

`stable_baselines3.PPO` via `training/agent.py:build_ppo_agent()`. Nothing is subclassed and
no custom feature extractor is used.

### Architecture

| Component | Value |
|---|---|
| Policy class | `"MlpPolicy"` (`ActorCriticPolicy`) |
| `net_arch` | `[256, 256]` |
| Actor (pi) | 34 → 256 → 256 → 2 (mean) |
| Critic (vf) | 34 → 256 → 256 → 1 |
| Shared trunk | **None.** In SB3 ≥ 1.8 a flat `net_arch` list gives actor and critic *separate* MLPs of that shape. |
| Activation | `nn.Tanh` (SB3 `ActorCriticPolicy` default — not set in config) |
| Action distribution | `DiagGaussianDistribution`, **state-independent** `log_std` parameter, initialised to 0 |
| Feature extractor | `FlattenExtractor` (identity for a Box obs) |
| Optimiser | Adam, `eps = 1e-5` (SB3 default) |

### Hyperparameters

Explicit in `config.json:training` and passed by `build_ppo_agent`:

| Parameter | Value | Note |
|---|---|---|
| `learning_rate` | `3e-4` | **Constant float** for a freshly built agent (see D5) |
| `n_steps` | `2048` | per environment → rollout buffer = `2048 * 8 = 16 384` transitions |
| `batch_size` | `64` | → 256 minibatches per epoch |
| `n_epochs` | `10` | |
| `gamma` | `0.99` | |
| `ent_coef` | `0.01` (config) / **`0.05` (curriculum override)** | see D6 |
| `policy_kwargs` | `{"net_arch": [256, 256]}` | |
| `seed` | `42` | `config.json:seed`, passed as `PPO(seed=42)` |
| `tensorboard_log` | `logs/` | |
| `verbose` | 1 | |

**Not set in the config — these are SB3 defaults and are part of the specification:**

| Parameter | SB3 default in use |
|---|---|
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `clip_range_vf` | `None` (value clipping disabled) |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `normalize_advantage` | `True` |
| `use_sde` | `False` |
| `target_kl` | `None` (no early stopping) |
| `device` | `"auto"` → CPU here |

### Learning-rate schedule

- **Fresh agent (curriculum stage 1 only):** constant `3e-4`.
- **Every `PPO.load()` (curriculum stages 2–5, and every retry):**
  `custom_objects={"learning_rate": linear_schedule(3e-4)}` where
  `linear_schedule(v)(progress_remaining) = progress_remaining * v`, decaying `3e-4 → 0`.

### Parallelism

| Property | Value |
|---|---|
| `n_envs` | **8** |
| Vec wrapper | `SubprocVecEnv` (true multiprocessing) → `VecMonitor(filename="logs/monitor", info_keywords=("is_success",))` |
| Eval env | `DummyVecEnv` with a **single** env → `VecMonitor(info_keywords=("is_success",))` |

### Seed handling

1. `train_curriculum()` sets `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`.
2. `PPO(seed=42)` → SB3 additionally seeds its action space and calls `env.seed(42)`.
3. `create_parallel_envs` builds worker `i` with `make_env(..., rank=i, seed=42)`, and the
   factory calls `env.reset(seed=42 + i)` **once, at construction**. Worker `i` therefore
   starts from stream `42 + i`; every *subsequent* `reset()` inside an episode loop draws
   from that worker's advancing `np_random` stream (no per-episode reseeding).
4. **Evaluation is fully seeded and deterministic**: `evaluate()` calls
   `eval_env.reset(seed=episode_index)` for `episode_index = 0 … 199`. The 200 evaluation
   scenarios (start, goal, obstacle placements, obstacle headings) are therefore **byte-identical
   between the original run and this reproduction** for a given `(N, speed)`. Only the policy
   differs. This is a very favourable property for reproduction.

Residual nondeterminism in *training*: `SubprocVecEnv` process scheduling and float
non-associativity in multithreaded PyTorch CPU kernels mean training is **not** bitwise
reproducible even at a fixed seed.

---

## 5. Reward

Read verbatim from `RobotNavEnv.step()`. Coefficients from `config.json:reward`.

```
reward = -0.02                                            # step_penalty, ALWAYS applied

if use_reward_shaping:                                    # <-- gates all three terms below
    reward += 0.3 * (previous_distance - current_distance)      # progress_scale

    if n_dynamic_obstacles > 0:                                 # proximity penalty
        clearance = min_i ||obstacle_i - agent|| - 0.3 - 0.3
        if clearance < 0.8:                                     # danger_zone_radius
            t = max(0.0, 1.0 - clearance / 0.8)
            reward -= 0.3 * t**2                                # proximity_penalty_scale

    action_delta = a_t - a_{t-1}                                # action smoothing
    reward -= 0.16 * (action_delta . action_delta)              # action_smoothing_scale

# terminal terms, mutually exclusive, goal wins ties
if current_distance < 0.6:   reward += +10.0 ; terminated = True     # goal
elif wall_collision or collision: reward += -10.0 ; terminated = True # collision
```

| Term | Coefficient | Config key | Gated by shaping? |
|---|---|---|---|
| Goal | `+10.0` | `reward.goal` | no |
| Collision | `-10.0` | `reward.collision` | no |
| Step penalty | `-0.02` / step | `reward.step_penalty` | **no** |
| Progress | `+0.3 * Δd` | `reward.progress_scale` | yes |
| Proximity | `-0.3 * t²`, `t = max(0, 1 - clearance/0.8)` | `reward.proximity_penalty_scale`, `reward.danger_zone_radius` | yes |
| Action smoothing | `-0.16 * ‖a_t − a_{t−1}‖²` | `reward.action_smoothing_scale` | **yes** (see D2) |

Details:

- `previous_distance` is the distance measured *at the end of the previous step*; `a_{t-1}`
  is initialised to `[0, 0]` at reset, so the first step is penalised for its full magnitude.
- Progress is measured **after** the obstacles have moved, but the target is static, so this
  does not matter for the progress term.
- The proximity `clearance` is *surface-to-surface* (both radii subtracted) and can go
  negative on the collision step, in which case `t > 1` and the penalty exceeds `0.3`.
- Proximity considers **only dynamic obstacles** — never static rectangles or walls.
- **The no-shaping variant's reward is exactly**: `-0.02` per step, `+10` goal, `-10`
  collision. Nothing else. This is the E4 ablation arm.

---

## 6. Curriculum

`training/train.py :: DEFAULT_CURRICULUM_STAGES`, driven by `train_curriculum()`.
`main.py` with `MODE = "all"` calls `train_curriculum(use_reward_shaping=True, log_prefix="shaping")`.

### Stages (exact, from code)

| Stage | `n_obstacles` | `speed` | `timesteps` | `threshold` | `max_retries` |
|---|---|---|---|---|---|
| 1 | **0** | 1.0 | 500 000 | 0.70 | **2** |
| 2 | **3** | 0.5 | 600 000 | 0.55 | 1 |
| 3 | **3** | 1.0 | 600 000 | 0.50 | 1 |
| 4 | **6** | 1.0 | **1 200 000** | 0.45 | 1 |
| 5 | **10** | 1.0 | 1 000 000 | `None` | 0 |

**Nominal total = 3 900 000 timesteps** (matches the README's "approximately 3 900 000";
contradicts the stale v3 README table that lists stage 4 as 800 000 — see D1).

The retry mechanism means the *actual* budget is data-dependent. A retry re-runs the
**full** stage budget, so the worst case is
`500k*3 + 600k*2 + 600k*2 + 1.2M*2 + 1M = 7 300 000` timesteps for a single curriculum.

### Environment configuration per stage

Each stage rebuilds both environments from scratch:

```python
training_env = create_parallel_envs(config, CONFIG_PATH, n_obstacles, speed, None, use_reward_shaping)
eval_env     = create_eval_env(CONFIG_PATH, n_obstacles, eval_speed, use_reward_shaping)   # eval_speed = speed
```

Everything else (world, static obstacles, LiDAR, reward coefficients, horizon) is stage-invariant
— only `n_dynamic_obstacles` and `obstacle_speed` change.

### Checkpoint transitions

1. **Stage 1** builds a fresh agent (`build_ppo_agent`), `ent_coef = 0.05`, constant LR 3e-4.
2. **Stages 2–5** call `PPO.load(pretrained_path, env=training_env,
   custom_objects={"ent_coef": 0.05, "learning_rate": linear_schedule(3e-4)})`.
   Only the *weights* carry over; the optimiser state, rollout buffer and LR schedule are
   rebuilt.
3. `agent.learn(..., reset_num_timesteps=False)` — the global timestep counter is preserved
   across stages.
4. During the stage, `EvalCallback` writes the best-so-far model to
   `models/best_obs{N}_spd{speed}/best_model.zip`.
5. At stage end the agent is also saved to
   `models/ppo_robot_nav_curriculum_stage{k}_obs{N}_spd{speed}_{shaping_tag}.zip` with a
   `_meta.json` sidecar.
6. **The next stage starts from `best_model.zip` if it exists**, else from the end-of-stage
   model. In practice `best_model.zip` essentially always exists.
7. After stage 5, the chosen model is copied to the canonical alias
   `models/ppo_robot_nav_curriculum_shaping.zip` (or `..._no_shaping.zip`).

### Threshold / retry logic

After a stage with a non-`None` threshold, `_read_success_rate()` reads
`logs/eval_{prefix}_obs{N}_spd{speed}_attempt{a}/evaluations.npz` and takes
`successes[-1].mean()` — the success rate of the **most recent** evaluation, over 20 episodes.

- If `rate >= threshold` → advance.
- Else if attempts remain → **reload `best_model.zip`** and re-run the *entire* stage budget.
- Else → print a warning and advance anyway.

Because the threshold is judged on only **20 episodes** (`config.curriculum.n_eval_episodes`),
it is a high-variance gate: a 45% threshold on 20 episodes has a standard error of ~11pp.
Whether a retry fires is therefore itself close to a coin flip near the threshold, and is a
genuine source of run-to-run divergence in total training budget.

### Callbacks (both `train()` and `train_curriculum()`)

| Callback | Setting |
|---|---|
| `EvalCallback` | `eval_freq=10_000`, `n_eval_episodes=20` (curriculum) / `200` (`train()`), `deterministic=True`, `render=False` |
| `CheckpointCallback` | `save_freq=50_000`, `name_prefix="ppo_robot_nav"` |

Note that SB3 counts `eval_freq` / `save_freq` in **callback calls, i.e. vectorised steps**,
so with `n_envs = 8` an evaluation happens every **80 000 environment steps** and a
checkpoint every **400 000 environment steps**.

---

## 7. Evaluation protocol

`evaluation/evaluate.py :: evaluate()`.

| Property | Value |
|---|---|
| Episodes | **200** (`config.evaluation.n_eval_episodes`, passed explicitly by every experiment) |
| Policy | **Deterministic** (`config.evaluation.deterministic = true` → `model.predict(obs, deterministic=True)`) |
| Environment | A **fresh, single, unwrapped `RobotNavEnv`** — no `VecNormalize`, no monitor |
| Seeding | `eval_env.reset(seed=episode_index)` for `episode_index = 0 … 199` |
| Obstacle count / speed | Passed per experiment; overrides config |
| Randomisation | Start, goal, obstacle positions and headings all re-randomised per episode, but **fully determined by the episode seed** |
| `use_reward_shaping` | Left at the constructor default `True` (does not affect success/collision metrics) |

Outcome classification (in this order):

| Outcome | Condition |
|---|---|
| `SUCCESS` | `info["success"]` — i.e. `distance_to_target < 0.6` on the terminal step |
| `TRUNCATED` | else if `truncated` — i.e. the episode hit 500 steps |
| `COLLISION` | else — terminated without reaching the goal |

Metrics written to `results/{experiment}/eval_obs{N}_spd{speed}[_{label}].json`:
`success_rate`, `avg_episode_length`, `avg_collision_count`, `n_success`, `n_collision`,
`n_truncated`, `n_episodes`, and a per-episode list. Since an episode ends on its first
collision, `avg_collision_count == n_collision / n_episodes` == the collision rate.

---

## 8. Experiments to reproduce

`main.py` with `MODE = "all"` sets `train_models = False`, so **all four experiments evaluate
the single curriculum-trained model**; no per-experiment training happens, with one exception
(E4's no-shaping arm).

### E1 — obstacle density
Model: `ppo_robot_nav_curriculum_shaping`. Speed fixed 1.0. `N ∈ {3, 6, 10}`. 200 episodes each.

### E2 — obstacle speed
Same model. `N` fixed 6. `speed ∈ {0.5, 1.0, 1.5}`. 200 episodes each.

### E3 — generalization to unseen configurations
Same model, **no retraining**. Seven configurations, exactly as listed in
`experiments/experiment_e3.py:UNSEEN_CONFIGS`:

| N | speed |
|---|---|
| 8 | 1.2 |
| 5 | 0.8 |
| 10 | 1.5 |
| 9 | 1.3 |
| 7 | 1 |
| 12 | 2 |
| 15 | 0.3 |

### E4 — reward-shaping ablation
- Arm A: the curriculum shaping model, evaluated at `N=6, speed=1.0`, label `shaping`.
- Arm B: **a second full 5-stage curriculum trained from scratch** with
  `use_reward_shaping=False` and `log_prefix="no_shaping"`, then evaluated at
  `N=6, speed=1.0`, label `no_shaping`.

Everything else is held fixed between the arms — same env, same PPO config, same curriculum,
same 3.9M budget, same evaluation seeds. The only variable is the `use_reward_shaping` flag,
which (per D2) toggles **progress + proximity + action smoothing** together.

**Total nominal training for the full reproduction = 2 × 3 900 000 = 7 800 000 timesteps.**

---

## 9. Software environment

The repository pins only lower bounds (`requirements.txt`), and specifies "Python 3.10+".

| Package | Repo requirement | Installed for this reproduction |
|---|---|---|
| Python | `>= 3.10` | **3.11.15** (via `uv`, project-local `.venv`) |
| `torch` | `>= 2.0.0` | **2.13.0+cpu** |
| `gymnasium` | `>= 0.29.0` | **1.3.0** |
| `stable-baselines3[extra]` | `>= 2.3.0` | **2.9.0** |
| `numpy` | `>= 1.24.0` | **2.4.6** |
| `matplotlib` | `>= 3.7.0` | 3.11.1 |
| `pygame` | `>= 2.5.0` | 2.6.1 |
| `tensorboard` | `>= 2.14.0` | 2.21.0 |

Notes and deviations:

- **CPU-only PyTorch wheel** was installed (`--index-url .../whl/cpu`). The two GPUs on this
  host are Tesla K80s (compute capability 3.7), which modern PyTorch no longer supports, and
  the user specified CPU. This is a packaging choice only; it does not change any numerics
  relative to the CPU path of a CUDA-enabled build.
- The system Python has no `ensurepip`, so `python3 -m venv` fails on this host. The venv was
  created with `uv` instead. No system-wide or `sudo` installs were performed.
- `requirements.txt` has open upper bounds, so `pip` today resolves to versions roughly two
  years newer than the ones the original results were produced with. This is a genuine and
  unavoidable reproduction risk (SB3 2.3 → 2.9, gymnasium 0.29 → 1.3, numpy 1.x → 2.x) and is
  recorded as a candidate explanation for any discrepancy in `REPRODUCTION_RESULTS.md`.

---

## 10. Reproduction procedure and output layout

The repository source is **not modified**. Two mechanical points had to be handled:

1. `evaluate()` hard-codes its output directory to `<repo>/results/{experiment}/`, which is
   where the *original* committed results live. To satisfy "do not overwrite the original
   repository results", the original `results/` tree is snapshotted to
   `reproduction_results/original_reported/` before any run; after each evaluation the newly
   written JSONs and figures are copied into `reproduction_results/evaluations/` and
   `reproduction_results/plots/`, and `results/` is restored with `git checkout -- results/`.
2. `config.json:paths` points training output at `models/` and `logs/`, both `.gitignore`d
   and both absent from the repository (D7), so training cannot overwrite anything original.
   Those directories are copied into `reproduction_results/checkpoints/` and
   `reproduction_results/logs/` afterwards.

Final layout:

```
rl-robot-navigation/
├── REPRODUCTION_PLAN.md          # this document
├── REPRODUCTION_RESULTS.md       # reported vs reproduced comparison
└── reproduction_results/
    ├── configs/                  # exact config.json + frozen pip list + curriculum spec
    ├── checkpoints/              # per-stage models, best_model, periodic checkpoints
    ├── logs/                     # tensorboard, monitor.csv, evaluations.npz, stdout/stderr
    ├── evaluations/              # reproduced eval_*.json, per experiment
    ├── plots/                    # reproduced trajectory + metric figures
    └── original_reported/        # untouched snapshot of the repo's committed results/
```

## 11. Smoke-test results (pre-training gate)

Run before any training, against `N=6, speed=1.0`:

| Check | Result |
|---|---|
| Observation space | `Box(-1.0, 1.0, (34,), float32)` — matches spec |
| Action space | `Box(-1.0, 1.0, (2,), float32)` — matches spec |
| Observation shape / dtype | `(34,) float32`, `observation_space.contains(obs) == True` |
| NaN / inf | none, over 200 episodes × all steps (`bad observations: 0`) |
| LiDAR block | 24 values, all within `[0, 1]`, varies with geometry |
| Obstacle-velocity block | 6 values; correctly zero-padded at `N=2`; all-zero at `N=0` |
| `reset(seed=k)` determinism | identical observations for repeated same-seed resets |
| Reward gating | shaping arm gives dense values (`-0.44, -0.21, -0.09, …`); no-shaping arm gives exactly `-0.02` per step and `-10.02` on the collision step — confirms D2 |
| Random-policy sanity | 200 episodes → 2 success / 198 collision / 0 timeout, mean length 4.1, mean return -10.77 (collision-dominated, as expected for a random walk) |
| Wall clamping | 50 consecutive `[1, 1]` commands at `N=0` leave the observation finite and in-box |

No shape mismatches, NaNs, invalid actions, broken LiDAR, collision bugs, reward bugs, or
reset problems were found. The environment is cleared for training.
