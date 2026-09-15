# Pursuit–Evasion: P0 repository audit and hunter design note

Status: **AUDIT + DESIGN ONLY.** No pursuit code written, no pursuit episode run.
Base commit `07d74ca` (branch `Week6`). Date 2026-09-12.

---

## 1. Frozen protagonist: exact implementation and configuration

| item | value / path |
|---|---|
| commit | `07d74ca` "Week 6 Continuation: tuned CLF-DR-CBF and Random-CLF-DR-CBF on M0" |
| tuned config | `results/week6_continuation/H2/tuned_frozen.json`, sha256 `c6893377…f705e98` |
| random config | `results/week6_continuation/R3/random_frozen.json`, sha256 `97e8cb92…378025c91` |
| tuned params | α = 0.8, λ_V = 1.0, r_W = 0.012, ε = 0.1, k_v = 0.10, N (k_scans) = 5 → τ_eff = 0.12, collapsed (εN = 0.5) |
| random recovery | K = 16 directions, H = 1 step, speed ladder {0.8, 1.0}, accept at m ≥ 0, else best over both speeds |
| policy class | `continuation/policy.py::TunableDRCBFPolicy` (subclass of frozen `dr_control/policy.py::DRCBFPolicy`; `predict()` inherited unchanged) |
| QP | `dr_control/drccp_controller.py::ClfCbfDrccpController` (frozen, SCS) |
| recovery | `continuation/random_recovery.py::RandomRecovery` (wraps `ctrl.generate_controller`; inert on feasible steps) |
| barrier source | `dr_control/estimated_cbf.py::EstimatedLidarBarrierSource`, k_scans = 5, tracker `LidarVelocityTracker`. **No T1 cap** (decision D2) |
| arm construction | `continuation/policy.py::build_arm("random")` + `attach_recovery` |
| RNG | `continuation/seeds.py::controller_rng(seed)` = `default_rng([0x52434246, seed])`, independent of `env.np_random` |
| LADDER | **not used**: `attach_recovery` returns `RandomRecovery`, never `RecoveryLadder` |

**Regression (P0 pass condition):** the frozen arm replayed 10 recorded final-evaluation seeds ×
2 motion conditions and reproduced **12/12 fields exactly on all 20 episodes** (outcome, steps,
success, collision, timeout, collision type, min-clearance, SPL, path length, n_infeasible,
n_recovery_events, recovery fraction). Mismatches = 0.

## 2. Canonical environment (verified against `config.json` + `exp6.make_env`)

10×10 world · 5 static rectangles · 6 dynamic circles · 24-ray LiDAR range 5.0 · agent radius 0.3 ·
obstacle radius 0.3 · dt 0.1 · max_steps 500 · single integrator · per-axis bound |u_x|,|u_y| ≤ 1 ·
obstacle speed 0.675. **No discrepancy** with the plan's §3 list.

Motion models (`robot_env/dynamic_obstacles.py::build_motion_model`):
`randomize=False` → `DeterministicMotion` (straight-line billiard); `randomize=True` →
`SmoothStochasticMotion` (OU heading noise σ 0.35, boundary steering). The plan asks for the
**stochastic** model, i.e. `randomize=True`, which is the `randomized` condition of the frozen work.

## 3. Mechanisms the pursuit layer must reuse

- **Dynamic obstacles are perceived, not read**: the protagonist's rows come from a rolling buffer
  of 5 LiDAR scans; each buffered scan contributes its nearest surface point
  (`h = range − r_robot`, `grad_h` unit, `dh_dt = −grad_h·v̂` from the LiDAR velocity tracker).
- **Fallback**: on a non-optimal QP solve the frozen controller commands **u = 0** and resets
  `prev_u` (`drccp_controller._solve`). That is the hunter's fallback too (no recovery).
- **Collision checks**: `RobotNavEnv._check_collision_type()` → "dynamic" (circle, d < 0.6),
  "static" (rectangle, d < 0.3); walls are checked pre-clamp in `step()`.
- **`cast_rays(origin, …)`** in `robot_env/lidar_core.py` takes an **arbitrary origin**, so the
  hunter can carry the same 24-ray sensor without touching frozen code.
- **`step()` order**: protagonist moves → obstacles move → outcomes evaluated. Obstacle motion reads
  only `np_random` and obstacle state, never the agent.

## 4. Hunter design

### 4.1 Nominal pursuit action
`u_nom = max_speed_h · unit(p_prot − p_hunter)` from the **shared pre-step state**. This is exactly
the frozen `nominal_action(p, gamma, max_v)` with `gamma = p_prot`, so no new nominal law is written.

### 4.2 Safety filter
The same `ClfCbfDrccpController`, with the CLF reference `gamma = p_prot` (pursuit replaces
goal-seeking; the CLF then drives the hunter toward the protagonist). Barrier rows come from the
hunter's **own** 24-ray LiDAR + its own `LidarVelocityTracker` and 5-scan buffer, i.e. an
independent `EstimatedLidarBarrierSource` instance. On infeasibility the hunter takes the frozen
**u = 0** fallback. **No LADDER, no random recovery** for the hunter in this experiment.

**Audited question from the plan (§2):** is the protagonist's Random recovery meaningful for the
hunter? Its acceptance rule (`m ≥ 0`, i.e. every kept barrier row non-negative) is defined purely on
the **safety** rows and is objective-agnostic, so it would transfer mechanically. But its *value*
was established against a goal-seeking objective, and for a pursuer "escape the infeasible set"
competes with "keep closing". It is therefore **deferred to a separate, documented hunter ablation**,
exactly as the plan requires.

### 4.3 What the protagonist sees of the hunter (known-state baseline)
The plan's §4 baseline gives the protagonist the hunter's **true position and velocity**. Implemented
as `HunterAugmentedSource`: a subclass of the frozen source whose `samples()` appends **one** barrier
row for the hunter, built with the frozen convention
`h = ‖p − p_h‖ − r_p − r_h`, `grad_h = (p − p_h)/‖·‖`, `dh_dt = −grad_h·v_h`,
from a `push_hunter(pos, vel)` channel fed by the harness. This mirrors the existing, already-audited
`OracleVelocitySource` pattern.

Consequences, stated explicitly:
- `DRCBFPolicy.predict()` is **still the frozen code path**; only the injected source object differs,
  which is the same mechanism `Phase7Policy` already uses.
- xi gains one row (6 total); the frozen controller still sorts by criticality and keeps
  `n_keep = 5`, so **εN_keep = 0.5 is unchanged and τ_eff stays 0.12**. The DR mathematics is
  untouched.
- Random recovery automatically scores candidates against the hunter row too, because it reads the
  QP's own `xi_kept`. **The recovery algorithm itself is not modified.**
- This is **privileged information by construction** and breaks the frozen information ledger. It is
  the plan's deliberate staged choice and every result carries the label **known-state**.
- **The hunter is therefore NOT rendered to the protagonist's LiDAR in the baseline**; the oracle row
  is its only channel. LiDAR-visible hunter detection is the separate later experiment (§4 "later
  extension"), which would then be the unprivileged arm.

### 4.4 Simultaneous update
Per step: (1) read one shared pre-step state; (2) compute the protagonist action; (3) compute the
hunter action **from that same pre-step state**; (4) apply both; (5) move dynamic obstacles;
(6) evaluate outcomes. Implementation: `PursuitEnv(RobotNavEnv)` computes nothing itself — the
harness hands it both actions, it calls the frozen `super().step(u_prot)` and integrates the hunter
with the pre-step-derived action in the same call, so neither agent can react to the other's updated
position.

### 4.5 Outcome semantics
- **Capture:** centre-to-centre `d(p_prot, p_hunter) ≤ d_capture = r_p + r_h + δ`, evaluated **after**
  both agents move.
- **Agent–agent contact** (`d < r_p + r_h = 0.6`) is recorded as its own physical flag. Because
  `d_capture > 0.6`, capture normally fires first; the flag exists so the two are never conflated.
- Every episode logs **all** flags (goal, capture, contact, static/dynamic/wall collision for each
  agent, timeout) so any outcome priority can be recomputed offline.

## 5. Files to be added (nothing frozen is touched)

```
continuation/pursuit/{__init__,config,env,hunter,source,harness}.py   new package
experiments/pursuit/{p0_regression,p1_sanity,p2_static,p3_main,report}.py
tests/test_pursuit.py
results/pursuit/… , MD_files/pursuit/…
```

## 6. Pre-declared choices that need no further debate

| item | value |
|---|---|
| motion model | stochastic (`randomize=True`) for P3; P1/P2 have no dynamic obstacles |
| episode cap | 500 steps (canonical) |
| hunter spawn | `_random_free_position` with min distance 1.5 from obstacles, drawn after the obstacles so the protagonist's own episode layout is unchanged; additionally ≥ `d_hunter_start` from the protagonist's start |
| hunter sensing | own 24-ray LiDAR from its own position, same range/rays; protagonist is **not** in the hunter's barrier rows (it is the target, not an obstacle) |
| protagonist controller | frozen Tuned + Random, verbatim, no LADDER |
| RNG | env seed stream separate from each controller's RNG; hunter has its own controller RNG only if a hunter recovery is ever added |
| seeds | fresh block, disjoint from every earlier block (10 000 000–10 999 999 are used by the navigation work) |

## 7. Decisions — RESOLVED by the user (2026-09-12), before any pursuit episode

| id | decision |
|---|---|
| capture tolerance | **δ = 0.05**, so `d_capture = 0.3 + 0.3 + 0.05 = 0.65 m`. Physical contact is `d < 0.60 m`, so there is a **5 cm capture margin before contact** and capture normally fires first |
| same-step priority | **own-collision → capture → goal → timeout**, identical in every condition. All flags are logged per episode, so any other convention is recomputable offline |
| hunter DR-CBF | the **tuned set**: α = 0.8, r_W = 0.012, ε = 0.1, k_v = 0.10 → τ_eff = 0.12. Both agents therefore carry a comparable safety filter |
| hunter speed | **\|u\| ≤ 1.0 for the primary experiment** (canonical bound, equal to the protagonist). A **separate \|u\| ≤ 1.25 speed sweep** is run as a clearly-labelled extra and is never pooled with the primary result |

**Consequence to watch, stated in advance:** at equal speed a pure-pursuit hunter starting behind
cannot close on a protagonist running a direct path, so the primary capture rate may be low and
captures will come mainly from interception and cornering. That is a finding, not a fault, and the
1.25 sweep exists so the experiment remains informative either way.

## 8. Pre-existing test failure, NOT caused by this work

`tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main` fails for all 4
obstacle counts (suite: 357 passed, 4 failed).

- **Cause:** the test replays the env from the **`main` branch ref**
  (`git show main:robot_env/robot_nav_env.py`). The pull on 2026-09-11 20:01 fast-forwarded `main`
  onto the merge of `DR_safe`, so that ref is now byte-identical to the working tree, and the test
  compares the current env against itself under a legacy config (52-d vs 34-d observation).
- **Intended baseline:** the pre-merge `main`, `6f1c2ed^`, which still holds the original 34-d env.
- **Not repaired here:** `tests/test_week4_env.py` is inside the Phase-7 frozen manifest, so
  re-pointing it would modify a frozen file and trip the frozen-file gate. Left for the user to
  decide.
- **Verified unrelated:** it reproduces on `DR_safe`, predates every pursuit file, and the frozen
  protagonist regression passes exactly (§1).
