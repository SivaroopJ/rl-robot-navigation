# Week 6 / Generalization Suite — map-suite specification

**Status: PROPOSAL. Nothing implemented, nothing run, no controller touched.**

A new experiment after Phase 7. It does **not** modify or rewrite any Phase 1–7 conclusion;
those stand as recorded. Neither controller is tuned, before or after seeing any result.

**Arms, fixed:**

| | configuration |
|---|---|
| **A. Original** | `DRCBFPolicy(use_planner=True)` — the frozen Phase-6 stack, exactly as at `1c47535` |
| **B. Updated** | `Phase7Policy(arm="T1_projection_cap", v_cap=0.96, use_planner=True)` + `RecoveryLadder(mode="ladder")`, exactly as committed at `91bb969` |

Both read `obs[0:2]` (goal offset), `obs[2:4]` (own velocity) and `obs[4:28]` (24 LiDAR
ranges) and nothing else. Both get ego pose and the static map for A*. Identical in every
other respect.

---

## §1 Inspection findings — what the environment can and cannot do

Measured, not assumed. Each finding is tagged with what it forces on the suite.

### 1.1 What varies through a config file alone (verified by smoke test)

| parameter | verified | notes |
|---|---|---|
| `world_size` + scaled `static_obstacles` | ✅ 20×20 ran | **7×7 FAILED**: `distance_bins` `[7.0,8.5]`/`[8.5,10.5]` exceed the 8.20 reachable maximum. Bins must be rescaled with the world. |
| `obstacle_radius` | ✅ 0.15 and 0.60 ran | flows to LiDAR (`circle_radius`) and collision (`AGENT_RADIUS+OBSTACLE_RADIUS`) consistently |
| `n_dynamic_obstacles` | ✅ 10 ran | |
| `static_obstacles` geometry | ✅ scaled and corridor layouts ran | axis-aligned `(cx, cy, hw, hh)` only |
| `max_steps` | ✅ 1000 ran | |
| `obstacle_speed` / `_range` | ✅ (already used in Phases 3–7) | |

**So the map-generation API is: emit a config JSON.** No environment code required for F2–F6.

### 1.2 Axis-alignment is structural — F1 as written is UNSUPPORTED

Rectangles are axis-aligned in **three independent places**:

- `lidar_core.cast_rays` — slab method over `(low, high)` bounds;
- `RobotNavEnv._check_collision_type` — `np.clip(pos, cx−hw, cx+hw)` AABB;
- `ShortestPathOracle._blocked` — `abs(x−cx) < hw + r`.

Rotation needs an angle in the rect tuple and new geometry in all three. **`robot_env/` and
`evaluation/` are frozen** (sha256 manifest, 60 files), so this cannot be done without breaking
the regression gate that every Phase 1–7 result depends on.

**Resolution:** F1 is re-specified as **F1′ — polygonal approximation of rotated obstacles**
(§3). A rotated rectangle is approximated by a staircase of small axis-aligned rectangles that
preserves area and orientation to a stated tolerance. This tests the *research* question —
do the controllers depend on axis-aligned structure? — without touching frozen code. The
approximation error is measured and reported. **F1′ is labelled an interface/generalization test,
explicitly NOT literal rotated-rectangle support** (decision 1): the environment still sees only
axis-aligned rectangles, and the staircase is a construction on top of that, not a new primitive.
Per-map staircase fidelity — area ratio, Hausdorff distance to the true rotated rectangle, and cell
count — is recorded in the manifest and reported beside every F1′ result.

### 1.3 The goal is fixed at reset — moving goals are UNSUPPORTED (family dropped)

`DRCBFPolicy.reset()` computes `self.goal = p + obs[0:2]·world_size` and plans A* **once**.
`predict()` never re-reads `obs[0:2]`. A moving goal is therefore invisible to both arms:
they navigate to wherever the goal was at t=0.

**Resolution:** classified **UNSUPPORTED**, and the family is **dropped from the suite**
(decision 4). This finding is retained as a documented boundary-of-applicability limitation —
established by code inspection, never presented as an experimental result. Fixing it would mean
re-reading the goal each step and changing the CLF reference: a controller change, which §24
forbids.

### 1.4 There is no replanning — F10 is an INTERFACE STRESS TEST

A* runs once at reset. `CarrotFollower.progress` is **monotone non-decreasing**, so the
reference cannot retreat along the path. When a wall appears across the planned route, the
planner cannot react and the carrot keeps pulling into it; only the CBF sees the wall, through
LiDAR, and only within 5 m.

**Resolution:** the replanning rule is fixed in advance as **no replanning, both arms** — the
honest reading of the committed controllers. F10 is evaluable without modification and tests
exactly what it should: can the safety layer save a route the planner no longer believes in.

### 1.5 The velocity tracker hard-codes obstacle radius 0.3 — F4 has a predicted mechanism

`r_nominal = 0.3` drives fixed-radius reconstruction, and `segment_is_compact` accepts a
segment only if its extent is `≤ 2·r_nominal + 0.15 = 0.75 m`:

| obstacle radius | true span | vs 0.75 m limit | consequence |
|---|---|---|---|
| 0.15 | 0.30 | accepted | centre pushed 0.3 along the ray but the true offset is 0.15 → ~0.15 m bias |
| 0.20 | 0.40 | accepted | ~0.10 m bias |
| **0.30** | 0.60 | accepted | canonical, unbiased |
| **0.45** | 0.90 | **REJECTED** | classified static → **velocity forced to 0** |
| **0.60** | 1.20 | **REJECTED** | classified static → **velocity forced to 0** |

So F4-large does not merely degrade the estimate — it **silently disables dynamic-obstacle
velocity entirely**, and `∂h/∂t` collapses to 0 for those rows. `SURFACE_MATCH_TOL = 0.15` binds
a point to a track only when `|‖q−c‖ − 0.3| < 0.15`, compounding it. This is a real limitation
of the committed controller and is **documented, not fixed** (§24).

**Both arms share this**, so the paired comparison stays fair; it is the *shared* boundary being
mapped.

### 1.6 The 6-slot observation limit does NOT affect this experiment

`_get_observation` truncates to `min(N_OBSTACLE_SLOTS=6, n_dynamic_obstacles)` nearest
obstacles. That block is `obs[28:52]` — which **neither DR-CBF arm reads**. Both work from the
24 LiDAR ranges, which see every obstacle in range regardless of count.

**F5 is therefore fully SUPPORTED for A vs B.** (It would confound a PPO comparison, which is
why PPO is out of scope here.) A test asserts the action is bit-identical under corruption of
`obs[28:52]` at every obstacle count.

### 1.7 `v_cap = 0.96` interacts with obstacle speed — the key F6 prediction

`v_cap = max_v − τ = 1.0 − 0.04 = 0.96` was derived under the canonical regime, where obstacles
move at 0.675 and no true closing rate can reach 0.96. Above that speed the cap begins clamping
**true** closing rates, not just over-estimates — the barrier is then told the obstacle is
approaching more slowly than it is. So the updated arm's T1 stage is predicted to turn from
protective to **actively harmful** somewhere between 0.675 and 1.2, and this is the one family
where B may lose to A on mechanism rather than noise.

`v_cap` is **not** changed to accommodate this (§24). Locating that crossover is the result.

**1.10 is the highest ordinary benchmark level** (decision 3). It is not described as
"unavoidable": it simply sits beyond the regime in which `v_cap = 0.96` was derived, and its
purpose is to test whether and where B's advantage reverses. Levels **1.20, 1.35 and 1.50** are
moved to the separate feasibility-stress category (X5–X7), where agent `max_speed = 1.0` makes the
obstacles strictly faster than the robot and success ceases to be the right question.
`v_cap = 0.96` stays fixed throughout.

### 1.8 New environment behaviour must live outside `robot_env/`

F10 (reactive walls) and F12 (structured trajectories) need behaviour the env does not have. `robot_env/` is frozen. All three are implemented as a **subclass in a new
module**, `generalization/scenario_env.py`, overriding only `step()`/`reset()` hooks — the same
pattern Phase 7 used for `Phase7Policy` and `ProjectionCappedSource`. The frozen manifest stays
green and the canonical env is byte-identical.

---

## §2 M0 — the canonical reference, and the regression gate

**M0 is the control, not a suite member.** `config.json` unchanged: world 10.0, agent radius 0.3,
obstacle radius 0.3, `max_speed` 1.0, `max_steps` 500, dt 0.1, 24 rays at range 5.0, the five
canonical rectangles, stratified start/goal, 6 dynamic obstacles at `obstacle_speed = 0.675`,
both motion models.

**Gate G0, must pass before any new map is generated:**

| arm | condition | must reproduce | source |
|---|---|---|---|
| A original | fixed | success 0.855, collision 0.140 | `results/phase6/exp6_fixed.json` |
| A original | randomized | success 0.725, collision 0.275 | `results/phase6/exp6_randomized.json` |
| B updated | fixed | success 0.925, collision 0.065 | `results/week5_phase7/final_comparison/final_fixed.json` |
| B updated | randomized | success 0.815, collision 0.180 | `.../final_randomized.json` |

200 paired episodes from `EVAL_SEED_BASE = 1_000_000`, **exact equality** — the pipeline is
deterministic given the seed, so anything other than an exact match means something moved.
Plus: frozen manifest PASS, 288 tests PASS. **If G0 fails, stop and diagnose; do not generate maps.**

---

## §3 The map families

Classification per §23: **S** = supported · **I** = interface stress test · **U** = unsupported.

### Level 1 — single-factor generalization

| id | family | levels | maps | class | assumption under test |
|---|---|---|---|---|---|
| **F1′** | static orientation (staircase-approximated) | 15°, 30°, 45°, 60°, 90°; 2 single-obstacle + 3 all-obstacle variants | 8 | **I** | axis-aligned static structure |
| **F2a** | static obstacle size | ×0.5, ×0.75, **×1.0**, ×1.25, ×1.5 | 4 (+M0) | **S** | free-space density |
| **F2b** | static obstacle count | 2, 3, **5**, 7, 9 rects | 4 (+M0) | **S** | clutter |
| **F3** | map size | 7×7, 10×10, 15×15, 20×20 | 3 (+M0) | **S** | fixed 10×10 geometry, LiDAR range vs world |
| **F4** | dynamic obstacle radius | 0.15, 0.20, **0.30**, 0.45, 0.60 | 4 (+M0) | **S** for ≤0.30, **I** for ≥0.45 (§1.5) | tracker `r_nominal = 0.3` |
| **F5** | dynamic obstacle count | 2, 4, **6**, 8, 10 | 4 (+M0) | **S** (§1.6) | multi-constraint load |
| **F6** | dynamic obstacle speed | 0.30, 0.45, **0.675**, 0.85, 0.96, 1.10 | 5 (+M0) | **S** to 0.96, **I** at 1.10 (§1.7) | `v_cap = 0.96` derivation |
| **F8** | start/goal configuration | open-space goal, goal behind a rect, goal in a corridor, goal in dynamic traffic, long diagonal | 5 | **S** | canonical start/goal geometry |
| **F9** | corridor width | 1.6, 1.2, 1.0, 0.85, 0.75 m | 5 | **S** | A*/CLF/CBF/recovery interaction |

**F9 is the major analysis family.** Robot diameter is 0.6 m, so clearance per side is
`(w − 0.6)/2`: 0.50, 0.30, 0.20, 0.125, 0.075 m. The 0.75 m case leaves 0.075 m — below the
`τ`-margin the DR constraint demands, so infeasibility should become the *normal* state, and this
is where Stage-4 recovery either earns its place or does not (**Q3**).

**F3 note:** LiDAR range is fixed at 5.0 m. In a 20×20 world that is a quarter of the arena
rather than half, so the horizon shrinks *relative to the map* — a genuine confound, reported
rather than corrected (correcting it would be tuning). Horizon scales as
`max_steps = ceil(500 · world_size / 10)` — 350 / 500 / 750 / 1000 — fixed now and applied
identically to both arms. `distance_bins` rescale by `world_size/10` (forced by §1.1).

**F2/F1′ rejection rule:** a generated map is rejected unless A* finds a path for **≥ 90 %** of
its 50 screening start/goal pairs on the inflated grid. Rejected maps are regenerated with the
next seed, not hand-edited. Rejections are logged.

### Level 2 — structured dynamics

| id | family | variants | maps | class |
|---|---|---|---|---|
| **F10** | reactive walls | behind, beside, blocking route, at next waypoint, just before arrival | 5 | **I** (§1.4) |
| **F12** | structured trajectories | crossing, head-on, parallel, converging, circling the goal, corridor traffic | 6 | **S** |

**F10 trigger rule, fixed in advance:** the robot entering a disc of radius 0.5 m centred at a
declared point activates a declared axis-aligned rectangle. Deterministic, identical for both
arms, defined in the map file. **No replanning for either arm** (§1.4). Recorded: trigger step,
distance to the wall at activation, whether the wall intersected the planned path.

**F11 (moving goal) is DROPPED from the suite** (decision 4). No moving-goal scenario code is
written and no screening or final budget is spent on it. The finding of §1.3 — that both arms fix
the goal and the A* path at `reset()` and never re-read `obs[0:2]`, so a moving goal is invisible
to them — is retained as a **documented boundary-of-applicability limitation** in the plan and the
report. It is a statement about the controllers' interface, established by code inspection, not an
experimental result, and it is never presented as one.

### Level 3 — combined stress (F7)

Run only after Level 1, on the levels Level 1 showed to be informative.

| id | combination |
|---|---|
| F7a | radius 0.45 + speed 0.96 |
| F7b | count 10 + speed 0.96 |
| F7c | radius 0.45 + count 10 |
| F7d | radius 0.45 + count 10 + speed 0.96 |

**I** — F7a/c/d inherit F4's tracker limitation. Purpose is compounding, not causal attribution.

### Level 4 — feasibility stress (§15) — reported separately, never in the benchmark

| id | scenario |
|---|---|
| X1 | obstacles fully block a 1.0 m corridor |
| X2 | an obstacle closes the only passage as the robot commits |
| X3 | a dynamic obstacle and a wall jointly remove every admissible control |
| X4 | the goal is temporarily enclosed |
| X5 | dynamic obstacle speed 1.20 — faster than the agent |
| X6 | dynamic obstacle speed 1.35 |
| X7 | dynamic obstacle speed 1.50 |

X5–X7 exist because decision 3 moved the 1.20–1.50 speed levels out of F6 and into this category;
they are not new maps invented to pad the suite. Agent `max_speed = 1.0` means the obstacles are
strictly faster than the robot, so success is not the right question for them.

Reported: DR infeasibility onset and duration, whether recovery activates, rung reached,
achieved CBC margin `m`, tier T0/T1/T2, collision, eventual recovery. **No success rate is
reported for X1–X7** — success is not the question. Arm A has no recovery, so its behaviour here
is the `u = 0` fallback by construction; that contrast is the point.

---

## §4 Seed manifest, selection rule and episode budget

Recomputed from the declared manifest after dropping F11. **The earlier "73" was wrong** — it did
not follow from the family tables — and is not preserved. No map was invented to restore it.

### 4.1 Configuration count, exact

| level | families | configurations |
|---|---|---|
| **1** single-factor | F1′ 8 · F2a 4 · F2b 4 · F3 3 · F4 4 · F5 4 · F6 5 · F8 5 · F9 5 | **42** |
| **2** structured dynamics | F10 5 · F12 6 | **11** |
| **3** combined stress | F7 4 | **4** |
| **4** feasibility stress | X1–X4 4 · X5–X7 3 | **7** |
| | **13 families** | **64 configurations** |

M0 is the control and is not counted. F11 is dropped and contributes nothing. X5–X7 are the
1.20/1.35/1.50 speed levels **moved here from F6 by decision 3**, not additions.

### 4.2 Motion models per family

The motion model is a second axis wherever it is free, and fixed wherever the family scripts the
trajectories. One **cell** = one configuration × one motion model.

| motion models | families | cells |
|---|---|---|
| 2 (deterministic + OU) | F1′, F2a, F2b, F3, F4, F5, F6, F8, F9, F10, F7, X5–X7 | 42+5+4+3 = 54 cfg → **108** |
| 1 (trajectories scripted) | F12, X1–X4 | 6+4 = 10 cfg → **10** |
| | **64 configurations** | **118 cells** |

### 4.3 The final-tier selection rule — one rule, no contradiction

The previous draft stated two incompatible rules. **This one supersedes both**, is fixed here
before any screening data exists, and is never revised on the basis of controller performance:

> **Every declared level of F1′, F2a, F2b, F3, F4, F5, F6, F8 and F9 runs the final tier.**
> **F10 and F12 run their first three declared variants**, by manifest index order, for budget
> control. **F7 runs all four combinations. X1–X7 run all seven scenarios.**
>
> **Episode allocation (budget lever (a), approved):** benchmark cells run **200** paired
> episodes; **X1–X7 run 50** paired episodes, being characterisation rather than benchmark.
>
> **AMENDMENT 1 (2026-09-09, recorded after screening and BEFORE the final tier):**
> **`F2b_n9` is excluded from the final tier**, on structural-feasibility grounds. It remains in
> the manifest and keeps its complete screening record. See §4.7.

Nothing is dropped for performing badly, and the F10/F12 truncation is by *declared index*, chosen
before any result exists.

Final-tier cells, after Amendment 1: F1′–F9 42 cfg **− F2b_n9** = 41 cfg → 82 · F10 3 cfg → 6 ·
F12 3 cfg → 3 · F7 4 cfg → 8 · X1–X4 4 cfg → 4 · X5–X7 3 cfg → 6 = **109 cells**, of which
**99 are benchmark cells at 200 episodes** and **10 are feasibility-stress cells (X1–X7) at 50**.

### 4.4 Seed blocks — disjoint by construction

| purpose | block | size |
|---|---|---|
| map generation | `GEN_SEED_BASE = 7_000_000` | layout sampling only; never an episode seed |
| screening | `SCREEN_SEED_BASE = 8_000_000 … +49` | 50 |
| final | `GEN_EVAL_SEED_BASE = 9_000_000 … +199` | 200 |

**`EVAL_SEED_BASE = 1_000_000` is used ONLY for the M0 regression gate (§2)** and is not extended.
No Week-6 decision touches a Phase 1–7 seed, and no Week-6 map is evaluated on the Phase-6 block.

### 4.5 Episode budget, exact

**Screening — 50 episodes per cell (decision 5, unchanged), both arms:**

```
118 cells x 50 episodes x 2 arms  =  11,800 episodes
```

**Final — split by tier, per the §4.3 allocation and Amendment 1:**

```
benchmark    99 cells x 200 episodes x 2 arms  =  39,600 episodes
stress       10 cells x  50 episodes x 2 arms  =   1,000 episodes   (X1-X7)
                                          final =  40,600 episodes
```

**Total: 52,400 episodes.** Screening stays at **11,800** — the entire 118-cell screening suite
was completed and F2b_n9 keeps its record there.

The 10 stress cells are X1–X4 (4 configurations × 1 scripted motion model) **and X5–X7**
(3 configurations × 2 motion models). Dropping them from 200 to 50 episodes saves
`10 × (200−50) × 2 = 3,000` episodes ≈ **4.71 h** single-process.

> **Correction recorded.** An earlier draft of this section claimed the saving was "~2.2 h". That
> was wrong: it counted only X1–X4's 4 cells and omitted X5–X7's 6. The figures above are the
> corrected ones, and the total falls from 56 200 to **53 200** episodes accordingly.

### 4.6 Runtime and storage

Measured from the Phase-7 final evaluation: arm B took 1 137 s / 200 episodes (fixed) and
1 374 s / 200 (randomized); arm A is cheaper, having no ladder. Blended estimate
**5.65 s/episode**.

| tier | episodes | single process | 8-way parallel |
|---|---|---|---|
| screening — **COMPLETE** | 11 800 | 17.8 h measured | **≈ 4 h actual wall-clock** |
| final — benchmark (200 ep) | 39 600 | 59.8 h | **7.5 h** |
| final — stress X1–X7 (50 ep) | 1 000 | 1.5 h | **0.2 h** |
| **total** | **52 400** | **79.2 h** | **9.9 h** |

Recomputed at the **measured 5.44 s/episode** from all 118 completed screening cells, replacing
the proposal's 5.65 and the smoke test's optimistic 4.68. Screening's actual wall-clock was ≈ 3 h
at 8-way plus ≈ 1 h for the F10 re-run.

**These runtimes are estimates from a prior measurement and are RECOMPUTED, not hard-coded.**
The runner derives them from its own measured per-episode rate on the first completed cells and
reports the revised projection before the final tier begins; if the measured rate differs from
5.65 s/episode the table above is superseded by the measured figure, and the episode counts —
which are exact — do not change.

**Storage ≈ 39 MB** of episode records (measured 0.75 KB per record from
`final_comparison/final_fixed.json`), plus ~0.25 MB of map configs. Results are written **per
cell** and never overwritten — the per-condition write pattern adopted after the Stage-4 OOM,
with only the fields the paired tests need retained.

**X1–X7 are characterisation, not benchmark.** They are excluded from every benchmark aggregate:
the suite-level success rate, the suite-level collision rate, the generalization scorecard of §5,
and every pooled paired test. No success rate is reported for them at all (§3, Level 4). They are
reported in their own section, on their own endpoints — infeasibility onset and duration, rung
reached, achieved CBC margin `m`, tier T0/T1/T2, collision, eventual recovery. Their reduced
50-episode budget is therefore not a loss of benchmark power: they were never in the benchmark.

### 4.7 AMENDMENT 1 — F2b_n9 excluded from the final tier (2026-09-09)

**Recorded after screening completed and BEFORE any final-tier episode was run.**

**Decision.** `F2b_n9` (the D+ level, 9 static rectangles) is **excluded from the benchmark and
the final tier**. It is **not replaced**. It remains in the manifest, in the screening suite and
in the screening report.

**Reason — structural feasibility, not controller performance.** Only **68 %** of its declared
screening start/goal pairs have an A\* path (34/50 per cell, in **both** motion models), against
the preregistered **≥ 90 %** criterion. The configuration does not satisfy the intended
navigable-map assumption.

**Why the preregistered remedy cannot be applied.** The rule repairs a rejected map by advancing
to the next declared generation seed. `F2b_n9` is a **fixed declared layout** — the canonical five
rectangles plus four declared `EXTRA_RECTS` — not a seed-generated one, so there is no seed to
advance. **No replacement map is invented**, because inventing one after seeing screening results
is precisely what the preregistration forbids.

**Why the exclusion cannot favour either arm.** The **same 16 seeds** fail for **both** arms in
**both** motion models; on a planner failure both arms degrade identically to direct-to-goal with
no waypoints; and on the 34 planned episodes the arms are near-identical (27 successes each,
deterministic). No controller advantage or disadvantage was created by the exclusion.

**What is retained.** F2b_n9's complete screening result stays in the report as a
**feasibility-boundary observation**, stating its 68 % A\* feasibility, the identical failed
seeds across arms, the identical behaviour where infeasible, the absence of any controller
advantage, and that it was excluded **before** final evaluation for failing the navigable-map
assumption.

**Derived counts changed by this amendment:**

| quantity | before | after |
|---|---|---|
| configurations | 64 | **64** (unchanged) |
| screening cells | 118 | **118** (unchanged) |
| final cells | 111 | **109** |
| final benchmark cells | 101 | **99** |
| final stress cells | 10 | **10** (unchanged) |
| final episodes | 41 400 | **40 600** |
| total episodes | 53 200 | **52 400** |

Implemented as `generalization.suite.FINAL_EXCLUDED`, honoured by `final_cells()` and **not** by
`screening_cells()`, and pinned by
`tests/test_generalization_maps.py::test_amendment1_excludes_f2b_n9_from_final_but_keeps_it_in_screening`.

## §5 Metrics, statistics, failure taxonomy

**Primary:** success rate, total collision rate.
**Secondary:** dynamic/static/wall collision split, timeout, minimum true clearance (mean and
worst), SPL, path length, mean `‖u − u_nom‖`, infeasibility rate, fraction of steps recovered,
rung distribution (R0/R1/R2/R2_lp), CBC margin `m` (median, min), tier T0/T1/T2 shares, solver
failures, mean step time.
**F10 additionally:** trigger step, distance to wall at activation, path-intersection flag.

Arm A has no recovery, so its rung/tier columns are `R0` and the tier of the frozen action —
reported as such, never blank.

**Statistics — Phase-6 methodology, unchanged:** exact McNemar for binary endpoints, paired
percentile bootstrap (10 000 resamples over episode pairs) for continuous ones, two-sided 95 % CI
on every paired difference **including non-significant ones**, α = 0.05 fixed now. Step-level
rates use an **episode-level cluster bootstrap** (steps within an episode are not independent).
Per-family results are primary; the aggregate is reported alongside and never replaces them.
**X1–X7 are excluded from every aggregate** — suite success rate, suite collision rate, the
scorecard below, and all pooled tests — and are reported separately on their own endpoints.

**Failure taxonomy**, assigned per episode: static collision · dynamic collision · wall collision
· timeout · DR infeasibility at termination · recovery/T2 event · planner failure · estimator
failure (no confirmed track within LiDAR range while an obstacle was present) ·
environment-trigger failure (F10).

**Generalization scorecard**, the actual deliverable:

```
degradation(arm, family) = performance(arm, M0) − performance(arm, family)
```

reported per family for both arms, with the **difference of degradations** and its CI. The
question is not "is B better" — Phase 7 answered that on M0 — but **does B degrade more slowly
than A as the environment departs from canonical (Q4)**.

---

## §6 Fairness invariants, asserted at runtime

Every episode asserts, for both arms: identical map file, identical seed, identical start and
goal, identical initial obstacle states, identical horizon, identical collision and success
definitions, identical motion model and RNG stream. Only the controller object differs.

Neither arm reads `obs[28:52]`; a test asserts bit-identical actions under corruption of that
slice, at every obstacle count and radius in the suite. No ground-truth obstacle state enters
either controller anywhere — it appears only in metrics channels, as in Phases 3–7.

---

## §7 Pre-registered predictions

Recorded now so they cannot be retrofitted.

| # | prediction |
|---|---|
| P1 | **F5 (count)** — B retains its advantage; both degrade smoothly. LiDAR sees all obstacles, and M2 conflicts grow with density, which is what LADDER was built for. |
| P2 | **F6 (speed)** — B's advantage shrinks with speed and may **reverse above 0.96**, where `v_cap` clamps true closing rates (§1.7). The crossover is the headline of this family. |
| P3 | **F4 (radius ≥ 0.45)** — both arms degrade sharply as velocity estimation silently returns 0 (§1.5). B's cap is inert on zeros, so the arms should converge — B's advantage **shrinks toward zero**, not reverses. |
| P4 | **F9 (corridors)** — infeasibility rises steeply as width falls; B's advantage **grows**, since recovery acts exactly where the frozen fallback commands `u = 0`. This is the strongest test of Q3. |
| P5 | **F1′/F2/F3 (static)** — smaller effects than the dynamic families. Neither arm's changes touch static geometry; A* handles it. |
| P6 | **F10 (reactive walls)** — both arms degrade badly; no replanning means the carrot pulls into the wall. B may collide *less* via recovery while still failing to reach the goal. |
| P7 | **X5–X7 (speed above the agent's own)** — both arms collide often; the informative quantity is the tier and margin distribution, not success. |
| P8 | **Aggregate** — B's M0 advantage (+0.070/+0.090 success) does **not** hold uniformly; expect it to persist in F5/F9, shrink in F4, and reverse in F6-fast. A uniform result either way would be surprising and would deserve scrutiny. |

**No prediction is a gate.** Any of them may be contradicted, and a contradiction updates the
interpretation, never the experiment — the standing rule since Phase 7 §11.1.

---

## §8 Files this would add (nothing yet written)

```
generalization/__init__.py
generalization/maps.py             # config-dict generators, one per family; pure functions
generalization/scenario_env.py     # RobotNavEnv subclass: reactive walls, moving goals,
                                   #   scripted trajectories. robot_env/ untouched.
generalization/suite.py            # the manifest: every map, seed, level, class, difficulty
experiments/exp9_generalization.py # screening + final runner, paired, per-map output
tests/test_generalization_maps.py  # generation regression + fairness invariants
MD_files/week6/generalization/     # this spec, then the reports
results/week6_generalization/      # per map, never overwritten
```

`dr_control/`, `optional_navigation/`, `robot_env/`, `evaluation/` and every Phase 1–7 artefact
are untouched. The frozen manifest gains the new modules as Phase-7 paths, not as frozen files.

---

## §9 Decisions recorded (approved; supersedes the open questions)

| # | question | decision |
|---|---|---|
| 1 | F1′ staircase approximation | **APPROVED.** Kept, and labelled an **interface/generalization test, NOT literal rotated-rectangle support**. Measured approximation error (area ratio, Hausdorff distance, cell count) preserved and reported per map. `robot_env/` and `evaluation/` untouched. |
| 2 | F3 LiDAR range | **KEEP FIXED at 5.0 m.** Not scaled with world size — the changing sensor-horizon-to-world ratio *is* part of the intended shift. World-size-scaled horizon (`max_steps = ceil(500·world/10)`) and distance-bin rescaling stand, applied identically to both arms. |
| 3 | F6 top level | **APPROVED through 1.10**, the highest ordinary benchmark level. 1.20/1.35/1.50 move to feasibility stress as X5–X7. 1.10 is **not** called "unavoidable" — it is beyond the regime in which `v_cap = 0.96` was derived, and exists to test whether and where B's advantage reverses. `v_cap = 0.96` fixed. |
| 4 | F11 moving goal | **DROPPED from the suite.** No scenario code, no screening or final budget. The §1.3 interface finding is retained as a documented boundary-of-applicability limitation — a code-inspection fact, never presented as an experimental result. |
| 5 | screening budget | **KEEP 50 episodes.** Not reduced to 30. Screening characterises maps; it never selects controllers. |

**Bookkeeping correction recorded.** The earlier draft claimed 49 Level-1 and 73 total
configurations. Both were wrong: the family tables sum to **42** Level-1 and, after dropping F11
and moving the F6 stress levels, **64** total. The count is now derived from the manifest in §4.1
rather than asserted, and no map was invented to restore the old figure. The §4.3 selection rule
replaces two contradictory rules with one.

---

## §10 Suite summary — the single source of truth

Every figure below is derived from §4 and must agree with it and with §3.

| quantity | value |
|---|---|
| families | **13** (F1′, F2a, F2b, F3, F4, F5, F6, F7, F8, F9, F10, F12, X) |
| configurations | **64** — Level 1: 42 · Level 2: 11 · Level 3: 4 · Level 4: 7 |
| M0 | control/reference, **not** counted among the 64 |
| F11 | **dropped**; documented limitation only, no code, no budget |
| screening cells | **118** (config × motion model) |
| final cells | **109** (§4.3 rule + Amendment 1) — **99 benchmark @ 200 ep**, **10 stress @ 50 ep** |
| screening episodes | **11 800** = 118 × 50 × 2 arms |
| final episodes | **40 600** = (99 × 200 + 10 × 50) × 2 arms |
| total episodes | **52 400** |
| runtime | **79.2 h** single-process · **9.9 h** at 8-way, at the **measured 5.44 s/ep** from 118 completed screening cells |
| storage | **≈ 39 MB** |
| seed blocks | gen 7 000 000 · screening 8 000 000+49 · final 9 000 000+199 · M0 gate only 1 000 000+199 |
| X1–X7 | characterisation/feasibility stress; **excluded from every benchmark aggregate**, no success rate reported |
| F2b_n9 | **excluded from the final tier** by Amendment 1 (§4.7) on structural-feasibility grounds; retained in screening |
| classification | **Supported (S):** F2a, F2b, F3, F5, F8, F9, F12, and F4 ≤ 0.30, F6 ≤ 0.96 · **Interface stress (I):** F1′, F10, F7, and F4 ≥ 0.45, F6 at 1.10 · **Feasibility stress:** X1–X7, reported separately, no success rate · **Unsupported (U):** F11 — dropped |
