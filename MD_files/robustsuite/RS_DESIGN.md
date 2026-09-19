# RS Experiment 1: design and pre-registration

**Robust A\*-Random on a fresh map suite.**

This document is committed **before any run**. Every rule and number in sections 3–9 is fixed here. Any later change goes in a separate, dated *Deviations* section of the results report and never edits this file. The only additions allowed are recorded approvals (section 6) and the candidate list (section 7.4), each in its own dated section that is appended here before the phase that depends on it.

- Decision record: `docs/adr/0002-test-first-robust-suite.md`
- Spec and tickets: `.scratch/rs-robust-astar-suite/`
- Branch: `robust-suite` (from `highdim` @ `cc470f0`)
- Date: 2026-09-19

---

## 1. Question

How does the frozen A\* + Random-CLF-DR-CBF stack behave on fresh, feasible and sensible maps with static, dynamic and region-triggered obstacles? And can one targeted improvement, chosen from its observed failures, beat it there without a safety cost and without regressing on M0?

The experiment runs **test first**:
1. characterise the frozen stack on the new suite;
2. choose improvement candidates from its failures;
3. select one on separate seeds;
4. compare it with the baseline once, on sealed seeds.

## 2. The baseline and what is known about it

The baseline is the frozen `astar_random` arm of HD Experiment 1. It is bit-identical to the Week 6 Continuation Random arm (the F-D arm), and a test proves this.

- **Planning.** A\* runs **once** at policy reset on the known static map, and never replans. The carrot follower's progress along the path only ever moves forward, and it emits γ 1 m ahead.
- **Safety layer.**
  - The CLF V = ½·k_v·‖p−γ‖² is a soft tracking objective.
  - The DR-CCP constraint runs in the collapsed regime, with τ_eff = 0.12.
  - The LiDAR barrier source and the velocity tracker see every obstacle (static and dynamic) through the 24 LiDAR rays only.
  - Random recovery (K 16, H 1, speeds {0.8, 1.0}) takes over when the QP is infeasible.
- **M0 results** (HD0, HD_FINAL, 400 episodes): success 0.9025, collision 0.095, almost all collisions dynamic, clustered early (steps ~60–80). Random finds a margin ≥ 0 in only about 12% of recoveries.
- **Known weaknesses:**
  - It never replans. In the Week 6 suite, a wall appearing across the route (F10c) dropped a closely related arm to 0.03–0.12 success.
  - The existing dynamic obstacles pass through static rectangles, which is not plausible in structured maps.
- **Earlier improvement attempts that changed no outcome on M0:** tuning (H1/H2), the Random perturbation (R1), persistence H > 1, the Track B slow-down supervisor, congestion detection.

## 3. What is frozen, new or a candidate

**Frozen.** No file is edited. The SHA-256 manifest `MD_files/robustsuite/PROTECTED_MANIFEST.sha256` (53 files) is verified by `tests/test_robustsuite.py` on every run. It covers:
- the canonical environment: dynamics, collision logic, LiDAR, observation, reward, and the M0 motion models;
- the A\* planner, the carrot follower and the shortest-path oracle;
- `dr_control/`: the CLF, the DR-CCP QP, the LiDAR barrier source, the velocity tracker, and the frozen policy's `predict`;
- the top-level `continuation/` modules: tuned-policy and Random-arm construction, Random recovery, seeds, harness, statistics;
- the HD modules that build `astar_random` and run its episodes (`policy`, `harness`, `blocks`, `subgoal`, `gates`) and the Week 6 parameter loaders;
- the package `__init__` files on that import path, and the reservation sources the RS seed blocks are proven disjoint from (`highdim/seeds.py`, the pursuit config);
- `config.json` and the frozen parameter files `H2/tuned_frozen.json` and `R3/random_frozen.json`.

Because the HD harness is frozen, the **RS harness is a new module**. It composes the frozen episode loop's pieces (the arm builder, recorders, episode record, metrics) around the scenario environment and adds the cell dimension. The HD harness files are not edited.

**New.**
- the scenario generator;
- the pedestrian motion model;
- the scenario environment, a subclass of the canonical environment;
- the RS harness and block runner;
- seed blocks and cells;
- the decision rules;
- the entry points under `experiments/robustsuite/`;
- tests;
- this document, ADR 0002 and the reports.

**Candidate.** A runtime composition around the frozen stack, approved in phase 3 (section 7). The CBF/DR formulation and the robot dynamics are never replaced.

## 4. The suite

### 4.1 World and robot

All cells keep M0's physical settings:
- 10 × 10 m world bounded by its outer walls;
- robot radius 0.3 m, maximum speed 1 m/s, holonomic single integrator;
- dt 0.1 s and a 500-step limit;
- LiDAR: 24 rays, range 5 m;
- the reward is the environment's own.

Static obstacles are axis-aligned rectangles (centre x, centre y, half-width, half-height), the canonical environment's format.

**Clearance radius.** Every feasibility and routing check uses the frozen shortest-path oracle (grid of 200 × 200, so 0.05 m cells) with obstacles inflated by **r_c = 0.45 m** (robot radius 0.3 + margin 0.15). A "clear path" means the oracle finds a path at that inflation.

### 4.2 Layout families

Each seed draws a new layout from its family.

| Family | Construction | Passage | Pedestrians |
|---|---|---|---|
| **open_clutter** | 5–7 rectangles, half-extents each in [0.25, 1.0] m | pairwise gap ≥ 1.2 m, ≥ 1.0 m from the outer wall | 5 |
| **rooms** | 3 or 4 rooms (equal probability). 4 rooms: a full vertical wall at x ∈ [4, 6] and a full horizontal wall at y ∈ [4, 6]. 3 rooms: one full wall, plus a second wall across one of its halves. Walls are 0.2 m thick. Each wall segment between two rooms gets exactly one doorway, at least 0.5 m from any wall junction. Each room gets 1–2 furniture rectangles (half-extents [0.2, 0.5] m), at least 1.0 m from any doorway centre | doorway width ∈ [1.2, 1.6] m | 3 |
| **aisles** | Orientation horizontal or vertical (equal probability). 3–4 shelf rows, depth ∈ [0.6, 0.8] m, centred in the world. A cross-aisle at both ends of the rows, plus, with probability 0.5, one mid cross-aisle cutting every row | aisle and cross-aisle width ∈ [1.4, 2.0] m | 4 |
| **corridors** | Free space is a union of corridor strips: a spine running the length of the world (orientation equally likely); 2–3 perpendicular branches off the spine (T-junctions); one L-bend at the end of one branch; and **one connector strip parallel to the spine joining the far ends of two branches**, so the network contains a loop. Everything else is filled with solid wall rectangles | corridor width ∈ [1.5, 2.0] m | 3 |
| **dense_clutter** | 12–18 rectangles, half-extents each in [0.15, 0.4] m | pairwise gap ≥ 1.1 m (0.2 m free at r_c, 4 grid cells), ≥ 0.8 m from the outer wall | 4 |

Widths, counts and positions are drawn uniformly within the stated ranges. A layout whose free space is not connected at r_c is resampled.

### 4.3 Obstacle conditions

The conditions are **cumulative**. For a given seed, all four conditions of a family use **one family draw** (layout, start and goal, pedestrian routes, trigger disc, block and spawn placements; section 5). A condition only switches parts of it on:
- `static` has no pedestrians;
- `trigger_block` fires the block;
- `trigger_spawn` fires the spawn.

So each triggered cell is its family's "+dynamic" cell plus exactly one event, paired seed by seed.

| Condition | Static layout | Pedestrians | Event |
|---|---|---|---|
| `static` | yes | none | none |
| `dynamic` | yes | the family count | none |
| `trigger_block` | yes | the family count | one triggered static block |
| `trigger_spawn` | yes | the family count | one triggered pedestrian spawn |

5 families × 4 conditions = **20 cells**. Each cell is run under **both motion conditions** (section 4.4). **M0** (the canonical map and M0's original motion) is the anchor cell, outside the pooled gate.

### 4.4 Pedestrians

Pedestrians have radius 0.3 m and speed 0.675 m/s (M0's values).

**Routes.** The episode spec gives each pedestrian a start and a cycle of **8 waypoints**.
- Waypoints are drawn uniformly in free space (inflation 0.45 m).
- Legs between them are oracle paths at inflation 0.45 m, simplified to line segments that keep that clearance.
- **The block zone is reserved.** Every family draw places a block (4.6) even in cells where it never fires. Pedestrian waypoints and legs treat that rectangle as an obstacle **in all four obstacle conditions**, so no pedestrian is ever inside or walking through a block. The four conditions keep identical pedestrians.
- A pedestrian walks its legs in order and loops.

**Starts.**
- Pedestrian starts are ≥ 1.5 m from the robot start and ≥ 1.0 m from the goal.
- Pedestrians never start inside or within 0.45 m of a static obstacle.

**Motion conditions.**
- **fixed:** the pedestrian follows its legs exactly at 0.675 m/s. Deterministic given the spec.
- **randomized:** the heading is perturbed by M0's randomized OU model (angular noise σ 0.35, angular-velocity decay 0.8, maximum turn rate 0.7 rad/s), plus steering towards the current leg with gain 4.0.
  - If a noisy step would bring the pedestrian within 0.3 m of a static obstacle, or outside the world, it takes the exact leg step instead.
  - The noise draws come from the environment's own RNG, which is seeded by the episode seed and never depends on the robot.

Pedestrians pass through each other. They never enter static obstacles; a test asserts this over long rollouts in every family.

### 4.5 Start and goal

- **Distance bins.** One of M0's four Euclidean distance bins, [4, 5.5], [5.5, 7], [7, 8.5] and [8.5, 10.5] m, is drawn uniformly. Start and goal are then drawn uniformly in free space (inflation 0.45 m) at a distance inside the bin. If a bin cannot be satisfied, the bin is redrawn; this counts towards the resample cap.
- **A clear path** must exist from start to goal.
- **Structural rule per family:**
  - rooms: start and goal in different rooms;
  - aisles: in different aisles, or at opposite ends of one aisle (at least 60% of the row length apart along it);
  - corridors: the oracle path turns at least once (a change of direction ≥ 60° sustained over ≥ 1 m);
  - open_clutter and dense_clutter: the bins only.

### 4.6 Triggered events

**Placement.** There is exactly one event per triggered episode.
- The **route** is the oracle path from start to goal on the static layout at r_c. It is arm-independent and computed by the generator.
- The trigger is a disc of radius **0.5 m** centred on the route at arc-length fraction **f ∈ [0.3, 0.7]** (uniform).
- It fires on the first step on which the robot's centre is inside the disc. The event log records the step and the robot's position.

**Triggered static block.**
- **Centre:** on the route, a further arc length **d ∈ [1.5, 2.5] m** (uniform) beyond the trigger centre.
- **Orientation (axis-aligned):** the block is an axis-aligned rectangle. Its long axis is the world axis closest to the perpendicular of the local route direction (averaged over ±0.5 m of arc). It is centred on the route point, so it always crosses the route.
- **It must span a passage.** Along its long axis, the free extent through the centre must end at a static obstacle or the outer wall **on both sides**, within a total free width **≤ 3.0 m**. Otherwise d, then f, is redrawn. This excludes free-standing walls in open space.
- **Size:** thickness 0.4 m; length = that free width + 0.2 m of overlap into the wall on each side, so at most 3.4 m.
- **Map knowledge:** the block is added to the environment's static obstacles on firing, so LiDAR and collision logic see it from that step on. **It never enters A\*'s map** or any candidate's known map.
- **Feasibility:**
  - a clear path from the trigger disc's centre to the goal must still exist with the block added;
  - the block keeps ≥ 0.45 m from the goal and from the start;
  - it must not overlap the trigger disc.
  - Otherwise d, then f, is redrawn.

**Triggered pedestrian spawn.**
- **Spawn point:** drawn uniformly in free space (inflation 0.45 m, block zone reserved) at a distance **[2.0, 3.0] m** from the trigger centre, so ≥ 1.5 m from every point of the trigger disc. It is also ≥ 1.5 m from the robot start and ≥ 1.0 m from the goal.
- **Occluder:** the straight line from the trigger centre to the spawn point must cross a static obstacle. This is the "behind an occluder" rule. If no point in that range qualifies, f is redrawn.
- **Entry variant**, drawn with equal probability and recorded:
  - **crossing:** the pedestrian walks an oracle path (inflation 0.45 m, block zone reserved) to the route point 1.0–2.0 m (uniform) beyond the trigger centre. It then continues in its arrival direction, perpendicular to the route snapped to the nearest world axis, for up to 2.0 m, stopping 0.45 m before any static obstacle. Then it joins its own waypoint cycle;
  - **head-on:** it walks an oracle path to the route point 2.0 m beyond the trigger centre, then back along the route towards the start for 3.0 m (the route keeps r_c clearance by construction), then joins its waypoint cycle.
- **Clearance:** every entry path must keep ≥ 0.45 m from static obstacles, or the spawn point, then f, is redrawn.
- **Behaviour:** the spawned pedestrian does not exist before the trigger fires (not in LiDAR, not in collision checks). It starts moving at 0.675 m/s on the firing step. After the entry it behaves like any pedestrian of its motion condition.
- **Slots:** the spawned pedestrian counts towards the environment's six obstacle slots. Family counts + 1 ≤ 6.

### 4.7 Feasibility and resampling

The generator returns an episode spec only if every rule in 4.2–4.6 holds:
- a clear path from start to goal;
- a clear path still exists after any block;
- no unavoidable spawn (≥ 1.5 m from the disc);
- pedestrians never faster than the robot, and not started on or near the start or goal;
- start and goal in free space with clearance;
- the family's structural rule.

**Resampling works at the level of the family draw.**
1. Draw the layout, start and goal.
2. Draw the trigger disc and both events: up to 50 attempts at f, d and the spawn point, before the layout itself is redrawn.
3. Draw the pedestrian routes.

A draw is accepted only if **every** condition's rules hold, including both events. So all four conditions of a family stay paired even though only two of them fire an event. Resampling is deterministic from the same seed stream. After **200** layout attempts the generator **raises**. It never returns an infeasible episode. The spec records the attempt counts.

### 4.8 Map validation report (phase 1 exit)

This report covers all 20 cells × 2 motion conditions on the RS_DIAG seed indices, plus the anchor, and is produced without running any robot episode. It gives:
- acceptance and resampling rates;
- distributions of clearance, passage width, route length, start/goal bin and pedestrian count;
- trigger fraction, block length and spawn distance;
- rendered samples of every cell;
- short pedestrian-only rollout renders showing that walls are respected.

**The researcher approves it before phase 2**, and the approval is appended to this document.

## 5. Seeds, cells and pairing

| Block | Range | Use |
|---|---|---|
| RS_DIAG | 15 000 000 + 0…49 | baseline diagnostic (phase 2); candidate parameter search (7.3) |
| RS_TUNE | 15 100 000 + 0…49 | candidate selection (phase 5) |
| RS_FINAL | 15 200 000 + 0…199, **sealed** | final comparison (phase 6) |

- **Disjointness.** The blocks are disjoint from each other and from every Continuation block and forbidden range, the pursuit, Track B and Track A reservations, HD_DEV, HD_FINAL and HD_TRAIN (tested). RS_FINAL opens only from the final-evaluation entry point `rs6_final.py` (tested).
- **Pairing.** Every cell uses the same seed indices of a block.
  - The generator's randomness is a SeedSequence with entropy **(seed, family index)**, so families draw independent layouts.
  - Neither the **obstacle condition** nor the **motion condition** is in the entropy. All four conditions and both motions of a family share one family draw (4.3); a condition only switches its parts on.
  - Every arm sees the same spec. The environment's reset seed is the episode seed, and the Random RNG is `continuation.seeds.controller_rng(seed)`, both unchanged. So paired arms face the same layout, start and goal, pedestrian routes, pedestrian noise and trigger.
- **Anchor.** The M0 anchor uses the block's seeds with M0's own map, start/goal sampler and motion models ("fixed" = M0's deterministic motion, "randomized" = M0's smooth stochastic motion). Through the scenario environment it must be bit-identical to the canonical environment.

## 6. Phases and transition rules

| Phase | Work | Rule to proceed |
|---|---|---|
| 0 | This document, ADR 0002, manifest, seed blocks, vault note, baseline test suite (section 11) | committed |
| 1 | Generator, pedestrian model, scenario environment, RS harness; map validation report (4.8) | **the researcher approves the report** (appended here) |
| 2 | **Baseline diagnostic** on RS_DIAG: 20 cells × 2 motions × 50 seeds + anchor | failure breakdown written (section 10) |
| 3 | **Candidate list** (section 7) | **the researcher approves it** (appended here) |
| 4 | Candidate implementation | all tests pass |
| 5 | **Selection** on RS_TUNE: baseline + candidates, 20 × 2 × 50 | selection rule (section 8) |
| 6 | **Sealed final** on RS_FINAL: baseline + winner (or baseline only), 20 × 2 × 200 + anchor | GO gate (section 9) |

Evaluation always uses the frozen SCS controller. Episodes are deterministic given the seed, so any block can be re-run exactly.

## 7. Candidates

### 7.1 What a candidate may change
- **Planning:** e.g. replanning when blocked, a Wait or Retreat behaviour layer, the carrot lookahead.
- **Parameters:** those of the tuned policy and of Random recovery, only within the search budget (7.3).
- **Perception and prediction:** e.g. a LiDAR-built occupancy of new static obstacles, a prediction-aware barrier source built from the frozen LiDAR pipeline's outputs.

**Never:** the CLF/CBF/DR-CCP formulation (constraint form, ξ construction, collapsed regime), the robot dynamics, the reward, or the environment.

At most **three** candidates. A combination of mechanisms is allowed only as one of the three.

### 7.2 Information and compute
- **Onboard information only:** the observation the frozen policy receives (goal offset, own velocity, LiDAR), the ego pose the frozen safety layer already uses, the **known static map without any triggered block**, and the candidate's own history.
- **Never:** the ground-truth obstacle block `obs[28:]`, the episode spec's trigger, block or spawn fields, or the scenario event log. Tests perturb each of these and require bit-identical actions.
- **Real time:** a candidate whose p95 control-step time on RS_TUNE exceeds **100 ms** is disqualified before selection.

### 7.3 Parameter search
A candidate with free parameters may evaluate at most **8 configurations**, including its **default** configuration (named in 7.4). Each is run on the **first 20 RS_DIAG seeds** of every cell and both motions. The chosen configuration is the one with the highest pooled success whose pooled collision rate is ≤ the baseline's + 0.01 on the same episodes. If none meets the collision limit, the default goes forward. Only the chosen configuration goes forward to RS_TUNE.

The search uses the diagnostic block, not RS_TUNE, so the selection block stays clean. The cost is disclosed: parameters are tuned on the seeds whose failures motivated the candidates. This departs from the spec's wording (spec story 43, "the tuning block only") and is recorded in section 14.

### 7.4 Candidate list
Appended here in phase 3, before any candidate is built. For each candidate: mechanism, changed components, information used, the failure modes it targets, its expected effect, its parameters, default configuration and search grid (≤ 8), and its expected compute.

**Changed components** are counted, for the tie-break in section 8, as the number of distinct items from this fixed list that the candidate replaces, wraps or re-parameterises:
1. the A\* planner / carrot follower;
2. a behaviour layer added around the policy;
3. the barrier source or its inputs;
4. Random recovery;
5. the tuned-policy parameter set;
6. the Random specification.

## 8. Selection rule (phase 5)

On RS_TUNE, paired with the baseline. Pooled rates weight **all 20 cells equally** and pool both motion conditions within a cell; the anchor is excluded.

- A candidate **qualifies** if both hold:
  - S_cand − S_base ≥ **0.02** (pooled success);
  - C_cand − C_base ≤ **0.01** (pooled collision).
  - Counts are compared in exact rational arithmetic.
- The **winner** is the qualifying candidate with the highest pooled success. Ties go to the candidate with fewer changed components, then to the earlier-listed one.
- **No qualifier:** no winner. The report says "no improvement found", and phase 6 runs the baseline only (a characterisation of the suite).

## 9. GO gate (phase 6)

On RS_FINAL, the winner against the baseline, paired by (cell, motion, seed). The pooled rates are equal-weight over the 20 cells. **Significance uses the exact two-sided McNemar test** (`continuation.stats.paired_binary`) on the episode-level paired outcomes; bootstrap CIs are reported but are not part of the gate.

1. **Success:** S_win > S_base with McNemar p < 0.05.
2. **Safety:** fails if C_win > C_base with McNemar p < 0.05.
3. **M0 anchor:** fails if S_win(M0) < S_base(M0) with McNemar p < 0.05 (the anchor's 400 paired episodes).

**GO** if 1 holds and neither 2 nor 3 fails. Otherwise NO-GO. **Per-cell** differences (400 pairs each) are reported with their McNemar p-values and a note that 20 cell-level tests will produce false positives. They are **not** gated.

## 10. Metrics and reporting

Every result file records the commit, the manifest hash, the frozen-parameter hashes, the generator version and the seed block.

**Per cell, per family and pooled:**
- success; collision (dynamic / static / wall); timeout;
- stuck (the Phase 5 definition, as in HD0); SPL; minimum clearance;
- infeasible-step rate, Random event rate and recovery margins;
- step-time mean and p95.

**Triggered cells:**
- everything above, both overall and **conditional on the trigger having fired**;
- the fired rate;
- outcome timing relative to the firing step;
- the robot's position at collision relative to the block or spawn.

**Other outputs:**
- rendered trajectories of the most common failure in each cell (phase 2);
- the final report adds per-cell arm tables, worst-cell success and a Deviations section.

## 11. Baseline test suite (phase 0)

Recorded on `robust-suite` (base `cc470f0`) on 2026-09-19, when the only RS code was the manifest, the seed blocks and their tests.

| Suite | Collected | Passed | Failed |
|---|---|---|---|
| Pre-existing tests | 630 | 626 | 4 |
| `tests/test_robustsuite.py` (manifest, seed blocks, cells) | 39 | 39 | 0 |
| **Total** | **669** | **665** | **4** |

The 4 failures are the known, pre-existing `tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main[0, 3, 6, 10]` (HD_DESIGN §10 explains them). The HD manifest test (`tests/test_highdim.py`) still verifies. Any other failure in later phases is new and gets reported.

## 12. Caveats stated in advance

- **One robot, one controller family.** The frozen parameters were tuned on M0. The suite keeps M0's physical settings so they stay in regime, but the new layouts are narrower than M0 in places (doorways down to 1.2 m).
- **Pedestrians are not reactive.** They ignore the robot and each other. A robot that stops in a pedestrian's path is hit only if the pedestrian's route crosses it, which is a property of the suite and not of any arm.
- **The events are LiDAR-only.** A spawned pedestrian or a new block is invisible until it is in LiDAR's line of sight. That is intended.
- **Axis-aligned rectangles only**, the canonical environment's obstacle model.
- **No moving goals, no multi-goal episodes, no adversarial obstacles.**

## 13. Out of scope

- Any change to the CLF, CBF/DR formulation, dynamics, reward, LiDAR or observation.
- Learned components (PPO or other RL) as candidates.
- The Week 6 generalization suite and its results.
- Pursuit, obstacle-speed stress, varying robot speed or world size.
- More than one event per episode; more than three candidates; any candidate added after RS_TUNE has been seen.

## 14. Clarifications and departures from the spec, fixed before any run

- **Pedestrian waypoints.** The spec says pedestrians draw their next waypoint from the environment's RNG. Here each pedestrian's 8-waypoint cycle is part of the episode spec (4.4), so pedestrian routes are fixed by the seed, reviewable in the validation report and identical across arms by construction. The environment's RNG drives only the randomized heading noise.
- **Parameter search on RS_DIAG** instead of the tuning block (7.3). This keeps RS_TUNE clean for selection.
- **One family draw shared by all four obstacle conditions** (4.3, 5). Triggered cells are paired with the "+dynamic" cell seed by seed, not only statistically.
- **The block zone is reserved for pedestrians in all conditions** (4.4). Pedestrians never walk through a block that appears, and the conditions stay paired.
- **The HD harness is frozen**, so the RS harness is a new module (3). Ticket 02 was worded accordingly.
- **The block spans a passage** (≤ 3.0 m, bounded on both sides) and is axis-aligned (4.6). This is the spec's "spanning the passage", made checkable in the rectangle format.

### 14.1 Researcher decisions and readings before any run (2026-09-19, tickets 02–05)

Recorded after the suite was built and before any robot episode was run on it. The researcher approved items 1 and 2. Items 3–8 are readings of wording the design leaves open. All of them apply from generator version `rs-gen-0.2` on.

1. **Start and goal clearance is 0.6 m, not 0.45 m (section 4.5).** The frozen controller's barrier value is h = LiDAR range − 0.3. So an obstacle within 0.6 m of the robot's centre gives h < 0, which turns the reference objective's weights p1 = 4h and p3 = 5h negative. CVXPY then raises DCPError, and the frozen policy executes u = 0. Random recovery triggers only on "infeasible", so it never acts on these steps. A robot that starts inside that zone in a static scene never moves. At 0.45 m, 20–44% of starts per family did. 0.6 m is M0's own start/goal wall margin, so M0 never produced such starts. Routes, pedestrian waypoints and every feasibility check stay at r_c = 0.45 m, so passages narrower than 1.2 m still exercise the zone mid-route.
2. **Randomized pedestrians (section 4.4).** The 0.7 rad/s turn-rate clip bounds the OU heading perturbation only. The steering is added on top of it, at gain 4.0 /s on the heading error to the route point 1 m ahead of the pedestrian's progress. Clipping the steering as well, as M0's boundary steer is clipped, left pedestrians stalled against interior walls 70–95% of the time. With this reading, randomized pedestrians stay a median of about 5 cm (95th percentile about 18 cm) from their route, so the randomized condition is mostly a small lateral and timing perturbation of the fixed routes. The OU angular velocity starts from its stationary distribution, as in M0.
3. **The exact leg step (section 4.4).** When a noisy step would end within 0.3 m of an obstacle or the outer wall, the pedestrian takes one step towards its route point one step ahead. If that step is not clear either, it steps towards its own route point, and if neither is clear it holds. Progress along the route is the pedestrian's projection onto it, searched over the next 1.5 m.
4. **Pedestrian routes.** A pedestrian walks an entry leg from its start to its first waypoint, then loops through its 8 waypoints. The spec carries the pedestrians in every condition; the static condition switches them off.
5. **Resample cap (section 4.7).** Each layout draw, each failed distance-bin attempt and each failed pedestrian draw counts as one of the 200. After 4 failed bins on one layout, the layout is redrawn.
6. **Corridors (section 4.2).**
   - The network is built constructively. The spine centre is 1.4–3.0 m from its side of the world. Corridors keep 0.5 m of wall from the outer boundary and 1.0 m of wall between corridors that do not join. The connector's far edge is at 7.0–9.5 m. The L-stub is 1.5–3.0 m long and as wide as its branch.
   - The stub branch is an outer branch, and the stub turns outward.
   - With 3 branches, the stub branch is the free one: a true dead-end L.
   - With 2 branches, both branches are in the loop. The stub then turns off a loop branch's far end, opposite the connector.
7. **Aisles (section 4.2).**
   - The mid cross-aisle leaves at least 1.5 m of row on each side.
   - The strips between the outermost rows and the outer wall follow from centring the rows. They are 0.4–2.7 m wide and are not held to the 1.4–2.0 m aisle range.
   - Start and goal never lie in those strips, because the structural rule needs an aisle between rows. Pedestrians may walk them.
8. **Corridors turn rule (section 4.5).** "A change of direction ≥ 60° sustained over ≥ 1 m" is read as follows: at some vertex of the simplified start–goal route, the 1 m chord before the vertex and the 1 m chord after it differ in direction by at least 60°.

### 14.2 Readings before any run (2026-09-19, ticket 06)

These are readings of wording in section 4.6 that the design leaves open. They were recorded when the triggered block was built, before any robot episode was run on the suite. The researcher approved item 2, the only one that tightens a pre-registered number. They apply from generator version `rs-gen-0.3` on, which adds the trigger and the block to every family draw.

1. **The route is the spec's route.** That is the simplified oracle path from start to goal at r_c, and the trigger fraction f is taken of its polyline arc length.
2. **Block to trigger clearance is 0.8 m (researcher-approved).** 4.6 requires only that the block not overlap the trigger disc (0.5 m). But when the block appears, the robot's centre can be anywhere in the disc, so a block closer than disc radius + robot radius (0.5 + 0.3 = 0.8 m) could appear on top of the robot and force a collision. The block therefore keeps ≥ 0.8 m from the trigger centre.
3. **The start lies outside the trigger disc,** so the robot has to enter it.
4. **A clear path from the trigger centre to the goal** is checked as start, trigger centre and goal lying in one free component of the frozen oracle's grid at r_c, with the block added. This also covers start to trigger.
5. **Attempts (4.7).** Each of the 50 attempts is one draw of f. Each f gets up to 5 draws of d before f is redrawn. After 50 failed draws of f, the family draw fails, counts one draw towards the resample cap of 200, and the layout is redrawn.
6. **Route direction and passage.**
   - The local route direction is the chord from 0.5 m of arc before the block centre to 0.5 m after it.
   - The long axis is x when |direction_y| ≥ |direction_x|, and y otherwise.
   - The free extent is measured along the long-axis line through the route point. A rectangle whose edge the line only grazes does not bound it.
   - Against the outer wall, the 0.2 m overlap lies outside the world.
7. **Firing time within a step.** The trigger is checked after the robot moves and before the pedestrians move, the collision check and the observation. So the collision check and LiDAR of the firing step already include the block.

### 14.3 Researcher decisions and readings before any run (2026-09-19, ticket 07)

These were recorded when the triggered spawn was built, before any robot episode was run on the suite. The researcher approved items 1 and 2. Items 3–8 are readings. All of them apply from generator version `rs-gen-0.4` on.

1. **The spawn ignores the block zone (a departure from 4.6).** 4.6 reserves the block zone for the spawn's start and entry paths. But the block lies 1.5–2.5 m beyond the trigger, where the entries go. The head-on target (2.0 m beyond) was always within 0.3 m of the block (40 draws measured), so a head-on entry could never be drawn, and only about 20% of crossing targets cleared the zone. The spawned pedestrian exists only in `trigger_spawn`, where the block never appears. The zone keeps the *regular* pedestrians paired across conditions (4.4), and they keep it. So the spawn's start, entry paths, waypoint cycle and motion use the static layout only.
2. **The spawned pedestrian's noise has its own stream (a departure from 4.4).** 4.4 draws pedestrian noise from the environment's RNG. The firing step depends on the robot, so extra draws from that RNG after firing would make the regular pedestrians' later noise depend on the robot. That breaks both arm pairing and the pairing of `trigger_spawn` with `dynamic`. The spawned pedestrian therefore draws from its own generator, seeded by (episode seed, 1). The environment's RNG and the regular pedestrians stay exactly those of the family's dynamic cell.
3. **The variant is a fair coin per seed.** It is the generator's first draw for the seed and is kept through every redraw, so feasibility cannot bias which variant a seed gets.
4. **Spawn start and occluder.**
   - The start is uniform over the area of the 2.0–3.0 m annulus around the trigger centre.
   - "Crosses a static obstacle" means the segment from the trigger centre to the start meets a rectangle of the layout; the outer wall does not count.
5. **Attempts (4.7).** Each draw of f tries the block first (up to 5 draws of d), then up to 20 spawn starts. Each start has its own crossing offset. The block and the spawn must both succeed for the same f.
6. **Crossing.**
   - The arrival direction is the last segment of the oracle approach path.
   - The walk goes along the world axis nearest the route's perpendicular (the chord over ±0.5 m of arc), in the sign of the arrival direction.
   - Its length is min(2.0 m, the free distance along that line − 0.45 m), and 0 if there is no room.
   - The whole walk must keep 0.45 m of exact clearance, or the start is redrawn.
7. **Head-on.** The walk back follows the route polyline from 2.0 m beyond the trigger centre to 1.0 m before it.
8. **After the entry.** The spawned pedestrian walks an oracle path from the end of its scripted entry into its own 8-waypoint cycle, drawn like any pedestrian's (on the static layout). In the environment it appears at its start on the firing step and takes its first step in that same step, as the last obstacle slot. Before firing it is in no array, so it is in no LiDAR ray, collision check or observation.

## Approval: map validation report (phase 1 → 2), 2026-09-19

The researcher approved the map validation report (`MD_files/robustsuite/RS1_MAP_VALIDATION_REPORT.md`, generated at `2d0d624` and committed at `f2234a8`: generator `rs-gen-0.4`, RS_DIAG, 50 seeds per family). Phase 2, the baseline diagnostic, may start.

The approval records two properties of the suite as known, not as defects. Neither changes a rule, and both apply equally to every arm.

1. **Accepted trigger fractions lean early.** f is drawn uniformly on [0.3, 0.7], but later fractions fail more often, so the accepted draws per 0.1 bin are 108 / 62 / 46 / 34 of 250. Every event still lies within the pre-registered 30–70% of the route.
2. **Some blocks leave a quick way round.** Blocks with a detour under 1 m: dense_clutter 14 / 50, open_clutter 5 / 50, 0–1 / 50 in the structured families. In those cells, trigger_block partly tests replanning around a nearby box rather than a shut passage.

The approval also accepts that the statistics are reported per family rather than per cell, since the four cells of a family are views of one draw (40 of 40 regenerated specs matched). Aisles give no start–goal pairs in the 8.5–10.5 m bin, a consequence of the family's geometry.

## Candidate list (section 7.4), 2026-09-20 (approved; see below)

Written in phase 3 from the baseline diagnostic (`MD_files/robustsuite/RS2_BASELINE_DIAGNOSTIC_REPORT.md`, run at `4c01512`) and the collision probe (`experiments/robustsuite/rs2_collision_probe.py`, `results/robustsuite/RS2/collision_probe.json`). No candidate has been built and no RS_TUNE seed has been opened. The list is in selection order (section 8's last tie-break).

**What the diagnostic says the candidates must address** (RS_DIAG, 2 000 episodes over the 20 cells):
- `timeout_at_block`: 321 episodes (16%), the dominant mode. A* never replans. The carrot stays on the far side of the fired block, and the robot stands 0.41–0.48 m from the block's face (5th–95th percentile) with a feasible QP until the timeout. The trigger_block cells succeed 0.04–0.31, against 0.78–0.99 for the dynamic cells on the same draws.
- Pedestrian collisions: 194 episodes (`collision_dynamic` 183, `collision_spawned` 11). 88 are in trigger_block cells, against 43 in the dynamic cells on the same draws. So about 45 come from waiting at the block; this is an estimate from the difference. The rest grow with narrowness (the corridors' dynamic cell 0.20). The collision probe (206 pedestrian collisions, anchor included) shows:
  - the pedestrian walks straight at the robot: cos > 0.7 in 87% of collisions;
  - the robot is backing away: its velocity points away from the pedestrian in 72% of collisions, within about 45° of straight away (cos < −0.7) in 47%. Those backing away move at a median 0.37 m/s, while the pedestrian closes at a median 0.63 m/s;
  - the pedestrian was within 2 m (centres) for a median 2.8 s before contact;
  - the QP was infeasible in the last 5 steps of only 33% of these collisions.
- `timeout_stuck`: 36 episodes. The same standstill without a block: the robot is wedged off its route with the carrot on the far side.
- DCPError freezes: 2 episodes, both at the outer wall. **Not targeted**: the phase 3 idea "Random on DCPError steps" has almost nothing to act on.

All three candidates:
- **Information:** onboard only.
  - The LiDAR ranges `obs[4:28]`, the ego pose, and the known static map. That is the snapshot the policy's planner is built from at reset (`pol.planner`), before any trigger can fire. A candidate never reads `env.static_obstacles`, which gains the block when it fires.
  - The frozen LiDAR velocity tracker's tracks. The frozen barrier source already computes them from LiDAR alone each step. They are read-only: a candidate reads a track's `position`, `velocity` and `confirmed`, and never calls anything that changes the tracker's state.
  - The candidate's own history.
  - Never `obs[28:]`, the spec's trigger, block or spawn fields, or the event log. The phase 4 tests perturb each of these.
- **What stays frozen:** the CLF/DR-CBF QP, its parameters, Random recovery and the dynamics are untouched. A candidate changes only γ, the CLF reference that the frozen `predict` reads from `pol.follower`, and whatever it needs to compute γ.

### Candidate 1: `replan`: a LiDAR map of new obstacles and A* replanning

- **Mechanism.**
  - **The new-obstacle map.** It is a 0.1 m grid. Each step, every LiDAR hit farther than 0.15 m from the known map (its rectangles and the outer wall) is *unexplained*. A cell becomes a *new static obstacle* once unexplained hits land in it on at least **K** of the last 20 steps. Cells are never cleared within an episode.
    - A block's face, once in view, is hit on nearly every step.
    - Hits on a pedestrian can stay in one cell for several steps: up to about 7 on its flank. A pedestrian that stays near the robot for a few seconds could therefore still leave phantom cells, so K is set high.
    - How many phantom cells pedestrians leave is measured in phase 4, on pedestrian-only replays of RS_DIAG.
  - **Triggers.** A replan happens on either of two triggers:
    - (a) the remaining planned path, from the robot's projection onward, passes within 0.3 m + **m** of a new cell;
    - (b) *stall* (if enabled): the robot moved less than 0.1 m in the last 30 steps.
  - **Replan.** A* (the frozen `StaticMapPlanner` class) runs on the known map plus every new cell as a 0.1 m square grown by **m**, from the current pose to the goal. The frozen `CarrotFollower` (lookahead 1 m) then follows the new path.
    - The planner inflates every obstacle by the robot's radius (0.3 m). A robot the CBF has stopped 0.45 m from the block therefore lies inside the inflated new cells whenever m > 0.1 (0.05 + m + 0.3 > 0.45). A* then starts from the free grid cell nearest the robot.
    - If A* finds no path, the current follower is kept.
    - Replans are at least 20 steps apart, at most 10 per episode.
- **Changed components:** 1. The A\* planner / carrot follower (item 1 of the list in 7.4).
- **Targets:**
  - `timeout_at_block` (321);
  - the collisions while waiting at the block (about 45);
  - `timeout_stuck` (36), through the stall trigger;
  - `collision_block` (4).
- **Expected effect.** The fired block is always passable by construction (4.6), so a replan exists.
  - If the trigger_block cells reach their dynamic cells' success, pooled success rises by about +19 pp (0.25 × (0.90 − 0.12)). The expected range is **+10 to +19 pp**, with pooled collisions down about 2 pp.
  - Little change is expected in cells without an event.
  - **Risk:** a lingering pedestrian could leave a phantom cell and cause a needless detour (lower SPL, more exposure to pedestrians).
- **Parameters and search (7.3, 8 configurations):** K ∈ {8, 14} × m ∈ {0.0, 0.15} m × stall ∈ {on, off}. **Default:** K = 14, m = 0.0, stall on.
- **Compute.**
  - The per-step map update is 24 rays: under 1 ms.
  - A replan with a freshly built frozen planner takes about 0.22 s (0.10 s to build the grid, 0.12 s for A*, measured on a rooms draw).
  - The implementation builds the known-map planner once per episode and adds the new cells to a copy of its grid, so a replan should cost about the A* time.
  - Replans are rare (event-driven, rate-limited), so the p95 step time should stay near the baseline's 40–47 ms. The individual replan steps will exceed 100 ms.
  - The 7.2 rule is on the p95 of every control step, replan steps included. It is measured as the harness measures it, as the whole `predict` call. The report will also state the maximum.

### Candidate 2: `yield`: step aside for an approaching pedestrian

- **Mechanism.**
  - **Threat.** A confirmed track of the frozen tracker, moving at ≥ 0.3 m/s, whose constant-velocity closest approach to the robot's current position within the next **H** s is under **D** (centre distance).
  - **Response.** While a threat exists, γ is replaced by a side point p + **L**·n, where n is the unit vector perpendicular to the threat's velocity:
    - the side is the one whose LiDAR rays within ±45° of n have the larger minimum range;
    - if that side point is not free on the known static map at 0.3 m clearance, the other side is used;
    - if neither is free, there is no yield and γ stays the follower's.
    - With several threats, the one with the earliest closest approach decides.
  - When no threat remains, the follower's γ returns.
- **Changed components:** 1. A behaviour layer around the policy (item 2).
- **Targets:** `collision_dynamic` (183) and `collision_spawned` (11). The probe shows head-on approaches with 2.8 s of warning, met by backing away along the pedestrian's line. A step of L = 1 m sideways at the robot's speed clears a 0.6 m contact diameter in about 1 s.
- **Expected effect: +2 to +5 pp** pooled success, and pooled collisions down 2–5 pp.
  - Less in corridors (1.5–2.0 m wide), where the side step is cut short.
  - Nothing on the block modes.
  - **Risk:** more timeouts and lower SPL from repeated yielding in busy cells. Stepping aside near walls stays guarded by the frozen CBF.
- **Parameters and search (8 configurations):** H ∈ {2, 3} s × D ∈ {0.9, 1.2} m × L ∈ {0.6, 1.0} m. **Default:** H = 3 s, D = 1.2 m, L = 1.0 m.
- **Compute:** a few tracks per step, well under 1 ms.

### Candidate 3: `replan_yield`: the combination

- **Mechanism:** candidate 1's follower with candidate 2's layer on top. Yield overrides γ while a threat exists; otherwise γ is the (re)planning follower's.
- **Changed components:** 2 (items 1 and 2).
- **Targets:** all the modes of candidates 1 and 2. They are largely disjoint, so their effects should roughly add up.
- **Expected effect: +12 to +22 pp** pooled success.
- **Parameters and search (2 configurations):**
  - both defaults;
  - the configurations chosen for candidates 1 and 2 by their own searches, taken together.
  - The 7.3 rule picks between the two on the same first 20 RS_DIAG seeds. Nothing else is searched.
  - Those seeds are then used a third time, after the searches of candidates 1 and 2. The cost of this is disclosed as in 7.3.
- **Compute:** the sum of the two.

**Tie-break counts (section 8):** `replan` 1, `yield` 1, `replan_yield` 2.

## Approval: candidate list (phase 3 → 4), 2026-09-20

The researcher approved the candidate list above (committed at `423de4b`), all three candidates as written: `replan`, `yield` and `replan_yield`, with their parameter grids and defaults. Phase 4, the candidate implementation, may start. Ticket 11 is replaced by one implementation ticket per candidate (11a, 11b, 11c).

## Correction: where h < 0 happens (to section 14.1 item 1), 2026-09-20

Recorded at the researcher's request. Section 14.1 is not edited.

Section 14.1 item 1 says an obstacle within 0.6 m of the robot's centre gives h < 0. Measured in ticket 09 (`RS2_BASELINE_DIAGNOSTIC_REPORT`, reading item 5):
- The environment's LiDAR reads static rectangles and pedestrians at their true distance from the robot's centre. So next to them, h = distance − 0.3 is negative only on contact.
- It reads the **outer wall** 0.3 m short. So h < 0, and with it DCPError, happens within 0.6 m of the outer wall.

In the RS2 diagnostic, 2 of 2 000 episodes froze, both 0.56 m from the outer wall. The approved 0.6 m start and goal clearance is kept: it is M0's own margin, and it keeps starts out of the band next to the outer wall. No rule changes.

## Revision of the candidate list (section 7.4), 2026-09-20 (approved; see below)

The researcher asked, after approving the list above and before any candidate was built, that **A\* is never run again after reset**. There are three reasons: the ~0.2 s replan breaks the 10 Hz step, the global plan should stay exactly the baseline's (planning once is part of the stack under test), and simplicity. The researcher chose a local LiDAR detour that follows walls (Bug2-style) over a simpler gap detour. The deciding fact was the block's detour lengths ([[RS1_MAP_VALIDATION_REPORT]]): a median 1.5 / 2.3 m in dense / open clutter, but 6.2–6.9 m in rooms, aisles and corridors. There the way round is through another doorway or aisle, usually behind a wall.

This revision **supersedes candidates 1 and 3** above. Candidate 2 (`yield`) is unchanged. The information rules, the frozen parts and the selection order are as above.

### Candidate 1 (revised): `detour`: follow the walls round a new obstacle, then rejoin the plan

- **Mechanism.** The reset-time A\* path and the frozen `CarrotFollower` stay exactly the baseline's. No planner is called after reset. A behaviour layer overrides γ during a detour.
  - **The new-obstacle map.** Unchanged from candidate 1 above: a 0.1 m grid; a LiDAR hit is *unexplained* when it is farther than 0.15 m from the known map; a cell is a *new static obstacle* once unexplained hits land in it on ≥ **K** of the last 20 steps; cells are never cleared.
  - **Trigger.** A detour starts on either of two triggers:
    - (a) the original path, from the follower's progress to 2 m of arc ahead, passes within 0.45 m of a new cell;
    - (b) *stall* (if enabled): the robot moved less than 0.1 m in the last 30 steps.
  - **The rejoin point q.** It is 0.5 m of arc past the last point where the original path comes within 0.45 m of a new cell. For a stall with no new cell nearby, it is the path point 1.5 m of arc past the follower's progress. The layer records d0 = |p − q| when the detour starts.
  - **The side.** The robot follows the obstacle keeping it on its left or right. It takes the side whose LiDAR rays 30–90° from the direction to q have the larger minimum range.
  - **Wall following.** The "wall" is the nearest LiDAR hit that is explained by the known map or lies in a new cell, so pedestrians are ignored. With n the unit vector from that hit to the robot and t the tangent (n turned 90° towards the chosen side), γ = p + **L**·t + (d − 0.6)·(−n), where d is the distance to the hit. This keeps the robot's centre about 0.6 m from the wall surface. The frozen CBF still filters every step.
  - **Leaving (the Bug2 rule).** The robot leaves the wall when both hold:
    - the straight segment from p to q is clear of the known map and the new cells by 0.35 m, on the layer's own grid;
    - |p − q| < d0.

    γ is then q itself until the robot is within 1.0 m of q. Then γ returns to the follower. The follower's projection searches the whole path and its progress only moves forward, so it continues from q.
  - **Limits.**
    - If a detour has not left the wall after 200 steps (20 s), it flips to the other side once.
    - After 400 steps it gives up and γ returns to the follower.
    - Only one detour runs at a time. A new trigger may start another once one ends.
- **Changed components:** 1. A behaviour layer around the policy (item 2). The planner / carrot follower is **not** changed.
- **Targets:**
  - `timeout_at_block` (321);
  - the collisions while waiting at the block (about 45, an estimate);
  - `timeout_stuck` (36), through the stall trigger;
  - `collision_block` (4).
- **Expected effect: +6 to +14 pp** pooled success.
  - In the clutter families the way round is short and visible. If open_clutter and dense_clutter trigger_block reached their dynamic cells, pooled success would rise by about 6.9 pp ((0.99 − 0.16) + (0.86 − 0.31), divided by 20 cells).
  - In rooms, aisles and corridors the most it could add is 12.5 pp, if every block were cleared. It is discounted because wall following may take the long side and is limited to 500 steps.
- **Risks:**
  - Long wall-following runs end in timeouts.
  - Following walls near moving pedestrians.
  - The stall trigger may start pointless detours in busy cells.
- **Parameters and search (7.3, 8 configurations):** K ∈ {8, 14} × L ∈ {0.6, 1.0} m × stall ∈ {on, off}. **Default:** K = 14, L = 1.0 m, stall on.
- **Compute:** a map update over 24 rays, a nearest-hit search and one line-of-sight check on a 0.1 m grid, well under 1 ms per step. No planner call after reset.

### Candidate 3 (revised): `detour_yield`: the combination

- **Mechanism:**
  - `yield`'s layer overrides γ while a threat exists;
  - otherwise `detour`'s layer does while a detour is active;
  - otherwise the follower's γ is used.
- **Changed components:** 1. Both parts are behaviour layers (item 2), and section 7.4 counts distinct items. So the tie-break against `detour` or `yield` falls to list order.
- **Targets:** all the modes of `detour` and `yield`.
- **Expected effect: +8 to +17 pp** pooled success.
- **Parameters and search (2 configurations):**
  - both defaults;
  - the configurations chosen by the searches of `detour` and `yield`, taken together.
  - It uses the same first 20 RS_DIAG seeds, a third use, disclosed as in 7.3.
- **Compute:** the sum of the two, well under 1 ms per step.

**Revised list, in selection order:** 1 `detour`, 2 `yield`, 3 `detour_yield`. **Tie-break counts:** 1, 1, 1.

## Approval: revised candidate list, 2026-09-20

The researcher approved the revision above (committed at `42fb512`). The candidates are 1 `detour`, 2 `yield` and 3 `detour_yield`, with their grids and defaults as written. Candidates 1 and 3 of the first list (`replan`, `replan_yield`) are withdrawn and will not be built. Phase 4 may start.
