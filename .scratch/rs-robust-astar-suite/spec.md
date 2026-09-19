Status: ready-for-agent

# RS Experiment 1: robust A*-Random on a fresh map suite

## Problem Statement

The validated safe-navigation stack is A* + Random-CLF-DR-CBF: a once-per-episode A* plan on the known static map, a carrot follower that feeds the CLF reference γ, the DR-CBF QP, and Random recovery. It has only been measured seriously on M0, a single fixed map with five rectangles and six dynamic obstacles that pass straight through walls. On M0 it reaches 0.9025 pooled success, and almost all of its failures are dynamic collisions.

The researcher does not know how the stack behaves in layouts that look like real spaces (rooms, aisles, corridors, clutter). Nor do they know how it behaves when the world changes during an episode: a passage becomes blocked, or a person steps out from behind a shelf.

Several structural weaknesses are known from code inspection:
- A* never replans, so the carrot keeps pointing through a newly blocked passage.
- Random recovery rarely finds a positive-margin direction.
- The dynamic obstacles used so far are not physically plausible in structured maps.

Earlier improvement attempts (parameter tuning, the Random perturbation, persistence, the risk supervisor, congestion detection) changed no outcome on M0. They were chosen without a failure analysis in varied environments.

The researcher wants:
1. an honest characterisation of the frozen stack on a fresh, feasible and sensible map suite with static, dynamic and region-triggered obstacles;
2. improvements chosen from the failures that suite reveals, tested against the frozen stack under pre-registered rules.

## Solution

One pre-registered experiment, run **test first**.

- **Build the suite.** Build a fresh procedural map suite: five layout families crossed with four obstacle conditions, so 20 cells, each run under two pedestrian motion conditions. Every episode is generated from its seed and must pass feasibility checks in code. The researcher approves a map validation report before any run.
- **Characterise the baseline.** Run the frozen A* + Random stack on a diagnostic block of the suite, plus the M0 anchor, to get a failure breakdown.
- **Choose candidates.** From that breakdown, choose at most three improvement candidates, each written down and approved before it is built. Candidates may change planning, parameters (within a limited search), perception and prediction. They may not change the CBF/DR formulation or the robot dynamics, and they may use onboard information only.
- **Select a winner.** Evaluate baseline and candidates on a separate tuning block and select at most one winner by a fixed rule.
- **Run the sealed comparison.** Run the baseline, and the winner if there is one, once on a sealed final block. Apply the GO gate:
  - success higher;
  - collisions not higher;
  - M0 not worse.
- **If no candidate qualifies,** the sealed run still produces the baseline's characterisation of the suite.

## User Stories

### Suite: layouts, obstacles, feasibility

1. As the researcher, I want five procedural layout families (open clutter, rooms + doorways, warehouse aisles, corridor network, dense clutter), so that the stack is tested in layouts that resemble real spaces.
2. As the researcher, I want every seed to draw a new layout from its family, so that the sealed block contains layouts no improvement was tuned on.
3. As the researcher, I want every family to generate sensible structure (walls meeting walls, doorways inside walls, aisles parallel), so that no episode is a random scatter pretending to be a building.
4. As the researcher, I want all cells to keep M0's physical settings (10 × 10 m world, robot radius 0.3 m, 1 m/s maximum speed, dt 0.1 s, 500-step limit), so that the frozen controller parameters stay inside the regime they were tuned for.
5. As the researcher, I want four obstacle conditions (static only; + dynamic; + triggered static block; + triggered dynamic spawn), crossed full-factorially with the five families, so that a failure can be attributed to the layout or to the event.
6. As the researcher, I want dynamic obstacles to behave like pedestrians that walk routes through free space between randomly drawn waypoints at 0.675 m/s, so that they never pass through walls or shelves.
7. As the researcher, I want the "fixed" motion condition to follow the pedestrian routes exactly and the "randomized" condition to add OU heading noise within the free space, so that both earlier motion conditions have structured-map equivalents.
8. As the researcher, I want the number of pedestrians set by family density (between 3 and 6), with the speed and radius kept at M0's values, so that density varies with layout but speed stress stays out of this experiment.
9. As the researcher, I want exactly one triggered event per episode in the triggered cells, so that event effects stay attributable.
10. As the researcher, I want the trigger region placed on the robot's planned A* route at a point drawn between 30% and 70% of the route length, so that the event happens mid-task and on the robot's way.
11. As the researcher, I want a triggered static block to appear 1.5–2.5 m further along the route and span the passage (a doorway, aisle or corridor), so that it forces a detour exactly where the static plan is committed.
12. As the researcher, I want a triggered dynamic spawn to appear behind the nearest occluding obstacle, 1.5–3 m from the trigger, and then either cross the planned route or come at the robot head-on (two variants), so that "someone steps out" and "someone comes round the corner" are both covered.
13. As the researcher, I want spawned pedestrians to continue as normal pedestrians after their scripted entry, so that the scene stays plausible.
14. As the researcher, I want no pursuing or adversarial obstacles, so that the task is hard but not adversarial.
15. As the researcher, I want every episode to satisfy feasibility rules checked in code, and to be resampled otherwise, so that failure rates are never inflated by unsolvable episodes.
16. As the researcher, I want a clear path (keeping at least robot radius + a margin from obstacles) to exist from start to goal on the static layout, so that every episode is solvable.
17. As the researcher, I want a clear path to still exist with the triggered block in place, so that the block forces a detour but never seals the goal off.
18. As the researcher, I want a triggered spawn never placed inside, or unavoidably close to, the robot's footprint (at least 1.5 m, beyond braking plus reaction distance at 1 m/s), so that no collision is physically forced.
19. As the researcher, I want pedestrians never faster than the robot and never spawned on the start or goal, so that the scene is fair.
20. As the researcher, I want start and goal drawn with M0's stratified distance bins (at least 4 m apart), so that trip lengths are comparable to earlier work.
21. As the researcher, I want one structural start/goal rule per family (different rooms; opposite aisle ends or different aisles; at least one corridor turn; bins only for the clutter families), so that every episode actually exercises its layout.
22. As the researcher, I want a map validation report, with acceptance and resampling rates, clearance distributions, passage-width distributions, trigger placement statistics and rendered samples of every cell, so that I can confirm the suite is feasible and sensible before any run.
23. As the researcher, I want to approve the map validation report before phase 2 starts, so that no result is produced on a suite I haven't accepted.
24. As the researcher, I want M0 kept as an anchor cell, with its original map and original motion, outside the pooled gate, so that new results connect to every earlier number.

### Pairing, seeds and determinism

25. As the researcher, I want the layout, start/goal, pedestrian routes and trigger derived from the episode seed and shared by both motion conditions and every arm, so that every comparison is paired.
26. As the researcher, I want the same seed (and candidate configuration) to reproduce an identical episode, so that reruns and paired statistics are exact.
27. As the researcher, I want fresh seed blocks (diagnostic, tuning, sealed final), disjoint from every existing block and reserved range, so that no earlier data is reused.
28. As the researcher, I want the sealed final block openable only from the final-evaluation entry point, so that it cannot be touched during development or selection.
29. As the researcher, I want the diagnostic and tuning blocks to be separate, so that the seeds used to find failures are not the seeds used to pick the winner.

### Baseline characterisation

30. As the researcher, I want the frozen A* + Random stack (bit-identical to the Week 6 Continuation Random arm) as the baseline, so that "frozen" is provable.
31. As the researcher, I want M0 run through the new scenario code path to be bit-identical to the existing arm on M0, so that the new environment code is proven not to change the canonical environment.
32. As the researcher, I want triggered blocks to never enter A*'s map, so that the baseline, like the robot, learns about them only through LiDAR.
33. As the researcher, I want the baseline run on the diagnostic block (20 cells × 2 motion × 50 seeds, plus the anchor), so that improvement candidates target observed failures.
34. As the researcher, I want the failure breakdown to include success, collision (dynamic/static/wall), timeout, stuck rate, SPL, minimum clearance, infeasible-step rate, Random event rate, recovery margins, and collision timing and location relative to the trigger, so that the candidate choice is grounded.
35. As the researcher, I want every triggered-cell result reported both overall and conditional on the trigger having fired, so that episodes that end before the event are not mistaken for event handling.
36. As the researcher, I want rendered example trajectories of the most common failure in each cell, so that I can see the failure, not just count it.

### Improvement candidates

37. As the researcher, I want at most three candidates, each written down (mechanism, components changed, information used, expected effect on the observed failures) and approved by me before it is built, so that the candidate set is fixed before any tuning data is seen.
38. As the researcher, I want candidates allowed to change planning (e.g. replanning when blocked, Wait or Retreat behaviours), parameters within a pre-registered limited search, and perception/prediction, so that the likely failure causes can be addressed.
39. As the researcher, I want the CBF/DR formulation and the robot dynamics kept frozen for every candidate, so that the validated safety core is not re-litigated.
40. As the researcher, I want a combination of mechanisms allowed only as one of the three candidates, so that combinations don't multiply the comparisons.
41. As the researcher, I want candidates to use onboard information only (LiDAR, own pose and velocity, the known static map without triggered blocks, their own history), never the ground-truth obstacle block or any trigger or spawn information, so that improvements are deployable.
42. As the researcher, I want a candidate disqualified if its p95 step time on the tuning block exceeds 100 ms, so that every winner runs in real time at 10 Hz.
43. As the researcher, I want any parameter search inside a candidate to have a pre-registered budget and to use only the tuning block, so that the Week 6 winner's curse is not repeated.

### Selection and gate

44. As the researcher, I want the baseline and every candidate run on the tuning block (20 cells × 2 motion × 50 seeds), so that selection is paired and independent of the diagnostic.
45. As the researcher, I want a candidate to qualify only if its pooled tuning success beats the baseline's by at least 2 pp and its pooled collision rate is no more than 1 pp higher, so that tuning noise does not produce a winner.
46. As the researcher, I want the winner to be the qualifying candidate with the highest pooled success, with ties going to the simpler candidate (fewer changed components), so that selection is mechanical.
47. As the researcher, I want the experiment to report "no improvement found" and still run the baseline on the sealed block when no candidate qualifies, so that the suite always produces a characterisation result.
48. As the researcher, I want the sealed final run (20 cells × 2 motion × 200 seeds, plus the M0 anchor) for the baseline and the winner, run once, so that the comparison uses unseen layouts.
49. As the researcher, I want GO to require: pooled success higher with exact McNemar p < 0.05; pooled collisions not significantly higher (fail if higher with p < 0.05); and M0 success not significantly lower (p < 0.05), so that an improvement is neither a safety trade nor a regression of the validated result.
50. As the researcher, I want pooled rates to weight all 20 cells equally and pool both motion conditions, so that no family dominates the gate.
51. As the researcher, I want per-cell regressions reported but not gated, with a multiple-comparison note, so that cell-level signals inform without false alarms deciding the verdict.

### Reporting and records

52. As the researcher, I want every result file to record the commit, frozen-file manifest hash, frozen-parameter hashes, suite-generator version and seed block, so that every number is traceable.
53. As the researcher, I want a hash manifest proving that the frozen environment, controller, planner, Continuation code and parameter files are unchanged, so that "frozen" is verifiable.
54. As the researcher, I want the design document, with every rule in this spec, committed before any run, and later changes recorded only in a dated Deviations section of the results report, so that the experiment is pre-registered.
55. As the researcher, I want the final report to show the per-cell arm table, the pooled gate, per-family summaries, worst-cell success, triggered-event analyses and the diagnostic distributions, so that the result can be understood beyond the verdict.
56. As the researcher, I want the vault experiment note created with the gate written before the first run, and updated at each phase, so that the tracking vault stays the living record.

## Implementation Decisions

**Categories.** Each change is one of three kinds:
- *Frozen:* not edited, and hash-checked.
- *New:* new modules and entry points.
- *Candidate:* a runtime composition of new components around the frozen stack, approved in phase 3.

*Frozen:*
- the canonical environment: dynamics, collision logic, LiDAR, observation, reward;
- the A* planner and carrot follower;
- the CLF-DR-CBF QP and its tuned parameters;
- Random recovery and its frozen specification;
- the Continuation modules the baseline arm uses;
- the HD harness pieces reused unchanged.

**Scenario generator (new; seam 2).** A deep module with a single entry point that takes (family, obstacle condition, motion condition, seed) and returns an **episode spec**. The spec contains:
- the static layout as rectangles;
- start and goal;
- pedestrian routes (waypoint sequences and speeds);
- for triggered cells, the trigger region and either the block rectangle or the spawn start and entry path;
- the feasibility verdict, with the resample count.

It is deterministic given the seed. The motion condition affects only the pedestrians' noise parameter, never the drawn geometry, routes or trigger, so both motion conditions share one spec. Each family is a small grammar:
- rooms from a partition with doorways cut in walls;
- aisles as parallel shelf rows with cross-aisles;
- corridors as a graph of L/T segments;
- open and dense clutter as non-overlapping rectangles with a minimum gap.

Passage widths are drawn from pre-registered ranges: doorways about 1.2–1.6 m, aisles about 1.4–2.0 m, corridors about 1.5–2.0 m, dense-clutter gaps down to about 1.0 m.

**Feasibility checks (inside the generator).**
- A grid planner with clearance (robot radius + margin) must find a path on the static layout.
- With a triggered block, a path must still exist with the block added.
- A spawn must be at least 1.5 m from the robot's position when it fires.
- Pedestrian speed must be at most the robot's maximum, with no spawn on the start or goal.
- Start and goal must be in free space with clearance, and satisfy the family's structural rule.

A failing draw is resampled deterministically from the same seed stream, up to a cap. If the cap is reached, the generator raises rather than returning an infeasible episode.

**Trigger placement.** The trigger is placed on the baseline A* route computed on the static layout at generation time (arm-independent), at a fraction drawn between 0.3 and 0.7 of the route length. Its location is part of the episode spec, not of any policy.

**Pedestrian motion model (new).** Same interface as the frozen motion models. Each pedestrian walks a free-space path between waypoints at 0.675 m/s and draws its next waypoint from the env's own RNG. "Fixed" follows the path exactly; "randomized" adds OU heading noise, with steering back towards the path so it never enters walls. Pedestrians may pass through each other.

**Scenario environment (new).** A subclass of the canonical environment, following the Week 6 scenario-env pattern:
- it installs the episode spec's layout, start/goal and pedestrian motion at reset;
- it fires the trigger when the robot enters the region, adding the block to the static obstacles (so the LiDAR and collision logic see it) or spawning the pedestrian;
- it logs scenario events: trigger fired (step and robot position), block added, spawn position and variant.

The known static map handed to A* is the layout without the triggered block. The M0 anchor goes through the same class with its original map and motion, and must be bit-identical to the canonical environment.

**Harness (seam 1, extended).** The HD episode harness gains a cell dimension: (arm, cell, motion condition, seed, optional candidate configuration) → episode record. The record carries the existing per-step trace plus step times and the scenario events. The block runner (parallel, fingerprinted, resumable, trace sidecars) is reused with the cell in the fingerprint. Evaluation always uses the frozen SCS controller.

**Arms.**
- `astar_random` is the frozen baseline, built exactly as in HD Experiment 1.
- Candidate arms are registered by name in phase 4. Each is a composition around the frozen stack (e.g. a replanning follower that re-runs A* on the known map plus a LiDAR-built occupancy of new static obstacles; a Wait/Retreat behaviour layer; a prediction-aware barrier source), with its declared components and information sources.
- The CBF/DR formulation and dynamics are never replaced.

**Information boundary.** Candidates receive the same inputs the frozen policy receives, plus their own state. Anything derived from the ground-truth obstacle block, the episode spec's trigger or spawn fields, or the scenario event log is unavailable to them.

**Seed blocks (new).** Three disjoint blocks at a fresh base (15 000 000 and up), asserted disjoint from every Continuation, HD, pursuit and Track block and forbidden range:
- a diagnostic block of 50 seeds;
- a tuning block of 50 seeds;
- a sealed final block of 200 seeds.

Every cell uses the same block of seed indices, with a cell-specific derivation, so cells draw independent layouts. The sealed block opens only from the final-evaluation entry point.

**Decision rules (new, pure functions).**
- The selection rule: qualify at ≥ +2 pp success and ≤ +1 pp collision, pooled with equal cell weights; highest success wins; ties go to the simpler candidate; no qualifier means no winner.
- The GO gate: success McNemar p < 0.05 and higher; collision not significantly higher; M0 not significantly lower.
- Exact counts are compared in rational arithmetic, as `highdim.gates` does. McNemar and bootstrap come from the existing paired statistics.

**Phases and approval points.**

| Phase | Work | Approval / rule |
|---|---|---|
| 0 | Design document, frozen-file manifest, seed blocks, vault note | committed before any run |
| 1 | Generator, pedestrian model, scenario env, map validation report | **researcher approves the report** |
| 2 | Baseline diagnostic | failure breakdown written |
| 3 | Candidate list | **researcher approves it** |
| 4 | Candidate implementation | tests pass |
| 5 | Tuning-block selection | selection rule |
| 6 | Sealed final | GO gate |

**Git and documents.**
- Branch `robust-suite` from `highdim`.
- Directory names use `robustsuite`.
- The design document and reports go in the reports folder under a `robustsuite` subfolder.
- Tickets live in this feature folder; the vault note is `rs-robust-astar-suite`.
- An ADR records the decision to test first and to keep the CBF/DR formulation frozen while planning and perception may change.

## Testing Decisions

A good test checks behaviour visible at a seam: what a generated episode spec contains, and what an episode returns and records. It does not test how modules are wired. Tests stay fast by truncating episodes and generating specs without running them.

**Seam 1: the episode harness** (arm, cell, motion condition, seed, candidate → episode record):
- M0 through the scenario env is bit-identical to the existing `astar_random` arm on a set of validation seeds.
- The same seed (and candidate configuration) reproduces an identical episode for every arm.
- Paired arms and both motion conditions face the same layout, start/goal, pedestrian routes and trigger.
- Candidate actions are bit-identical when the ground-truth obstacle block is perturbed, and when the trigger or spawn fields they must not see are perturbed.
- The A* map never contains a triggered block.
- A fired static block is visible to LiDAR and to collision checks from the firing step on.
- Scenario events are recorded exactly when the robot enters the trigger region.
- Evaluation refuses any controller other than frozen SCS.
- The frozen-file manifest verifies.

**Seam 2: the scenario generator** (family, obstacle condition, motion condition, seed → episode spec):
- Determinism given the seed.
- An identical spec across motion conditions.
- Every feasibility rule, on a large sample of seeds per cell:
  - the clear path exists;
  - it still exists with the block added;
  - spawn distance is at least 1.5 m;
  - there is no spawn on the start or goal;
  - start and goal clearance;
  - each family's structural start/goal rule.
- Passage widths within their ranges.
- Trigger placed within 30–70% of the route.
- Resampling never returns an infeasible spec, and the cap raises.

**Seed blocks and rules:**
- The blocks are disjoint from all existing blocks and reserved ranges.
- The sealed block raises unless opened from the permitted entry point.
- The selection rule and the gate are tested on constructed records at their thresholds, as the HD gate tests do.

**Prior art:**
- the HD tests (the frozen-manifest gate, the sealed block, paired runs, the ground-truth-block invariance test, the gate functions);
- the Week 6 generalization tests for the scenario env (reactive walls, scripted motion);
- the Continuation frozen-equivalence tests.

## Out of Scope

- Changing the CBF/DR formulation, the CLF, the robot dynamics, the reward, the LiDAR model or the observation of the canonical environment.
- Learned components (PPO or other RL) as candidates. HD Experiment 1 closed that direction for now.
- Reusing the Week 6 generalization map suite or its results, apart from the scenario-env pattern.
- Adversarial or pursuing obstacles, varying obstacle speed, varying robot speed or world size.
- More than one triggered event per episode, and moving goals.
- Multi-agent settings and multi-goal episodes.
- Using ground-truth obstacle information, trigger or spawn information, or a map that includes triggered blocks, in any candidate.
- More than three candidates, and any candidate added after the tuning block has been seen.

## Further Notes

- **Motivation from earlier results.**
  - The A* + Random arm's M0 failures are mostly dynamic collisions, clustered early.
  - Random recovery finds a positive-margin direction only about 12% of the time.
  - In the Week 6 suite, a wall appearing across the route (F10c) dropped a closely related arm to 0.03–0.12 success. That is the expected weakness of planning once per episode, and the strongest prior for a replanning candidate.
  - The Track B Wait/Retreat/Reroute candidates were specified but never built.
- **Cost estimate.** Episodes take about 8 s on frozen SCS. With 14 workers: diagnostic about 0.3 h, tuning with 3 candidates about 1.3 h, sealed final about 2.5 h. Candidate compute (replanning, prediction) adds to this and is bounded by the 100 ms p95 rule.
- **Existing dynamic obstacles pass through static rectangles** and each other; only the outer walls reflect them. This is why structured families need the new pedestrian model while M0 keeps its original motion.
- **Defaults for the design document to fix precisely:** pedestrian counts per family, passage-width ranges, block sizes, clearance margin, resampling cap, candidate parameter-search budgets.
