# Week 6 / Generalization Suite — implementation notes

Implements `MAP_SUITE_SPEC.md` as approved. **No experiment has been run beyond the validation
gate.** The 118-cell screening run has **not** been started.

## Files added

| file | role |
|---|---|
| `generalization/__init__.py` | package marker |
| `generalization/maps.py` | pure config generators, one per family, plus the F1′ staircase |
| `generalization/scenario_env.py` | `RobotNavEnv` **subclass**: fixed start/goal, reactive walls, scripted motion |
| `generalization/suite.py` | the declarative manifest — 64 configurations, seeds, selection rule |
| `experiments/exp9_generalization.py` | runner: `--mode g0 / smoke / screen / final` |
| `tests/test_generalization_maps.py` | 31 tests over generation, fidelity, fairness, leakage |
| `MD_files/week6/generalization/` | this file and the spec |
| `results/week6_generalization/` | output tree, written per cell, never overwritten |

**Nothing else changed.** `git status` shows only untracked additions; `robot_env/`,
`evaluation/`, `dr_control/`, `optional_navigation/` and every Phase 1–7 artefact are untouched,
and the frozen sha256 manifest verifies clean over all 60 files.

## How each constraint is met

**F11 / moving goals — not implemented.** `scenario_env.py` has no goal-trajectory code and the
tests assert no manifest entry declares one. The §1.3 finding — both arms fix the goal and the A*
path at `reset()`, so a moving goal is invisible to them — remains a documented
boundary-of-applicability limitation, established by code inspection rather than by experiment.

**F1′ — staircase, not real rotation.** A cell is occupied when its centre lies inside the true
rotated rectangle; occupied cells are run-length merged along x. Every static obstacle in every
F1′ map is still a 4-tuple `(cx, cy, hw, hh)` — a test asserts none gained a fifth element,
because that would be real rotation and would need the frozen geometry code to change. Measured
fidelity, recorded per map and reported beside every result:

| angle | cells | rects after merge | area ratio | Hausdorff |
|---|---|---|---|---|
| 15° | 301 | 32 | 1.0033 | 0.0860 |
| 30° | 300 | 31 | 1.0000 | 0.1041 |
| 45° | 301 | 28 | 1.0033 | 0.0925 |
| 60° | 300 | 23 | 1.0000 | 0.0844 |
| 90° | 300 | 10 | 1.0000 | **0.0000** |

90° is exactly representable and comes out exact, which is the check that the construction is
sound rather than merely close. All Hausdorff distances sit under the `cell·√2 = 0.1414` bound.

**F10 — no replanning, either arm.** A* runs once at policy reset, from the base static map;
`CarrotFollower.progress` is monotone. A wall activates when the robot enters a declared trigger
disc and is then visible to both arms through LiDAR only. `ScenarioEnv.step()` re-reads the
observation on the activation step so the timing is identical for both. Verified: the wall fires
exactly once, at a recorded distance, and wall state resets between episodes.

**`v_cap = 0.96` fixed everywhere.** The runner passes `V_CAP_DERIVED` and nothing else; a test
parses the runner's AST with docstrings stripped and asserts no literal `v_cap=0.…` survives in
executable code.

**LiDAR range 5.0 m at every world size.** Asserted for all 64 configurations, alongside
`max_speed = 1.0`, 24 rays and `agent_radius = 0.3`.

**Fairness.** `run_cell` runs both arms on the same seed and asserts identical start, goal,
initial obstacle states, horizon, world size and static-geometry count per episode; a mismatch
raises rather than warns. A leakage test corrupts `obs[28:52]` and asserts both arms return
bit-identical actions.

**Counts.** `suite.budget()` is computed from the manifest, not asserted:
64 configurations · 118 screening cells · 111 final cells (101 benchmark + 10 stress) ·
11 800 + 41 400 = **53 200 episodes**. A test pins the whole dictionary.

## Two bugs the tests caught before any run

1. **`ScriptedMotion` crashed on a single waypoint.** X2 and X3 declare `waypoints=[[5.0, 5.0]]`
   — a destination, not a patrol — and the ping-pong index walked off the array. Fixed: a
   one-waypoint spec now holds position on arrival, which is what "close the passage and stay
   closed" requires.
2. **The `v_cap` test read the module docstring.** The docstring legitimately states
   `v_cap = 0.96`; the test now scans executable code with docstrings stripped, as the Stage-5
   leakage tests do.

## Decisions that needed a concrete choice

* **F1′'s eight maps** are five single-obstacle rotations (obstacle 0 at 15/30/45/60/90) plus
  three all-obstacle rotations (30/45/90). The spec gave the angle set and the count 8; this is
  the concrete reading, and it uses every declared angle.
* **F8c** was originally "goal in a narrow corridor" at `[3.75, 7.5]`. That point is inside the
  inflated canonical rectangle — the gap there is 0.5 m wide against a 0.6 m inflated robot, so
  no valid goal exists in it. Replaced with `F8c_goal_confined_slot`, goal `[6.0, 7.0]`, which is
  a confined pocket reached by a six-waypoint route: clearance 0.70 m, A*-feasible. Narrow
  corridors proper are F9's job. All five F8 pairs are verified collision-free and A*-feasible by
  test, not by inspection.
* **F2b's extra rectangles** for the D+ levels are declared constants in `maps.EXTRA_RECTS`, not
  sampled, so the maps are reproducible without a generation seed.

## Not yet done

Screening has not started. The runtime projection in the spec (5.65 s/episode) is still the
proposal's estimate; the runner recomputes it from its own measured rate and reports the revised
projection before the final tier.
