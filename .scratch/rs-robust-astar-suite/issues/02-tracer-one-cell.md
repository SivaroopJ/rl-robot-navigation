# 02: Tracer: one cell end to end

**What to build:** The researcher can generate one open-clutter, static-only episode from a seed and run the frozen A* + Random baseline through it. The result is an episode record with its trace and scenario events. The M0 anchor runs through the same new code path and is bit-identical to the existing arm.

**Blocked by:** 01

**Status:** done (see Comments for the commit)

- [x] The scenario generator returns an episode spec for (open clutter, static only, motion condition, seed): layout, start/goal, and a feasibility verdict with the resample count
- [x] The scenario env subclass installs a spec at reset. The canonical environment files are not edited
- [x] A new RS harness and block runner (the HD harness files are frozen, RS_DESIGN §3) compose the frozen episode loop's pieces around the scenario env and accept a cell. The record carries the existing trace, step times and a scenario-event log; the fingerprint includes the cell
- [x] Seam 1: M0 through the scenario env is bit-identical to the existing `astar_random` arm on a set of validation seeds
- [x] Seam 1: the same seed reproduces the same episode, and both motion conditions share the layout and start/goal
- [x] Seam 1: evaluation refuses any controller other than frozen SCS
- [x] Seam 2: the spec is deterministic given the seed and identical across motion conditions
- [x] Seam 2: a clear path with clearance exists; start and goal are in free space; the distance bins hold
- [x] Seam 2: resampling never returns an infeasible spec, and hitting the cap raises

## Comments

- 2026-09-19. New modules, no frozen file edited (the manifest verifies):
  - `robustsuite/scenario.py`: the generator (seam 2). Entry point `generate(cell, motion, seed)` returns an `EpisodeSpec`, generator version `rs-gen-0.1`. Only open_clutter / static is built; other cells raise `NotImplementedError`.
  - `robustsuite/scenario_env.py`: `RSScenarioEnv(motion)`, a subclass of `RobotNavEnv`. `install(spec)` goes before `reset`; with no spec installed it is M0, unchanged.
  - `robustsuite/harness.py`: seam 1, `run_episode(arm, cell, motion, seed)`. It is HD's evaluation loop, built from the frozen HD and Continuation pieces.
  - `robustsuite/blocks.py`: the block runner. The cell and the generator version are in the fingerprint, and it reuses the HD checkpoint and trace helpers.
  - `robustsuite.seeds.check_motion`.
- **Tests** (32 new, in `tests/test_robustsuite.py`):
  - The M0 anchor is bit-identical to HD `astar_random`: trajectory, trace and every shared record key, on 5 seeds × 2 motions, truncated at 60 steps.
  - Spec determinism, and the same spec under both motions.
  - The family rules, start/goal clearance, bins and the clear path, on 100 draws.
  - Resampling and the cap.
  - Reproducible episodes, the frozen-SCS guard, and the A* map equal to the layout.
  - The block runner: resume, parallel = serial including the trace digest, the cell in the fingerprint.
- **Reading of the cap (RS_DESIGN §4.5, §4.7), to repeat in the map validation report's Deviations and clarifications:**
  - Each layout draw and each failed bin attempt counts as one of the 200 draws.
  - After 4 failed bins on one layout, the layout is redrawn (`BIN_REDRAWS_PER_LAYOUT`).
  - The sampler budgets `RECT_TRIES = 200` and `POINT_BATCH = 4000` are implementation details: exhausting one is a failed draw.
- **For tickets 05–07:**
  - Adding events and pedestrians to the family draw (§4.7) changes which draws are accepted, the static cells included. `GENERATOR_VERSION` must be bumped, and no records from an earlier version may be paired with later ones.
  - In generated cells, the env RNG has already been advanced by the canonical stratified sampler (its result is discarded) before the pedestrian noise starts. This is deterministic.
  - The block must reach the env as a new `static_obstacles` list, not an in-place append, so that the policy's map, taken at build time, never contains it.
- **Two-axis review before the commit.** Fixes:
  - The harness no longer accepts an outside spec. It let a record's cell disagree with its map.
  - An env the harness creates is closed on every path.
  - The M0 map is a public `m0_static`.
  - One `check_motion` replaces three copies.
  - The distance bins are tested equal to `config.json`.
  - The clear-path test covers all 100 draws.
  - The docstring states which checks are the oracle's.
  - Deferred to the entry points (ticket 09): the commit, manifest hash and parameter hashes in result files (§10). The block's `tag` carries the commit.
- **Finding:** all 50 RS_DIAG open_clutter draws were accepted on the first layout (about 0.17 s each). In 6 full baseline episodes, 5 succeeded. The sixth (seed 15 000 005) timed out without moving: its start sits in a 1.03 m gap between the outer wall and a rectangle, which the 1.0 m wall margin of §4.2 allows. It is not a harness fault. The diagnostic should watch for it.
