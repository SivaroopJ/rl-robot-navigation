# 01: Pre-registration and scaffolding

**What to build:** The researcher gets a committed pre-registration for RS Experiment 1, written before any code runs. It covers every rule in the spec, with each default pinned to an exact value, and the scaffolding that makes "frozen" and "sealed" verifiable.

**Blocked by:** None (can start immediately)

**Status:** done (see Comments for the commit)

- [x] A `robust-suite` branch is created from `highdim`
- [x] A design document fixes every rule in the spec: the families, obstacle conditions, pedestrian motion, trigger placement, feasibility rules, blocks, selection rule, GO gate, aggregation and phases with their approval points
- [x] The design document pins the defaults to exact values: pedestrian counts per family, passage-width ranges, block sizes, clearance margin, resample cap, and parameter-search budgets for candidates
- [x] ADR 0002 records the decision to test first, with the CBF/DR formulation and dynamics frozen and planning, parameters and perception open to change. It lists the rejected alternatives
- [x] A hash manifest covers the frozen environment, controller, planner, Continuation modules, HD harness pieces and parameter files. A test verifies it
- [x] The diagnostic (50), tuning (50) and sealed final (200) seed blocks are defined at a fresh base. Tests show they are disjoint from every existing block and reserved range, and the sealed block raises unless opened from the final-evaluation entry point
- [x] The baseline test suite is recorded (the known legacy failures only)
- [x] The vault experiment note `rs-robust-astar-suite` is created from the template, with the gate written in it

## Comments

- 2026-09-19. Branch `robust-suite` from `highdim` @ `cc470f0`.
  - Design document: `MD_files/robustsuite/RS_DESIGN.md`.
  - ADR 0002.
  - Manifest: 53 files. The HD harness is frozen, so the RS harness is new (ticket 02 reworded).
  - Seed blocks: RS_DIAG 15 000 000 (50), RS_TUNE 15 100 000 (50), RS_FINAL 15 200 000 (200, sealed to `experiments/robustsuite/rs6_final.py`, checked across the whole repo).
  - Baseline suite: 665 passed plus the 4 known legacy failures.
- **The two-axis review changed the design before commit:**
  - **The corridor family now has a loop.** Without one, a block on the route always sealed the goal off, so every corridors/trigger_block draw hit the cap.
  - **The block must span a passage.** Its free width must be ≤ 3.0 m and bounded on both sides, and it is snapped to an axis, because the environment's obstacles are axis-aligned rectangles.
  - **Crossing spawns** stop 0.45 m before any wall.
  - **Entry paths** are oracle paths.
  - **The spawn is ≥ 1.5 m from the start and ≥ 1.0 m from the goal.**
  - **The block zone is reserved for pedestrians** in all conditions, so they never walk through a block that appears.
  - **The minimum dense-clutter gap is now 1.1 m.** At 1.0 m it left only 2 grid cells of free space.
  - **The generator entropy is now (seed, family).** All four obstacle conditions share one family draw, so a triggered cell pairs with its +dynamic cell seed by seed.
  - **The parameter-search fallback** (the default configuration goes forward) and a countable definition of "changed components" were added.
  - **Package `__init__` files and the reservation sources** were added to the manifest.
  - **The sealed-block scan now covers the whole repo.**
  - Departures from the spec's wording are listed in RS_DESIGN §14.
