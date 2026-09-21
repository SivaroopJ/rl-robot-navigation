# 03: Families: rooms + doorways, warehouse aisles

**What to build:** The researcher can generate and run sensible rooms-and-doorways and warehouse-aisle layouts (static only), each feasible by construction, through the harness with the baseline.

**Blocked by:** 02

**Status:** done (see Comments for the commit)

- [x] The rooms family builds 3–4 rooms from a partition, with doorways cut in the walls, widths in the pre-registered range
- [x] The aisles family builds parallel shelf rows with cross-aisles, widths in the pre-registered range
- [x] Structural start/goal rules: rooms put start and goal in different rooms; aisles put them at opposite ends or in different aisles
- [x] Seam 2, on a large seed sample per family: feasibility holds, the structural rule holds, widths are within range, and the spec is deterministic
- [x] Seam 1: a baseline episode runs in each family, and the A* map equals the generated layout

## Comments

- 2026-09-19. Built together with tickets 04, 05 in one commit. Generator version `rs-gen-0.2`.
  - `robustsuite/families.py`: the five families and their structural start/goal rules.
  - `robustsuite/geometry.py`: exact segment clearance; grid paths on the frozen oracle's own occupancy and moves (the lengths equal `ShortestPathOracle.path_length`); the corridor wall complement.
  - `robustsuite/pedestrians.py`: routes and the motion model.
  - `robustsuite/scenario.py`: the generator, with the start→goal route in the spec.
- **Researcher decisions, recorded in RS_DESIGN §14.1 before any run:**
  - Start and goal clearance is 0.6 m. Closer than that, the frozen controller raises DCPError every step (h < 0) and never moves; at 0.45 m, 20–44% of starts did.
  - Randomized pedestrians: the turn-rate clip bounds the OU noise only; steering is at 4.0 /s towards 1 m ahead.
  - The other readings (exact leg step, corridor construction, aisle outer strips, turn rule) are listed there too.
- **Candidate idea for phase 3, not a decision:** run Random recovery on DCPError steps as well as infeasible ones. The diagnostic should count DCPError freezes per cell, at the start and mid-route.
- **Tests** (50 RS_DIAG draws per family). They re-derive the structure independently from the raw rectangles:
  - rooms: wall lines, doorway gaps of 1.2–1.6 m, 3 or 4 doorways, doorways ≥ 0.5 m from junctions, furniture ≥ 1.0 m from doorway centres, and start and goal separated by a wall line;
  - aisles: row depths, aisle and cross-aisle widths, centring, a mid cross-aisle in both states, and the different-aisle / opposite-end rule;
  - seam 1: a baseline episode runs per family on its own layout, static and dynamic.
- **Review fix:** `_wall_lines` checks were extended to the junction margin, which had not been tested.
