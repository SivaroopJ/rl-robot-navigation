# 04: Families: corridor network, dense clutter

**What to build:** The researcher can generate and run sensible corridor-network and dense-clutter layouts (static only), each feasible by construction, through the harness with the baseline.

**Blocked by:** 02

**Status:** done (see Comments for the commit)

- [x] The corridors family builds a graph of L- and T-shaped corridors, widths in the pre-registered range
- [x] The dense-clutter family builds many small non-overlapping obstacles, with gaps no smaller than the pre-registered minimum
- [x] Structural start/goal rules: corridors need at least one turn between start and goal; dense clutter uses the distance bins only
- [x] Seam 2, on a large seed sample per family: feasibility holds, the structural rule holds, widths and gaps are within range, and the spec is deterministic
- [x] Seam 1: a baseline episode runs in each family

## Comments

- 2026-09-19. Built together with tickets 03, 05 in one commit. Generator version `rs-gen-0.2`.
  - `robustsuite/families.py`: the five families and their structural start/goal rules.
  - `robustsuite/geometry.py`: exact segment clearance; grid paths on the frozen oracle's own occupancy and moves (the lengths equal `ShortestPathOracle.path_length`); the corridor wall complement.
  - `robustsuite/pedestrians.py`: routes and the motion model.
  - `robustsuite/scenario.py`: the generator, with the start→goal route in the spec.
- **Researcher decisions, recorded in RS_DESIGN §14.1 before any run:**
  - Start and goal clearance is 0.6 m. Closer than that, the frozen controller raises DCPError every step (h < 0) and never moves; at 0.45 m, 20–44% of starts did.
  - Randomized pedestrians: the turn-rate clip bounds the OU noise only; steering is at 4.0 /s towards 1 m ahead.
  - The other readings (exact leg step, corridor construction, aisle outer strips, turn rule) are listed there too.
- **Candidate idea for phase 3, not a decision:** run Random recovery on DCPError steps as well as infeasible ones. The diagnostic should count DCPError freezes per cell, at the start and mid-route.
- **Corridors are built constructively.** Before the review fix, 3-branch networks never succeeded (0 of 1462 attempts), so the dead-end L never appeared. Now every draw is accepted, split 50/50 between 2 and 3 branches.
- **Tests:**
  - strip widths of 1.5–2.0 m;
  - the complement is exactly the walls (point sampling);
  - topology: every branch meets the spine; the connector meets exactly two branches; one stub; with 3 branches, the stub is a dead end off the free branch; both branch counts occur;
  - the route turns ≥ 60° over 1 m chords (an independent check at 2 cm steps);
  - dense clutter: counts, half-extents, 1.1 m gaps and the 0.8 m wall margin.
- **Review fix:** a float sliver (a wall of width 2e-16) was removed by snapping strip edges.
