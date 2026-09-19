# 04: Families: corridor network, dense clutter

**What to build:** The researcher can generate and run sensible corridor-network and dense-clutter layouts (static only), each feasible by construction, through the harness with the baseline.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] The corridors family builds a graph of L- and T-shaped corridors, widths in the pre-registered range
- [ ] The dense-clutter family builds many small non-overlapping obstacles, with gaps no smaller than the pre-registered minimum
- [ ] Structural start/goal rules: corridors need at least one turn between start and goal; dense clutter uses the distance bins only
- [ ] Seam 2, on a large seed sample per family: feasibility holds, the structural rule holds, widths and gaps are within range, and the spec is deterministic
- [ ] Seam 1: a baseline episode runs in each family
