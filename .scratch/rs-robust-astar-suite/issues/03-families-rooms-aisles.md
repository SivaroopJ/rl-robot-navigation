# 03: Families: rooms + doorways, warehouse aisles

**What to build:** The researcher can generate and run sensible rooms-and-doorways and warehouse-aisle layouts (static only), each feasible by construction, through the harness with the baseline.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] The rooms family builds 3–4 rooms from a partition, with doorways cut in the walls, widths in the pre-registered range
- [ ] The aisles family builds parallel shelf rows with cross-aisles, widths in the pre-registered range
- [ ] Structural start/goal rules: rooms put start and goal in different rooms; aisles put them at opposite ends or in different aisles
- [ ] Seam 2, on a large seed sample per family: feasibility holds, the structural rule holds, widths are within range, and the spec is deterministic
- [ ] Seam 1: a baseline episode runs in each family, and the A* map equals the generated layout
