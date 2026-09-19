# 02: Tracer: one cell end to end

**What to build:** The researcher can generate one open-clutter, static-only episode from a seed and run the frozen A* + Random baseline through it. The result is an episode record with its trace and scenario events. The M0 anchor runs through the same new code path and is bit-identical to the existing arm.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] The scenario generator returns an episode spec for (open clutter, static only, motion condition, seed): layout, start/goal, and a feasibility verdict with the resample count
- [ ] The scenario env subclass installs a spec at reset. The canonical environment files are not edited
- [ ] A new RS harness and block runner (the HD harness files are frozen, RS_DESIGN §3) compose the frozen episode loop's pieces around the scenario env and accept a cell. The record carries the existing trace, step times and a scenario-event log; the fingerprint includes the cell
- [ ] Seam 1: M0 through the scenario env is bit-identical to the existing `astar_random` arm on a set of validation seeds
- [ ] Seam 1: the same seed reproduces the same episode, and both motion conditions share the layout and start/goal
- [ ] Seam 1: evaluation refuses any controller other than frozen SCS
- [ ] Seam 2: the spec is deterministic given the seed and identical across motion conditions
- [ ] Seam 2: a clear path with clearance exists; start and goal are in free space; the distance bins hold
- [ ] Seam 2: resampling never returns an infeasible spec, and hitting the cap raises
