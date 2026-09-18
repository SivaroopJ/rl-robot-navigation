# 02: Floor arm tracer: goal-only + Random through the episode harness

**What to build:** The researcher can run the floor arm (goal-only + Random-CLF-DR-CBF: planning off, no static map, γ = goal) for any condition and seed through a new episode harness. The harness returns a complete episode record with per-step diagnostics. The new seed blocks exist and HD_FINAL is sealed.

**Blocked by:** 01 (Pre-registration)

**Status:** ready-for-agent

- [ ] HD seed blocks exist and are asserted disjoint from every Continuation block and forbidden range, and from the 11M, 12M and 13M blocks
- [ ] HD_FINAL raises unless opened from the two permitted entry points (baseline runs and final evaluation), following the Continuation FINAL-block pattern
- [ ] The harness runs the floor arm on the canonical M0 env (6 obstacles at 0.675 m/s, fixed or randomized) with the frozen tuned parameters and frozen Random spec
- [ ] The per-episode Random RNG is the existing controller RNG derived from the episode seed
- [ ] The episode record reuses the existing episode-record, true-clearance and step-recorder functions
- [ ] The episode record carries per-step γ, V, u_nom, u_exec, QP status, Random events and min CBC
- [ ] Seam-1 test: γ equals the goal at every step
- [ ] Seam-1 test: the planner's path method, patched to raise, is never called
- [ ] Seam-1 test: replacing the static map handed to the policy leaves every action identical
- [ ] Seam-1 test: the same seed reproduces an identical episode
- [ ] Seam-1 test: the controller used in evaluation is the frozen SCS class, and the fast solver is refused
- [ ] No frozen file is edited, and the manifest test from 01 still passes
