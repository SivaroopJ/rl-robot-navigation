# 09: Baseline diagnostic

**What to build:** The researcher gets the frozen A* + Random baseline's failure breakdown on the diagnostic block of the new suite, the evidence for choosing improvement candidates.

**Blocked by:** 08 (researcher approval)

**Status:** ready-for-agent

- [ ] The baseline runs on the diagnostic block: 20 cells × 2 motion conditions × 50 seeds, plus the M0 anchor, frozen SCS, provenance recorded (commit, manifest, parameter hashes, generator version)
- [ ] Per cell, per family and pooled: success, collision (dynamic/static/wall), timeout, stuck, SPL, minimum clearance, infeasible-step rate, Random event rate and recovery margins
- [ ] Triggered cells are reported overall and conditional on the trigger having fired, with collision timing and location relative to the trigger
- [ ] Rendered trajectories of the most common failure in each cell
- [ ] The report (reports folder) ranks the failure modes by frequency and names the likely mechanism of each
- [ ] The M0 anchor matches the known F-D numbers within sampling noise
