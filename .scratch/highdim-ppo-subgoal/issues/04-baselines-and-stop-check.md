# 04: Run floor then ceiling on HD_DEV and HD_FINAL, then apply the stop check

**What to build:** The researcher gets paired floor and ceiling results on the fresh M0 blocks, plus a written, pre-registered stop verdict saying whether A* contributes enough on M0 to justify training PPO.

**Blocked by:** 01 (Pre-registration), 03 (Ceiling arm)

**Status:** ready-for-agent

- [ ] Run order, as pre-registered, using the frozen SCS controller and 2 conditions throughout:
  1. Floor on HD_DEV (100 × 2)
  2. Floor on HD_FINAL (200 × 2)
  3. Ceiling on the same HD_FINAL seeds
  4. Ceiling on HD_DEV
- [ ] Results files record the commit, manifest hash and frozen-parameter hashes
- [ ] The report gives success, collision (dynamic/static/wall), timeout, stuck rate, min clearance, SPL and infeasible-step rate for each arm and condition
- [ ] The ceiling vs floor comparison is paired (McNemar and bootstrap CI via the existing statistics functions)
- [ ] A written stop verdict: pooled S_A* − S_floor < 0.05 → STOP, otherwise proceed
- [ ] The historical F-D numbers are cited as context only
- [ ] The vault experiment note is updated with the baseline numbers and the verdict
- [ ] If the verdict is STOP, tickets 05–10 are marked `wontfix` with a pointer to the verdict
