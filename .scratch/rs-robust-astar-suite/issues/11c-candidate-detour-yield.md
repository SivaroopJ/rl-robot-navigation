# 11c: Candidate 3, `detour_yield`

**What to build:** the combination: `yield`'s layer over `detour`'s layer over the frozen follower.

**Blocked by:** 10 (approved 2026-09-20), 11a, 11b

**Status:** ready-for-agent

Design: `MD_files/robustsuite/RS_DESIGN.md`, "Revision of the candidate list (section 7.4)", candidate 3. Build it exactly as written; any reading of open wording goes into a dated readings section in RS_DESIGN before any tuning run.

- [ ] The candidate is a registered arm (`robustsuite.harness.ARMS`), composed around the frozen `astar_random` stack at runtime; no frozen file is edited (the manifest verifies)
- [ ] It runs through the harness and the block runner on every cell and the M0 anchor
- [ ] Seam 1: actions are bit-identical when `obs[28:]` and the spec's trigger, block and spawn fields are perturbed; the same seed (and configuration) reproduces the episode; the controller is the frozen SCS one; the frozen QP, Random and parameters are unchanged
- [ ] Its default configuration and its search grid are named configurations the harness accepts
- [ ] Its p95 step time is measured on a short sample
- [ ] Tests: yield overrides γ while a threat exists, otherwise the detour's γ while a detour is active, otherwise the follower's; with no threat, no new obstacle and no stall it equals `astar_random` step for step
