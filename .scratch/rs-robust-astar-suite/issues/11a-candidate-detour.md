# 11a: Candidate 1, `detour`

**What to build:** the `detour` arm: a LiDAR map of new static obstacles and a Bug2-style wall-following layer that rejoins the reset-time A* path. No planner call after reset.

**Blocked by:** 10 (approved 2026-09-20, revised and re-approved the same day)

**Status:** ready-for-agent

Design: `MD_files/robustsuite/RS_DESIGN.md`, "Revision of the candidate list (section 7.4)", candidate 1. Build it exactly as written; any reading of open wording goes into a dated readings section in RS_DESIGN before any tuning run.

- [ ] The candidate is a registered arm (`robustsuite.harness.ARMS`), composed around the frozen `astar_random` stack at runtime; no frozen file is edited (the manifest verifies); the planner is called only at reset (tested)
- [ ] It runs through the harness and the block runner on every cell and the M0 anchor
- [ ] Seam 1: actions are bit-identical when `obs[28:]` and the spec's trigger, block and spawn fields are perturbed; the same seed (and configuration) reproduces the episode; the controller is the frozen SCS one; the frozen QP, Random and parameters are unchanged
- [ ] Its default configuration and its search grid are named configurations the harness accepts
- [ ] Its p95 step time is measured on a short sample
- [ ] Unit tests at the layer's seam: a fired block becomes new cells within K steps and starts a detour; the rejoin point is past the new cells on the original path; the side with more LiDAR room is chosen; wall following holds about 0.6 m and ignores pedestrian hits; the Bug2 leave rule (clear line of sight and closer than d0); the side flip at 200 steps and give-up at 400; with no new obstacle and no stall, actions equal `astar_random` step for step
- [ ] Pedestrian-only replays of RS_DIAG: phantom new cells at K counted and reported
