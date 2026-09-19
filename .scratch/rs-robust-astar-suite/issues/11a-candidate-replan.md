# 11a: Candidate 1, `replan`

**What to build:** the `replan` arm: a LiDAR map of new static obstacles and A* replanning on a blocked path or a stall.

**Blocked by:** 10 (approved 2026-09-20)

**Status:** ready-for-agent

Design: `MD_files/robustsuite/RS_DESIGN.md`, "Candidate list (section 7.4)", candidate 1. Build it exactly as written; any reading of open wording goes into a dated readings section in RS_DESIGN before any tuning run.

- [ ] The candidate is a registered arm (`robustsuite.harness.ARMS`), composed around the frozen `astar_random` stack at runtime; no frozen file is edited (the manifest verifies)
- [ ] It runs through the harness and the block runner on every cell and the M0 anchor
- [ ] Seam 1: actions are bit-identical when `obs[28:]` and the spec's trigger, block and spawn fields are perturbed; the same seed (and configuration) reproduces the episode; the controller is the frozen SCS one; the frozen QP, Random and parameters are unchanged
- [ ] Its default configuration and its search grid are named configurations the harness accepts
- [ ] Its p95 step time is measured on a short sample
- [ ] Unit tests at the candidate's seam: a fired block becomes new cells within K steps and triggers a replan around it; pedestrian-only replays of RS_DIAG leave few or no phantom cells at K (count reported); A* starts from the nearest free cell when the robot is inside the inflated new cells; the known map is the reset-time snapshot and never contains the block
- [ ] The replan cost is measured (mean and maximum per replan)
