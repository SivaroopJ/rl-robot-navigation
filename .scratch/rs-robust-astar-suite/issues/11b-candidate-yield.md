# 11b: Candidate 2, `yield`

**What to build:** the `yield` arm: a behaviour layer that steps aside when a tracked pedestrian will come too close.

**Blocked by:** 10 (approved 2026-09-20)

**Status:** ready-for-human

Design: `MD_files/robustsuite/RS_DESIGN.md`, "Candidate list (section 7.4)", candidate 2. Build it exactly as written; any reading of open wording goes into a dated readings section in RS_DESIGN before any tuning run.

- [x] The candidate is a registered arm (`robustsuite.harness.ARMS`), composed around the frozen `astar_random` stack at runtime; no frozen file is edited (the manifest verifies)
- [x] It runs through the harness and the block runner on every cell and the M0 anchor
- [x] Seam 1: actions are bit-identical when `obs[28:]` and the spec's trigger, block and spawn fields are perturbed; the same seed (and configuration) reproduces the episode; the controller is the frozen SCS one; the frozen QP, Random and parameters are unchanged
- [x] Its default configuration and its search grid are named configurations the harness accepts
- [x] Its p95 step time is measured on a short sample
- [x] Unit tests at the candidate's seam: a constructed head-on track inside H and D triggers a side point on the freer side; the side point is free on the known map or the other side is taken, else no yield; the follower's γ returns when no threat remains; the tracker's state is not changed by the layer

## Comments

**2026-09-20, built.** `robustsuite/candidates.py`: `YieldLayer`, reading the frozen LiDAR velocity
tracker's confirmed tracks (position, velocity, confirmed) and never calling it. Tests:
`tests/test_robustsuite_candidates.py`.

**For the researcher, before the tuning run.** Two properties of the rule as written, both measured
in `MD_files/robustsuite/RS4_CANDIDATE_CHECKS.md` and recorded as readings 11 and 14 of
`RS_DESIGN.md`:
- the closest approach is taken over [0, H], so a track already inside D is a threat while it
  moves away;
- the frozen tracker confirms moving tracks built out of static geometry (single LiDAR points
  sliding along a wall as the robot moves, at 0.3-0.7 m/s), so the layer steps aside in cells that
  hold no pedestrian at all.

Both were built as written rather than filtered: a filter would change the approved mechanism.
