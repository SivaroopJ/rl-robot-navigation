# 11a: Candidate 1, `detour`

**What to build:** the `detour` arm: a LiDAR map of new static obstacles and a Bug2-style wall-following layer that rejoins the reset-time A* path. No planner call after reset.

**Blocked by:** 10 (approved 2026-09-20, revised and re-approved the same day)

**Status:** ready-for-human

Design: `MD_files/robustsuite/RS_DESIGN.md`, "Revision of the candidate list (section 7.4)", candidate 1. Build it exactly as written; any reading of open wording goes into a dated readings section in RS_DESIGN before any tuning run.

- [x] The candidate is a registered arm (`robustsuite.harness.ARMS`), composed around the frozen `astar_random` stack at runtime; no frozen file is edited (the manifest verifies); the planner is called only at reset (tested)
- [x] It runs through the harness and the block runner on every cell and the M0 anchor
- [x] Seam 1: actions are bit-identical when `obs[28:]` and the spec's trigger, block and spawn fields are perturbed; the same seed (and configuration) reproduces the episode; the controller is the frozen SCS one; the frozen QP, Random and parameters are unchanged
- [x] Its default configuration and its search grid are named configurations the harness accepts
- [x] Its p95 step time is measured on a short sample
- [x] Unit tests at the layer's seam: a fired block becomes new cells within K steps and starts a detour; the rejoin point is past the new cells on the original path; the side with more LiDAR room is chosen; wall following holds about 0.6 m and ignores pedestrian hits; the Bug2 leave rule (clear line of sight and closer than d0); the side flip at 200 steps and give-up at 400; with no new obstacle and no stall, actions equal `astar_random` step for step
- [x] Pedestrian-only replays of RS_DIAG: phantom new cells at K counted and reported

## Comments

**2026-09-20, built.** `robustsuite/candidates.py`: `DetourLayer` over the frozen `CarrotFollower`,
installed as the policy's follower after reset by `candidates.attach`. The harness and the block
runner take a `config`; the configuration joins the block fingerprint, and the baseline's
fingerprint is unchanged. Tests: `tests/test_robustsuite_candidates.py`.

Readings of open wording are in `RS_DESIGN.md`, "Readings before any tuning run (2026-09-20,
tickets 11a-11c)". Two are worth the researcher's eye:
- the known map's outer wall is the wall LiDAR sees, 0.3 m inside the world boundary; against the
  true boundary every wall hit would be unexplained and the whole wall would turn into new cells;
- after a detour rejoins at q, later triggers look at the path only from q on. Without it a robot
  that arrives within 1 m of q can project back just short of the blocked stretch and immediately
  re-trigger the detour it has just finished.

Measurements: `MD_files/robustsuite/RS4_CANDIDATE_CHECKS.md`
(`experiments/robustsuite/rs4_candidate_checks.py`).
