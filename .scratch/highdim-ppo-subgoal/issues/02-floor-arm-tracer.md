# 02: Floor arm tracer: goal-only + Random through the episode harness

**What to build:** The researcher can run the floor arm (goal-only + Random-CLF-DR-CBF: planning off, no static map, γ = goal) for any condition and seed through a new episode harness. The harness returns a complete episode record with per-step diagnostics. The new seed blocks exist and HD_FINAL is sealed.

**Blocked by:** 01 (Pre-registration)

**Status:** done (`4b7e304`)

- [x] HD seed blocks exist and are asserted disjoint from every Continuation block and forbidden range, and from the 11M, 12M and 13M blocks
- [x] HD_FINAL raises unless opened from the two permitted entry points (baseline runs and final evaluation), following the Continuation FINAL-block pattern
- [x] The harness runs the floor arm on the canonical M0 env (6 obstacles at 0.675 m/s, fixed or randomized) with the frozen tuned parameters and frozen Random spec
- [x] The per-episode Random RNG is the existing controller RNG derived from the episode seed
- [x] The episode record reuses the existing episode-record, true-clearance and step-recorder functions
- [x] The episode record carries per-step γ, V, u_nom, u_exec, QP status, Random events and min CBC
- [x] Seam-1 test: γ equals the goal at every step
- [x] Seam-1 test: the planner's path method, patched to raise, is never called
- [x] Seam-1 test: replacing the static map handed to the policy leaves every action identical
- [x] Seam-1 test: the same seed reproduces an identical episode
- [x] Seam-1 test: the controller used in evaluation is the frozen SCS class, and the fast solver is refused
- [x] No frozen file is edited, and the manifest test from 01 still passes

## Comments

- 2026-09-18, done in `4b7e304`.
  - Standards review follow-ups:
    - min CBC is taken from `StepRecorder`, not recomputed;
    - `spec` dropped from the arm builder;
    - test imports moved to the top.
  - Spec review follow-ups:
    - the determinism test now uses a seed with Random events, and compares `events` (phi/theta) as well;
    - new test: the Random draws match `controller_rng(seed)`, and a mutation check confirms it catches a wrong or nondeterministic stream;
    - new test: the HD_TRAIN range stays clear of the other blocks;
    - `run_episode` rejects a condition that doesn't match the env.
  - The HD_TRAIN cap (100 runs × 1000 envs) is an implementation limit, documented in the code; it is not a pre-registered rule.
  - Two tests go slightly beyond this ticket and are kept for ticket 06 to reuse on the PPO arm: the recorded V equals the frozen formula, and the frozen parameter values.
  - Full suite: 526 passed, 4 known failures.
