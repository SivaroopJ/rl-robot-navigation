# 03: Ceiling arm: A* + Random through the same harness

**What to build:** The researcher can run the ceiling arm (frozen A* + carrot + Random-CLF-DR-CBF, i.e. the F-D arm) through the same episode harness, and it is provably the frozen baseline.

**Blocked by:** 02 (Floor arm tracer)

**Status:** done (`bf0b157`)

- [x] The ceiling arm is built by delegating to the existing Continuation construction of the Random arm and its recovery attachment, verbatim
- [x] Seam-1 test: ceiling trajectories are bit-identical to the Continuation harness's Random arm on 5 existing validation seeds
- [x] Seam-1 test: the same seed reproduces an identical ceiling episode
- [x] Running both arms on the same seeds asserts identical start and goal across arms
- [x] No frozen file is edited, and the manifest test still passes

## Comments

- 2026-09-18, done in `bf0b157`.
  - The Spec review found the bit-identity test compared episode summaries, not trajectories. The test now records the F-D arm's position at every step through the frozen Continuation harness (via an env.step wrapper, measurement only) and requires the ceiling trajectory to match exactly, on 5 seeds × both conditions, as well as the 43 shared record keys.
  - Mutation checks (lookahead 1.1; Random K = 8) are both caught.
  - `run_paired` now checks seed, start and goal, raises RuntimeError on a mismatch, and closes envs in `finally`.
  - Full suite: 540 passed, 4 known failures.
