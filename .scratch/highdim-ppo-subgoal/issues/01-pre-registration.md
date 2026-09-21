# 01: Pre-registration: design doc, ADR 0001, frozen-file manifest

**What to build:** Before any experiment runs, the researcher can point to one committed pre-registration that fixes every rule of HD Experiment 1. There is also a committed ADR recording that PPO replaces the CLF reference source and nothing else, plus a hash manifest proving the safety stack and the A* baseline stay frozen. See the parent spec (`spec.md` in this folder).

**Blocked by:** None (can start immediately)

**Status:** done (`2bfdf05`)

- [x] The design document (HD_DESIGN) states the question, the four arms, the seed blocks (HD_DEV 14 000 000 × 100, HD_FINAL 14 100 000 × 200 sealed, HD_TRAIN 14 500 000+) and the phase order
- [x] It pre-registers the stop check: pooled S_A* − S_floor < 0.05 on HD_FINAL means stop, computed after both floor and ceiling have run
- [x] It pre-registers the fast-solver training-suitability criteria:
  - E-a: feasibility category agrees on 100% of steps, and ‖Δu‖∞ ≤ 1e-3 on ≥ 99% of optimal steps
  - E-b: closed-loop McNemar p > 0.05 and the success-difference CI lies within ±0.03
  - If the check fails, training falls back to SCS
- [x] It states the fast solver is an acceleration mechanism only, and that the Phase 7 Stage 1 verdict is not reopened
- [x] It pre-registers the pilot rule: dev success at 1M steps ≥ the floor's dev success; one re-pilot with 5-step holding; if that fails, NO-GO and HD_FINAL stays unopened
- [x] It pre-registers the extension rule: ≥ 2 of 3 seeds gain ≥ 2 pp between the 2.0–2.5M and 2.5–3.0M dev windows
- [x] It pre-registers the GO gate, applied per seed and passing in ≥ 2 of 3 seeds:
  - McNemar p < 0.05 against the floor
  - gap closure ≥ 0.5
  - collisions not significantly higher than the re-run ceiling
- [x] It records that the final checkpoint is used, not the best one on dev
- [x] It documents three caveats:
  - the CLF is a soft tracking objective, not a convergence certificate
  - the filter's hidden state makes the problem mildly non-Markov for PPO
  - on M0 the map is learned, not unknown
- [x] ADR 0001 records the decision (PPO → γ; CLF, CBF, DR and Random unchanged; no map access) and rejects options A, C and D with reasons
- [x] A hash manifest covers the planner, the DR-CBF code, the Continuation code and the frozen tuned/Random parameter files, and a test verifies it
- [x] The baseline test suite result is recorded: only the 4 known `test_legacy_path_is_bit_identical_to_main` failures
- [x] Everything is committed on `highdim`, and the commit hash is recorded in the vault experiment note

## Comments

- 2026-09-18, done in `2bfdf05`.
  - The two-axis review found nothing blocking.
  - The Spec review found five ambiguities in the rules, all tightened before the commit (pre-registration, not deviations):
    1. E-b: both arms must pass.
    2. Pilot sanity has concrete checks, and a sanity failure does not use up the hold-5 re-pilot.
    3. Pilot learning is a pooled 200-episode point estimate; a tie passes.
    4. Extension windows are W1 = {2.25M, 2.5M} and W2 = {2.75M, 3.0M}, pooled over conditions, applied once.
    5. The gate uses McNemar only; bootstrap CIs are reported but not gated.
  - Existing Continuation and Phase 7 manifests verified.
  - Baseline suite: 496 pre-existing tests, 492 passed + 4 known failures.
