# 01: Pre-registration: design doc, ADR 0001, frozen-file manifest

**What to build:** Before any experiment runs, the researcher can point to one committed pre-registration that fixes every rule of HD Experiment 1. There is also a committed ADR recording that PPO replaces the CLF reference source and nothing else, plus a hash manifest proving the safety stack and the A* baseline stay frozen. See the parent spec (`spec.md` in this folder).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The design document (HD_DESIGN) states the question, the four arms, the seed blocks (HD_DEV 14 000 000 × 100, HD_FINAL 14 100 000 × 200 sealed, HD_TRAIN 14 500 000+) and the phase order
- [ ] It pre-registers the stop check: pooled S_A* − S_floor < 0.05 on HD_FINAL means stop, computed after both floor and ceiling have run
- [ ] It pre-registers the fast-solver training-suitability criteria:
  - E-a: feasibility category agrees on 100% of steps, and ‖Δu‖∞ ≤ 1e-3 on ≥ 99% of optimal steps
  - E-b: closed-loop McNemar p > 0.05 and the success-difference CI lies within ±0.03
  - If the check fails, training falls back to SCS
- [ ] It states the fast solver is an acceleration mechanism only, and that the Phase 7 Stage 1 verdict is not reopened
- [ ] It pre-registers the pilot rule: dev success at 1M steps ≥ the floor's dev success; one re-pilot with 5-step holding; if that fails, NO-GO and HD_FINAL stays unopened
- [ ] It pre-registers the extension rule: ≥ 2 of 3 seeds gain ≥ 2 pp between the 2.0–2.5M and 2.5–3.0M dev windows
- [ ] It pre-registers the GO gate, applied per seed and passing in ≥ 2 of 3 seeds:
  - McNemar p < 0.05 against the floor
  - gap closure ≥ 0.5
  - collisions not significantly higher than the re-run ceiling
- [ ] It records that the final checkpoint is used, not the best one on dev
- [ ] It documents three caveats:
  - the CLF is a soft tracking objective, not a convergence certificate
  - the filter's hidden state makes the problem mildly non-Markov for PPO
  - on M0 the map is learned, not unknown
- [ ] ADR 0001 records the decision (PPO → γ; CLF, CBF, DR and Random unchanged; no map access) and rejects options A, C and D with reasons
- [ ] A hash manifest covers the planner, the DR-CBF code, the Continuation code and the frozen tuned/Random parameter files, and a test verifies it
- [ ] The baseline test suite result is recorded: only the 4 known `test_legacy_path_is_bit_identical_to_main` failures
- [ ] Everything is committed on `highdim`, and the commit hash is recorded in the vault experiment note
