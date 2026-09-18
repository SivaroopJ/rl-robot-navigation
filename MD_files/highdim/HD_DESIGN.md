# HD Experiment 1 — design and pre-registration

**PPO subgoal source replaces A\* in Random-CLF-DR-CBF (high-dimensionality axis).**

This document is committed **before any run**. Every rule in sections 5–9 is fixed here. Any later change goes in a separate, dated *Deviations* section of the results report, and never edits this file.

- Decision record: `docs/adr/0001-ppo-replaces-reference-source-not-clf.md`
- Spec and tickets: `.scratch/highdim-ppo-subgoal/`
- Branch: `highdim` (from `Week6` @ `3ef5d03`)
- Date: 2026-09-18

---

## 1. Question

Can a PPO policy take over A\*'s role as the navigation component while Random-CLF-DR-CBF stays the safety layer? And how much of A\*'s contribution on M0 does it recover?

The high-dimensionality axis will later move to episodes with several ordered goals (section 12). Experiment 1 isolates one change: where the CLF reference comes from.

## 2. What the current system is (verified from code)

| Step | Component | Uses A\*? |
|---|---|---|
| LiDAR ranges `obs[4:28]` → scan buffer (5 scans) + LiDAR velocity tracker → ξ rows `[dh_dt, h, ∇h_x, ∇h_y]` | `EstimatedLidarBarrierSource`, `LidarVelocityTracker` | no |
| γ = carrot 1 m ahead on the A\* polyline, planned once on the known static map; γ = goal when there is no follower | `StaticMapPlanner`, `CarrotFollower`, via `DRCBFPolicy.predict` | **yes: this is the only place** |
| CLF `V(p, γ) = ½·k_v·‖p−γ‖²`, constraint `∇V·u + λ_V·V ≤ δ`, δ ≥ 0; `u_nom = max_v·(γ−p)/‖γ−p‖` | `ClfCbfDrccpController` | through γ only |
| Objective `3‖u−u_prev‖² + 4h_crit‖u−u_nom‖² + 5h_crit·δ²` | same | no |
| DR-CCP constraint, collapsed regime: `min_i CBC_i ≥ τ_eff` | same | no |
| On infeasibility: K random directions, best `min CBC` margin | `RandomRecovery` | no |
| Executed `[vx, vy]`, holonomic single integrator | `RobotNavEnv` | no |

Four consequences follow:

- The CLF is a **soft tracking objective**. δ is unbounded and its weight falls to zero near obstacles, so it certifies nothing about reaching the goal.
- `u_nom` is always saturated, so a "local target" and a "desired velocity" are the same thing.
- Removing A\* has been measured already. Phase 5 standalone harness, estimated velocities: goal-direct 72% success / 24% stuck, against the A\* carrot 91% / 12%.
- M0 has one fixed static map, so a PPO policy trained on it learns the map implicitly.

## 3. What changes, and what does not

**Frozen.** No file is edited; the SHA-256 manifest `MD_files/highdim/PROTECTED_MANIFEST.sha256` (41 files) is verified by `tests/test_highdim.py` on every run. It covers:

- the CLF, DR-CCP and objective;
- the LiDAR barrier source and tracker;
- Random recovery and its frozen specification (`R3/random_frozen.json`: K 16, H 1, speeds {0.8, 1.0}, accept m ≥ 0);
- the tuned parameters (`H2/tuned_frozen.json`: α 0.8, λ_V 1, r_W 0.012, ε 0.1, k_v 0.10, 5 scans, τ_eff 0.12, collapsed regime);
- the environment's dynamics, collision logic, observation and reward;
- the A\* planner and carrot follower;
- the fast DR-CCP solver;
- the exp6 helpers the harness reuses.

The manifest covers the top-level `continuation/*.py` modules only. The Continuation code used by the arms (`policy`, `random_recovery`, `seeds`, `params`, `harness`, `stats`) imports only top-level modules. The subpackages (`congestion`, `pursuit`, `perception`, `supervisor`, `unlabelled_threat`) belong to other tracks, and nothing in this experiment imports them.

**Modified, by runtime composition only.**

- *The reference source.* The PPO arm and the floor arm build the tuned policy with planning off and an empty static map. The PPO arm then assigns a subgoal source as the policy's `follower` after each reset. That assignment is the entire replacement of A\* → carrot → γ.
- *The training controller.* The fast solver may be swapped in, in training only (section 6).

**New.**

- the subgoal source;
- the arm builder;
- the PPO environment wrapper and its diagnostics;
- the PPO trainer;
- seed blocks;
- the entry points under `experiments/highdim/`;
- tests;
- this document, ADR 0001 and the results report.

## 4. Arms

All arms use the canonical M0 environment: `exp6.make_env`, 6 dynamic obstacles at 0.675 m/s, the 5 rectangles, dt 0.1, 500 steps, stratified start/goal. Both conditions are run: `fixed` and `randomized` obstacle motion.

| Arm | γ source | Safety layer | Map |
|---|---|---|---|
| 1 Ceiling: A\* + Random | A\* carrot (frozen F-D arm, re-run) | CLF-DR-CBF + Random | known static map, A\* only |
| 2 Floor: goal-only + Random | γ = goal | CLF-DR-CBF + Random | none |
| 3 PPO-subgoal + Random | γ = p + L·a, or the goal inside L | CLF-DR-CBF + Random | none |
| 4 PPO, no filter (diagnostic) | same checkpoint as arm 3 | bypassed: `u = nominal_action(p, γ, max_v)/max_v` | none |

The subgoal rule:

- The action is `a = a_env / max(1, ‖a_env‖)`, where `a_env` is the SB3-clipped action.
- L = 1.0 m.
- γ = goal exactly when ‖goal − p‖ < L.
- γ is never projected into free space.

PPO observes only `obs[0:28]`: goal offset / 10, own velocity, and 24 LiDAR rays / 5. The safety layer keeps the ego pose it uses today. All arms use the Random RNG `continuation.seeds.controller_rng(episode_seed)`.

## 5. Seed blocks

| Block | Range | Use |
|---|---|---|
| HD_DEV | 14 000 000 + 0…99 | floor and ceiling dev runs, fast-solver check, pilot and in-training evaluation (50 seeds), extension rule |
| HD_FINAL | 14 100 000 + 0…199, **sealed** | floor, ceiling, PPO and no-filter arms; the same seeds are used in both conditions |
| HD_TRAIN | 14 500 000 + 1000·run + env index | initial seeds for the training environments; they then auto-reset from their own `np_random` |

The blocks must be disjoint from every Continuation block and forbidden range, and from the pursuit (11M), Track B (12M) and Track A (13M) blocks.

HD_FINAL may be opened only by the baseline entry point (arms 1 and 2) and the final-evaluation entry point (arms 3 and 4). PPO seeds are 0, 1 and 2.

## 6. Phases and transition rules

| Phase | Work | Rule to proceed |
|---|---|---|
| 0 | This document, ADR 0001, manifest, baseline test suite recorded, existing Continuation and Phase 7 manifests verified (section 10) | committed |
| 1 | **Floor** on HD_DEV (100 × 2), then on HD_FINAL (200 × 2); frozen SCS | always go on to phase 2 |
| 2 | **Ceiling** on the same HD_FINAL seeds, then HD_DEV; frozen SCS; then the **stop check** (section 7) | stop check not triggered |
| 3 | **Fast-solver training-suitability check** (section 8) | never a stop; it only selects the training solver |
| 4 | Implementation of the PPO arm, wrapper and trainer | all `tests/test_highdim.py` pass |
| 5 | **Pilot**: seed 0, 1M steps, 10 Hz (section 9.1) | pilot rule passes |
| 6 | **Main run**: 3 seeds × 3M steps, 8 environments; extension rule (section 9.2) | done |
| 7 | **Sealed evaluation** of arms 3 and 4 on HD_FINAL; GO gate (section 9.3) | — |

Arms 1 and 2 are deterministic given the seed: the environment's `np_random`, the controller RNG and SCS are all deterministic. Their HD_FINAL results are therefore produced once, in phases 1–2, and reused in phase 7.

## 7. Stop check (end of phase 2)

On HD_FINAL, pooled over both conditions (400 paired episodes per arm):

> **If S_ceiling − S_floor < 0.05, STOP.** A\* contributes too little on M0 for Experiment 1 to answer its question. Phases 3–7 are cancelled.

The historical F-D numbers (pooled success 0.910, collision 0.090) are context only.

## 8. Fast-solver training-suitability check (phase 3)

**Question:** is the existing `FastClfCbfDrccpController` (variant `reduced_osqp`, unmodified) faithful enough to the frozen SCS controller to speed up PPO training?

**Background, not re-litigated.** Phase 7 Stage 1 (`MD_files/week5/phase7/STAGE1_FINAL_REPORT.md`) found:

- feasible-set equality: passed;
- guarantee-tier agreement: passed;
- closed-loop equivalence: passed;
- strict KKT certificate: **failed** (maximum 5.429e−5 against 1e−6, three attempts; stopping rule applied).

**That verdict stands. This check does not reopen, amend or reinterpret it.** The fast solver is an acceleration mechanism, not a formally equivalent replacement.

**Parameters:** exactly the frozen tuned and Random parameters (section 3). The solver's own preconditions hold there (ε·n_keep = 0.5 < 1; max_v = 1 ≤ max(1, α)), giving τ = 0.12.

**Corpus:** HD_DEV, 100 seeds × 2 conditions, driving arm 1 and arm 2 with the frozen controller.

- **E-a, per step.** At every step the fast solver solves the identical inputs (p, γ, ξ, u_prev) in shadow. Pass requires both:
  - feasibility category (optimal vs infeasible) agrees on **100%** of steps;
  - ‖u_fast − u_frozen‖∞ ≤ 1e−3 on ≥ 99% of steps where both are optimal. The maximum is reported.
- **E-b, closed loop.** Each arm runs with the fast solver against the same arm with the frozen solver, paired by seed and pooled over conditions (200 pairs per arm). That makes two tests, and **both arms must pass**. Pass requires McNemar p > 0.05 on success, and the 95% bootstrap CI of the success difference inside ±0.03.

**Decision:**
- E-a and E-b both pass → training uses the fast solver.
- Otherwise → training uses frozen SCS.
- The fast solver is never modified.

**Evaluation and every reported number always use the frozen SCS controller.**

## 9. PPO rules

**Setup.**
- SB3 PPO with the `config.json` hyperparameters: lr 3e−4 linear, n_steps 2048, batch 256, 10 epochs, γ 0.99, λ 0.95, entropy 0.01, target KL 0.02.
- MLP [256, 256], 8 environments, no observation normalisation.
- 10 Hz: a new subgoal every step.
- Environment reward unchanged; no intervention penalty.

**Rollout.**
- The buffer stores the observation, the sampled subgoal action, its log-probability, reward, done/truncated flags, the value estimate and the next observation.
- It never stores the executed velocity.
- Per-step diagnostics: γ, u_nom, u_exec, ‖u_exec − u_nom‖, QP status, Random activation, minimum CBC.

**Checkpoints.**
- A checkpoint every 250k steps, plus the final model.
- Metadata records the commit, solver, L, hold length, frozen-parameter hash, seed and step count.
- In-training evaluation: 50 HD_DEV seeds × 2 conditions, deterministic, every 250k steps.

### 9.1 Pilot rule (phase 5)

Implementation sanity must hold:
- no exceptions;
- no non-finite loss, value or action anywhere in training;
- every diagnostic key present on 100% of logged steps;
- explained variance > 0 at 1M steps.

approx_kl, the fraction of updates stopped early at target_kl, and throughput are **reported, not gated**.

A sanity failure is an implementation bug. It is fixed and the pilot is re-run at the same hold length; this is logged as a deviation and **does not use up** the single hold-5 re-pilot, which is reserved for a learning failure.

**Learning:** the 1M-step checkpoint's deterministic success on HD_DEV (100 seeds × 2 conditions, pooled into 200 episodes) must be **≥ the floor's pooled success on the same 200 episodes**. This is a comparison of point estimates with no statistical test; a tie passes.

- If not, re-run the pilot **once** with 5-step subgoal holding. γ is frozen in world coordinates between decisions, and the QP still runs every step. The same comparison applies.
- If the hold-5 pilot also fails, the verdict is **NO-GO**. Phases 6–7 are cancelled and HD_FINAL is never opened for PPO.

If the pilot passes, the hold length that passed is used in phase 6.

### 9.2 Extension rule (phase 6)

The in-training evaluations fall every 250k steps, each on 50 HD_DEV seeds × 2 conditions pooled (100 episodes). For each seed, average them over two windows:
- W1: the evaluations at 2.25M and 2.5M steps;
- W2: the evaluations at 2.75M and 3.0M steps.

If **W2 − W1 ≥ 0.02 in at least 2 of 3 seeds**, all three seeds extend to 5M steps. The rule is applied **once**, at 3M steps; there is no further extension after 5M. The decision uses dev data only. The sealed evaluation uses each seed's **final** checkpoint, never the best-on-dev one.

### 9.3 GO gate (phase 7)

The gate is evaluated **per PPO seed**. Each seed's arm 3 is compared with arms 1 and 2 on HD_FINAL, pooled over both conditions (400 paired episodes). The gate uses only the exact two-sided McNemar test (`continuation.stats.paired_binary`). The paired-bootstrap CIs are reported but are not part of the gate.

1. **Beats the floor:** S_PPO > S_floor, with McNemar p < 0.05 on success.
2. **Closes the gap:** (S_PPO − S_floor) / (S_ceiling − S_floor) ≥ 0.5.
3. **Safety:** collisions are not significantly higher than the ceiling's. This fails if C_PPO > C_ceiling with McNemar p < 0.05 on collision.

**GO if all three hold in ≥ 2 of 3 seeds.** Otherwise the verdict is NO-GO. The report gives the mean ± sd over seeds and the gap closure for each seed. Arm 4 is descriptive only and is not part of the gate.

## 10. Baseline test suite (phase 0)

Recorded on `highdim` (base `283ec6d`) on 2026-09-18. At that point the only HD code was the manifest and its tests.

| Suite | Collected | Passed | Failed |
|---|---|---|---|
| Pre-existing tests | 496 | 492 | 4 |
| `tests/test_highdim.py` (manifest) | 18 | 18 | 0 |
| **Total** | **514** | **510** | **4** |

The 4 failures are the known, pre-existing `tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main[0, 3, 6, 10]`. They fail because the `main` ref that test replays moved on 2026-09-11 (see `MD_files/week6_tracks_REPO_AUDIT.md`). They are inside the Phase 7 frozen manifest and are left alone. Any other failure in later phases is new and gets reported.

The existing manifests were verified on the same date:
- the Continuation protected manifest (`experiments/week6_continuation/protect_manifest.py`): modified [], missing [];
- the Phase 7 frozen manifest tests (`tests/test_frozen_phase1_6.py -k manifest`): 3 passed.

## 11. Caveats stated in advance

- **Soft CLF.** V is a soft tracking objective towards a policy-selected local reference. It is **not** a certificate of reaching the goal, and no such claim is made.
- **Non-Markov filter state.** The filter's hidden state (5-scan buffer, velocity tracker, u_prev, Random persistence) is not in PPO's observation. The wrapped environment is therefore mildly non-Markov. This is documented, not engineered away.
- **Map learned on M0.** The static map is fixed, so PPO learns it implicitly. The claim is "PPO replaces the explicit planner", not "PPO navigates unknown maps". Held-out layouts are a later experiment (1b).
- **Ego pose.** The safety layer keeps using the ego pose, as every earlier arm did. PPO never receives it.

## 12. Out of scope

- Multi-agent settings.
- Multi-goal episodes. They are design only: ordered goals supplied externally, the next goal's offset added to the observation, +10 per goal, CLF and DR-CBF unchanged.
- Held-out maps.
- Frame stacking or recurrent PPO.
- SAC, TD3 and hierarchical RL.
- Intervention penalties. At most a later, separately labelled ablation.
- Any change to the CLF, the CBF/DR formulation, Random recovery, dynamics, reward or fast solver.
