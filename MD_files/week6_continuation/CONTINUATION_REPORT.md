# Week 6 Continuation — Tuned CLF-DR-CBF and Random-CLF-DR-CBF on M0

Branch `DR_safe` @ `427e589` plus untracked continuation files (nothing committed). Run dates 2026-09-10 → 2026-09-11.

- **Scope:** every rollout is on the canonical M0 map, in both motion conditions, paired by seed.
- **Frozen inputs untouched:** 466 protected files, verified by SHA-256 (`PROTECTED_MANIFEST.sha256`),
  and the 60-file Phase-7 manifest. No git-tracked file was modified. PPO was not rerun.
- **Tests:** 361 passed (322 pre-existing + 39 new).
- **Size:** 26 300 episodes in total, exactly the budget declared in AUDIT.md section 7.
- **Machine-readable manifest:** `results/week6_continuation/MANIFEST_ALL.jsonl` (one row per episode).
- **Full generated tables:** `CONTINUATION_REPORT_DATA.md`.

Positive differences below mean "the first-named arm is higher". **Binary outcomes** use exact
McNemar. **Continuous outcomes** use a paired bootstrap (10 000 replicates, 95 % CI). **Step-level
rates** use an episode-cluster bootstrap. **p-values are unadjusted**; F makes 4 pre-declared
comparisons.

---

## 1. Implementation audit

See `AUDIT.md`. The findings that shaped the design:

- **A1.** The scan buffer N is the DR sample count.
- **A2.** For εN ≤ 1, r_W and ε enter only through τ_eff. This leaves 50 distinct collapsed-regime
  controllers in the 120-cell grid.
- **A3.** ε = 0.30 is a different, non-collapsed constraint.
- **A4.** τ_eff = r_W·max(1, α)/ε, so LADDER must be handed τ explicitly.
- **A5.** k_v = 0.05.
- **A6.** m ≥ τ is unattainable on an infeasible step. This was confirmed empirically: **0 of
  5 046 infeasible steps across all arms in F reached T0.**

**Reference-equivalence gate** (`results/week6_continuation/gate/gate.json`): the tunable policy at
the reference parameters reproduced **260/260 published dev/validation episodes exactly**. The
compared fields were outcome, steps, min-clearance, SPL, solver failures and LADDER rungs.

## 2. Hyperparameter search table (as executed)

| stage | parameter | values | n/cond | selection |
|---|---|---|---|---|
| C1 | N | {1, 3, 5} | 100 | diagnostic; kept N = 5 |
| H1 | α × r_W × ε | {0.2, 0.4, 0.8, 1.2, 2.0} × {0, .001, .002, .004, .008, .012} × {.05, .1, .2 \| .3} | 20 | top-10 distinct (α, τ_eff) classes |
| H2 | λ_V × k_v | {0.25, 0.5, 1, 2, 4} × {0.025, 0.05, 0.10} on 10 classes | 50 | rank 1 → **Tuned** |
| RANDOM_EXP_R1 | K, H, v | 8, 1, {0.2} | 100 | report only |
| RANDOM_EXP_R2 | speed ladder | {0.2 … 1.0}, accept m ≥ 0 | 100 | decides the R3 axis |
| RANDOM_EXP_R3 | K × H × v₀ | {4, 8, 16} × {1, 3, 5} × escalation start {0.2 … 1.0} | 20 → 100 | top-4 → rank 1 → **Random** |

## 3. Seed allocation

| block | seeds | used by |
|---|---|---|
| C0 | 10 000 000 + 0..99 | C0 (+ a 5-episode smoke run) |
| C1 | 10 100 000 + 0..99 | C1 |
| H1 | 10 200 000 + 0..19 | H1 |
| H2 | 10 300 000 + 0..49 | H2 |
| R1 | 10 400 000 + 0..99 | RANDOM_EXP_R1 |
| R2 | 10 500 000 + 0..99 | RANDOM_EXP_R2 |
| R3 screen | 10 600 000 + 0..19 | R3 screen |
| R3 validation | 10 700 000 + 0..99 | R3 validation |
| FINAL (sealed) | 10 900 000 + 0..199 | F only, opened once |

- **Earlier blocks never used for any decision:** 1 000 000, 7M / 8M / 9M, and 20 000 / 21 000. The
  20 000 / 21 000 blocks were read-only, for the gate.
- **Controller RNG:** `default_rng([0x52434246, episode_seed])`, independent of `env.np_random`.

## 4. C0 — frozen baselines (100 eps/condition)

| arm | cond | success | collision | dyn / stat / wall | P(coll \| inf) | ms mean / p95 |
|---|---|---|---|---|---|---|
| C0-A original | fixed | 0.900 | 0.100 | 0.100 / 0 / 0 | 0.122 | 37.1 / 42.1 |
| | randomized | 0.790 | 0.210 | 0.210 / 0 / 0 | 0.253 | 37.3 / 42.5 |
| | pooled | 0.845 | 0.155 | 0.155 / 0 / 0 | 0.186 | 37.2 / 42.3 |
| C0-B T1 + LADDER | fixed | 0.890 | 0.110 | 0.110 / 0 / 0 | 0.159 | 38.0 / 46.6 |
| | randomized | 0.870 | 0.130 | 0.130 / 0 / 0 | 0.164 | 38.3 / 47.2 |
| | pooled | 0.880 | 0.120 | 0.120 / 0 / 0 | 0.162 | 38.2 / 46.9 |

**C0-B vs C0-A:**

| condition | collision diff | p |
|---|---|---|
| fixed | +0.010 | 1.0 |
| randomized | **−0.080** | **0.0215** |
| pooled | −0.035 | 0.12 |

This is consistent with the historical Week-5 result.

## 5. C1 — buffer diagnostic (100 eps/condition; decision **KEEP N = 5**)

| N | pooled success | pooled collision | static coll | infeasible-step rate | inf. events/ep | ms mean |
|---|---|---|---|---|---|---|
| 1 | 0.870 | 0.130 | **0.020** | **0.0071** | 0.69 | 31.3 |
| 3 | 0.860 | 0.135 | 0.000 | 0.0242 | 1.70 | 34.2 |
| 5 | 0.820 | 0.175 | 0.000 | 0.0351 | 2.27 | 36.9 |

- **Infeasibility:** reducing N removes infeasibility strongly. N = 1 cuts the infeasible-step rate
  by 80 % (cluster CI [−0.032, −0.024]).
- **Collisions:** there is no significant collision change (N1 vs N5: −0.045, p = 0.21; N3 vs N5:
  −0.040, p = 0.18).
- **New failure mode:** N = 1 introduces **static collisions** that N = 5 never had.
- **Timing:** N = 1 is 15 % faster.
- **Interpretation:** the 0.5 s buffer is a real source of infeasibility, but it also carries
  geometric memory, and removing it is not a safety win. The C1 stop condition did not fire.

## 6. H1 — coarse search (20 eps/condition, 4 800 episodes)

- **Eligibility:** 42/50 collapsed classes passed G1 + G2. All 8 rejections are low-α (0.2–0.4)
  with large τ_eff; they are freeze/timeout failures (up to 40 % timeout).
- **Reference on this block:** the reference cell had 0.275 pooled collision.
- **Shortlist, in rank order** (α, r_W, ε → τ_eff):

| rank | α | r_W | ε | τ_eff |
|---|---|---|---|---|
| 1 | 0.4 | .001 | .2 | 0.005 |
| 2 | 1.2 | .001 | .1 | 0.012 |
| 3 | 2 | .008 | .05 | 0.32 |
| 4 | 0.8 | .012 | .1 | 0.12 |
| 5 | 1.2 | .008 | .05 | 0.192 |
| 6 | 1.2 | .004 | .05 | 0.096 |
| 7 | 1.2 | .002 | .05 | 0.048 |
| 8 | 2 | .004 | .05 | 0.16 |
| 9 | 1.2 | 0 | .05 | 0 |
| 10 | 0.4 | .012 | .2 | 0.06 |

- **Degeneracy check:** duplicate cells, which are the same controller up to SCS tolerance, agreed
  on **90.1 % of episodes on average (min 47.5 %, in the α = 0.2, τ = 0 class)**.
  - A ~10⁻⁵ solver difference is chaotically amplified in closed loop.
  - Class rates nevertheless agree to within ≈ ±5 pp.
  - **This is the empirical noise floor of a 20-episode screen.** H1 ranking is coarse by
    construction.
- **ε = 0.30 (non-collapsed):** reported separately and not forwarded. Its top cells are all α = 0.2.

## 7. H2 — fine search (50 eps/condition, 15 100 episodes) → Tuned-CLF-DR-CBF

- **Eligibility:** 118/150 candidates passed.
- **Selected configuration:**

  **Tuned-CLF-DR-CBF: α = 0.8, r_W = 0.012, ε = 0.1 (τ_eff = 0.12, collapsed), λ_V = 1.0,
  k_v = 0.10, N = 5.**

- **H2 block performance:** pooled collision 0.12 vs anchor 0.22; success 0.88 vs 0.77.
- **Tie-break:** it tied rank 2 (α 2, r_W .008, ε .05, λ_V 4, k_v 0.1) on collision, success and
  timeout, and won on clearance (0.202 vs 0.152).
- **Where the effect lives:** λ_V and k_v changed results only within noise. The effect is carried
  by (α, τ_eff), i.e. **a 3× larger DR margin than the reference (0.12 vs 0.04)**.
- **Freeze:** the configuration is frozen with its hash in `H2/tuned_frozen.json`.

## 8. RANDOM_EXP_R1 — K = 8, H = 1, v = 0.2 (100 eps/condition)

- **Headline:** R1 vs Tuned: **collision diff 0.000 in every condition** (pooled 4 vs 4 discordant,
  p = 1.0).
- **P(collision | infeasibility):** 0.232 for both arms.
- **Why:** at 0.2 m/s only 2 % of events find a candidate with m ≥ 0. A 0.02 m nudge per step does
  not change the geometry.
- **H2 (small-perturbation form): not supported.**

## 9. RANDOM_EXP_R2 — speed escalation {0.2 … 1.0} (100 eps/condition)

| comparison | fixed | randomized | pooled |
|---|---|---|---|
| R2 vs Tuned, collision | −0.080 (p = .039) | **−0.140 (p = .0005)** | **−0.110 (p < .0001; 3 vs 25)** |
| R2 vs R1, collision | −0.040 (p = .29) | −0.110 (p = .003) | **−0.075 (p = .0015)** |

- **Speeds executed:** 1.0 m/s in 303/560 events (54 %).
- **Acceptance:** 14 % of events found m ≥ 0.
- **P(collision | infeasibility):** 0.137 vs 0.295 for Tuned.
- **H3 (speed escalation): supported.**
- **R3 axis:** escalation-start.

## 10. RANDOM_EXP_R3 → Random-CLF-DR-CBF

- **Screen (45 configs × 20 eps/condition):**
  - The top of the ranking is dominated by **H = 1** and a high escalation start (v₀ = 0.8).
  - Persistence H ∈ {3, 5} never reached the shortlist.
  - Shortlist: K8/H1/v₀ 0.8, K4/H1/v₀ 0.8, K16/H1/v₀ 0.8, K16/H1/v₀ 0.4.
- **Validation (100 eps/condition, vs Tuned):** all 4 passed the gates.

| config | pooled collision | vs Tuned (0.160) | p |
|---|---|---|---|
| **K16 / H1 / v₀ 0.8** | **0.090** | −0.070 | 0.0013 |
| K8 / H1 / v₀ 0.8 | 0.090 | −0.070 | 0.0005 |
| K4 / H1 / v₀ 0.8 | 0.100 | −0.060 | 0.0075 |
| K16 / H1 / v₀ 0.4 | 0.115 | −0.045 | 0.049 |

- **Selected configuration:**

  **Random-CLF-DR-CBF = Tuned + random recovery, K = 16, H = 1, speeds {0.8, 1.0}, accept m ≥ 0,
  otherwise the best over both speeds.**

- **Tie-break:** it tied K8 on collision, success and timeout, and won on clearance by 0.0006,
  which is well inside noise. K ∈ {8, 16} are practically interchangeable.
- **Freeze:** `R3/random_frozen.json`.

## 11. F — final paired comparison (sealed block, 200 eps/condition, 1 600 episodes)

| arm | cond | success | collision | dyn | stat | wall | timeout | min-clr | SPL | inf-step rate |
|---|---|---|---|---|---|---|---|---|---|---|
| F-A Original | fixed | 0.840 | 0.160 | 0.160 | 0 | 0 | 0 | 0.237 | 0.742 | 0.035 |
| | randomized | 0.770 | 0.230 | 0.230 | 0 | 0 | 0 | 0.188 | 0.646 | 0.036 |
| | **pooled** | **0.805** | **0.195** | 0.195 | 0 | 0 | 0 | 0.213 | 0.694 | 0.036 |
| F-B Tuned | fixed | 0.880 | 0.120 | 0.120 | 0 | 0 | 0 | 0.218 | 0.820 | 0.029 |
| | randomized | 0.775 | 0.225 | 0.225 | 0 | 0 | 0 | 0.174 | 0.708 | 0.025 |
| | **pooled** | **0.828** | **0.172** | 0.172 | 0 | 0 | 0 | 0.196 | 0.764 | 0.027 |
| F-C Tuned + LADDER | fixed | 0.955 | 0.045 | 0.045 | 0 | 0 | 0 | 0.232 | 0.845 | 0.018 |
| | randomized | 0.875 | 0.125 | 0.125 | 0 | 0 | 0 | 0.192 | 0.756 | 0.018 |
| | **pooled** | **0.915** | **0.085** | 0.085 | 0 | 0 | 0 | 0.212 | 0.800 | 0.018 |
| F-D Random-CLF-DR-CBF | fixed | 0.950 | 0.050 | 0.050 | 0 | 0 | 0 | 0.233 | 0.850 | 0.020 |
| | randomized | 0.870 | 0.130 | 0.130 | 0 | 0 | 0 | 0.189 | 0.754 | 0.018 |
| | **pooled** | **0.910** | **0.090** | 0.090 | 0 | 0 | 0 | 0.211 | 0.802 | 0.019 |

**Every collision in every F arm is dynamic.** The improvements below are therefore
dynamic-obstacle improvements, not static or wall effects.

## 12. Statistical tests (F, pre-declared comparisons)

| comparison | cond | Δ collision [95 % CI] | discordant (x-only / y-only) | McNemar p |
|---|---|---|---|---|
| **Tuning:** Tuned vs Original | fixed | −0.040 [−0.090, +0.010] | 9 / 17 | 0.169 |
| | randomized | −0.005 [−0.065, +0.055] | 17 / 18 | 1.000 |
| | pooled | −0.022 [−0.060, +0.018] | 26 / 35 | **0.306** |
| **LADDER after tuning:** Tuned + LADDER vs Tuned | fixed | −0.075 [−0.115, −0.035] | 2 / 17 | 0.0007 |
| | randomized | −0.100 [−0.145, −0.055] | 2 / 22 | < 0.0001 |
| | pooled | **−0.087 [−0.120, −0.058]** | 4 / 39 | **< 0.0001** |
| **Random after tuning:** Random vs Tuned | fixed | −0.070 [−0.110, −0.035] | 1 / 15 | 0.0005 |
| | randomized | −0.095 [−0.140, −0.055] | 1 / 20 | < 0.0001 |
| | pooled | **−0.083 [−0.113, −0.055]** | 2 / 35 | **< 0.0001** |
| **H4:** Random vs LADDER | fixed | +0.005 [−0.010, +0.020] | 2 / 1 | 1.000 |
| | randomized | +0.005 [−0.015, +0.025] | 3 / 2 | 1.000 |
| | pooled | **+0.005 [−0.007, +0.020]** | 5 / 3 | **0.727** |

- **Success:** differences are the exact mirror of collision in every row, because timeout = 0 in
  every F arm.
- **Paired continuous and step-level metrics, pooled:**

| comparison | min-clearance | SPL | infeasible-step rate (cluster) |
|---|---|---|---|
| Tuned vs Original | −0.017 [−0.028, −0.005] | **+0.070 [+0.038, +0.102]** | −0.0087 [−0.012, −0.005] |
| LADDER vs Tuned | +0.016 [+0.010, +0.022] | +0.037 [+0.012, +0.062] | −0.0091 |
| Random vs Tuned | +0.015 [+0.010, +0.021] | +0.038 [+0.016, +0.061] | −0.0080 |
| Random vs LADDER | −0.001 [−0.004, +0.002] | +0.002 [−0.010, +0.013] | +0.0011 [−0.0007, +0.0027] |

## 13. Timing (F, measured wall-clock `perf_counter` around `predict()`, 8 concurrent workers)

| arm | mean ms | median | p95 | worst | QP ms | recovery cost |
|---|---|---|---|---|---|---|
| Original | 36.33 | 35.76 | 41.03 | 242 | 27.06 | — |
| Tuned | 36.08 | 35.55 | 40.71 | 156 | 26.82 | — |
| Tuned + LADDER | 36.93 | 35.71 | **43.99** | 250 | 26.87 | **+38 ms per infeasible step** (extra QP / LP solves) |
| Random-CLF-DR-CBF | 36.34 | 35.78 | 41.03 | 245 | 26.94 | **0.23 ms per event** (31 candidates on average, vectorised) |

Paired, Random vs LADDER:

| metric | difference |
|---|---|
| mean step time | **−0.58 ms [−0.78, −0.39]** |
| p95 step time | **−2.96 ms [−3.88, −2.13]** |

LADDER vs Tuned adds +3.28 ms to p95 [+2.44, +4.21]. Random vs Tuned adds +0.32 ms [+0.10, +0.54].
Worst-case steps (~150–250 ms) occur in every arm, including the ones without recovery. They are
first-step canonicalisation and host-contention outliers, not recovery.

## 14. Failure / infeasibility analysis (F, pooled)

| | Original | Tuned | T + LADDER | Random |
|---|---|---|---|---|
| infeasibility events (runs) | 898 | 518 | 505 | 522 |
| mean event duration (steps) | 2.36 | 2.62 | 1.75 | 1.85 |
| P(collision \| episode had infeasibility) | 0.229 | 0.230 | **0.098** | **0.106** |
| P(collision within 1 s of an event) | 0.184 | 0.197 | 0.081 | 0.079 |
| recovery success (feasible next step, no collision) | 0.436 | 0.368 | 0.542 | 0.531 |
| collisions after an infeasible step / while feasible | 61 / 17 | 53 / 16 | 13 / 21 | 15 / 21 |
| infeasible steps: T0 / T1 / T2 | 0 / 0 / 100 % | 0 / 0.2 / 99.8 % | 0 / 6.7 / 93.3 % | 0 / 12.3 / 87.7 % |
| mean margin of the u = 0 fallback on infeasible steps | −1.03 | −1.15 | −1.07 | −1.10 |
| mean margin actually executed on infeasible steps | −1.03 | −1.15 | **−0.24** | **−0.39** |

- **Where the gain comes from.** Both recovery mechanisms remove the same failure: the frozen
  **u = 0 fallback against moving obstacles**.
  - Collisions that follow an infeasible step fall from 53 to 13 (LADDER) and 15 (Random).
  - Collisions while feasible rise slightly, 16 → 21 in both. These are episodes that survived one
    crisis and met another.
- **LADDER's rungs:** R2 530, R1 359, R2_lp 17.
- **Random's recoveries:**
  - 72 % were executed at 1.0 m/s, 28 % at 0.8.
  - Only 12 % found a candidate with m ≥ 0.
  - The selected margin averaged −0.37, versus −1.03 for u = 0.
- **Summary:** both mechanisms mostly execute a *least-violating motion* rather than a certified-safe
  one. Neither recovered action is ever described as safe.

## 15. Interpretation

1. **How much of the gap was untuned hyperparameters? Little, on this evidence.**
   - The selection chain found configurations that looked much better in-sample: H2 pooled
     collision 0.12 vs 0.22.
   - On fresh seeds the tuned controller is **−2.2 pp vs Original, not significant (p = 0.31)**.
   - The H2 → F drop (0.12 → 0.172) is the winner's curse, and it is consistent with the ±5 pp
     episode-level noise floor measured in H1.
   - Tuning did improve efficiency significantly: SPL +0.07 and a lower infeasibility rate, at a
     small clearance cost.
   - **H1 (tuning substantially improves safety): not supported.**
2. **Recovery still matters after tuning.**
   - LADDER: −8.7 pp collision (p < 10⁻⁴).
   - Random: −8.3 pp (p < 10⁻⁴).
   - Both effects are significant in **each** condition separately.
   - So the Week-5 recovery result is not an artefact of a poorly tuned base controller.
3. **H2 (random recovery replaces LADDER):**
   - **Supported in its speed-escalated form.** Random-CLF-DR-CBF reaches 0.090 vs LADDER's 0.085
     pooled collision (Δ +0.5 pp, CI [−0.7, +2.0] pp, only 8 discordant episodes of 400).
   - Clearance, SPL and infeasibility are indistinguishable.
   - It is cheaper: −3.0 ms p95 per step, and 0.23 ms vs ~38 ms per recovery event.
   - **Not supported in its small-perturbation form:** R1 at 0.2 m/s had zero effect.
4. **H3 (speed escalation): supported.** R2 beat R1 by 7.5 pp (p = 0.0015). R3 preferred starting
   escalation at 0.8 m/s.
5. **H4 (structured vs stochastic): the mechanisms are practically equivalent here, and the data
   say why.**
   - LADDER's dominant rung R2 solves max_u min_i CBC_i exactly.
   - Random with K = 16 directions at 0.8–1.0 m/s approximates the same argmax. It reaches a slightly
     worse executed margin (−0.39 vs −0.24) with no outcome difference.
   - "Structured relaxation vs randomness" therefore mostly reduces to "which approximation of the
     least-violating action". The random one is a 16-row matrix product.
   - The randomisation of the base angle is not what does the work. Coverage of directions at high
     speed is. Persistence (H > 1) never helped.
6. **Limitations.**
   - Single map (M0), where every collision is dynamic.
   - The H1 screen is noise-limited (degeneracy check).
   - p-values are unadjusted. Both significant F comparisons (LADDER vs Tuned, Random vs Tuned) also
     hold at Bonferroni α = 0.0125; the tuning and H4 rows are null either way.
   - The H4 result is equivalence *within ±2 pp on 400 episodes*; no formal equivalence margin was
     pre-registered.
   - Random's advantage in computation time depends on LADDER being implemented as CVXPY re-solves.

### Operational notes

- **H1 was killed once** by host memory pressure at 3 200 / 4 800 episodes, with nothing saved. It
  was rerun from scratch, unchanged.
  - Worker memory was measured flat (≈ 124 MB), so the pressure came from outside this job.
  - Episode checkpointing and worker recycling were then added to `run_block`. They are covered by
    `test_checkpoint_resume_reproduces_the_uninterrupted_block` and change no outcome field.
- **No selection rule, seed, or frozen configuration was changed after results were seen.**
  `SELECTION_RULES.md` sha256 = `86432497…ddb91` was recorded before C0, and every stage verified it.
