# Week 6 Continuation — pre-declared selection rules

Written **before any C0/C1/H1 episode was run** (the only rollouts so far were the equivalence gate on
published dev/validation seeds and 24 smoke episodes on dev seeds). The SHA-256 of this file is stored
in `results/week6_continuation/selection_rules.sha256` before C0 starts. Every selecting script
refuses to run if the hash differs. Implemented in `continuation/stats.py` (`rank_key`, `gates`,
`select`).

## 0. Common rule

- **Ranking key**, lexicographic on **pooled** (fixed + randomized) metrics:
  1. collision rate ↓
  2. success rate ↑
  3. timeout rate ↓
  4. mean true min-clearance ↑
  5. H1/H2: infeasible-step rate ↓; R2/R3: mean step time ↓
  - Remaining ties: the lexicographically smaller configuration tag.
- **G1 (pooled):**
  - pooled timeout ≤ 10 %;
  - pooled success ≥ reference − 5 pp.
- **G2 (each condition):**
  - success ≥ reference − Δ;
  - collision ≤ reference + Δ;
  - Δ = 10 pp at ≥ 50 episodes per condition, 15 pp at 20.
- **Reference:** the paired reference arm on the same seed block (listed per stage below).
- **Rejections:** every rejected candidate is listed with its failed gate. No candidate that fails
  G1 or G2 can be selected.
- **Reporting:** fixed, randomized and pooled metrics are reported for every candidate.

## C1 — buffer diagnostic (no automatic change)

- N = 5 is kept for H1.
- **STOP for review** if some N′ ∈ {1, 3} has both:
  - pooled collision lower than N = 5 with exact McNemar p < 0.05;
  - a pass on G1 and G2 against N = 5.

  In that case the user decides whether to change N. N is never changed automatically.

## H1 — coarse search (H1 block, 20 episodes/condition)

1. **Run** all 120 cells (5 α × 6 r_W × 4 ε).
2. **Reference:** the cell (0.4, 0.004, 0.1). It is bit-identical to C0-A (gate G-B).
3. **Main regime:** ε ∈ {0.05, 0.10, 0.20}, collapsed (εN ≤ 1).
   - Group cells by `effective_class()` = (α, τ_eff, regime, λ_V, k_v, N), which gives 50 classes.
   - Each class is represented by its **lowest-r_W cell, then lowest ε**.
   - The agreement of duplicate cells is reported as the empirical degeneracy check. It is never
     used for selection.
4. **Gates:** G1 + G2 (Δ = 15 pp) against the reference.
5. **Shortlist:** the **top 10 eligible classes** by the ranking key. If fewer than 10 are
   eligible, all eligible classes go through. If none is eligible, **STOP**.
6. **Non-collapsed regime:** ε = 0.30 (30 cells) is ranked and reported **separately**. It is never
   forwarded to H2.

## H2 — fine search (H2 block, 50 episodes/condition)

1. **Candidates:** each shortlisted representative × λ_V ∈ {0.25, 0.5, 1.0, 2.0, 4.0} ×
   k_v ∈ {0.025, 0.05, 0.10}.
2. **Anchor:** a C0-A arm on the same block is the reference.
3. **Gates:** G1 + G2 (Δ = 10 pp) against the anchor.
4. **Tuned-CLF-DR-CBF** = the **rank-1 eligible candidate**. If none is eligible, **STOP**.
5. **Freeze:** the configuration is written to `results/week6_continuation/H2/tuned_frozen.json`
   with its SHA-256. It cannot change afterwards.
6. **Caveat:** the H2 figures of the selected configuration are optimistically biased (winner's
   curse). Only F, on fresh seeds, is used for inference.

## RANDOM_EXP_R1 (R1 block, 100 episodes/condition)

- **Arms:** Tuned vs Tuned + random(K = 8, H = 1, speeds = (0.2,)).
- Report only; nothing is selected.
- R2 runs regardless of the R1 outcome. It is cheap, and it decides the R3 speed axis.

## RANDOM_EXP_R2 (R2 block, 100 episodes/condition)

- **Arms:** Tuned, R1 (as above), and R2 = random(K = 8, H = 1,
  speeds = (0.2, 0.4, 0.6, 0.8, 1.0)), accepting at m ≥ 0 and otherwise executing the best over
  all speeds.
- **R3 speed axis:**
  - If `rank_key(R2) ≤ rank_key(R1)` (5th key = mean step time), the axis is the **escalation
    start** v₀ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, with schedule {v ≥ v₀}.
  - Otherwise it is **fixed speed** v ∈ {0.2, 0.4, 0.6, 0.8, 1.0}.

## RANDOM_EXP_R3

- **Screen** (R3 screen block, 20 episodes/condition):
  - 45 configurations = K ∈ {4, 8, 16} × H ∈ {1, 3, 5} × 5 speed schedules.
  - Gate: G1 timeout clause only, because there is no paired reference on the screen block.
  - **Shortlist:** the top 4 by the ranking key (5th = mean step time).
- **Validation** (R3 validation block, 100 episodes/condition):
  - Arms: the 4 shortlisted configurations + Tuned (the reference).
  - Gates: G1 + G2 (Δ = 10 pp) against Tuned.
  - **Random-CLF-DR-CBF** = the rank-1 eligible configuration. If none is eligible, **STOP**.
  - The configuration is frozen to `results/week6_continuation/R3/random_frozen.json`.
- **Inspected and reported, never used for selection:** collision after infeasibility, recovery
  success, samples, speed, duration, tiers, CBF margin, path deviation.

## F — final paired comparison (sealed FINAL block, 200 episodes/condition)

- **Arms:**
  - F-A: C0-A, the frozen original.
  - F-B: Tuned.
  - F-C: Tuned + LADDER, with τ = τ_eff and no T1.
  - F-D: Random-CLF-DR-CBF.
- **Comparisons**, per condition and pooled:
  - A→B: the effect of tuning;
  - B→C: LADDER after tuning;
  - B→D: random recovery after tuning;
  - C vs D: H4.
- **Statistics:**
  - binary outcomes: exact McNemar;
  - continuous outcomes: paired bootstrap, 10 000 replicates, 95 % CI, α = 0.05;
  - step-level rates: episode-cluster bootstrap.
  - p-values are reported unadjusted, and the number of comparisons is stated.
- **Seal:** the FINAL block is opened only by `cont_final.py`, once, after both frozen files exist
  and their hashes verify.
