# 05: Fast-solver training-suitability check

**What to build:** The researcher gets a recorded, pre-registered decision on whether PPO training may use the existing fast DR-CCP solver or must use frozen SCS. The check answers only one question: is the existing fast solver faithful enough to the frozen SCS controller to accelerate PPO training?

**Blocked by:** 04 (Baselines and stop check)

**Status:** done (`1558344` code, `4a2ec26` results). Verdict: FAIL → train with frozen SCS

- [x] The check uses the unmodified fast solver (reduced OSQP variant) at the exact frozen Random parameters: α 0.8, r_W 0.012, ε 0.1, k_v 0.10, 5 scans, τ_eff 0.12, K 16, H 1, speeds {0.8, 1.0}
- [x] E-a (shadow, per step): on 100 HD_DEV seeds × 2 conditions driving both the floor and the ceiling arm, the fast solver solves the identical inputs (p, γ, ξ, previous u) as the frozen solver
  - reports feasibility-category agreement (must be 100%)
  - reports the ‖Δu‖∞ distribution (≤ 1e-3 on ≥ 99% of optimal steps; maximum reported)
- [x] E-b (closed loop): each arm runs with the fast solver against the same arm with the frozen solver, paired by seed; passes if McNemar p > 0.05 and the success-difference 95% CI lies within ±0.03
- [x] Verdict recorded: pass → train with the fast solver; fail → train with frozen SCS. No change to the fast solver in either case
- [x] The write-up cites Stage 1 as background (feasible-set equality, tier agreement and closed-loop checks passed; strict KKT failed) and states that Stage 1 is not reopened or amended
- [x] The write-up states the fast solver is an acceleration mechanism, not a formally equivalent replacement

## Comments

- 2026-09-18. Code `1558344`, results `4a2ec26`. Report: `MD_files/highdim/HD1_SOLVER_CHECK_REPORT.md`.
- **Verdict: FAIL → PPO training uses frozen SCS.**
  - E-a category agreement was 58 301 / 58 667 (99.38 %), against a required 100 %.
  - E-a action: 99.81 % of both-optimal steps within 1e−3, max 3.2e−3 (pass).
  - E-b passed for both arms: floor 0.620 → 0.615, ceiling 0.880 → 0.875, both p = 1, CIs inside ±0.03.
- **The cause is the fast solver, not the category definition.**
  - 365 of 366 disagreements are steps where frozen was optimal and the fast solver was not: `solver_error` (OSQP iteration cap, 306) or `optimal_inaccurate` (59).
  - They fail under any reading of the categories.
- **Interpretations fixed before the run** (disclosed in the report):
  - three categories (optimal / infeasible / other), following what the policy does with each;
  - E-a decided on the corpus pooled over both arms;
  - the E-b frozen side reuses the E-a episodes (the shadow is proven not to change them).
- **Cost consequence:** training runs at about 28 ms per QP on frozen SCS. The spec estimates ~90 CPU-hours for 3 × 3M steps.
- **Review follow-ups (not done, judgement calls):**
  - hd0/hd1 share a provenance/run/write skeleton that could be extracted;
  - the solver string is branched on in three places.
