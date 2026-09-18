# 05: Fast-solver training-suitability check

**What to build:** The researcher gets a recorded, pre-registered decision on whether PPO training may use the existing fast DR-CCP solver or must use frozen SCS. The check answers only one question: is the existing fast solver faithful enough to the frozen SCS controller to accelerate PPO training?

**Blocked by:** 04 (Baselines and stop check)

**Status:** ready-for-agent

- [ ] The check uses the unmodified fast solver (reduced OSQP variant) at the exact frozen Random parameters: α 0.8, r_W 0.012, ε 0.1, k_v 0.10, 5 scans, τ_eff 0.12, K 16, H 1, speeds {0.8, 1.0}
- [ ] E-a (shadow, per step): on 100 HD_DEV seeds × 2 conditions driving both the floor and the ceiling arm, the fast solver solves the identical inputs (p, γ, ξ, previous u) as the frozen solver
  - reports feasibility-category agreement (must be 100%)
  - reports the ‖Δu‖∞ distribution (≤ 1e-3 on ≥ 99% of optimal steps; maximum reported)
- [ ] E-b (closed loop): each arm runs with the fast solver against the same arm with the frozen solver, paired by seed; passes if McNemar p > 0.05 and the success-difference 95% CI lies within ±0.03
- [ ] Verdict recorded: pass → train with the fast solver; fail → train with frozen SCS. No change to the fast solver in either case
- [ ] The write-up cites Stage 1 as background (feasible-set equality, tier agreement and closed-loop checks passed; strict KKT failed) and states that Stage 1 is not reopened or amended
- [ ] The write-up states the fast solver is an acceleration mechanism, not a formally equivalent replacement
