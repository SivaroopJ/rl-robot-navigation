# Week5-Phase7 / Stage 1 — SECOND RUN under Amendment 1

**Dated 2026-09-06. Amendment recorded at `3c85f9d` BEFORE this run. The FIRST run and its
FAILED gate stand permanently at `998eeaa`; nothing here rewrites or relabels them.**

## Amended gate result: **FAIL** (primary FAIL, safety/L4 FAIL, closed loop PASS)

| criterion | pre-registered pass condition | measured | verdict |
|---|---|---|---|
| **P1** feasible-set equality | 0 disagreements | **0** in 516 088 comparisons, 248 steps × 2 081 probes | **PASS** |
| **P2** coefficients + ordering | bitwise, every corpus step | **0** mismatches in **17 492** steps (kept 0, sort 0, weights 0) | **PASS** |
| **P3** bounds, τ, barrier, objective | exact | `‖ū‖∞ ∈ [1.0, 1.0]` over 10⁶ admissible `u`; τ = 0.04; εN = 0.5; box precondition holds | **PASS** |
| **P4a** `u_V1` in the V0 feasible set | 100 % | **573 / 573** | **PASS** |
| **P4b** suboptimality gap ≤ `J_tol = 1e−9` | 0 exceedances | max **8.93e−9**, **17 / 573 = 2.97 %** exceed | **FAIL** |
| **L4** guarantee-tier regressions | 0 | **1 406** of 14 612 (all T0→T1) | **FAIL** |
| **Closed loop** | McNemar p > 0.05 and CI ⊂ ±0.03, all outcomes both conditions | all 6 tests pass | **PASS** |

Secondary (reported, not a gate), n = 573:

| | median | p95 | p99 | max |
|---|---|---|---|---|
| **E0** = ‖u_V0 − u_arbiter‖∞ | 5.840e−7 | 2.045e−5 | 5.555e−5 | 9.158e−4 |
| **E1** = ‖u_V1 − u_arbiter‖∞ | 1.266e−11 | 5.311e−10 | 2.171e−9 | 3.199e−8 |

**Fraction of cases where E1 < E0: 0.9983.** The original failed thresholds
(`≤1e−6` vs V0, `≤1e−8` vs arbiter, objective `≤1e−8`) remain recorded as failed in
`STAGE1_REPORT.md` §1 and are not re-scored here.

Status matrix, five statuses kept distinct: `V0=optimal|V1=optimal` 14 612;
`V0=infeasible|V1=infeasible` 541; `V0=infeasible_inaccurate|V1=infeasible` 2. No step moved
into or out of `optimal`.

Closed loop (the **same** 400 episodes from `998eeaa`; the criterion changed, the data did not):

| condition | outcome | V0 → V1 | McNemar | CI95 | verdict |
|---|---|---|---|---|---|
| fixed | success | 0.8550 → 0.8650 | n01=2 n10=0, p=0.50 | [+0.0000, +0.0250] | PASS |
| fixed | collision | 0.1400 → 0.1300 | n01=0 n10=2, p=0.50 | [−0.0250, +0.0000] | PASS |
| fixed | timeout | 0.0050 → 0.0050 | p=1.00 | [0, 0] | PASS |
| randomized | success | 0.7250 → 0.7250 | n01=1 n10=1, p=1.00 | [−0.0150, +0.0150] | PASS |
| randomized | collision | 0.2750 → 0.2750 | n01=1 n10=1, p=1.00 | [−0.0150, +0.0150] | PASS |
| randomized | timeout | 0.0000 → 0.0000 | p=1.00 | [0, 0] | PASS |

The 4 divergent episodes remain reported: fixed 1000010 and 1000165 collision→success;
randomized 1000118 collision→success, 1000127 success→collision.

---

## Diagnosis of the two failures

### P4b — the threshold sits below the noise floor of the instrument that measures it, and my calibration was flawed

Gap distribution: min **−1.443e−8**, median **−2.22e−16**, p95 6.08e−10, p99 3.03e−9,
max 8.93e−9.

The median is one machine epsilon: `u_V1` is at the arbiter optimum. But the **minimum is
negative at −1.44e−8** — on some steps `u_V1` achieves a *lower* objective than the arbiter,
which is only possible if **the arbiter itself is suboptimal by ~1e−8 there**.

`J_tol = 1e−9` was calibrated by solving with CLARABEL at `tol 1e−14` versus `tol 1e−12` and
taking 300× the observed max spread (3.2e−12). **That calibration was wrong, and the error is
mine.** Comparing a solver against *itself* at two tolerances measures its tolerance
*sensitivity*, not its distance from the true optimum; the two runs share an algorithm and
their errors are correlated. The arbiter's true error, revealed here by `u_V1` beating it,
is ~1.4e−8 — four orders larger than my calibration suggested. A 1e−9 threshold on a
difference measured with a ±1.4e−8 instrument cannot be met by any implementation.

Note P4a, the part of P4 that needs no threshold, passes **573/573**.

### L4 — the tier boundary coincides exactly with where the optimizer puts the solution

All 1 406 regressions are T0→T1. **None reaches T2.** Distances from τ = 0.04:

| | n | min | median | max |
|---|---|---|---|---|
| all V0 steps, `minCBC − τ` | 14 612 | **−9.437e−5** | −6.686e−8 | +1.668 |
| all V1 steps, `minCBC − τ` | 14 612 | **−1.962e−9** | +2.480e−12 | +1.668 |
| regression steps, V1 side | 1 406 | −1.962e−9 | −1.779e−11 | −2.08e−17 |
| improvement steps, V0 side | 3 978 | −9.413e−5 | −4.259e−6 | −3.80e−12 |

96.44 % of the "regressions" are within **1e−9** of τ and 18.99 % within **1e−12**. The worst
excursion below τ under V1 is **1.96e−9**; under V0 it is **9.44e−5**, five orders larger.
There are 3 978 improvements against 1 406 regressions.

The mechanism: the barrier constraint is **active** at the optimum — the optimizer drives
`min CBC` to exactly τ — so the T0/T1 boundary lies exactly where the solution sits. Which
side of it a step lands on is decided by floating-point noise, in both implementations. The
criterion "0 tier regressions", evaluated with an exact `≥ τ` test, is therefore unsatisfiable
by *any* implementation that solves the problem accurately; the more accurate the solver, the
closer to the boundary it lands and the more often it straddles.

This is a defect in the tier metric used as an *equivalence* criterion. It is **not** a safety
regression: no step moved into T2, and V1's worst excursion below the theoretical DR-CBF floor
is 1.96e−9 (0.0000049 % of τ) against V0's 9.44e−5.

---

## What this does and does not establish

**Established, and not threshold-dependent:**
* the two formulations define the **same feasible set** — 516 088 independent membership
  comparisons, 0 disagreements, the V0 side decided by an LP oracle that never consults the
  accelerated implementation;
* **identical** reconstructed coefficients and critical-row ordering over all 17 492 corpus steps;
* **identical** bounds, τ, barrier formulation and objective formulation;
* `u_V1` lies in the V0 feasible set on **573/573** sampled solved steps;
* feasibility classification is unchanged: no step moved into or out of `optimal`;
* the closed loop is **statistically/distributionally equivalent** under the pre-registered
  paired protocol.

**Not established:** any claim resting on a numerical threshold. Two attempts have now failed,
and both failed the same way — a threshold set against a reference whose own error exceeds it.
The first time it was V0's SCS error; the second time it was the arbiter's, plus a calibration
procedure of mine that measured the wrong quantity.

**The frozen-implementation observation, worded as amended (§11.4.7):** V0 attains
`min CBC < τ` on 68.5 % of solved steps, reaching 9.44e−5 below τ. This is a **numerical
violation of the theoretical DR-CBF floor in the frozen SCS implementation**, of magnitude
~0.24 % of τ. It is a solver-accuracy artefact; no claim about physical safety follows from it.

Throughout, the candidate is described as a **mathematically equivalent accelerated
implementation**. It is *not* called behaviour-preserving or trajectory-identical.

---

## Recommendation

I am **not** proposing a third threshold. The pattern — two failures, both from thresholds set
against noisy references — says the remaining criteria should be made **threshold-free**, and I
put two structural replacements to you rather than adopting either:

* **P4b → KKT residual certificate.** For a strictly convex QP the optimality of `u_V1` can be
  certified *directly* from its own KKT residual, with no arbiter and no reference solver.
  Report the residual and require it below the problem's conditioning-scaled machine precision.
  This removes the measuring instrument from the criterion entirely.
* **L4 → tier of the exact optimum.** Assign the tier from the **arbiter's** solution and
  require V1's tier to match it, reporting the raw `minCBC − τ` excursion for both
  implementations alongside. Absolute, not relative to V0. The `≥ τ` test still needs a
  resolution statement, which the KKT route above would supply.

**Alternative, entirely legitimate:** hold both criteria as written, record the second run as
failed as well, and keep V0 as the Stage 2–7 substrate. The costs stated in
`STAGE1_FORENSIC.md` §6 apply unchanged.

**Provenance.** First run `998eeaa` FAILED (original gate). Amendment 1 `3c85f9d`. Second run
this document, FAILED (amended gate). No result file has been overwritten; both runs stand.

**Stage 2 has not been started.**
