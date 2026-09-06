# Week5-Phase7 / Stage 1 — forensic diagnosis of the failed gate

**Dated 2026-09-06. Linked to the original Stage 1 run, commit `998eeaa`
(`Week5-Phase7-Stage1`), whose failed result stands permanently and is not amended by this
document.** This is a diagnosis, not a rerun. No tolerance, status mapping, controller,
objective, sample selection, τ or α was changed. No failing case was excluded. No implementation
error was found, so nothing was fixed.

Evidence: `experiments/exp7_1_forensic.py`, `results/week5_phase7/stage1_equivalence/stage1_forensic.json`.

---

## 1. Exact gate results

| | criterion (pre-registered) | tolerance | measured | failures | verdict |
|---|---|---|---|---|---|
| **L1** | exact status-string agreement on the enriched infeasible corpus | 100 % | 0.99786 (reduced_osqp) | **5 / 2 337 = 0.214 %** | **FAIL** |
| **L2** | `‖u − u_V0‖∞` on solved steps | ≤ 1e−6 | median 6.36e−7, max 1.695e−3 | **2 468 / 5 380 = 45.87 %** | **FAIL** |
| **L2′** | `‖u − u_arbiter‖∞` (amended criterion, also pre-registered) | ≤ 1e−8 | mean 1.8e−10, max 1.80e−8 | exceeds by 1.8× | **FAIL** |
| **L3** | objective agreement | ≤ 1e−8 | max 2.7e−4 – 5.2e−4 | tracks L2 exactly | **FAIL** |
| **L4** | guarantee-tier changes | 0 | **0 / 5 380** | **0** | **PASS** |
| **L5** | identical episode outcomes, 200 paired seeds × 2 conditions | 1.000 | 0.990 / 0.990 | **4 / 400 episodes** | **FAIL** |
| **L6** | runtime | — | 27.4–29.3 → 0.408–0.440 ms | — | measured |

### L1 by corpus part (reduced_osqp), no status summarised away

| part | disagreements | steps | % |
|---|---|---|---|
| dev_fixed | 0 | 2 046 | 0.0000 |
| dev_randomized | 1 | 1 861 | 0.0537 |
| validation_fixed | 1 | 5 559 | 0.0180 |
| validation_randomized | 0 | 5 689 | 0.0000 |
| **enriched infeasible** | **5** | **2 337** | **0.2139** |
| non-enriched total | 2 | 15 155 | 0.0132 |

Every disagreement, verbatim, with the LP-exact verdict on the same step:

| step | V0 status | accelerated status | LP-exact | action identical |
|---|---|---|---|---|
| 102 | `infeasible_inaccurate` | `infeasible` | infeasible | yes |
| 253 | `infeasible_inaccurate` | `infeasible` | infeasible | yes |
| 2153 | `infeasible_inaccurate` | `infeasible` | infeasible | yes |
| 1622 | `infeasible` | `infeasible_inaccurate` | infeasible | yes |
| 1467 | `infeasible` | **`solver_error`** (OSQP max_iter) | infeasible | yes |

param_scs additionally produced `infeasible → optimal_inaccurate` on steps 103 and 1061, both
LP-exact infeasible — V1 is the least reliable variant on status. param_clarabel produced only
`infeasible_inaccurate → infeasible` (3 steps), i.e. **100 % agreement under the coarse
predicate**.

### Where L2 failures concentrate

Error is **not** uniform. Breakdown of `‖u_V0 − u_arbiter‖∞` (603 solved steps, validation_fixed):

| h_crit regime | n | V0 error mean | V0 error max | accelerated error max |
|---|---|---|---|---|
| [0, 0.2) — near contact | 39 | 3.12e−5 | **8.92e−4** | 1.70e−8 |
| [0.2, 0.5) | 307 | 5.86e−6 | 6.21e−5 | 1.80e−8 |
| [0.5, 1.0) | 232 | 3.66e−6 | 5.09e−5 | 1.27e−8 |
| [1.0, 2.0) | 25 | 1.19e−6 | 1.87e−5 | 9.74e−10 |

| active barrier rows at the optimum | n | V0 error mean | V0 error max |
|---|---|---|---|
| 0 | 76 | 4.97e−7 | 2.60e−5 |
| 1 | 405 | 4.43e−6 | 6.21e−5 |
| 2 (degenerate vertex) | 121 | **1.70e−5** | **8.92e−4** |
| 3 | 1 | 9.76e−6 | 9.76e−6 |

Error concentrates where `h_crit` is small and where two barrier rows are simultaneously
active — the numerically hardest and the safety-critical regimes. `n_active` explains more of
the variance than `h_crit` does. No dependence on obstacle condition (fixed vs randomized) or
on position within the trace was found.

---

## 2. Classification of each failed criterion

### L1 — **C (solver-status/termination semantics)**, with one instance of **D**

Evidence:
* All 5 disagreements are between labels that the frozen controller treats identically. Its
  `_solve` returns `u = 0` for **every** status other than `"optimal"`, so `infeasible`,
  `infeasible_inaccurate`, `optimal_inaccurate` and solver errors are one behavioural class.
* The LP-exact arbiter (scipy HiGHS, independent of both implementations) calls **all 2 337**
  enriched steps infeasible, including all 5 disagreeing ones. **V0's own misclassification
  rate against LP-exact is 0/2 337.** The two implementations and the arbiter agree on the
  substance; only the certificate spelling differs.
* 4 of 5 are `infeasible ↔ infeasible_inaccurate` — a tolerance-driven distinction (**D**)
  internal to how each solver certifies infeasibility.
* 1 of 5 (step 1467) is `infeasible → solver_error`: OSQP hit `max_iter = 20 000` without
  certifying. That is a genuine reliability difference **in kind** (C), not a mathematical one.
  It is 1 step in 2 337 (0.043 %) and it produced the identical action.

Not A: no code path differs — the frozen `_solve` and the accelerated `_solve_reduced` both map
"not optimal" to `u = 0`. Not B/E: §3 rules these out.

### L2, L2′, L3 — **G (pre-registered tolerance too strict), on top of D**

The decisive evidence, computed per step and not merely at the maximum:

```
correlation( ‖u_reduced − u_V0‖ , ‖u_V0 − u_arbiter‖ )  =  1.000000
median ratio ‖u_reduced − u_V0‖ / ‖u_V0 − u_arbiter‖    =  1.000000
```

| quantity | mean | p95 | max |
|---|---|---|---|
| ‖u_V0 − u_arbiter‖ | 6.456e−6 | 2.084e−5 | 8.915e−4 |
| **‖u_reduced − u_arbiter‖** | **1.835e−10** | **4.858e−10** | **1.799e−8** |
| ‖u_reduced − u_V0‖ | 6.456e−6 | 2.084e−5 | 8.915e−4 |

The third row equals the first to every printed digit. **The L2 discrepancy is not a difference
between two answers; it is V0's distance from the true optimum, measured through a more
accurate instrument.** The arbiter — the frozen formulation transcribed unchanged, solved by
CLARABEL at tol 1e−14 — belongs to neither implementation.

A criterion of 1e−6 against a reference carrying up to 8.9e−4 of its own error is unsatisfiable
by any implementation, including an exact one. L3 tracks L2 because the objective is evaluated
at those actions.

L2′ (≤ 1e−8 against the arbiter) fails by a factor of 1.8. That threshold was chosen *a priori*
with no data; the measured 1.8e−8 is 4.7 orders below V0's own error. This is **G** as well —
a badly chosen number, not a defect.

Corollary worth recording: **V0 undershoots its own DR floor τ = 0.04 on 413 of 603 solved
steps (68.5 %)**, reaching `min CBC = 0.03996336`. The accelerated controller reaches
`0.04000000`. The frozen controller violates the distributionally-robust constraint it exists
to enforce, by up to 3.7e−5, on two-thirds of steps.

### L5 — **D + G**, a consequence of L2, not an independent failure

Trajectories diverge at step ~6.5 on average. A per-step action difference of order 1e−5–1e−3
— which §2 shows is V0's error — compounds through a 150-step closed loop with six moving
obstacles. 4 of 400 episodes end differently: 3 collision→success, 1 success→collision;
exact McNemar p = 0.50 (fixed) and p = 1.00 (randomized).

Not F: the per-step *solutions* are not genuinely different — they agree with the true optimum
to 1.8e−8 (accelerated) and 8.9e−4 (frozen). Exact trajectory equality across two different
convex solvers is not achievable for this system, so the criterion tests solver bit-agreement
rather than equivalence.

### L4 — PASS, and it is the informative one

L4 is the only criterion that does **not** reference V0's numerical output: it asks whether the
guarantee tier changes. **0 changes in 5 380 solved steps.**

---

## 3. Independent recheck of mathematical equivalence

The accelerated implementation was **not** consulted in any check below.

**3.1 τ / r_W / ε.** `‖ū‖∞ = max(1, α, |uₓ|, |u_y|)` sampled at 200 000 admissible `u`:
range **[1.0, 1.0]**. So `τ = r_W·‖ū‖∞/ε = 0.004/0.1 = 0.04` is exactly constant on the
admissible set. Preconditions `εN = 0.5 < 1` and `max_v = 1.0 ≤ max(1, α) = 1.0` both hold.

**3.2 Feasible-set equality — the central test.** For 48 steps (all 8 gate-failure steps plus
40 random) and 2 081 probe points each (41×41 grid + 400 random), membership was decided twice:

* **V0 side:** for the fixed `u`, solve the `(t, s₁..s_N)` feasibility LP with scipy HiGHS —
  the literal constraint block with `u` substituted in.
* **reduced side:** direct arithmetic, `minᵢ (ū·ξᵢ) ≥ r_W‖ū‖∞/ε`.

**99 888 membership comparisons, 0 disagreements.** The two formulations define the same set.

**3.3 Coefficients, ordering, and h_crit** on every gate-failure step, re-derived from the
recorded `xi`:

* kept-sample matrix identical to what the corpus recorded V0 using: **True, all 8 steps**
* criticality sort `α·h + ∂h/∂t` ascending: **True, all 8**
* `‖∇h‖ = 1.000000` for every kept row (unit gradients as constructed)
* `p0 = 3`, `p1 = 4·h_crit`, `p3 = 5·h_crit` reproduced exactly
* `h_crit > 0` on all 8 (range +0.0989 to +1.0684) — so the `h_crit < 0` delegation branch
  is not involved in any failure
* `h_crit < 0` on **0 of 2 337** enriched steps, confirming the branch is dead in practice

**3.4 Variable bounds.** `|u| ≤ max_v` per axis in both; `δ ≥ 0` and the CLF row identical.

**3.5 Status semantics.** V0 reports CVXPY's status verbatim and treats anything ≠ `"optimal"`
as failure. The accelerated path maps OSQP statuses through an explicit table and applies the
same `≠ "optimal" → u = 0` rule. The mapping is documented in `OSQP_STATUS` and is the only
place the two vocabularies meet.

**Conclusion of §3: B (incorrect reduction) and E (different problem) are ruled out by direct
measurement, not by appeal to the implementation.**

---

## 4. The enriched infeasible corpus

2 337 steps, all recorded by V0 as `infeasible` (2 336) or `infeasible_inaccurate` (1).

| partition | count |
|---|---|
| genuinely infeasible per LP-exact | **2 337 / 2 337 (100 %)** |
| feasible per LP-exact (V0 misclassification) | **0** |
| V0 `optimal` | 0 (by construction of the corpus) |
| V0 `optimal_inaccurate` | 0 |
| V0 `infeasible` | 2 336 |
| V0 `infeasible_inaccurate` | 1 |
| accelerated `optimal` | **0** |
| accelerated `optimal_inaccurate` | **0** (reduced_osqp) / 2 (param_scs) |
| accelerated `infeasible` | 2 333 |
| accelerated `infeasible_inaccurate` | 3 |
| accelerated `solver_error` | 1 |
| `h_crit < 0` | 0 |

**Does the accelerated implementation change feasibility classification?**
For `reduced_osqp`: **no.** Zero steps move into or out of the "solvable" class — all 2 337
remain non-optimal, and the action is `u = 0` in every case. Under the coarse predicate the
policy acts on, 2 336/2 337 agree; the single exception (`solver_error`) is still a failure and
still yields `u = 0`.
For `param_scs`: **2 steps (0.086 %) move to `optimal_inaccurate`** on LP-exact-infeasible
problems. The frozen semantics treat that as a failure too, so the action is unchanged — but
this is a real status-level regression and V1 should not be used as a substrate.

Reverse direction, checked on the 15 155 non-enriched corpus steps (which contain 12 923 V0
`optimal` steps): **0 steps where V0 solved and the accelerated implementation reported
infeasible.**

---

## 5. Changes made

**None.** No tolerance changed, no case excluded, no status redefined, no controller, objective,
sample selection, τ or α touched, no post-hoc action-matching rule added. No implementation
error was identified, so §5's "fix only that error and rerun" does not apply.

Two items are recorded as *observations*, not fixes:
* OSQP `max_iter` was reached once in 2 337 infeasible problems. Raising it is a tuning change
  and was **not** made.
* `E1′ ≤ 1e−8` was set without data. It was **not** relaxed.

---

## 6. Recommendation: **B — amend before rerun**

The formulations are mathematically equivalent (§3: 99 888 independent membership comparisons
with 0 disagreements, τ constant, coefficients identical, ordering identical), and four
pre-registered numerical criteria are demonstrably inappropriate because each measures the
reference's own solver error rather than a difference between the two.

The case does **not** rest on speed. The load-bearing evidence is:

1. `corr(‖u_reduced − u_V0‖, ‖u_V0 − u_arbiter‖) = 1.000000` with median ratio 1.000000 — the
   discrepancy *is* V0's error, per step, not merely in aggregate.
2. Feasible-set equality proven independently of the accelerated code.
3. **L4, the only criterion that does not reference V0's numerical output, passes with 0/5 380.**
4. Feasibility classification is unchanged on 2 337/2 337 infeasible and 12 923/12 923 solved steps.
5. V0 violates its own DR floor on 68.5 % of solved steps; the accelerated implementation does not.

**Caveat that must be carried, and that argues against calling this "behaviour-preserving".**
Even with exact equivalence, L5 shows 4/400 episodes ending differently, because the closed
loop amplifies solver-level differences. **B1 is therefore distributionally equivalent to B0,
not trajectory-identical**, and no reimplementation of this controller in any solver could be.
Two consequences: (a) the Stage 7 B0-vs-B1 check must be read as a statistical test, which is
how the plan already specifies it; (b) `n_infeasible` shifts +0.9 %/+2.2 %, so Direction 1 must
be measured against B1 and never against B0's published counts.

**Not A**, because no implementation or comparison error was found — the comparison was
correct, the *criterion* was mis-specified. **Not C**, because equivalence is established by
measurement independent of the implementation under test, and because `param_scs` (V1) is the
only variant showing a true status-level regression; the recommendation covers `reduced_osqp`
(V3) only.

### Proposed amendment, for approval before any rerun

Not adopted. To be dated and linked to `998eeaa` if approved.

| # | pre-registered | proposed | justification |
|---|---|---|---|
| 1 | L1 exact status = 100 % | coarse predicate (`"infeasible" in status`; `optimal` vs not) ≥ 99.9 % **and** action agreement = 100 % on the enriched corpus, every disagreement enumerated and LP-arbitrated | the exact string is not what the system acts on; §4 shows classification is unchanged |
| 2 | L2 `‖u − u_V0‖ ≤ 1e−6` | `‖u − u_arbiter‖ ≤ ‖u_V0 − u_arbiter‖` on ≥ 99 % of solved steps | scale-free; does not privilege the reference's noise; V3 beats it by 4.5 orders |
| 3 | L2′ `≤ 1e−8` | withdrawn, superseded by #2 | chosen without data |
| 4 | L3 objective ≤ 1e−8 | evaluate at the arbiter optimum, same form as #2 | same cause |
| 5 | L5 identical outcomes | agreement ≥ 0.98 **and** paired McNemar p > 0.05 on success and collision, both conditions | exact trajectory equality is unachievable across solvers |
| 6 | — | **new:** report the `n_infeasible` shift; D1 measured against B1 | quantified at +0.9 %/+2.2 % |
| 7 | — | **new:** substrate is `reduced_osqp` only; `param_scs` excluded on the §4 status regression | measured |

**Alternative, if you prefer to hold the original criteria:** record Stage 1 as failed and keep
V0 as the Stage 2–7 substrate. Cost: ~6× on every subsequent experiment and the n = 500 final
evaluation becomes unaffordable. It also means retaining a controller that violates its own DR
floor on 68.5 % of steps. Legitimate, and your call.

---

## 7. Provenance

* The original Stage 1 run and its **failed** gate stand as commit `998eeaa`
  (`Week5-Phase7-Stage1`), with `STAGE1_REPORT.md` and
  `results/week5_phase7/stage1_equivalence/stage1_equivalence.json` unmodified.
* This diagnosis is dated 2026-09-06 and adds only `STAGE1_FORENSIC.md`,
  `experiments/exp7_1_forensic.py` and `stage1_forensic.json`. No prior result file was
  rewritten.
* If the amendment is approved, it will be recorded in `PHASE7_PLAN.md` §11.1.4 with this date,
  the triggering measurements, the original text retained, and an explicit link to `998eeaa`;
  any rerun will be reported as a *second* run alongside the first, never as a replacement.

**Stage 2 has not been started.**
