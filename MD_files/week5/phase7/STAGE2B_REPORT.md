# Week5-Phase7 / Stage 2b — velocity-estimation causal counterfactual

**Dated 2026-09-07. Diagnostic only.** V0 completely frozen. No IMM, CT tracking, uncertainty
margin, predictive barrier, recovery ladder, slack, or estimator redesign was implemented.
Amendment 3 (`e2cda71`) recorded the design before this run.

Chain preserved, nothing overwritten: `998eeaa` → `4c45c08` → `3c85f9d` → `00587e3` →
`4e55219` → `cc692ab` → `df42437` → `5415bc7` → `e2cda71` → this.

## Verdict, against the pre-registered interpretation rule (§11.6.3)

> *If ground-truth velocity substantially reduces infeasibility while everything else is held
> fixed, velocity estimation becomes the leading supported mechanism.*

**Ground-truth velocity reduces infeasibility by 66.1 %, removing 69.4 % of the baseline's
infeasible steps. Velocity estimation is therefore the LEADING SUPPORTED MECHANISM.**

It is **not the whole explanation**: a third of the baseline rate survives perfect velocity.

---

## 1. Result

All 69 226 corpus steps, four arms, identical recorded states, geometry and constraints; only
column 0 of `ξ` differs.

| arm | infeasible | rate | vs A1 | fixed | randomized | super-physical step rate | MIS = 1 |
|---|---|---|---|---|---|---|---|
| **A1** frozen estimate | 2 339 | 0.0338 | — | 0.0330 | 0.0346 | 0.0664 | **884** |
| **A2** ground truth | **792** | **0.0114** | **−66.1 %** | 0.0075 | 0.0154 | 0.0000 | **0** |
| **A3** zero | 4 | 0.0001 | −99.8 % | 0.0001 | 0.0001 | 0.0000 | 0 |
| **A4** capped estimate | 1 121 | 0.0162 | −52.1 % | 0.0122 | 0.0202 | 0.0000 | 0 |

Paired, on the same recorded step:

| arm | removes of A1's 2 339 | introduces new |
|---|---|---|
| A2 ground truth | **1 624 (69.43 %)** | **77** |
| A3 zero | 2 335 (99.83 %) | 0 |
| A4 capped estimate | 1 218 (52.07 %) | 0 |

Minimal infeasible subset sizes and remediability:

| arm | MIS=1 | MIS=2 | MIS=3 | slack median | slack max | more control would help |
|---|---|---|---|---|---|---|
| A1 | 884 | 1 404 | 51 | 0.1853 | **1.9480** | 2 115 |
| A2 | **0** | 738 | 54 | 0.1183 | 0.5256 | 614 |
| A3 | 0 | 3 | 1 | 0.0143 | 0.0299 | 1 |
| A4 | 0 | 1 066 | 55 | 0.1007 | 0.5229 | 912 |

`h_crit` is essentially unmoved (median 0.4860 with the recorded order held fixed, 0.4827–0.4860
re-sorted), confirming the geometry channel was not disturbed.

## 2. The single-constraint result is the decisive one

**All 884 cardinality-1 minimal infeasible subsets vanish in every arm that makes the velocity
physical — A2, A3 and A4 alike.** Stage 2 derived that a single row should never be
unsatisfiable inside the action box unless `∂h/∂t` exceeds what an obstacle can physically
produce. Stage 2b confirms the derivation causally: supply a physical velocity, by any of three
independent routes, and the entire class disappears. The maximum required slack also falls from
1.948 to 0.526.

## 3. What survives perfect velocity

792 steps (0.0114, one third of the baseline rate) remain infeasible under exact ground-truth
velocity. They are **entirely multi-constraint** (738 pairwise, 54 three-way; zero singletons),
milder (slack median 0.1183 vs 0.1853, max 0.526 vs 1.948), and concentrated in the randomized
condition — 0.0154 vs 0.0075 fixed, a **2.05× ratio** where the baseline showed almost none
(0.0346 vs 0.0330). Genuine multi-obstacle geometric conflict is therefore a real, separate
residual mechanism, and it is the part of the problem that gets harder under stochastic motion.

## 4. Evidence against over-reading the result

* **A2 introduces 77 new infeasible steps.** The estimator is not uniformly pessimistic; on
  those steps it *understates* the true closing rate, and perfect velocity makes the constraint
  harder. Any intervention that only suppresses overshoot would not help there.
* **A3 (zero velocity) removes 99.8 %** — more than ground truth. Infeasibility is trivially
  eliminated by discarding the `∂h/∂t` channel altogether, which is *not* a candidate remedy:
  Phase 3 measured `∂h/∂t = 0` at 0.44 collision rate against 0.12 for oracle velocity. A3 is
  included as a bound on the channel's contribution, not as an option.
* **A4 is a lower bound.** It clamps the projected closing rate, which for closing rows is at
  most what clamping the velocity vector would give. So "respecting the physical speed bound
  removes ≥ 52 % of infeasibility" is conservative.
* The result is about **infeasibility only**. Nothing here shows that reducing infeasibility
  reduces collisions; Phase 6's attribution (86 % / 67 % of collisions follow an infeasible
  step) is suggestive but was measured on the frozen system, not on any intervention.

## 5. Method and validation

* **Feasibility oracle.** An exact vertex-enumeration test for the 2-D polytope
  `{u : |u| ≤ max_v, Gu ≥ b}`, used for speed. Validated against the scipy-HiGHS LP definition
  on 1 500 random instances with both branches exercised (0 disagreements), and against the
  **actual frozen V0 solver on 2 400 arm-steps: 100.00 % agreement**.
* **Order invariance.** Feasibility depends on the constraint *set*, not its ordering, and with
  `K = 5, n_keep = 5` all rows are always kept — so substituting `∂h/∂t` can permute the
  criticality order but cannot change which rows are present. The infeasibility rate is
  identical whether the recorded ordering is held fixed or recomputed; only `h_crit` differs,
  and it is reported both ways.
* **Ground-truth attribution.** 93 488 of 340 730 buffered rows (27.44 %) attribute to a
  dynamic obstacle at tolerance 0.15; the rest are static structure and receive `v = 0`.
  Sensitivity: 27.25 % at 0.10, 27.55 % at 0.15, 28.20 % at 0.25 — the attribution is not
  tolerance-sensitive.
* **Baseline count.** The LP finds 2 339 genuinely infeasible steps against the 2 337 that SCS
  *labelled* infeasible. The two extra are exactly the `optimal_inaccurate` steps identified in
  Stage 0's correction C2, which the frozen controller already answered with `u = 0`. The
  discrepancy is fully accounted for and is not a new finding.

## 6. Limitations

* A2 uses ground truth, so it is an upper bound on what any realisable estimator could achieve.
  It bounds the mechanism's size; it is not an achievable target.
* Owner attribution assigns a buffered point to the true obstacle whose surface it lies on at
  scan time and reads that obstacle's *current* velocity, mirroring the frozen pipeline. A point
  produced by one obstacle and later occluded or straddled by another could be misattributed;
  the tolerance sweep bounds but does not eliminate this.
* `V_PHYS` is the configured obstacle speed, with the randomized regime's upper bound 0.75 used
  throughout, making A4 conservative.
* Infeasibility is a property of the recorded states. Substituting velocity in closed loop would
  change the trajectory and hence the states visited; that is an intervention and is out of
  scope.

## 7. Consequence

The mechanism is now **supported causally, not merely by association**, and it sits in
**Direction 3** (velocity estimation), not Direction 1 (infeasibility recovery). Two thirds of
infeasibility is attributable to the velocity channel; one third is genuine multi-constraint
conflict that no velocity fix will remove and that worsens under stochastic motion.

**No stage has been reordered and no intervention has been implemented.** Stage 3 has not been
started.
