# Week5-Phase7 / Stage 2 — D1a: why the frozen V0 DR-CBF goes infeasible

**Dated 2026-09-07. Diagnostic only.** No recovery policy, slack variable, τ relaxation, α
modification, predictive barrier, uncertainty margin, tracker change or controller change was
implemented. V0 is untouched. Stage 1 remains closed and FAILED (`cc692ab`); V0 is the
substrate and the accelerated implementation is used nowhere in this analysis.

Provenance preserved: `998eeaa` → `4c45c08` → `3c85f9d` → `00587e3` → `4e55219` → `cc692ab`.

## Conclusion: the staleness hypothesis is **CONTRADICTED** in its stated mechanism. A weaker cross-age effect is real but is not the dominant driver. The dominant correlate is the velocity estimator reporting physically impossible closing rates.

---

## 0. Scope

All 6 corpus parts, **69 226 steps**, **2 337 infeasible** (3.376 %), plus **15 761 sampled
feasible steps** as a contrast and a **66 889-step** feasible base-rate population.

Infeasibility rate is stable across conditions and tiers: dev_fixed 3.030 %, dev_randomized
4.514 %, validation_fixed 3.220 %, validation_randomized 3.832 %, final_fixed 3.336 %,
final_randomized 3.300 % (fixed 1 143 / randomized 1 194 of the 2 337).

---

## 1. Distribution — infeasibility is **not** where one would expect

| | infeasible | feasible control |
|---|---|---|
| `h_min` (median) | **0.5194** | 0.4845 |
| true TTC < 1 s | **0.3 %** | 2.2 % |
| distinct track ids per step | 2 → 1 088, 3 → 1 060, 4 → 184, 5 → 5 | — |
| ≥1 row unsatisfiable alone | 884 / 2 337 (37.83 %) | — |

**Infeasible steps are farther from obstacles than feasible ones and are seven times *less*
likely to be inside 1 s of contact.** Infeasibility is therefore not a proximity or
imminent-collision phenomenon, which already argues against any explanation resting on
geometry being tight.

## 2. Minimal infeasible subsets

| cardinality | count | share |
|---|---|---|
| **1** | 884 | **37.83 %** |
| 2 | 1 402 | 59.99 % |
| 3 | 51 | 2.18 % |

**Over a third of infeasible steps need only ONE constraint.** A single row `i` is
unsatisfiable inside the action box when `‖∇hᵢ‖₁·max_v < τ − α·hᵢ − ∂h/∂tᵢ`. With unit
gradients, `h ≥ 0` and a physical obstacle speed of 0.675–0.75 m/s the right side cannot exceed
≈ 0.79 < 1, so this **should be impossible** — unless `∂h/∂t` is larger in magnitude than any
obstacle can produce. §7 shows that is exactly what happens.

## 3. Temporal staleness

Age membership in the minimal infeasible subset (uniform ≈ 20 % per age):

| | age 0 | age 1 | age 2 | age 3 | age 4 |
|---|---|---|---|---|---|
| all MIS rows | **34.55 %** | 14.32 % | 15.49 % | 15.49 % | 20.15 % |
| cardinality-1 (control-bound) | 33.7 % | 27.0 % | 22.5 % | 10.4 % | 6.3 % |
| cardinality-2 (pairwise) | **34.9 %** | 10.7 % | 13.3 % | 16.7 % | **24.4 %** |

Owning-obstacle displacement since its scan: median **0.270 m**, p95 0.270 m — staleness of the
expected magnitude (0.675 m/s × 0.4 s) is definitely present in the buffer.

**But the newest scan is the most implicated age, not the oldest.** Pairwise conflicts show a
U-shape: newest (34.9 %) and oldest (24.4 %) dominate, with the middle ages under-represented.

## 4. Counterfactuals

| counterfactual | feasible fraction |
|---|---|
| **K = 1, newest scan only** (plan D1a-M3) | **0.8725**, CI95 [0.8588, 0.8858] |
| age-prefix {0..j} | j=0: 0.873 · j=1: 0.713 · j=2: 0.530 · j=3: 0.328 · j=4: 0.000 |
| **leave-one-out, remove age 0** | 0.1836 |
| leave-one-out, remove age 1 | 0.1472 |
| leave-one-out, remove age 2 | 0.1716 |
| leave-one-out, remove age 3 | 0.1669 |
| **leave-one-out, remove age 4 (oldest)** | **0.3282** |
| any single removal helps | 0.8451 |

**K = 1 is confounded and must not be read as the staleness share.** Dropping from 5
constraints to 1 makes any problem easier whether or not the removed rows were stale; 0.8725
measures "fewer constraints", not "less staleness".

**Leave-one-out is the controlled test** — every subset keeps 4 rows, so constraint count is
fixed and only the removed row's age varies. Removing the oldest restores feasibility 32.8 % of
the time versus 18.4 % for the newest: a **real but modest 1.79× age gradient**, nowhere near
dominant. The middle ages sit at 14.7–17.2 %, so the effect is concentrated at the extremes
rather than increasing monotonically with age.

## 5. Attribution

| class | count | share |
|---|---|---|
| control-bound limitation (cardinality 1) | 884 | 37.83 % |
| cross-age, involves an **untracked/static** return | 866 | 37.06 % |
| cross-age, **different** tracks | 587 | 25.12 % |
| **stale-vs-current, SAME track** — the hypothesis's signature | **0** | **0.00 %** |

Restricting to the 1 402 pairwise conflicts, which is where the hypothesis would live:

* **same track id: 0 (0.00 %)**
* same scan age: 0 (0.00 %) — *every* pairwise conflict is cross-age
* involves an untracked/static return: 835 (59.56 %)
* ≥1 super-physical row: 701 (50.00 %)

So conflicts *are* structurally cross-age, but **never** between a stale and a current view of
the same tracked object. They are between different objects, and most often one of them is a
static/untracked return.

## 6. Remediability (measured, nothing adopted)

Required uniform slack: median 0.1853, p95 1.3029, max 1.9480 against τ = 0.04. Tiers:
T0 0, T1 315 (13.5 %), **T2 2 022 (86.5 %)**. More control authority would help on 2 113 / 2 337
(90.42 %) — note this **differs sharply from Phase 1.5's analytic finding** of 95.3 %
*un*helpable, which is direct evidence that the analytic taxonomy does not transfer to the
LiDAR pipeline. Some `α ≥ 0.4` restores feasibility on 2 337 / 2 337 (100 %).

## 7. The competing explanation, with a proper control

`∂h/∂tᵢ = −∇hᵢ·v̂_track` is the **estimated** closing rate. If the Kalman tracker reports a
speed above what an obstacle can physically travel (0.675 m/s fixed, ≤ 0.75 m/s randomized),
a single row can demand more than the action box can deliver.

| | infeasible | feasible | risk ratio |
|---|---|---|---|
| P(≥1 super-physical row) | **0.6795** (1 588/2 337) | **0.0450** (3 010/66 889) | **15.1×** |
| overshoot magnitude, median | **0.8625 m/s** | 0.2423 m/s | — |
| overshoot magnitude, max | 2.8790 m/s | 1.0310 m/s | — |

Read the other way:

```
P(infeasible | >=1 super-physical row) = 1588/4598 = 0.3454
P(infeasible | no  super-physical row) =  749/64628 = 0.01159      risk ratio 29.8x
```

Among the 884 control-bound (cardinality-1) steps the association is **100 %** — every single
one has a super-physical row. Among non-control-bound steps it is 48.5 %.

The projected estimator error `e = ∂h/∂t_est − ∂h/∂t_true` on infeasible steps has a median
worst-row value of **−1.114 m/s** (min −3.554): the estimator reports obstacles closing
**over a metre per second faster than they truly are**. Note the sign — this is *pessimistic*
error, which tightens the constraint and creates infeasibility, the mirror image of the
optimistic error Phase 4 tracked for collisions.

---

## 8. Evidence for and against the hypothesis

**Supporting.** Staleness of the predicted magnitude is present (0.270 m median displacement).
Every pairwise conflict is cross-age. Prefix feasibility falls monotonically as older scans are
added. Removing the oldest row restores feasibility 1.79× more often than removing the newest.

**Contradicting.** The specific mechanism — historical geometry of an object combined with that
object's current velocity — occurs **0 times in 1 402** pairwise conflicts. The newest scan is
the most-implicated age, not the oldest. 37.8 % of infeasibility needs only one row and
therefore has no cross-age component at all. Infeasible steps are *farther* from obstacles and
7× less likely to be near contact. And the estimator-overshoot correlate is an order of
magnitude stronger (15× / 29.8× risk ratios) than the age gradient (1.79×).

## 9. Limitations

* **Association, not causation.** §7 is observational. Establishing causation would require
  suppressing the super-physical estimates and re-running, which is an intervention and is out
  of Stage 2's scope.
* Track identity comes from the frozen tracker; a track that fragments across scans would look
  like "different tracks" even for one physical object, which could mask same-object staleness.
  Mitigation: the geometric channel (owner displacement, via ground truth) agrees that the
  objects involved are distinct, but this remains the weakest link in the "0.00 % same-track"
  figure.
* `V_PHYS` is the configured obstacle speed; the randomized regime samples per episode from
  [0.6, 0.75] and 0.75 was used throughout, making the super-physical test *conservative*.
* Ground truth was used only as a measurement channel; no reconstructed problem was altered.
* The plan's gate G-D1a is stated on the K = 1 share, which this report shows is confounded.

## 10. Verdict and consequence for Stage 3

**CONTRADICTED as stated** — the named mechanism is absent (0/1402). **A weaker cross-age
effect is real** (all pairwise conflicts are cross-age; oldest-row removal helps 1.79× more
than newest) but explains far less than the estimator's super-physical closing rates.

**Gate G-D1a is ambiguous as written.** The confounded K = 1 share is 87.25 % (> 50 %, "run
de-staling first"); the controlled leave-one-out-oldest share is 32.8 %, which falls between
the plan's 20 % and 50 % thresholds. I recommend reading the controlled number and treating
the gate as *not* triggering the "de-staling first" branch — but that is a plan decision and I
have not made it.

Per §11.1, the interpretation is updated and the pre-registered experiment stands as run.
Stage 3 (D3a predictive barrier) remains independently motivated as a measurement, but this
report removes its billing as *the* fix for infeasibility. The finding that most naturally
follows the evidence — that the velocity estimator's pessimistic overshoot is the dominant
driver — falls under **Direction 3**, not Direction 1, and would reorder Stages 3–6 again.
**No such reordering has been made.**

**Stage 3 has not been started, and no intervention has been implemented.**
