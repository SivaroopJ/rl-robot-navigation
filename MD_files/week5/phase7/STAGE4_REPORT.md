# Week5-Phase7 / Stage 4 — M2 recovery ladder

**Dated 2026-09-08.** Pre-registered in `PHASE7_PLAN.md` §11.8 (`aa66cd7`); implementation and
tests committed at `0073790` before the run. V0 frozen; α, τ, ε, `v_cap`, the objective, sample
selection, the planner and the LiDAR pipeline unchanged; no estimator change, IMM/CT,
covariance, uncertainty or governor logic. `u = 0` unchanged outside the treatment arms.
Baseline is the **Stage-3 T1 configuration** (`v_cap = 0.96`), not frozen C0.

Gates: frozen manifest **PASS**; full suite **266 passed**; feasible-step inertness pinned by
test (bit-identical to B4 single-step and over 20-step sequences).

Final tier, 200 paired episodes per condition, Phase 6 protocol and seeds.

---

## 1. Primary claim — characterisation of the M2 population and the ladder

### Recovery fraction by rung

| condition | arm | infeasible steps | recovered | rungs |
|---|---|---|---|---|
| fixed | B4 | 561 | 0 (0.0 %) | `fallback_u0` 561 |
| fixed | R1 | 566 | **75 (13.3 %)** | `fallback_u0` 491, `R1` 75 |
| fixed | R2 | 377 | **377 (100 %)** | `R2` 358, `R2_lp` 19 |
| fixed | LADDER | 383 | **383 (100 %)** | `R2` 280, `R1` 83, `R2_lp` 20 |
| fixed | R1.5 *(ablation)* | 388 | 357 (92.0 %) | `R1.5` 357, `fallback_u0` 31 |
| randomized | B4 | 671 | 0 (0.0 %) | `fallback_u0` 671 |
| randomized | R1 | 655 | **79 (12.1 %)** | `fallback_u0` 576, `R1` 79 |
| randomized | R2 | 580 | **580 (100 %)** | `R2` 551, `R2_lp` 29 |
| randomized | LADDER | 573 | **573 (100 %)** | `R2` 409, `R1` 139, `R2_lp` 25 |
| randomized | R1.5 *(ablation)* | 561 | 524 (93.4 %) | `R1.5` 524, `fallback_u0` 37 |

`R2_lp` denotes the 19–29 steps where the lexicographic second stage failed and the max–min LP
solution was returned; it is reported separately rather than folded into `R2`.

### Guarantee tier and achieved margin `m`

| condition | arm | T0 | T1 | **T2** | m median | m min | worst decay floor |
|---|---|---|---|---|---|---|---|
| fixed | R1 | 0 | 11 | **64 (85.3 %)** | **−0.0000** | −0.0000 | −0.0000 |
| fixed | R2 | 0 | 76 | **301 (79.8 %)** | −0.0703 | −0.5722 | −1.4306 |
| fixed | LADDER | 0 | 14 | **369 (96.3 %)** | −0.0742 | −0.5722 | −1.4306 |
| randomized | R1 | 0 | 8 | **71 (89.9 %)** | **−0.0000** | −0.0000 | −0.0001 |
| randomized | R2 | 0 | 135 | **445 (76.7 %)** | −0.0587 | −0.6377 | −1.5943 |
| randomized | LADDER | 0 | 18 | **555 (96.9 %)** | −0.0646 | −0.6377 | −1.5943 |

**No recovered action reached T0 in any arm.** Recovery buys an action, not the DR guarantee.

**The R1 boundary cases, reported exactly as the implemented tier rule classifies them.** R1's
recovered actions carry `m ≈ −0.0000` — values on the order of **−1e−9**, i.e. the τ = 0
constraint is active and the solver lands a hair below zero. Under the implemented rule
(`m < 0 ⇒ T2`) they are classified **T2**, and 64/75 and 71/79 of them are. **They are not
rounded into T1 and they are not called safe.** The magnitude is stated alongside the tier so
the difference between a −1e−9 boundary case and R2's −0.57 is visible: the same tier label
covers both, and only `m` distinguishes them. This is the same active-constraint boundary
behaviour documented in Stage 1, and no post-hoc tolerance was introduced to hide it.

**R2's worst-case decay floors reach −1.43 (fixed) and −1.59 (randomized).** These are bounds
on `h` implied by `ḣ ≥ −αh + m`, not predictions, and they say that some recovered actions carry
a very weak guarantee indeed.

### MIS cardinality — mechanism characterisation

| condition | arm | card 1 | card 2 | card 3 |
|---|---|---|---|---|
| fixed | B4 | **0** | 538 | 23 |
| fixed | R2 | **0** | 317 | 60 |
| fixed | LADDER | **0** | 315 | 68 |
| randomized | B4 | **0** | 657 | 14 |
| randomized | R2 | **0** | 523 | 57 |
| randomized | LADDER | **0** | 516 | 57 |

**No arm creates a singleton class.** The population remains M2 throughout, as Stage 3 left it.

---

## 2. P1–P4 verdicts against the pre-registered criteria

| | prediction | result | verdict |
|---|---|---|---|
| **P1** | R1 resolves < 25 % of M2 steps | 13.3 % (fixed), 12.1 % (randomized) | **CONFIRMED** |
| **P2** | R2 always returns an action, predominantly T2 | 377/377 and 580/580; T2 79.8 % / 76.7 % | **CONFIRMED** |
| **P3** | collision not significantly reduced vs B4 | see below | **CONTRADICTED for R2 / LADDER / R1.5; HOLDS for R1** |
| **P4** | residual MIS ≥ 2; no singleton class created | card 1 = 0 in every arm and condition | **CONFIRMED** |

---

## 3. P3 — paired inference, reported separately by condition

Exact paired McNemar against B4 on the same seeds:

| condition | arm | B4 → arm | p | verdict |
|---|---|---|---|---|
| fixed | R1 | 0.135 → 0.110 | 0.1797 | not significant |
| fixed | **R2** | **0.135 → 0.055** | **0.0004** | **significant** |
| fixed | **LADDER** | **0.135 → 0.065** | **0.0013** | **significant** |
| fixed | R1.5 *(abl.)* | 0.135 → 0.055 | 0.0000 | significant |
| randomized | R1 | 0.280 → 0.270 | 0.7266 | not significant |
| randomized | **R2** | **0.280 → 0.175** | **0.0000** | **significant** |
| randomized | **LADDER** | **0.280 → 0.180** | **0.0002** | **significant** |
| randomized | R1.5 *(abl.)* | 0.280 → 0.180 | 0.0002 | significant |

**The apparent collision reduction survives paired testing, in both conditions, for R2, LADDER
and R1.5.** P3 predicted a null and is contradicted. The pre-registered expectation was based on
Stage 3, where removing ~40 % of infeasible steps moved fallback-associated collisions
essentially not at all; that reasoning does not carry over, because Stage 3 removed infeasible
*steps* whereas Stage 4 replaces the *action taken* at them.

The mechanism is visible in the attribution: collisions **after an infeasible step** fall
22 → 4 (fixed) and 39 → 12 (randomized) under R2, while collisions **while feasible** rise
slightly, 5 → 7 and 17 → 23. The net is a large reduction. So the `u = 0` fallback *was* a
substantial collision pathway after all — a finding that **re-strengthens the conjecture Stage 3
weakened**, and which Stage 3 could not have detected because it never changed the fallback.

Descriptive arm-level outcomes (not paired inference): success 0.860 → 0.935 (fixed) and
0.720 → 0.820 (randomized) under R2; SPL 0.772 → 0.810 and 0.619 → 0.675; mean minimum clearance
0.252 → 0.273 and 0.183 → 0.205.

---

## 4. Findings that qualify the headline

* **The infeasibility rate itself falls in the recovering arms** — 0.0232 → 0.0124 (fixed),
  0.0282 → 0.0189 (randomized). This is **not** recovery reducing M2: it is a closed-loop
  effect. A recovered action moves the robot somewhere B4 never went, so fewer infeasible states
  are visited. The M2 population these arms face is therefore not the same population B4 faced,
  and the recovery fractions in §1 are conditional on each arm's own trajectory.
* **Worst-case clearance does not uniformly improve.** Fixed: −0.0510 → −0.0398 (R2, better) but
  −0.0580 (LADDER, worse). Randomized: −0.0629 → −0.0702 (R2 and LADDER, worse), and −0.1451
  for the R1.5 ablation — more than twice B4's worst excursion.
* **Static collisions appear where B4 had none:** 0 → 2 (R2, randomized), 0 → 1 (LADDER and
  R1.5). Small, but a regression in a category that had been exactly zero since Phase 6.
* **R1.5's admission checks passed** and it performed comparably to R2, with `α'` reaching 5.99
  (fixed) and 7.20 (randomized) against the `1/dt = 10` cap. It remains an **ablation**, not a
  primary arm, and its worst-clearance regression above is a reason not to promote it.
* Rung-switching rates are low (0.017–0.028 per step), so chatter is not evident.

---

## 5. Execution infrastructure (not an experimental result)

The first final-tier attempt was terminated by a **system-wide low-memory condition** after
completing the fixed condition's four primary arms. The experiment was changed to write results
per condition and to retain only the fields the paired tests need, and was re-run from scratch;
every number above comes from complete re-runs. Separately, a shell wrapper intended to chain
the two conditions deadlocked because `pgrep -f` matched the very shell that had created it; it
was removed and the second condition launched directly. **Neither event affected any measurement**
— both are execution infrastructure and are recorded here only for completeness.

---

## 6. What this does and does not establish

**Establishes:** the residual M2 population is fully recoverable in the sense that R2 always
returns an action (100 %); the guarantee obtained is weak and quantified — no T0, predominantly
T2, median `m` ≈ −0.06 to −0.07 with worst decay floors near −1.6; no arm creates a singleton
class; and the collision reduction is real under paired testing in both conditions.

**Does not establish:** that recovered actions are safe — they are not, and none is described
that way. `m < 0` means forward invariance is lost, and 77–97 % of recoveries are in that state.
The collision result is an outcome measurement, not a guarantee: the ladder trades a hard
constraint for a quantified violation and, in this environment, that trade happened to pay.
Worst-case clearance and the new static collisions are the visible cost.

**Not attempted:** any estimator, IMM/CT, covariance or uncertainty work, which remains gated;
and the `v_cap` sensitivity arms, which were pre-registered but not run.

**Stage 5 has not been started.**
