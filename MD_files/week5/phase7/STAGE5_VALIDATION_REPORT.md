# Week5-Phase7 / Stage 5 / VALIDATION TIER — the A4 admission gate

**Pre-registration:** `PHASE7_PLAN.md` §11.9 (`9d38008`), with the `n = 1` derivation corrected
in §11.9.18 (`3b3a629`) **before** any code was written. Implementation `c867f59`.

**VERDICT: A4 FAILS. The 200-episode final run is NOT authorised.** Per §11.9.13-D2 the
estimator branch **terminates at the diagnostic**, `EVAL_SEED_BASE` was never touched, and the
null is the reported result. LADDER (Stage 4, `91bb969`) remains the established empirical
fallback baseline, and every Stage 1–4 conclusion stands unchanged.

**No Stage-5 conclusion is drawn from validation alone**, and nothing here was used to tune
anything — E1 has no free parameter to tune.

---

## 1. The gate, as pre-registered

> A4: on 50 held-out validation episodes per condition, E1 must show **both** (i) a lower
> `P(e > 0 | TTC < 1 s)` than A0 **and** (ii) a **non-increase** in the super-physical rate —
> jointly, in **both** conditions.

A4 is **directional, not a significance test**; significance is reserved for the final tier
(§11.9.13-B2). CIs below are printed for information. Requiring them to exclude zero would be
tightening the gate after seeing data, which §11.9.15 forbids.

Seeds `DEV_SEED_BASE + 1000 … +1049`, both conditions `make_env(rnd)` at the default
`obstacle_speed = 0.675` — identical to Stages 3 and 4.

| condition | criterion | A0 | A1 (E1) | Δ | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| fixed | (i) `P(e>0\|TTC<1s)` pooled | 0.4178 | 0.4076 | **−0.0102** | [−0.0446, +0.0263] | **PASS** |
| fixed | (ii) super-physical step rate | 0.0781 | 0.0847 | **+0.0067** | [−0.0021, +0.0159] | **FAIL** |
| randomized | (i) `P(e>0\|TTC<1s)` pooled | 0.4719 | 0.4812 | **+0.0093** | [−0.0348, +0.0506] | **FAIL** |
| randomized | (ii) super-physical step rate | 0.0535 | 0.0561 | **+0.0026** | [−0.0047, +0.0096] | **FAIL** |

**Three of the four directional criteria fail. Both conditions fail. A4 = FAIL.**

Episode-mean optimistic tail, the form comparable to the Stage-3 T1 reference: fixed
0.4250 → 0.4034 (reference 0.397); randomized 0.4711 → 0.4919 (reference 0.498). **Phase 4's
0.142 is a different harness and subset and is not used as a comparator anywhere.**

**Every one of the four deltas has a 95 % CI spanning zero.** At n = 50 E1's closed-loop effect
on these endpoints is not distinguishable from zero, and A4 failed on the *sign* of
noise-magnitude differences. That is a property of the gate as written; it is recorded as an
observation, **not** as a proposal to change the verdict, the gate, or any threshold.

---

## 2. Diagnosis — the pre-registered failure mode is CONTRADICTED

§11.9.8 pre-registered the competing explanation for exactly this shape of result:

> Inflating `R` makes the filter sluggish; a sluggish filter reports **smaller** velocities,
> which mechanically lowers the super-physical rate while **increasing** optimistic error.

That is **not** what happened, and the sluggish-filter hypothesis is **contradicted**.

The A4 numbers compare two **closed-loop** arms, so A0 and A1 visit different states and the
estimator is confounded with the trajectory. `experiments/exp7_5_diagnosis.py` removes the
confound: one trajectory is driven by the frozen A0 policy and the **identical `(p_ego, ranges)`
stream** is fed to both trackers in parallel. Ground truth is a metrics channel only; neither
tracker sees it.

| open loop, identical inputs | fixed frozen → E1 | randomized frozen → E1 |
|---|---|---|
| mean estimated speed | 0.6683 → 0.6783 (true 0.675) | 0.6475 → 0.6688 (true 0.675) |
| estimated/true speed ratio | 0.9901 → **1.0048** | 0.9593 → **0.9907** |
| mean `‖v̂ − v‖` | 0.3758 → **0.3251** (−13.5 %) | 0.3363 → **0.2971** (−11.7 %) |
| `P(optimistic projection)` | 0.5292 → **0.5108** | 0.5423 → **0.5177** |
| mean projected error | 0.0625 → **0.0265** | 0.0096 → **0.0004** |
| **`P(optimistic)` 1-point** | 0.5434 → **0.5079** | 0.4993 → **0.4518** |
| **mean `‖err‖` 1-point** | 0.4734 → **0.3940** | 0.4277 → **0.3548** |
| `P(optimistic)` 2-point | 0.5119 → 0.5309 | 0.5935 → 0.5949 |
| `P(optimistic)` 3+-point | 0.4519 → 0.4742 | 0.6051 → 0.6250 |

**On identical inputs E1 is a better estimator on every aggregate measure**, and the improvement
is **concentrated exactly where it was designed to act** — the 1-point returns, which is where
the frozen isotropic `R` is geometrically wrong. Speeds moved **toward** the true value, not
below it: the filter is **less** biased, not more sluggish.

**What actually failed is the transfer from estimator accuracy to the closed-loop endpoint**, not
the estimator. Three observations, none of which is a conclusion:

1. The 2-point and 3+-point splits move slightly the **wrong** way (+0.019 to +0.022), partly
   offsetting the 1-point gain in the pooled figure.
2. The closed-loop `P(e>0|TTC<1s)` is measured over the barrier's binding rows on trajectories
   that **differ between arms**, so it is not the same population as the open-loop measure.
3. The closed-loop effects are at noise magnitude at n = 50 in every endpoint family.

---

## 3. All seven endpoint families (DESCRIPTIVE — not the gate, not a conclusion)

n = 50 resolves far less than n = 200. Nothing below is a Stage-5 finding.

| endpoint | fixed A0 → A1 | randomized A0 → A1 |
|---|---|---|
| total infeasibility | 0.0201 → 0.0239 (CI [−0.0015, +0.0095]) | 0.0255 → 0.0302 (CI [−0.0011, +0.0119]) |
| **M1 singleton** | **0 → 0** | **0 → 0** |
| M2 multi-constraint | 0.0201 → 0.0239 | 0.0255 → 0.0302 |
| **T0 share** | 0.3869 → 0.3887 (CI [−0.0117, +0.0151]) | 0.3965 → 0.4021 (CI [−0.0136, +0.0252]) |
| T2 share | 0.0201 → 0.0239 | 0.0255 → 0.0302 |
| collision | 0.180 → 0.200, McNemar n01=2 n10=1 p=1.0000 | 0.340 → 0.380, n01=3 n10=1 p=0.6250 |
| collision split dyn/stat/wall | 9/0/0 → 10/0/0 | 17/0/0 → 19/0/0 |
| mean min clearance | 0.2713 → 0.2644 | 0.1663 → 0.1572 |
| worst-case clearance | −0.0578 → −0.0588 | −0.0675 → −0.0621 |
| success / timeout | 0.820 → 0.800 / 0.000 | 0.660 → 0.620 / 0.000 |
| `e` p95/p99/max | 0.062/0.199/0.350 → 0.074/0.216/0.452 | 0.137/0.410/0.598 → 0.176/0.440/0.678 |
| pessimistic tail | 0.1291 → 0.1383 | 0.1531 → 0.1562 |
| T1 clamp rate | 0.0455 → 0.0496 | 0.0395 → 0.0416 |

**M1 stays exactly 0 in every arm** — the §11.9.10-3 regression check passes; E1 did not break
T1's singleton elimination.

**No safety veto is invoked**, because the veto (§11.9.12-1) applies to the final tier. The
collision differences here are **not significant** (p = 1.0000 and 0.6250) and are reported as
**failed to detect harm at n = 50**; the CIs do not exclude an increase of up to +0.100 (fixed)
and +0.120 (randomized). **This is not evidence of safety.**

---

## 4. Two implementation errors found and corrected, with the invalid artefacts preserved

Both were found before any conclusion was drawn. Neither is a tolerance change or a hypothesis
change, and no threshold was touched.

1. **Fixed-condition environment mismatch.** The first fixed-condition run used
   `obstacle_speed = 1.0`, but Stages 3 and 4 define **both** conditions via `make_env(rnd)` at
   the default **0.675**. That made the run non-comparable to the Stage-3 T1 reference and to
   Stage-4 B4, and put the 0.675 super-physical threshold *below* the true obstacle speed, which
   inflated the fixed-condition super-physical rate to 0.1779. Corrected to `make_env(rnd)`; the
   fixed condition and the diagnosis were **re-run from scratch**. The invalid artefacts are
   preserved and marked, never deleted:
   `stage5_validation_fixed_INVALID_env_mismatch.json`,
   `stage5_validation_diagnosis_INVALID_env_mismatch.json`.
   The randomized condition was already correct and was **not** re-run; the diagnosis re-run
   reproduced its randomized numbers exactly, which doubles as a reproducibility check.
2. **Gate pairing bug.** `admission_a4`'s positional signature paired **fixed-A0 against
   randomized-A0** instead of fixed-A0 against fixed-A1. Fixed by naming the parameters
   explicitly. Preserved as `stage5_validation_a4_gate_INVALID_pairing_bug.json`.

**Both invalid gate computations also reported A4 = FAIL**, but for different and wrong reasons.
The verdict above rests only on the corrected data.

---

## 5. Regression gates

* Frozen sha256 manifest: **PASS**, 60 files.
* Test suite: **288 passed**.
* Admission A1 (inertness, bit-identical with E1 disabled, plus a negative control), A2
  (anisotropy, closed forms vs Monte Carlo, rotation equivariance, SPD, eigenvalue floor,
  isotropic limit, 2-point and n≥3 covariances vs Monte Carlo of the exact map, degenerate-chord
  fallback, FD step-insensitivity, `POINT_COUNT_FACTOR` absent from executable code) and A3
  (signature, AST import scan, no obstacle-speed constant, runtime tripwire, action invariance
  under corruption of `obs[28:52]`): **all PASS**. A4 is the gate that failed.

V0 frozen. `v_cap = 0.96` unchanged. `u = 0` fallback unchanged. Stage-4 LADDER unchanged. R1.5
excluded. `dr_control/uncertainty.py` untouched and reserved. No IMM, CT, covariance or
uncertainty work implemented.

---

## 6. What this does and does not establish

**Establishes:** the pre-registered A4 gate is **not** satisfied, so §11.9.13-D2 applies and the
200-episode run is not authorised. Separately, on identical inputs, the geometry-derived
covariance **is** a better estimator than the frozen isotropic rule, most clearly at the 1-point
returns it targets, and it does **not** make the filter sluggish.

**Does not establish:** that E1 improves — or fails to improve — the closed-loop endpoints. At
n = 50 every closed-loop delta is at noise magnitude with CIs spanning zero. It also does not
establish anything about collisions, in either direction.

**Does not change:** any Stage 1–4 conclusion. H-c (that velocity work reaches the M2 population)
remains untested — M2 is unmoved here, but at n = 50 that is uninformative.

**Not attempted:** the 200-episode final tier; IMM/CT; covariance or uncertainty margins; the
`v_cap` sensitivity arms.

**§11.9.13-D2 terminates the estimator branch. Escalation to IMM/CT (§11.9.13-E) is expressly
NOT triggered by this failure** — E requires an ACCEPTED E1, and "the branch terminates, it does
not escalate" is the pre-registered rule. Any further Stage-5 work requires explicit approval.
