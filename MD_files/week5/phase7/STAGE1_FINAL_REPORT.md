# Week5-Phase7 / Stage 1 — FINAL run under Amendment 2

**Dated 2026-09-07. Amendment 2 recorded at `4e55219` BEFORE this run.**

Provenance chain, unchanged and not overwritten:
`998eeaa` original gate **FAILED** → `3c85f9d` Amendment 1 → `00587e3` Amendment 1 rerun
**FAILED** → `4e55219` Amendment 2 → **this run, FAILED**.

---

## Result: **STAGE 1 FAILED.** Per the pre-registered stopping rule (§11.5.4), no further amendment or tolerance is introduced, and **V0 is kept as the Stage 2–7 substrate.**

| criterion | pass condition | measured | verdict |
|---|---|---|---|
| **P1** feasible-set equality | 0 disagreements | 0 in 516 088 comparisons | **PASS** |
| **P2** coefficients + ordering | bitwise, all steps | 0 in 17 492 steps | **PASS** |
| **P3** bounds, τ, barrier, objective | exact | `‖ū‖∞ ∈ [1,1]` over 10⁶ `u` | **PASS** |
| **P4a** `u_V1` in the V0 feasible set | 100 % | 573/573 | **PASS** |
| **P4b** KKT certificate, `max δ_cert ≤ 1e−6` | 0 exceedances | **max 5.429e−5**, **5 031 / 14 612 exceed** | **FAIL** |
| **L4** certified-optimum tier | 0 differing determinate tiers | **0** | **PASS** |
| **Closed loop** | McNemar p > 0.05, CI ⊂ ±0.03 | all 6 tests pass | **PASS** |

---

## P4b — the KKT certificate, 14 612 certified solved steps

| residual | | median | p95 | p99 | max |
|---|---|---|---|---|---|
| primal feasibility | V1 | 0.000e+0 | 2.949e−10 | 9.907e−10 | **1.962e−9** |
| | V0 | 6.686e−8 | 1.894e−5 | 3.376e−5 | **9.437e−5** |
| dual feasibility | V1 | 0.000e+0 | 1.060e−11 | 7.067e−10 | 1.422e−8 |
| | V0 | 0.000e+0 | 1.057e−9 | 5.437e−8 | 9.947e−8 |
| stationarity | V1 | 1.193e−12 | 6.346e−10 | 2.419e−9 | 1.907e−8 |
| | V0 | 0.000e+0 | 2.827e−8 | 7.639e−8 | 9.984e−8 |
| complementary slackness | V1 | 1.940e−11 | 1.747e−9 | 6.019e−9 | 2.845e−8 |
| | V0 | 1.492e−6 | 3.654e−5 | 8.421e−5 | 5.191e−4 |
| **certified `‖x*−x_opt‖`** | **V1** | **5.229e−8** | 1.231e−5 | 2.565e−5 | **5.429e−5** |
| | V0 | 2.332e−4 | 1.168e−3 | 2.417e−3 | 7.520e−3 |

V0 is shown for reference only and is in no pass condition. Dual-recovery fallbacks: 0.

### Why it fails, stated factually

`δ_cert = sqrt(2·(c_sum + ‖r_stat‖²/(2μ))/μ)` is a **bound**, and the square root makes it
structurally loose: near an optimum a convex objective is flat, so a certified suboptimality of
`s` only certifies a distance of `≈ sqrt(2s/μ)`. Attaining `δ_cert ≤ 1e−6` with `μ ≥ 6` requires
certified suboptimality `≤ 3e−12`; V1's is `~1e−8`, itself an upper bound.

The secondary (arbiter) diagnostic measures V1's *actual* action error at max 3.199e−8 — about
three orders tighter than the bound the certificate can prove. **The certificate is honest
about what it can prove; the pre-registered threshold assumed a tightness a square-root bound
cannot deliver.** That is a statement about the criterion, recorded here without proposing a
replacement, because §11.5.4 forbids one.

### One implementation error was found and fixed in the gate machinery, not in the candidate

The first execution of this certificate recovered multipliers by plain non-negative least
squares and produced complementarity residuals up to **47** with `δ_cert` **identical for V0
and V1** — the signature of a degenerate dual, not of the points being certified. Cause: the
five buffered barrier rows are near-parallel (five temporal copies of one surface point), so
`Cᵀ` has near-collinear columns, the multiplier vector is under-determined, and NNLS spread
weight onto inactive rows with large slack.

Fixed by recovering multipliers as the LP `min_{λ≥0} Σ_i(−slack_i)λ_i  s.t.  Cᵀλ = −grad`,
which selects, among all valid dual certificates, the one giving the tightest bound. Any
`λ ≥ 0` yields a valid bound, so this is a correctness fix, not a relaxation: **the
pre-registered tolerance of 1e−6 was not changed**, and only that error was fixed before the
same gate was rerun. Fallbacks to NNLS: 0 of 14 612.

---

## L4 — certified-optimum tier: **PASS**

| transition | count |
|---|---|
| INDET → INDET | 5 683 |
| INDET → T1 | 3 535 |
| T0 → T0 | 3 574 |
| T1 → INDET | 1 184 |
| T1 → T1 | 621 |
| INDET → T0 | 15 |

**Steps where V0's and V1's certified tiers differ and both are determinate: 0.**
Indeterminate on 10 417 / 14 612 (71.29 %) — the certificate's interval straddles a boundary,
which is the expected consequence of the barrier constraint being *active* at the optimum.
Indeterminate steps are reported, not counted as regressions.

### Raw CBC residuals, reported separately with no thresholds attached

| | median | p95 | p99 | min | max |
|---|---|---|---|---|---|
| `CBC_V0 − τ` | −6.686e−8 | 5.666e−1 | 1.101e+0 | **−9.437e−5** | 1.668e+0 |
| `CBC_V1 − τ` | 2.480e−12 | 5.666e−1 | 1.101e+0 | **−1.962e−9** | 1.668e+0 |

Counts below τ: V0 8 377, V1 5 805 (of 14 612). **Counts below zero: V0 0, V1 0.**

These are **numerical residuals**, not guarantee-tier regressions — that distinction is exactly
what the certificate establishes. V1's smaller residuals are **not** a safety improvement; they
are a statement about numerical accuracy only. V0's excursions remain a **numerical violation
of the theoretical DR-CBF floor in the frozen SCS implementation**, of magnitude ~0.24 % of τ.

---

## Consequence for Phase 7

Stage 1 is failed and **V0 remains the substrate**. Two planning consequences follow, for your
decision at Stage 2:

1. **Cost.** Every subsequent experiment runs at ~27–29 ms/step of controller time instead of
   ~0.4 ms — end-to-end ~37 ms/step instead of ~6 ms.
2. **Statistical power.** The plan's §6.4 argued for n = 500 final paired episodes on the
   grounds that D1's plausible effect (0.03–0.10) is below what n = 200 resolves (~0.08–0.10).
   With V0 as the substrate, n = 500 across 5 arms × 2 conditions is roughly 7.5 h of solve per
   sweep. **The n = 500 design in §6.4 should be re-costed or re-scoped before Stage 2 is
   planned**, and I have not done so unilaterally.

`dr_control/fast_drccp.py` and its 29 tests remain in the tree as a Phase 7 **extension**, not
as a substrate, and are excluded from the control path of every experiment. It is described, as
required, as a **mathematically equivalent accelerated implementation** — never as
trajectory-identical or behaviour-preserving.

**What the three runs did establish**, independently of any threshold: the two formulations
define the same feasible set (516 088 comparisons, 0 disagreements); the reconstructed
coefficients and critical-row ordering are identical across all 17 492 corpus steps; bounds, τ,
barrier and objective formulations are identical; `u_V1` lies in the V0 feasible set on
573/573 sampled steps; feasibility classification is unchanged; certified guarantee tiers never
differ; and the closed loop is statistically/distributionally equivalent. What was never
established is any claim resting on a numerical threshold — three attempts, three failures,
each because the threshold was tighter than what the available instrument can certify.

**Stage 2 has not been started.**
