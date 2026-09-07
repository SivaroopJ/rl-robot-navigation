# Phase 7 — research plan: infeasibility recovery, computational acceleration,
# and uncertainty-aware velocity estimation

**Status: APPROVED 2026-09-06, with eleven clarifications applied (see §12 for the review
record). Implementation is STAGE-BY-STAGE: Stage 0 only, then stop and report.**
Phases 1–6 are frozen and remain reproducible: no existing controller, environment,
evaluation, experiment, or result file is modified by any part of Phase 7.

**Naming.** All Phase 7 work is Week 5 work. Plans live under `MD_files/week5/phase7/`;
every result directory, report header and result-file record is labelled
`Week5-Phase7 / Stage <n> / <name>`.

Provenance labels used throughout, as in Phases 1–6:
`[reference reproduction]` · `[implementation correction]` · `[adaptation]` · `[extension]`.
Everything proposed here is `[extension]` unless explicitly marked otherwise. No extension is
ever renamed into reference behaviour.

---

## 0. Executive summary of the recommendation

Three findings from inspecting the frozen code drive the entire plan. Each is a *derivation*
from the implementation, not a guess, and each is stated as a testable claim.

**Finding A — under the frozen configuration the DRCCP is exactly a small linear-constrained
QP, and the Wasserstein term is a constant.**
With `n_keep = 5` and `eps = 0.1`, `eps*N = 0.5 < 1`, so the CVaR at level `eps` collapses onto
the single worst sample (proof in §1.2). The feasible set is *exactly*

```
{ u : |u_x|,|u_y| <= max_v ,  min_i ( grad_h_i . u + alpha*h_i + dh_dt_i ) >= tau }
tau = r_W * ||u_bar||_inf / eps ,  ||u_bar||_inf = max(1, alpha, |u_x|, |u_y|) = 1
tau = 0.004 / 0.1 = 0.04   (constant, because max_v = 1 and alpha = 0.4 <= 1)
```

Measured `min_cbc_solved` is 0.03989–0.03996 in every phase — `tau` to four decimals. This
gives Direction 2 a *provably equivalent* fast path and gives Direction 1 an exact,
two-dimensional geometry to reason in.

**Finding B — the guarantee lost during recovery is not binary, and `tau` defines the tiers.**
Because the DR margin is a constant `tau` sitting on top of the nominal CBF condition, a
recovery that needs slack `s` degrades in ordered steps:

| tier | condition | what still holds | what is lost |
|---|---|---|---|
| T0 | `s = 0` | DR chance constraint `inf_P P(CBC>=0) >= 1-eps` | nothing |
| T1 | `0 < s <= tau` | nominal CBF `min_i CBC_i >= 0`, forward invariance of `{h>=0}` under the sampled constraints | distributional robustness only |
| T2 | `s > tau` | nothing; `min_i CBC_i = tau - s < 0` | forward invariance. Worst-case decay bound `h(t) >= (h_0 + m/alpha) e^{-alpha t} - m/alpha`, `m = s - tau` |

This is the exact answer to "what guarantee is lost when recovery is invoked", and it is
*measurable per step*, not asserted. Phase 1.5 measured median required slack 0.096 and p90
0.234 against `tau = 0.04` — i.e. mostly T2 — but that was the **analytic** pipeline; the
LiDAR pipeline has never been diagnosed and must be, first.

**Finding C — LEADING HYPOTHESIS (not an established finding): infeasibility in the LiDAR
pipeline is dominated by stale buffered geometry rather than by genuine multi-obstacle
conflict.** Everything below marked "Finding C" is a hypothesis with a designed test in
Stage 2. It has NOT been measured. The derivation that follows shows the mechanism is
*available*; it does not show that it dominates. Stage 2 decides, and if Stage 2 refutes it
the interpretation is updated and the later stages are re-motivated rather than forced to
follow it (§11.1).
Single-constraint infeasibility is essentially impossible here: one row needs
`||g||_1 * max_v >= tau - alpha*h - dh_dt`, and with `||g||_1 >= 1`, `h >= 0`,
`|dh_dt| <= v_obs ~ 0.75`, the right side is at most `0.79 < 1`. So every infeasible step in
Phase 6 is a *directional conflict between the five buffered samples*. But those five samples
are five temporal copies of the nearest surface point (scheme S-T), and
`LidarBarrierSource.samples` measures from the **current** robot position to **old** point
clouds while `EstimatedLidarBarrierSource` attaches the **current** track velocity. At 10 Hz,
`K = 5` spans 0.5 s; an obstacle at 0.675 m/s has moved 0.34 m — larger than its own radius.
So `h_i` describes where the obstacle *was* and `dh_dt_i` describes how fast it is closing
*now*: the two are mutually inconsistent, and for an **approaching** obstacle the stale `h`
would be **optimistic**. Whether this mechanism accounts for a large share, a small share, or
none of the observed infeasibility is exactly what Stage 2 measures. The reference ran this buffer at 50 Hz where `K = 5` spans 0.1 s; the defect is
an artefact of the rate adaptation, already flagged in `dr_control/lidar_cbf.py`'s docstring as
a limitation and never measured.

**Consequence for ordering.** The listed order 1 → 2 → 3 is *not* the scientifically correct
one. The recommended order is:

```
Stage 0  regression substrate (freeze manifest, trace corpus)
Stage 1  DIRECTION 2   acceleration, validated as equivalence  <- enables everything else
Stage 2  DIRECTION 1a  DIAGNOSIS of infeasibility in the LiDAR pipeline (no remedy)
Stage 3  DIRECTION 3a  de-staled barrier buffer (uses estimated velocity; no new estimator)
Stage 4  DIRECTION 1b  recovery ladder, designed against the residual, post-3a
Stage 5  DIRECTION 3b/c  anisotropic measurement model, IMM motion model
Stage 6  DIRECTION 3d  velocity covariance into dh_dt, with the ladder present
Stage 7  factorial D1 x D3 final evaluation, then reporting
```

Three reasons, spelled out in §5: Direction 2 must be validated **before** Direction 1 because
Direction 1's dependent variable *is* the infeasibility label and the solver decides it;
Direction 3a must precede Direction 1's remedy because building a recovery for an artefact is
fixing a symptom; Direction 3d must follow Direction 1 because injecting an uncertainty margin
*tightens* constraints and will *raise* infeasibility, so its benefit is unmeasurable while the
`u = 0` fallback is still in place.

**The 2^3 factorial the brief proposes is not the right design** (§6.1). Direction 2 is a
substrate, not a treatment: if it passes its equivalence gate it has no behavioural effect by
construction, so the four cells that differ only by D2 measure nothing and would consume
200-episode budget to do it. The correct design is an equivalence experiment for D2, then a
2×2 factorial in D1 × D3 on the D2 substrate, plus the frozen baseline: **5 arms, not 8**. The
full 2^3 becomes warranted only if D2 *fails* equivalence, at which point it genuinely is a
treatment.

---

## 1. Current architecture, as it bears on the three directions

### 1.1 The pipeline and where each direction cuts into it

```
                                                          D2 cuts here
                                                               |
StaticMapPlanner (A*)  ->  CarrotFollower  ->  gamma           v
                                                    ClfCbfDrccpController.generate_controller
LiDAR obs[4:28] -> EstimatedLidarBarrierSource.push/samples -> xi = [dh_dt, h, gx, gy]
                          ^                    ^                        |
                          |                    |                        v
                   LidarVelocityTracker    D3a cuts here          u, or u = 0 on failure
                          ^                (buffer staleness)            |
                          |                                              v
                    D3b/c/d cut here                              D1 cuts here
```

Concretely, per step, `dr_control/policy.py::predict`:

1. `ranges = obs[4:28] * lidar_range` — nothing past index 28, ever.
2. `src.push(p, ranges)` — advances `LidarVelocityTracker`, buffers world-frame surface points
   and binds a track id per point **at push time**.
3. `h, g, dd = src.samples(p)` — for each of the K buffered clouds, nearest point to the
   **current** `p`; `h = dist - r_robot`; `g = (p - q)/dist`; `dd = -g . v_track_now`.
4. `xi = build_xi(h, g, dd)`; `u = ctrl.generate_controller(p, gamma, xi)`.
5. `ctrl` sorts by `alpha*h + dh_dt`, keeps the 5 most critical, builds a fresh CVXPY problem,
   solves with SCS, and returns `u = 0` on any non-optimal status.
6. `action = clip(u / max_speed, -1, 1)`.

### 1.2 The exact reduction (proof), because both D1 and D2 rest on it

`_dro_constraints` imposes, with `c_i := u_bar . xi_i`:

```
r_W ||u_bar||_inf / eps  <=  t - (1/(N eps)) sum_i s_i ,   s_i >= 0 ,  s_i >= t - c_i
```

For fixed `u`, the tightest choice is `s_i = max(0, t - c_i)`, so feasibility in `(t, s)` is

```
exists t :  r_W ||u_bar||_inf / eps  <=  f(t) := t - (1/(N eps)) sum_i (t - c_i)_+
```

`f` is concave piecewise linear with slope `1 - (1/(N eps)) * #{i : c_i < t}`. With
`N eps = 0.5`, the slope is `+1` for `t <= min_i c_i` and `1 - 2k <= -1` above it, so
`max_t f(t) = f(min_i c_i) = min_i c_i`. Hence

```
DR constraint  <=>  min_i c_i >= r_W ||u_bar||_inf / eps      (exactly, not approximately)
```

`||u_bar||_inf = max(1, alpha, |u_x|, |u_y|)`. Under the frozen configuration `alpha = 0.4` and
`max_v = 1`, so it equals 1 for **every admissible** `u`, and `tau = r_W/eps = 0.04` is a
constant. **Precondition, to be asserted in code and refused if violated:
`eps * n_keep < 1` and `max_v <= max(1, alpha)`.** If `eps*n_keep >= 1` the CVaR no longer
collapses and the general `(t, s_i)` form is required; if `max_v > max(1, alpha)` then `tau`
becomes `u`-dependent but the set stays convex and LP-representable via the epigraph
`z >= 1, z >= alpha, z >= +-u_x, z >= +-u_y`. Both fallbacks must exist and be tested.

### 1.3 What the five samples actually are

Scheme S-T: five *temporal* copies of the nearest surface point, from the last five scans, all
re-measured from the current position. Consequences, all already documented in the frozen
docstrings and now load-bearing:

* they are not i.i.d., so the Wasserstein radius formula's premise does not hold (this is why
  §4.4 refuses to inflate `r_W` to represent velocity uncertainty);
* they usually describe **one** obstacle, so the constraint set is often rank-1 — which is why
  the Finding C hypothesis attributes conflicts to staleness rather than to genuine
  multi-obstacle geometry — to be tested, not assumed;
* they carry stale `h` against current `dh_dt` (Finding C).

### 1.4 Measured baseline numbers this plan is anchored to

| quantity | value | source |
|---|---|---|
| total controller latency | 27.6–28.9 ms (standalone), 36.0–36.4 ms (RobotNavEnv) | `results/phase5/exp5_cv.json`, `results/phase6/*.json` |
| SCS solve time | 1.14–1.49 ms | same |
| canonicalization share | **95.0–95.9 %** | same, `mean_canon_time / mean_total_time` |
| `min_cbc_solved` | 0.03989–0.03996 | Phases 2–6 |
| infeasible steps, Phase 6 | 902 (fixed), 892 (randomized) | `results/phase6/*.json` |
| collisions after an infeasible step | 24/28 = 86 % (fixed), 37/55 = 67 % (randomized) | same |
| frac infeasible, Phase 4 estimated | 0.0257 (cv), 0.0318 (ou) | `results/phase4/*.json` |
| `P(e>0 | h<0.6 & binding)`, estimated | 0.142 (cv) | `results/phase4/exp4_cv.log` |
| environment | cvxpy 1.9.2, scipy 1.17.1, numpy 2.4.6; solvers CLARABEL, SCS, SCIPY, HIGHS, OSQP | measured |

---

## 2. Files: what changes, what must not

### 2.1 New files only (nothing frozen is edited)

```
dr_control/fast_drccp.py            D2  accelerated controller, new class
dr_control/recovery.py              D1  the recovery ladder, pure functions + a policy object
dr_control/predictive_cbf.py        D3a de-staled barrier source (subclasses EstimatedLidarBarrierSource)
dr_control/tracking2.py             D3b/c anisotropic R, IMM {CV, CT}; new tracker class
dr_control/uncertainty.py           D3d  Sigma_v -> dh_dt margin, substitution arms
dr_control/policy_phase7.py         composed policy; policy.py untouched
tests/test_frozen_phase1_6.py       regression: sha256 manifest of every frozen module
tests/test_dr_control_phase7_d1.py
tests/test_dr_control_phase7_d2.py
tests/test_dr_control_phase7_d3.py
experiments/exp7_0_trace_corpus.py  Stage 0
experiments/exp7_1_equivalence.py   Stage 1 (D2)
experiments/exp7_2_infeas_diag.py   Stage 2 (D1a)
experiments/exp7_3_destale.py       Stage 3 (D3a)
experiments/exp7_4_recovery.py      Stage 4 (D1b)
experiments/exp7_5_estimator.py     Stage 5 (D3b/c)
experiments/exp7_6_uncertainty.py   Stage 6 (D3d)
experiments/exp7_7_factorial.py     Stage 7
MD_files/week5/phase7/FROZEN_MANIFEST.sha256
MD_files/week5/phase7/STAGE<n>_REPORT.md   one report per stage
results/week5_phase7/stage<n>_<name>/      new directories; no Phase 1-6 file is ever touched
```

### 2.2 Must remain untouched (verified by test, not by convention)

* everything under `robot_env/` — `tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main`
  must keep passing unchanged;
* everything under `evaluation/` — `ShortestPathOracle` is the SPL denominator of every prior
  result and is subclassed, never edited;
* `dr_control/{drccp_controller,cbf_sources,baselines,diagnostics,lidar_cbf,dynamic_cbf,
  velocity_tracker,estimated_cbf,policy}.py`;
* `optional_navigation/planner.py`;
* `tests/test_dr_control{,_phase2..6}.py`;
* `results/phase1..6/**` and `experiments/exp0..6*.py`.

`tests/test_frozen_phase1_6.py` hashes all of the above against a committed manifest and fails
loudly on any drift. This replaces the ad-hoc checksum step used in Phases 5–6.

---

## 3. DIRECTION 1 — infeasibility recovery

### 3.1 What the current fallback actually does (and why it is not neutral)

On any non-optimal status the frozen controller returns `u = 0` and sets `prev_u = 0`. At
`u = 0`, `CBC_i = alpha*h_i + dh_dt_i`, which is negative exactly when
`dh_dt_i < -alpha*h_i` — i.e. whenever an obstacle closes faster than the barrier's own decay
budget. So `u = 0` is not "do nothing safe": it is a *specific* action that violates the CBF
in precisely the situation that produced the infeasibility. Phase 6's 86 % / 67 % attribution
is the empirical shadow of this. **The frozen `u = 0` behaviour is never changed in place; it
remains the control arm.**

### 3.2 Stage 2 first: diagnose before designing (D1a, no remedy)

The Phase 1.5 taxonomy was measured on the **analytic** barrier source, which emits one
constraint per object. The LiDAR pipeline emits five temporal copies of one point. **The
taxonomy has never been shown to transfer, and the Finding C hypothesis predicts it does
not.** So Stage 2
runs, offline, over recorded Phase 6 traces (no new rollouts, no controller change):

For every infeasible step, using the LP machinery already in
`experiments/exp0_5_infeasibility.py` (`is_feasible`, `min_uniform_slack`, `min_control_scale`,
`min_alpha_feasible`, `categorise`) applied to the LiDAR `xi`:

* **D1a-M1** minimal infeasible subset and its cardinality;
* **D1a-M2** *staleness attribution*: for each row in the minimal infeasible subset, the age of
  the buffered scan it came from (0–4) and the track id bound to it. Classify the conflict as
  **stale-vs-current** (rows of different age bound to the *same* track id), **genuine
  multi-object** (different track ids), or **static-vs-dynamic**;
* **D1a-M3** counterfactual: recompute feasibility with `K = 1` (newest scan only). The
  fraction of infeasible steps that disappear is the direct estimate of the staleness share;
* **D1a-M4** required uniform slack `s`, and the **tier** it lands in (T0/T1/T2 of Finding B);
* **D1a-M5** `min_control_scale` (does more speed help — Phase 1.5 said no in 95.3 % of analytic
  cases; re-measure here);
* **D1a-M6** `min_alpha_feasible`;
* **D1a-M7** time from the infeasible step to the collision, and whether the same track id was
  the eventual collider.

**Gate G-D1a:** the staleness share (M3) is reported with a bootstrap CI. If it exceeds 50 %,
Stage 3 (de-staling) runs before any recovery mechanism is built, and the recovery ladder is
then designed and evaluated against the *residual* infeasibility. If it is below 20 %, Stage 3
still runs (it is cheap and independently motivated) but the recovery ladder is the headline.

### 3.3 Candidate recovery mechanisms

Throughout, the geometry is `P = {u in [-max_v,max_v]^2 : G u >= b}` with `G` the `N x 2`
gradient matrix and `b_i = tau - alpha*h_i - dh_dt_i`.

#### A. Minimum-violation / soft constraint

**A1 — uniform slack (max–min margin).**
`min s  s.t.  G u + s*1 >= b,  s >= 0, |u| <= max_v`. Equivalently
`max_u min_i (g_i.u - b_i)`: the *least-unsafe admissible action*. Already implemented as
`slack_fallback_u` in Phase 1.5 and measured there (analytic: collision 0.36 -> 0.28).

* *Mathematics*: an LP; always feasible; unique in `s`, generally non-unique in `u`.
* *Safety interpretation*: exact and reportable — achieved margin `m = min_i CBC_i = tau - s`,
  hence tier T0/T1/T2 per Finding B, plus the explicit decay bound of §0.
* *Relation to DR*: the DR constraint is `min_i CBC_i >= tau`; A1 solves for the largest
  attainable `min_i CBC_i`. So it is the *natural continuation of the same object*, not a
  different one. The DR interpretation survives iff `s = 0`, and the loss is graded, not binary.
* *Cost*: one 3-variable LP, sub-millisecond.
* *Multi-obstacle conflict*: this is exactly the case it is designed for; it splits the
  violation evenly rather than sacrificing one obstacle entirely.
* *Guarantees an action*: yes, always. *Can still fail*: yes — T2 actions can collide.
* *Tuning parameters*: none. **This is its main advantage.**
* *Fairness*: it is an `[extension]`; the reference has no recovery. Reported as such, and the
  frozen `u = 0` arm is always run alongside.
* *Weakness*: it ignores the objective, so the recovered `u` may be far from `u_nom` and may
  chatter between steps. Addressed by A1-L below.

**A1-L — lexicographic minimum-violation (RECOMMENDED as rung R2).**
Two-stage: (i) `s* = min s` as above; (ii) among `{u : G u + s* 1 >= b - eta}` minimise the
original Phase 1 objective `p0||u-u_prev||^2 + p1||u-u_nom||^2 + p3 delta^2`. Convex, one extra
QP, one tolerance `eta` (set to `1e-9`, not tuned). Keeps the controller pursuing its task and
suppresses chatter through the retained `p0||u - u_prev||^2` term. The safety report is
unchanged from A1, because `s*` is unchanged by construction.

**A2 — weighted slack** `min sum_i w_i s_i`. Rejected as primary: `w` is a free tuning vector,
and any criticality weighting duplicates the criticality sort already inside the controller.
Retained as a single ablation with `w_i = 1/(h_i + h_0)`.

#### B. Emergency backup / evasive controller

Fixed evasive laws: (B1) `u = 0` — the frozen baseline; (B2) maximum-clearance retreat
`u = max_v * g_crit` along the most critical gradient; (B3) braking along `-grad V`.

* *Mathematics*: no optimisation; a closed-form direction.
* *Relation to DR*: none retained. The DR constraint is simply abandoned.
* *Cost*: free.
* *Multi-obstacle*: B2 is A1 restricted to one constraint, so it can drive the robot into a
  second obstacle — precisely the failure mode the Finding C hypothesis predicts. That makes B2 an
  *informative* ablation: if B2 ≈ A1 then conflicts are effectively single-obstacle.
* *Guarantees an action*: yes. *Can fail*: yes, and more often than A1 by construction.
* *Tuning*: none. Retained as ablations, never as primary.

#### C. Constraint prioritisation / hierarchical safety

Drop constraints in criticality order (or in staleness order) until feasible.

* *Mathematics*: a discrete relaxation; the retained subproblem is the original DRCCP on a
  sub-sample, so **the DR guarantee survives exactly, but only over the retained subset** —
  the cleanest guarantee statement of any candidate, and also the most easily misread. The
  dropped constraints may be violated without bound.
* *Cost*: up to `N` LPs, still sub-millisecond.
* *Guarantees an action*: yes, given §0's argument that a single row is essentially always
  feasible. Must nonetheless fall back to A1 if even one row is infeasible.
* *Tuning*: none, but the *drop order* is a design choice. Two orders are meaningfully
  different and both are tested: **C1 criticality order** (drop the least critical last) and
  **C2 staleness order** (drop the oldest scan first) — C2 is the direct hypothesis test of
  the Finding C hypothesis and costs nothing.
* *Weakness*: discrete switching invites chatter; measured via a switching-rate metric.

#### D. Other candidates considered

**D1 — drop the Wasserstein margin only (`tau -> 0`).** The minimal possible relaxation: it
gives up *exactly* distributional robustness and nothing else, and by Finding B it lands the
step in tier T1 by construction. Phase 1.5 measured 80/344 analytic cases recovered by this
alone. **Recommended as rung R1** — it is the only relaxation whose lost guarantee is
nameable in one word.

**D2 — locally relaxed barrier rate.** Solve for the smallest `alpha' >= alpha` restoring
feasibility (`min_alpha_feasible`, already implemented). The retained guarantee is
`min_i CBC_i^{alpha'} >= tau`, i.e. `h_dot >= -alpha' h`: forward invariance of `{h >= 0}` is
**preserved**, only the permitted decay rate increases, and the violation of the original
`alpha` constraint is `(alpha' - alpha) h`, which **vanishes as `h -> 0`**. That is a
qualitatively stronger safety property than A1's uniform slack, which does not vanish at
contact. `alpha'` is computed, never chosen, so no tuning parameter is introduced. **Recommended
as rung R1.5 ONLY IF it survives the three checks below**; until then it is an ablation, not
a primary rung.

**R1.5 admission checks (all three must pass before it enters the primary ladder).**

1. *Feasibility monotonicity in `alpha`.* For `h_i >= 0`, raising `alpha` shrinks the RHS
   `b_i = tau - alpha h_i - dh_dt_i`, so the feasible set grows monotonically and bisection
   upward is valid. **But this holds only while every `h_i >= 0`.** If any kept sample has
   `h_i < 0`, raising `alpha` *tightens* that row and monotonicity fails, so `alpha'` may not
   exist and bisection is invalid. The implementation must detect `min_i h_i < 0` and refuse
   R1.5 outright on that step, falling through to R2. Measured `frac_h_negative = 0` in
   Phases 4–6, so this is expected to be rare — but "rare" is not "impossible" and the branch
   is tested.
2. *Coupling to the objective and the rest of the controller.* `alpha` is NOT confined to the
   barrier row. In the frozen implementation it appears in three further places:
   (a) `u_bar = [1, alpha, u_x, u_y]`, so `||u_bar||_inf = max(1, alpha, |u|)` — for
   `alpha' > 1` the Wasserstein term `tau = r_W*alpha'/eps` GROWS, which partially cancels the
   relaxation and makes the effect non-monotone above `alpha' = 1`;
   (b) the criticality sort key `alpha*h + dh_dt`, so a different `alpha` can select a
   DIFFERENT set of five kept samples, which changes the problem being solved rather than
   relaxing it;
   (c) through (b), `h_crit` and hence the objective weights `p1 = 4 h_crit`,
   `p3 = 5 h_crit` — so the objective itself moves.
   **Therefore R1.5 must be defined as: re-solve with `alpha'` substituted ONLY in the barrier
   rows, holding the sample selection, `tau`, and the objective weights fixed at their
   `alpha`-derived values.** Otherwise it is not a relaxation of the original problem at all.
   A unit test asserts that under this definition the feasible set is nested in `alpha'` and
   the objective is unchanged; if that cannot be made to hold cleanly, R1.5 is demoted.
3. *Forward invariance in DISCRETE time.* The continuous-time claim `h_dot >= -alpha' h`
   implies invariance of `{h >= 0}`; the implementation is a `dt = 0.1` Euler step with a
   sampled, piecewise-constant `u`. The correct discrete statement is
   `h_{k+1} >= (1 - alpha' dt) h_k + O(dt^2)`, which preserves non-negativity only while
   `alpha' dt <= 1`, i.e. `alpha' <= 10`, and only up to the neglected second-order term and
   the sampling error in `dh_dt`. **The plan does not claim discrete-time invariance.** What
   is claimed and reported is the one-step bound above, plus the measured `min h_true` — and
   `alpha'` is capped at `1/dt = 10` for this reason. A unit test checks the one-step bound
   numerically against rollouts.

**D3 — control-bound relaxation.** Rejected: Phase 1.5 found 95.3 % of infeasible cases are
unfixable by any finite speed, and `max_v` is the environment's own action clip — raising it
would break the comparison to PPO. Retained only as the diagnostic `min_control_scale`.

**D4 — predictive / backup-set CBF.** Verify safety by simulating a known-safe backup policy
over a horizon. Principled but requires a forward model of the obstacles, i.e. exactly the
thing Direction 3 is trying to estimate; it would import Direction 3's error into Direction 1's
guarantee. **Deferred, out of Phase 7 scope**, recorded here so the omission is deliberate.

**D5 — sampling the uncertainty into the ambiguity set.** Belongs to Direction 3; see §4.4.

### 3.4 Recommended primary: the recovery ladder

A strict ladder, evaluated in order, stopping at the first rung that is feasible. Each rung
gives up exactly one named guarantee, and the rung index is logged for every step.

The DEFAULT primary ladder is R0 -> R1 -> R2. R1.5 sits between R1 and R2 only on promotion.

```
R0  full DR:        min_i CBC_i >= tau                 guarantee: DR chance constraint
R1  tau := 0:       min_i CBC_i >= 0                   lost: distributional robustness
R1.5 alpha -> alpha': min_i CBC_i^{alpha'} >= 0        ABLATION unless the three admission
     (barrier rows only)                               checks of 3.3-D2 pass. lost: the decay
                                                       RATE. Violation (alpha'-alpha)h -> 0 at
                                                       contact. NO discrete-time invariance is
                                                       claimed; the one-step bound is reported.
R2  A1-L:           max-min margin, then objective     lost: forward invariance;
                                                       reports achieved margin m and decay bound
```

`u = 0` never appears. It remains the frozen baseline arm and nothing else.

Properties: every rung is convex and parameter-free; R2 always produces an action; the ladder
costs at most three small solves and, on the D2 substrate, well under 1 ms; and the reported
state per step is a 4-valued rung plus a scalar margin, which is exactly the five-way
distinction the brief demands:

| brief's requirement | recorded field |
|---|---|
| 1. feasible CBF solution | `rung == R0` |
| 2. infeasible CBF problem | `rung > R0`, with the LP-verified infeasibility flag |
| 3. recovered action | `u_recovered`, `rung`, `alpha_prime`, `s_star` |
| 4. actual CBF violation of the recovered action | `m = min_i CBC_i` evaluated **on the full sample set**, and `tier in {T0,T1,T2}` |
| 5. subsequent collision | `collision`, `steps_since_last_recovery`, `same_track_as_collider` |

**A recovered action is NEVER described as safe merely because it exists.** No report,
log line, field name, docstring or summary may call a recovered action "safe", "recovered
safely", or "handled". The only admissible statements are quantitative: the rung that produced
it, the achieved margin `m = min_i CBC_i` measured post hoc from the returned action against
the samples the controller actually saw, the guarantee tier T0/T1/T2 that `m` implies, and —
as a separately labelled diagnostic channel — the ground-truth clearance. Slack is never
introduced and then relabelled as safety. An action that exists is an action that exists; what
it guarantees is `m`, and `m` is always printed next to it.

### 3.5 Direction 1 unit tests

1. `tau` identity: for random `xi` with `eps*N < 1`, the CVXPY feasible set and
   `min_i c_i >= tau` agree on 10^4 random `u` (this also underpins D2).
2. R0 reproduces the frozen controller **exactly** when the step is feasible (bit-identical
   action) — the ladder must be inert on feasible steps.
3. R1 is feasible whenever R0 is (nested sets), and strictly larger when `tau > 0`.
4. R1.5 monotonicity: feasibility is monotone non-decreasing in `alpha` for `h >= 0`, and
   `alpha' = alpha` exactly when R0 is feasible. (Phase 1.5 got this direction backwards once;
   the test exists because of that.)
5. R2 always returns an action, for adversarially constructed empty polytopes.
6. Tier classification: hand-built cases with known `s` land in T0/T1/T2 as derived.
7. Decay bound: simulate `h_dot = -alpha h - m` and assert the closed-form bound of §0 holds.
8. No-hidden-slack: on a feasible step the reported `m >= tau` and `rung == R0`.
9. Determinism: the ladder contains no RNG; repeated runs are bit-identical.
10. Chatter: on a fixed adversarial trace, the rung-switching rate is recorded (no threshold
    asserted; it is a measurement).

---

## 4. DIRECTION 2 — computational acceleration

### 4.1 Where the 36 ms goes

Measured, not assumed: 95.0–95.9 % is CVXPY canonicalization, 1.1–1.5 ms is SCS. The frozen
controller constructs `cp.Variable`, `cp.Problem`, and `N + 3` constraint objects **every
step**, and `build_objective` rebuilds the objective with data-dependent scalar weights.

### 4.2 Stage 0 — the validation substrate, built before any optimisation

`experiments/exp7_0_trace_corpus.py` replays the frozen **Phase 6 configuration** — the real
`RobotNavEnv` driven by the frozen `DRCBFPolicy`, in both the fixed and randomized
obstacle-motion conditions — and records, per step, the exact controller inputs and outputs:

*Scope note, flagged rather than made silently.* The plan as drafted said "Phase 5 and Phase 6
configurations". Stage 0 records the Phase 6 configuration only. Reason: the Phase 6
configuration is the one Stage 7 evaluates and the one Stage 2 diagnoses, so it is the corpus
every gate actually consumes; adding the Phase 5 standalone harness would roughly double the
corpus and the code surface without serving any gate. If a later stage needs standalone-harness
traces, they are added then, as an amendment under §11.1.4.

```
inputs :  p, gamma, xi (full, pre-truncation), u_prev, alpha, r_W, eps, max_v, k_v
outputs:  u, status, delta, h_crit, weights, box_overshoot, min_i CBC_i, timings
```

Corpus tiers: **dev** 20 episodes, **validation** 50 episodes, **final** 200 episodes, both
motion models, plus a deliberately enriched **infeasible corpus** containing every infeasible
step found (~1800 from Phase 6) so that status agreement is tested where it matters rather than
where it is easy. Stored under `results/week5_phase7/stage0_trace_corpus/` as `.npz`,
checksummed, each with the label `Week5-Phase7 / Stage 0 / trace corpus`.

**No accelerated variant is written until this corpus exists.**

### 4.3 Candidate accelerations

| id | approach | problem mathematically unchanged? | expected latency | complexity |
|---|---|---|---|---|
| V0 | frozen: SCS, rebuilt per step | — (reference) | 28–36 ms | — |
| V1 | `cp.Parameter` + cached `Problem`, SCS | yes, identical cone program | 2–5 ms | low |
| V2 | V1 + CLARABEL | yes; different algorithm | 1–3 ms | low |
| V3 | **exact analytic reduction** (Finding A) to a 3-variable QP, OSQP | yes, *provably* — §1.2 | 0.1–0.5 ms | medium |
| V4 | V3 + warm start from `u_prev` | yes; different iterate path | 0.05–0.3 ms | low |
| V5 | hand-rolled KKT / active-set | yes | <0.05 ms | high |

**V1 has a concrete obstacle worth stating now.** DPP requires parameters to enter affinely.
`p1 * sum_squares(u - u_nom)` is parameter × quadratic and is **not** DPP-compliant, so CVXPY
would re-canonicalize every step and V1 would deliver no speedup. The fix is to carry
`sqrt(p1)` as the parameter and write `sum_squares(sqrt_p1 * (u - u_nom))`, which is DPP. This
requires `p1 = 4*h_crit >= 0`. Frozen measurements give `frac_h_negative = 0` in Phases 4–6, so
in practice this never binds — but the code must **detect `h_crit < 0` and fall back to the
frozen non-parameterized path**, so that the reference's `DCPError` behaviour is preserved
exactly rather than silently repaired. That fallback is itself a tested branch.

**V3 is the recommended primary.** It is the only variant whose equivalence is a theorem rather
than a tolerance. Preconditions asserted at construction (`eps*n_keep < 1`,
`max_v <= max(1, alpha)`); if either fails the class refuses to build and the caller must use
V1/V2. Two additional general paths are implemented and tested but not used by default: the
epigraph form for `max_v > max(1, alpha)`, and the full `(t, s_i)` form for `eps*N >= 1`.

V5 is **deferred**: if V3 reaches ~0.3 ms the controller is no longer the bottleneck (the
environment step and the tracker will be), so further work would optimise the wrong thing. The
plan states the stopping criterion in advance rather than optimising indefinitely.

### 4.4 Equivalence is graded, and "bitwise" is not achievable — say so up front

Bit-identical output across solvers, or even across canonicalization orders with the same
solver, is not attainable: SCS is first-order and the frozen controller already records
`box_overshoot ~ 1e-6`. Promising bitwise equivalence and then quietly reporting closeness
would be exactly the failure the brief warns against. The plan therefore pre-registers four
tiers and states which is required of which variant:

```
E0  bit-identical            byte equality of u.  Required of: nothing. Achievable only if the
                             identical problem data reaches the identical solver in the identical order.
E1  numerically equivalent   status identical on 100% of corpus steps AND ||u - u_ref||_inf <= 1e-6
                             AND |objective - objective_ref| <= 1e-8 AND |min CBC - min CBC_ref| <= 1e-6.
                             Required of: V1, V2, V3, V4.
E2  behaviourally equivalent identical episode OUTCOME (success/collision/timeout and collision
                             type) on 200 paired seeds, both motion models.
                             Required of: whichever variant becomes the substrate.
E3  statistically indistinguishable  paired McNemar p > 0.05 on collision and success, with the
                             CI reported. NOT accepted as evidence of equivalence on its own —
                             it is reported only as a companion to E2.
```

### 4.5 The validation hierarchy, run in this order

```
L1  feasibility-status agreement   exact, categorical, on the ENRICHED INFEASIBLE CORPUS first.
                                   Any disagreement is a hard stop: Direction 1's dependent
                                   variable is this label.
L2  action agreement               ||u - u_ref||: max, p99, p50, over the full corpus
L3  objective agreement            optimal value, same statistics
L4  CBC agreement                  min_i CBC_i, and the count of steps where the tier changes
L5  trajectory agreement           closed loop, paired seeds: first-divergence step, terminal
                                   outcome agreement, SPL difference
L6  runtime                        only now
```

**No runtime number is reported before L1–L5 pass.** This is the "do not optimise first and
assume behaviour is unchanged" requirement, made procedural.

### 4.6 Benchmarking methodology

* `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, as the existing run scripts already use.
* `time.perf_counter` around: (a) sample construction, (b) canonicalization/setup,
  (c) solve, (d) post-processing, (e) total. Reported separately — the frozen controller
  already splits canon from solve and that split is preserved.
* Statistics: mean, p50, p95, p99, **max** (worst-case latency is the control-relevant number,
  not the mean), over ≥ 20 000 steps per variant.
* Steady state vs first call reported separately: V1/V3 pay a one-time setup cost.
* Solver reliability: counts of `optimal`, `optimal_inaccurate`, `infeasible`,
  `infeasible_inaccurate`, `SolverError`, `DCPError`, and iteration counts where exposed.
* The PPO 0.54 ms figure is re-measured under identical thread settings in the same process so
  the ratio is not an artefact of different environments.

### 4.7 Direction 2 unit tests

1. The `tau` identity of §1.2, as a standalone theorem test over random `xi`, `alpha`, `eps`,
   `N`, including cases where `eps*N >= 1` (identity must NOT hold there — a negative control).
2. Precondition assertions: V3 refuses to construct when `eps*n_keep >= 1` or
   `max_v > max(1, alpha)`.
3. V3 vs V0 on 10^4 random `xi`: E1 on status, action, objective, CBC.
4. V3 vs V0 on the enriched **infeasible** corpus: status agreement exactly 100 %.
5. DPP compliance of V1 asserted programmatically (`problem.is_dcp(dpp=True)`), plus a test
   that the second solve does not re-canonicalize (compile-count or timing ratio).
6. `h_crit < 0` fallback: V1 routes to the frozen path and reproduces `DCPError`.
7. Warm-start determinism: V4 with a fixed start sequence is reproducible across runs and
   across process restarts.
8. The frozen controller is untouched — covered by `tests/test_frozen_phase1_6.py`.
9. Box feasibility: `|u| <= max_v` exactly after clipping, and `box_overshoot` recorded.
10. Numerical stress: near-degenerate `G` (parallel gradients), `h` near 0, `h < 0`.

---

## 5. DIRECTION 3 — improved and uncertainty-aware velocity estimation

### 5.1 Stage 3 (D3a) — predictive barrier construction. `[extension]`, NOT a correction.

**Provenance, stated before the mathematics.** This is a **new `[extension]`: a predictive
barrier construction.** It is emphatically *not* an `[implementation correction]` and must
never be described as one. The reference's S-T buffer is not a bug — it is a deliberate design
that is defensible at the 50 Hz it was written for. D3a changes `h` and `grad_h`, i.e. it
changes the controller's *input*, and therefore changes the controller's behaviour on every
step, feasible or not. It is a different barrier, not a repaired one. Consequences that follow
from that classification and are honoured throughout: the frozen source stays in the tree and
in every comparison; D3a is always an arm, never a replacement; and any improvement it shows is
reported as the benefit of a new method, not as the removal of a defect.

The Finding C hypothesis says `h_i` is measured to where an obstacle *was* up to 0.5 s ago
while `dh_dt_i` is
its current closing rate. The fix uses only quantities the pipeline already computes:

```
frozen :  q_i = buffered surface point (age k scans)
          h_i = ||p - q_i|| - r_robot ,  g_i = (p - q_i)/||p - q_i|| ,  dh_dt_i = -g_i . v_track

D3a    :  q_i' = q_i + v_track(bound at push time) * (k * dt)     # propagate the point forward
          h_i = ||p - q_i'|| - r_robot , g_i = (p - q_i')/||...|| , dh_dt_i = -g_i . v_track
```

* Uses **no ground truth** — `v_track` is the same estimated velocity the frozen `dh_dt`
  already uses, so the information ledger is unchanged and the existing leakage tests apply
  verbatim.
* **It changes the controller input.** `h`, `grad_h` and `dh_dt` all move, so `xi` moves, so
  the criticality sort, the kept subset, `h_crit`, the objective weights and the action can all
  differ on steps that were perfectly feasible before. D3a is therefore NOT behaviour-preserving
  and is never eligible for an equivalence gate — only for a paired outcome comparison.
* Static structure has `v_track = 0`, so static geometry is untouched — a required unit test.
* Predicted effects, all falsifiable: fewer stale-vs-current directional conflicts (lower
  `frac_infeasible`); less **optimistic** `h` error for approaching obstacles; possibly *more*
  conservative behaviour overall (higher `mean_u_dev`, lower SPL).
* Risk: it makes `h` depend on estimated velocity, so estimator error now contaminates the
  *geometry* channel and not only the `dh_dt` channel. That is a real cost and must be measured,
  not waved away: `D3a-M` includes `h` error against ground truth, split by track quality.
* Ablation `K = 1` (newest scan only) is run alongside: it removes staleness by removing the
  buffer, at the cost of collapsing the DR sample set to a single point (`eps*N = 0.1 < 1`, so
  the reduction still holds and `tau` is unchanged). If `K = 1` matches D3a, the buffer is
  simply not earning its place at 10 Hz and that is a publishable negative result about the
  reference's S-T scheme at a rate it was not designed for.

### 5.2 Stage 5 (D3b/c) — a better measurement model and a better motion model

**D3b — anisotropic measurement covariance from the reconstruction geometry.**
`Track.measurement_noise` currently returns `diag(s^2, s^2)` with `s` scaled by a hand-set
`POINT_COUNT_FACTOR = {1: 3.0, 2: 1.5}`. That isotropy is geometrically wrong: a one-point
fixed-radius reconstruction `c = q + r*u_ray` has near-zero **radial** error and up to
`r*sin(theta_incidence)` **tangential** error. The principled replacement propagates the
surface-point noise through the reconstruction Jacobian:

```
c = f(q_1..q_n)          R_c = J Sigma_q J^T  ,  Sigma_q = sigma_range^2 * u_ray u_ray^T (+ beam width)
```

with closed forms for the 1-point and 2-point cases and the Gauss-Newton `J` for `n >= 3`.
This removes two hand-set constants and replaces them with geometry. It is expected to help
most exactly where the frozen estimator is weakest (1–2 point returns, i.e. `h < 1 m`).

**D3c — motion model.** The frozen filter is CV with white acceleration `SIGMA_A = 0.5`.
`SmoothStochasticMotion` is an OU *angular velocity* process at nearly constant speed, so the
mismatch is in **turning**, not in speed — a constant-turn (CT) model matches it structurally.
Recommended: **IMM over {CV, CT}**, which adapts without requiring us to know which regime the
environment is in and which is honest about the fact that the deterministic arm really is CV.
Cost is roughly 2× the filter, negligible against even the accelerated controller.

Ablations retained: CV only (frozen), CT only, IMM{CV,CT}; sliding-window batch least squares
over M frames as a non-Bayesian comparator; Huber-robust innovation gating.

### 5.3 Stage 6 (D3d) — velocity uncertainty, and exactly how it enters `dh_dt`

**The estimator already computes `Sigma_v`. The frozen code throws it away.** `Track.P` is a
4×4 covariance; `Sigma_v = P[2:,2:]`. Exposing it costs nothing and is the prerequisite for
everything below.

**How uncertainty enters the barrier, derived.** With `v_hat_i ~ (v_i, Sigma_i)` and `g_i` a
unit gradient,

```
dh_dt_i = -g_i . v_i        =>   dh_dt_i ~ ( -g_i . v_hat_i ,  sigma_i^2 )
                                  sigma_i^2 = g_i^T Sigma_i g_i
```

So velocity uncertainty enters the barrier as a **scalar variance projected onto the barrier
gradient** — per sample, state dependent, and confined to slot 0 of `xi`. That is the whole of
the mathematical interface, and it immediately tells us where uncertainty does *not* belong.

**Four ways to use it, and why only one is primary.**

* **U1 — ignore it.** Frozen behaviour. Control arm.
* **U2 — deterministic margin (RECOMMENDED).**
  `xi[i,0] := -g_i . v_hat_i - z_beta * sigma_i`, i.e. assume the obstacle closes `z_beta`
  standard deviations faster than estimated. This is the standard chance-constrained
  tightening of a single constraint and it is transparent: the sample becomes conservative,
  the controller is otherwise untouched, and `alpha`, `r_W`, `eps` are not modified.
  One declared parameter `z_beta`, fixed a priori at `1.0` (≈84 %) and *not* tuned; a
  sensitivity arm at `z_beta in {0, 0.5, 1, 2}` is reported as a curve, never used to select.
* **U3 — enlarge the Wasserstein ball (`r_W` <- `r_W + f(sigma)`). REJECTED as primary.**
  Three independent reasons: (i) `r_W` multiplies `||u_bar||_inf` and applies **uniformly** to
  all samples, whereas `sigma_i` is **per-sample** — the shapes do not match; (ii) the radius
  formula `r_N(eps)` is derived for i.i.d. samples and our five are a temporal lag window, so
  the object being inflated has no valid frequentist interpretation here (§1.3); (iii) it
  changes a frozen parameter. Retained only as a clearly labelled ablation.
* **U4 — sample the uncertainty into the empirical distribution (scheme S-N).** Draw K i.i.d.
  `v_hat + Sigma^{1/2} zeta` and build `xi` from them. Theoretically the cleanest use of DRO —
  the ambiguity set would then genuinely surround a sampled distribution. But with
  `eps*N < 1` the CVaR is the *minimum*, so this reduces to U2 with a **random** `z`, and it
  makes the controller stochastic, which costs reproducibility. Retained as an ablation with a
  fixed, logged RNG stream, and reported with that caveat attached.

**Avoiding double counting, treated as an experiment rather than an assertion.**
`tau = 0.04` m/s already buys a constant margin on the worst sample. U2 adds `z_beta*sigma_i`
on top. These are different objects — one uniform and distribution-level, one per-sample and
state-level — so applying both is not literally the same margin twice. But the *total*
conservatism does increase, and asserting "therefore it is fine" would be exactly the kind of
hand-wave the brief forbids. The design therefore tests **substitution** directly:

```
S1  r_W = 0.004, z_beta = 0     frozen                    (control)
S2  r_W = 0.004, z_beta = 1     margins ADD
S3  r_W = 0,     z_beta = 1     uncertainty REPLACES the Wasserstein margin
S4  r_W = 0.004, z_beta = 0, but with the improved estimator only   (isolates estimator from margin)
```

If S3 ≈ S2 the two margins are redundant and the DR term is doing no work beyond a constant
offset — a substantive finding about the method. If S3 is clearly worse, `r_W` earns its place
for a reason unrelated to velocity uncertainty. S3 is the only arm that touches `r_W`, it does
so at a single declared value, and it is reported as an ablation, never as the primary system.

### 5.4 Direction 3 metrics — the primary endpoint is not mean error

The headline is **not** mean `||v_hat - v||`. It is the safety-critical optimistic tail:

```
PRIMARY   P(e > 0 | TTC < 1 s)   where e = dh_dt_est - dh_dt_true, e > 0 means optimistic
```

computed on the binding-constraint subset exactly as `experiments/exp4_dynamic_estimated.py`
already does (`ttc = h_true / max(-dh_dt_true, 1e-6)`), so it is directly comparable to the
frozen 0.142 figure. Secondary: `p95`, `p99`, `max` of `e` on the same subset; the same split
by range and by point count; and — new for D3d — **uncertainty calibration**, which is the
claim that `Sigma_v` is meaningful at all:

* NEES / NIS over the tracks (should sit inside the chi-square band if the filter is
  consistent), reported per motion model;
* empirical coverage of the `z_beta`-sigma interval: the fraction of steps with
  `e <= z_beta * sigma`. If a "1-sigma" margin covers 60 % rather than 84 %, `Sigma_v` is
  optimistic and U2's interpretation fails.

**`Sigma_v` is assumed UNCALIBRATED until proven otherwise.** Gate G6 is ordered: NEES, NIS and
empirical coverage are computed and reported FIRST, on validation seeds, and only if they pass
may any *safety* interpretation be attached to the `z_beta`-sigma margin. If they fail, U2 may
still be run — a margin that reduces optimistic error is useful even when miscalibrated — but
it is then reported strictly as **a heuristic tightening of unknown confidence level**, and the
phrase "chance-constrained" is not used for it anywhere.

### 5.5 Direction 3 unit tests

1. **Leakage, all four frozen layers, re-applied to every new module**: signature (no obstacle
   argument), AST import scan, runtime tripwire, and bit-identical action under corruption of
   ground-truth velocities. Non-negotiable; these are copied from
   `tests/test_dr_control_phase4.py` and pointed at the new classes.
2. D3a: static structure (`v_track = 0`) leaves `h`, `g` bit-identical to the frozen source.
3. D3a: a synthetic obstacle on a known straight line, propagated forward, reproduces the true
   current surface point to within the estimator error — a direct correctness test.
4. D3b: the reconstruction covariance is anisotropic in the predicted direction (tangential
   variance > radial) for a 1-point return, with a hand-derived numeric case.
5. D3b: `R` reduces to the frozen isotropic form in the limit of many well-spread points.
6. D3c: on a pure CV trajectory the IMM converges to the CV mode weight > 0.9; on a pure
   constant-turn trajectory, to CT.
7. D3c: both motion models are run with **no retuning between them** — the same constants.
8. D3d: `sigma_i^2 = g^T Sigma g` verified against a Monte Carlo projection.
9. D3d: sign convention — with a *known* `Sigma`, U2 makes `dh_dt` more negative, never less
   (an optimistic margin would be a catastrophic sign error; Phase 4 had one).
10. Calibration: on a synthetic track with known noise, NEES lies inside the 99 % band.
11. Determinism of U1–U3; explicit RNG logging for U4.

---

## 6. Experimental design

### 6.1 Why the 2^3 factorial is the wrong design as stated

Direction 2 is a **substrate**, not a treatment. Its own success criterion (E1/E2) is that it
has *no behavioural effect*. Cells 3, 5, 7 and 8 of the proposed 2^3 differ from cells 1, 2, 4
and 6 only by D2 and would therefore be, by construction, identical up to solver tolerance —
spending 200-episode budget to measure nothing. The correct structure:

```
D2 evaluated as an EQUIVALENCE experiment      (metrics: agreement + runtime, NOT success rate)
then, on the accelerated substrate, a 2x2 factorial:

  B0  FROZEN Phase 6 controller (V0)     u=0 fallback, frozen estimator   <- reproduces Phase 6
  B1  accelerated only                   u=0 fallback, frozen estimator
  B2  accelerated + D1                   recovery ladder, frozen estimator
  B3  accelerated + D3                   u=0 fallback, improved estimator
  B4  accelerated + D1 + D3              recovery ladder, improved estimator
```

Five arms, with two distinct roles:

* **B0 vs B1 is the acceleration equivalence check**, not a factorial cell. It asks only
  whether the substrate moved the baseline. If B0 ≡ B1 within E2, every later comparison is
  attributable to D1/D3 alone; if not, D2 has failed its gate and becomes a treatment factor.
* **The factorial proper is B1 / B2 / B3 / B4**, a 2x2 in D1 x D3 on a fixed substrate.
  Marginals: D1 = (B2 - B1) and (B4 - B3); D3 = (B3 - B1) and (B4 - B2).
  Interaction: B4 - (B2 + B3 - B1).

The interaction term A4 − (A2 + A3 − A1) is the scientifically interesting quantity and is
**expected to be non-zero**, for a reason that is derivable rather than incidental: D3d
*tightens* constraints and therefore *raises* infeasibility, while D1 changes what happens when
infeasibility occurs. Additivity must not be assumed. If D2 fails equivalence it becomes a
genuine treatment and the full 2^3 is reinstated — that contingency is pre-registered here.

### 6.2 Evaluation tiers and seed hygiene

| tier | episodes | seeds | used for |
|---|---|---|---|
| development | 5–20 | `DEV_SEED_BASE = 20_000` | iterating within a stage |
| validation | 50 | `DEV_SEED_BASE + 1_000` | clearing a stage gate, ablations, sensitivity curves |
| final | 200 (compat) + 300 (extension) | `EVAL_SEED_BASE = 1_000_000 .. 1_000_499` | Stage 7 only |

`1_000_000..1_000_199` is reused so Phase 7 pairs one-to-one with the Phase 6 record;
`1_000_200..1_000_499` extends it for power (§6.4). **No tuning decision may touch any
`EVAL_SEED_BASE` seed**, exactly as in Phases 1–6. Every sensitivity curve (`z_beta`, `w`,
`K`) is reported from validation seeds and never used to select a value.

Pairing is guaranteed as in Phases 4–6: obstacle trajectories are pre-computed per seed with a
dedicated per-episode RNG, so every arm faces an identical world frame for frame.

### 6.3 Per-stage gates

```
G0  Stage 0   frozen manifest verifies; trace corpus written and checksummed; full suite green
G1  Stage 1   L1 status agreement 100% on the enriched infeasible corpus; L2-L4 within E1;
              L5 E2 on 200 paired seeds both motion models; only then L6 runtime reported
G2  Stage 2   staleness share reported with CI; taxonomy of the LiDAR pipeline established;
              NO controller change made in this stage
G3  Stage 3   D3a: frac_infeasible, optimistic h error, and collision reported on validation
              seeds; static-geometry invariance test passes; K=1 ablation reported alongside
G4  Stage 4   recovery ladder: rung distribution, tier distribution, achieved margin m,
              and collision, on validation seeds; ladder inert on feasible steps (bit-identical)
G5  Stage 5   estimator metrics FIRST (P(e>0|TTC<1s) primary) on validation seeds, before any
              closed-loop claim; both motion models with no retuning between them
G6  Stage 6   Sigma_v calibration (NEES/NIS/coverage) passes BEFORE U2 is evaluated;
              substitution arms S1-S4 on validation seeds
G7  Stage 7   final 500 paired episodes, 5 arms, 2 motion models; pre-registered primary endpoint
```

Each gate is reported and reviewed before the next stage begins, as in Phases 1–6.

### 6.4 Statistical methodology

* **Pre-registered primary endpoint: collision rate**, paired by seed, tested with **exact
  McNemar**. Everything else is secondary and reported descriptively with CIs.
* Paired bootstrap (10 000 resamples over seeds) for CIs on differences of rates, SPL, and
  clearance — the same functions Phases 4–6 already use.
* **Multiplicity**: 5 arms × 2 motion models. Holm–Bonferroni within the family of primary
  comparisons (4 contrasts against A1 per motion model). Secondary metrics are explicitly
  labelled exploratory and are not corrected — and are not used to make claims.
* **Clustering**: per-infeasible-event analyses (recovered-action violation, time to collision)
  involve many events inside one episode, which are *not* independent. These use a
  **cluster bootstrap over episodes**, never a naive per-event CI.
* **Power, stated honestly in advance.** Phase 6 could not resolve 0.275 vs 0.335 at n = 200
  (p = 0.14). With observed discordance of roughly 40–70 pairs, n = 200 detects about a
  0.08–0.10 absolute difference. n = 500 brings that to roughly 0.05. Since the plausible
  effect of the recovery ladder is of the order of the 24/28 and 37/55 post-infeasibility
  collisions, i.e. 0.03–0.10 absolute, **n = 200 is underpowered and n = 500 is the minimum
  defensible final size.** That is affordable only because of Direction 2 — a concrete
  instance of the dependency argument.
* **Non-significance is never reported as equivalence.** The Phase 4 and Phase 6 language is
  reused verbatim: "not resolved at this sample size", with the CI shown.

### 6.5 Mandatory metrics, and where each comes from

*Navigation*: success, collision, dynamic / static / wall collision, timeout, SPL, minimum true
clearance — all already produced by `episode_record` in `experiments/exp6_ppo_comparison.py`
and `EpisodeDiagnostics`.

*DR-CBF*: `frac_infeasible`, `n_infeasible`, `n_solver_fail`, `n_dcp_error`, `min_cbc_solved`,
`frac_cbc_negative`, `frac_delta_positive`, `mean_u_dev`, `max_u_dev` — all already in
`EpisodeDiagnostics`.

*Infeasibility (new for Phase 7)*: infeasibility category and minimal-subset cardinality;
**staleness attribution** (stale-vs-current / multi-object / static-vs-dynamic); required
uniform slack `s`; **tier** T0/T1/T2; **rung** R0/R1/R1.5/R2 distribution; achieved margin
`m = min_i CBC_i` of the recovered action, measured against both the sampled `xi` and the
ground-truth clearance (labelled diagnostic); rung-switching rate; time from infeasibility to
collision; fraction of collisions preceded by infeasibility within `k` steps for
`k in {1, 5, 10}`.

*Velocity estimation*: position error, velocity error, directional error, `P(e>0)`,
**`P(e>0 | TTC<1s)`**, p95/p99/max of `e`, error by range and by point count, track loss rate,
confirmation latency, association failures, plus **NEES/NIS and `z_beta`-sigma coverage** for
D3d.

*Computation*: total controller latency, sample-construction time, canonicalization/setup time,
solver time, post-processing time, **worst-case (max) latency**, p95/p99, and solver-status
counts including `optimal_inaccurate`.

---

## 7. Expected failure modes, named in advance

| # | failure | direction | how it would show | response |
|---|---|---|---|---|
| 1 | Staleness explains most infeasibility, so the recovery ladder has little left to do | D1/D3a | G2 staleness share high; G3 drops `frac_infeasible` sharply; B2 ≈ B1 | Report it. A negative result about the ladder is still the answer, and the finding about S-T at 10 Hz is the contribution |
| 2 | The ladder converts infeasibility into *silent* T2 violations that collide anyway | D1 | rung R2 frequent, `m` strongly negative, collision unchanged | The tier instrumentation makes this visible rather than hidden; report `m` distribution and the decay-bound violation |
| 3 | R1.5 is not actually safer than R2 | D1 | ordered comparison of collision at matched rung frequency | The ordering was flagged as a claim to verify; drop R1.5 to an ablation |
| 4 | Rung chatter destabilises the controller | D1 | switching rate high, `mean_u_dev` up, SPL down | The `p0||u-u_prev||^2` term in R2 is the designed mitigation; if insufficient, report and stop |
| 5 | V3 disagrees with V0 on infeasibility status | D2 | L1 < 100 % | **Hard stop.** D1 cannot proceed on a substrate that redefines its dependent variable |
| 6 | V1 is not DPP-compliant and delivers no speedup | D2 | L6 shows no improvement | Anticipated in §4.3; the `sqrt(p1)` reformulation is the planned response, with the `h_crit<0` fallback |
| 7 | Acceleration succeeds but the environment/tracker becomes the bottleneck | D2 | total step time dominated by non-controller terms | Pre-declared stopping criterion: V5 is not attempted |
| 8 | D3a imports estimator error into the geometry channel and makes `h` worse | D3a | `h` error against ground truth rises, especially for poor-quality tracks | Measured explicitly in G3; if it dominates, D3a is rejected and `K=1` becomes the recommendation |
| 9 | `Sigma_v` is badly calibrated, so U2's "1 sigma" means nothing | D3d | NEES outside band, coverage far from 84 % | G6 blocks U2 until calibration passes; if it cannot be fixed, U2 is reported as uncalibrated and demoted |
| 10 | U2 raises infeasibility more than it improves safety | D3d | `frac_infeasible` up, collision flat or worse | This is precisely why D3d is sequenced *after* D1; report the interaction |
| 11 | IMM helps on OU and hurts on CV | D3c | opposite signs across motion models | Both models are reported separately and never pooled; no retuning between them |
| 12 | Improvements are non-additive and one masks another | all | interaction term `B4 - (B2 + B3 - B1)` large | The 2×2 exists to detect exactly this; report the interaction rather than the marginals alone |
| 13 | Effects are real but below the resolution of n=500 | all | CIs straddle zero | Reported as unresolved with the CI, never as equivalence |

---

## 8. Safety and guarantee interpretation — the summary claim table

| configuration | guarantee that holds | guarantee that does not |
|---|---|---|
| frozen, feasible step | `inf_{P in B_r(P_N)} P(CBC >= 0) >= 1-eps` over the five sampled constraints | anything about unsampled obstacles, or about the true `h` when `h_est` is optimistic |
| frozen, infeasible step (`u = 0`) | **none.** `CBC_i = alpha h_i + dh_dt_i < 0` whenever `dh_dt_i < -alpha h_i` | forward invariance |
| R1 | nominal CBF `min CBC >= 0`, forward invariance of `{h >= 0}` w.r.t. the sampled constraints | distributional robustness |
| R1.5 (ablation) | the ONE-STEP discrete bound `h_{k+1} >= (1 - alpha' dt) h_k + O(dt^2)` at the relaxed rate, with `alpha' <= 1/dt`; violation `(alpha'-alpha)h -> 0` at contact | the original decay rate; and **continuous-time forward invariance is NOT claimed** |
| R2 | an explicit worst-case bound `h(t) >= (h_0 + m/alpha) e^{-alpha t} - m/alpha`, `m = s - tau` | forward invariance |
| D3a | consistency between the `h` and `dh_dt` channels | independence of `h` from estimator error — this is a **new** dependency and is a real cost |
| U2 | a `z_beta`-sigma chance-constrained tightening of each sampled `dh_dt`, **conditional on `Sigma_v` being calibrated** | anything if calibration fails; hence gate G6 |

Two statements that must appear in every Phase 7 report, because they bound all of the above:
the guarantees are over the **five sampled constraints**, not over the true world; and `h` is
estimated from a 24-ray sensor with a measured far-field miss rate, so an undetected obstacle
is a perception failure that no barrier argument can cover.

---

## 9. Dependencies between the directions — the scientific ordering argument

```
                 +-------------------+
                 |  D2 acceleration  |  substrate: must be equivalence-validated FIRST
                 +---------+---------+
                           | (1) D1's dependent variable is the infeasibility LABEL,
                           |     which the solver decides. (2) D1 and D3 need ~10x the
                           |     experiment volume; 36 ms/step makes that unaffordable.
                           v
                 +-------------------+
                 |  D1a  DIAGNOSIS   |  no remedy; establishes whether the Phase 1.5
                 +---------+---------+  taxonomy transfers to the LiDAR pipeline at all
                           |
                           | (3) if staleness dominates, a recovery mechanism would be
                           |     treating a symptom
                           v
                 +-------------------+
                 |  D3a  de-staling  |  uses ESTIMATED velocity; belongs to D3, but its
                 +---------+---------+  payoff is measured in D1's currency
                           |
                           v
                 +-------------------+
                 |  D1b  the ladder  |  designed against the RESIDUAL infeasibility
                 +---------+---------+
                           |
                           | (4) D3d TIGHTENS constraints and RAISES infeasibility;
                           |     its benefit is unmeasurable while u=0 is the fallback
                           v
                 +-------------------+
                 |  D3b/c then D3d   |
                 +---------+---------+
                           v
                    2x2 factorial, Stage 7
```

Four dependencies, stated as falsifiable propositions rather than intuitions:

1. **D2 → D1.** The infeasibility label is a solver output. If V3 and V0 disagree on any step,
   a measured change in `frac_infeasible` cannot be attributed to D1. Gate G1's L1 requirement
   (100 % status agreement on the enriched infeasible corpus) exists solely for this.
2. **D2 → everything.** n = 500 final episodes × 5 arms × 2 motion models × ~150 steps is
   roughly 750 000 controller calls. At 36 ms that is ~7.5 hours of pure solve per full sweep
   and makes the ablation programme impractical; at 0.3 ms it is minutes. The power argument
   of §6.4 is therefore *contingent* on D2.
3. **D3a → D1b.** Building a recovery ladder for infeasibility that is an artefact of a stale
   buffer would optimise the wrong object. G2 measures the share before D1b is designed.
4. **D1b → D3d.** An uncertainty margin makes constraints harder to satisfy. Evaluating it
   while `u = 0` is still the fallback would confound "the margin is too conservative" with
   "the fallback is bad", which is the exact confound Phase 6 already identified.

One dependency deliberately **not** created: D3's estimator improvements are *not* allowed to
feed back into `alpha`, `r_W` or `eps`. Those stay frozen except in the single labelled
substitution arm S3, which exists to test redundancy and is never the primary system.

---

## 10. Staged implementation order, with deliverables

| stage | direction | deliverable | gate | est. compute |
|---|---|---|---|---|
| 0 | — | frozen manifest, trace corpus, regression harness | G0 | ~1 h |
| 1 | D2 | `fast_drccp.py` (V1, V3, +V2/V4 ablations), equivalence report | G1 | ~4 h |
| 2 | D1a | `exp7_2_infeas_diag.py`, LiDAR-pipeline taxonomy | G2 | ~1 h (offline, no rollouts) |
| 3 | D3a | `predictive_cbf.py`, de-staling report, `K=1` ablation | G3 | ~1 h on D2 substrate |
| 4 | D1b | `recovery.py`, ladder report with rung/tier distributions | G4 | ~2 h |
| 5 | D3b/c | `tracking2.py`, estimator-metrics-first report | G5 | ~3 h |
| 6 | D3d | `uncertainty.py`, calibration + substitution arms S1–S4 | G6 | ~3 h |
| 7 | all | `exp7_7_factorial.py`, B0–B4 × 2 motion models × 500 paired seeds | G7 | ~6 h |

Each stage stops and reports. Nothing downstream begins until the gate is reviewed and
approved, exactly as Phases 1–6 were run.

**Reporting discipline, carried over unchanged:** every deviation labelled; no tuning on
`EVAL_SEED_BASE`; no optimisation against the PPO comparison; PPO numbers re-run only as a
closing appendix in Stage 7 and never as a signal; oracle channels used only as explicitly
labelled diagnostics; information asymmetry restated in every report.

---

## 11. Protocol

### 11.1 When a stage contradicts the plan

Every hypothesis in this document is pre-registered so that it can be *refuted*, and a
refutation is a result, not a problem. The protocol, fixed in advance:

1. The **pre-registered experiment is preserved and reported as run.** It is never rewritten,
   re-scoped or quietly dropped after seeing its outcome, and its result files are never
   overwritten.
2. The **scientific interpretation is updated** in the stage report, which states plainly which
   hypothesis was contradicted, by which measurement, and with what confidence.
3. **Subsequent stages are re-motivated from the new interpretation**, not forced to follow the
   original one. Concretely: if Stage 2 refutes the Finding C staleness hypothesis, Stage 3
   still runs (D3a is independently motivated as a predictive barrier and its cost/benefit is
   worth measuring either way) but it stops being framed as the fix for infeasibility, and
   Stage 4's recovery ladder becomes the headline rather than the residual treatment.
4. Any change to a **later** stage's design that follows from an earlier stage's result is
   written down as an amendment in this file, with the date, the triggering measurement, and
   the old text retained, before that stage is implemented.
5. Hypothesis-contradicting results are reported with the **same prominence** as confirming
   ones. A stage report that refutes its own hypothesis is a successful stage.

### 11.2 Stage-by-stage execution

Stages are implemented **one at a time**. Each stage: implement -> run its gate -> stop ->
report -> await approval. No stage's code is written before its predecessor's gate has been
reviewed. Stage 0 is implemented first and alone; Stage 1 and every Phase 7 controller change
wait on explicit approval of the Stage 0 report.

### 11.3 Naming and result layout

All Phase 7 output is Week 5 output and is labelled `Week5-Phase7`:

```
MD_files/week5/phase7/PHASE7_PLAN.md            this plan
MD_files/week5/phase7/FROZEN_MANIFEST.sha256    the frozen-file manifest
MD_files/week5/phase7/STAGE<n>_REPORT.md        one report per stage
results/week5_phase7/stage0_trace_corpus/
results/week5_phase7/stage1_equivalence/
results/week5_phase7/stage2_infeasibility_diagnosis/
results/week5_phase7/stage3_predictive_barrier/
results/week5_phase7/stage4_recovery_ladder/
results/week5_phase7/stage5_estimator/
results/week5_phase7/stage6_uncertainty/
results/week5_phase7/stage7_factorial/
```

Every result JSON carries a top-level `"label": "Week5-Phase7 / Stage <n> / <name>"` and every
console report opens with that line. No Phase 1–6 result file is written to, renamed or moved.

## 11.4 AMENDMENT 1 — Stage 1 gate criteria (dated 2026-09-06)

**Linked to:** the original Stage 1 run, commit `998eeaa` (`Week5-Phase7-Stage1`), and the
forensic diagnosis `MD_files/week5/phase7/STAGE1_FORENSIC.md` (commit `4c45c08`).

**Status of the original result: PERMANENTLY RECORDED AS FAILED.** The original gate text is
preserved verbatim in §4.4/§4.5 above and in `STAGE1_REPORT.md`; the original failed
measurements stand in `results/week5_phase7/stage1_equivalence/stage1_equivalence.json`. This
amendment does not rewrite, relabel or supersede them. Any rerun is reported as a *second* run
alongside the first.

**Trigger.** The forensic diagnosis established (a) independent feasible-set equality —
99 888 membership comparisons, 0 disagreements, decided without consulting the accelerated
implementation; (b) identical reconstructed coefficients and critical-row ordering; (c) that
the numerical discrepancies are primarily V0/SCS solver error rather than a different
mathematical problem — `corr(‖u_acc − u_V0‖, ‖u_V0 − u_arbiter‖) = 1.000000`, median ratio
1.000000. No implementation error was found; classification **G** (pre-registered tolerance
inappropriate) on top of **D**, with **B** and **E** ruled out by measurement.

**Approved by the reviewer on 2026-09-06 as Amendment B, with revisions**, which are
incorporated below. In particular the relative criterion
`‖u_acc − u_arbiter‖ ≤ ‖u_V0 − u_arbiter‖` proposed in `STAGE1_REPORT.md` §7 was **rejected as
a primary criterion** because it is defined relative to V0's error; it is retained as
diagnostic evidence only.

### 11.4.1 Notation, corrected to remove a collision

`V0` = the frozen reference controller. `V1` = **the accelerated candidate**, which is
`reduced_osqp` and nothing else. The four accelerated variants were previously also called
V1–V4; those labels are **retired** and renamed to avoid ambiguity:

| old | new | role after this amendment |
|---|---|---|
| V1 | `PAR-SCS` | **excluded** from substrate candidacy (§11.4.6) |
| V2 | `PAR-CLARABEL` | ablation only |
| V3 | **`RED-OSQP` = V1** | the sole substrate candidate |
| V4 | `RED-OSQP-WS` | ablation only (measured null on speed) |

### 11.4.2 PRIMARY gate — mathematical equivalence (all four mandatory)

| # | requirement | pass condition |
|---|---|---|
| **P1** | independent feasible-set equality on the pre-registered corpus | **0 disagreements**. Membership decided twice per probe: V0 side by solving the `(t, s_i)` feasibility LP with scipy HiGHS for the fixed `u`; reduced side by direct arithmetic. **The V0 side must remain independent of the accelerated implementation.** Any disagreement is enumerated with its distance to the constraint boundary. |
| **P2** | identical reconstructed constraint coefficients and critical-row ordering | bitwise `array_equal` between the kept-sample matrix re-derived from the recorded `xi` and the matrix the corpus recorded V0 using; criticality sort `α·h + ∂h/∂t` ascending. Over **every** corpus step, not a sample. |
| **P3** | identical variable bounds, τ, α/barrier formulation, and objective formulation | `‖ū‖_∞ = max(1, α, \|u_x\|, \|u_y\|) = 1` exactly over ≥ 10^6 admissible `u`, hence `τ = r_W/ε = 0.04` constant; `p0 = 3`, `p1 = 4·h_crit`, `p3 = 5·h_crit` exact; `\|u\| ≤ max_v` per axis and `δ ≥ 0` identical; preconditions `ε·n_keep < 1` and `max_v ≤ max(1, α)` asserted |
| **P4** | independent arbiter confirmation that the accelerated solution solves the **same** mathematical problem | for each sampled solved step: (a) `u_V1` lies in the V0 feasible set per the **independent LP oracle** of P1, and (b) suboptimality gap `J(u_V1) − J* ≤ J_tol` where `J*` is the arbiter optimum |

**`J_tol` is pre-registered here, before the rerun, at `1e-9`, with justification.**
It is anchored to the *arbiter's own* reproducibility, not to the candidate's performance:
solving the frozen formulation with CLARABEL at `tol 1e-14` and at `tol 1e-12` over 194 solved
steps gives a self-inconsistency of median 5.1e−14, p99 8.2e−13, **max 3.2e−12** in the
objective (and max 7.1e−12 in the action). `J_tol = 1e-9` is ~300× that maximum — loose enough
that the arbiter's own noise cannot cause a failure, and 7 orders below `τ = 0.04` and below
any quantity the study interprets. *Disclosure:* the candidate's action accuracy (~1.8e−8) was
already known from the failed run, so this threshold cannot be claimed to have been set blind;
it is however derived solely from arbiter measurements and is not a function of any candidate
result.

### 11.4.3 SECONDARY — numerical solution accuracy (reported, not a pass/fail gate)

Compare **both** implementations independently against the arbiter:

```
E0 = ||u_V0 - u_arbiter||_inf        E1 = ||u_V1 - u_arbiter||_inf
```

Report for each: **median, p95, p99, maximum**, and the **fraction of cases where E1 < E0**.
No tolerance is attached to E0/E1 and none is invented. The original failed thresholds
(`‖u − u_V0‖ ≤ 1e−6`, `‖u − u_arbiter‖ ≤ 1e−8`, objective `≤ 1e−8`) remain recorded in
`STAGE1_REPORT.md` §1 as failed, and are re-reported unchanged in the rerun for continuity.
The `corr = 1.000000` / `ratio = 1.000000` result is retained as **diagnostic evidence only**
and is explicitly **not** promoted into any criterion.

### 11.4.4 SAFETY — L4 retained unchanged, mandatory

**0 guarantee-tier regressions.** A regression is a step whose tier under V1 is strictly worse
than under V0 (T0→T1, T0→T2, T1→T2). Improvements are reported separately and are not failures.
Evaluated over every solved corpus step.

### 11.4.5 CLOSED LOOP — statistical, never trajectory identity

**Trajectory identity is NOT required and is not tested.** The 4/400 divergent episodes from
the original run remain reported. B0 vs B1 is evaluated with the paired protocol of §6.4:
**success, collision and timeout**, each with an **exact paired McNemar test** and a
**paired bootstrap 95 % CI** (10 000 resamples over seeds), on the same 200 paired seeds per
condition.

Pass condition: for all three outcomes in both conditions, McNemar `p > 0.05` **and** the 95 %
CI on the paired difference contained within **±0.03** absolute. The ±0.03 margin is
pre-registered here with justification: it must be strictly smaller than the smallest effect
Direction 1 aims to detect (0.03–0.10 per §6.4), so that substrate noise can never be mistaken
for a D1 effect.

If it passes, the result is stated as **"statistically/distributionally equivalent"**.
**The terms "behaviour-preserving" and "trajectory-identical" are forbidden.** The system is
described throughout as a **"mathematically equivalent accelerated implementation"**.

### 11.4.6 STATUS HANDLING

`RED-OSQP` is the **only** substrate candidate. **`PAR-SCS` is excluded** on the measured
status-level regression: 2 steps moved to `optimal_inaccurate` on problems the LP-exact oracle
calls infeasible.

The five statuses — `optimal`, `optimal_inaccurate`, `infeasible`, `infeasible_inaccurate`,
`solver_error` — are **preserved as distinct** in all evidence and are never silently collapsed.
The coarse policy-level predicate is reported **separately and additionally**, never as a
substitute.

### 11.4.7 The τ observation, and how it must be worded

The finding that V0 attains `min CBC < τ` on 413 of 603 solved steps (68.5 %, reaching
0.03996336) while the reduced formulation reaches 0.04000000 is **retained**. It is to be
described as a **numerical violation of the theoretical DR-CBF floor in the frozen SCS
implementation**, of magnitude up to 3.7e−5. It must **not** be called "unsafe" without
qualification: it is a solver-accuracy artefact, its magnitude is ~0.09 % of τ, and no claim
about physical safety follows from it.

### 11.4.8 Seed hygiene, unchanged

No tuning of the accelerated solver on `EVAL_SEED_BASE` seeds. Solver settings were fixed
before the original run and are **not** modified by this amendment. The closed-loop analysis
re-uses the *same* 400 paired episodes recorded at `998eeaa`; only the criterion applied to
them changes, so no new randomness enters.

## 11.5 AMENDMENT 2 — Stage 1 gate, structural replacement of P4b and L4 (dated 2026-09-07)

**Provenance chain, unchanged and not overwritten:**
`998eeaa` original Stage 1 gate **FAILED** → `3c85f9d` Amendment 1 → `00587e3` Amendment 1
rerun **FAILED** → this amendment → final rerun.

**Trigger.** Both prior failures were properties of the *criteria*, not evidence that V1 solves
a different problem: P4b's threshold sat below the noise floor of the arbiter measuring it
(the arbiter's own error is ~1.4e−8, revealed by V1 achieving a lower objective than it on some
steps), and L4's exact `≥ τ` test straddles a boundary that coincides with where the optimizer
places the solution. **No third arbitrary numerical threshold is introduced.** Both criteria are
replaced by threshold-free or scale-derived constructions.

P1, P2, P3, P4a, the secondary E0/E1 diagnostics and the closed-loop test are **unchanged**
from Amendment 1 and their Amendment-1 results stand.

### 11.5.1 P4b → solver-independent KKT optimality certificate

The arbiter-distance threshold is **withdrawn**. V1's returned solution is certified from the
recorded problem and the solution alone, with no reference solver anywhere in the criterion.

The reduced QP is written `min ½xᵀPx + qᵀx  s.t.  Cx ≤ d`, `x = [u_x, u_y, δ]`, with the `N`
barrier rows, the CLF row, `δ ≥ 0` and the four box rows — all reconstructed from the recorded
`ξ`, not from the solver. Multipliers are recovered **independently of the solver's own duals**
by non-negative least squares, `λ = argmin_{λ≥0} ‖Cᵀλ + (Px* + q)‖₂`, which uses only
`(P, q, C, d, x*)` and needs no active-set tolerance.

Reported for every certified step:

| residual | definition |
|---|---|
| primal feasibility | `r_prim = max(0, max_i (Cx* − d)_i)` |
| dual feasibility | `r_dual = max(0, −min_i λ_i)` (0 by construction of NNLS; reported anyway) |
| stationarity | `r_stat = ‖Px* + q + Cᵀλ‖∞` |
| complementary slackness | `r_comp = max_i |λ_i · (Cx* − d)_i|`, and `c_sum = Σ_i λ_i (d − Cx*)_i` |

From these, two **certified bounds** are derived — rigorous consequences of strong convexity,
not estimates. With `μ = λ_min(P) = min(2(p0+p1), 2p3) > 0`:

```
f(x*) − f*        ≤  c_sum + ‖r_stat‖₂² / (2μ)
‖x* − x_opt‖₂     ≤  sqrt( 2 (c_sum + ‖r_stat‖₂²/(2μ)) / μ )  =:  δ_cert
```

**PRE-REGISTERED PASS CONDITION, recorded here before the rerun:**

> `max` over sampled solved steps of `δ_cert` **≤ 1e−6**, in action units.

**Justification, from the problem's own scales and from nothing else.** The tolerance is not
calibrated against V0, against the arbiter, or against any solver's behaviour. `1e−6` is:
4 orders below `τ = 0.04`, the smallest quantity the DR-CBF formulation defines; 6 orders below
the action box `max_v = 1.0`, the scale of the controller's output; and below the resolution at
which any quantity this study interprets — actions, `min CBC`, guarantee tier, episode outcome —
could change. A certified distance-to-optimum of 1e−6 means no interpreted result of Phase 7
could differ had the exact optimum been used.

`r_prim` is reported but carries no pass condition, because primal feasibility is already tested
exactly and threshold-free by **P4a** (the LP oracle, 573/573 in the Amendment-1 run). Where
`r_prim > 0` the bounds above are stated for the problem with constraints relaxed by `r_prim`,
and `r_prim` is reported alongside so the reader can see the size of that relaxation.

The arbiter results are retained as **secondary evidence only** and appear in no pass condition.

### 11.5.2 L4 → guarantee tier from the certified optimum

The tier is **no longer classified from a raw floating-point solver CBC value**.

For each implementation the certificate of §11.5.1 bounds the distance to the exact optimum by
`δ_cert`. Since every barrier gradient is a unit vector, the induced uncertainty in
`min_i CBC_i` is bounded by the same `δ_cert`. The tier is therefore assigned to an **interval**:

```
CBC_interval = [ min CBC(x*) − δ_cert ,  min CBC(x*) + δ_cert ]
tier = T0 if the whole interval is >= tau
       T1 if the whole interval is in [0, tau)
       T2 if the whole interval is < 0
       INDETERMINATE if the interval straddles a boundary
```

**PASS CONDITION:** **0 steps at which V0's certified tier and V1's certified tier differ and
both are determinate.** Steps where either tier is indeterminate are **reported as
indeterminate**, with counts, and are **not** counted as regressions. This is the structural
point: an excursion of ~1.96e−9 below `τ` is a **numerical residual**, not a guarantee-tier
regression, and the certificate is what distinguishes the two.

Separately and additionally, the **raw** quantities are reported with no thresholds attached:

```
CBC_V0 − tau     and     CBC_V1 − tau
median, p95, p99, min, max, and counts below zero and below tau
```

**Wording constraint.** V1 having smaller residuals than V0 is **not** to be described as a
safety improvement. It is a statement about numerical accuracy only.

### 11.5.3 Terminology, unchanged

V1 is described throughout as a **"mathematically equivalent accelerated implementation"**.
The terms "trajectory-identical" and "behaviour-preserving" remain forbidden.

### 11.5.4 Stopping rule, pre-registered

If the KKT certificate or the certified-optimum tier test fails, **no further amendment or
tolerance is introduced.** Stage 1 is then recorded as failed and **V0 is kept as the Stage 2–7
substrate.**

## 11.6 AMENDMENT 3 — G-D1a verdict and stage reordering (dated 2026-09-07)

**Linked to:** Stage 2 diagnostic `df42437`, provenance correction `5415bc7`. Chain preserved:
`998eeaa` → `4c45c08` → `3c85f9d` → `00587e3` → `4e55219` → `cc692ab` → `df42437` → `5415bc7`.
No previous result is overwritten.

### 11.6.1 Decision 1 — G-D1a is recorded as CONTRADICTED in the stated mechanism

The Finding C staleness hypothesis (§0, §3.2, §5.1) is **CONTRADICTED in its stated
mechanism**. The 32.8 % oldest-sample leave-one-out restoration rate is **not** reinterpreted
as a pass: it establishes that sample age can have *some* influence, and nothing more. It does
not establish the proposed stale-geometry/current-velocity mechanism.

Decisive evidence, retained in full:

* same-track stale-vs-current conflict: **0 of 1 402** pairwise conflicts (0.00 %)
* the **newest** sample is the most implicated age (34.55 % of MIS rows) — not the oldest
* infeasibility is **not** concentrated near contact: `h_min` median 0.5194 vs 0.4845 on
  feasible controls, and 0.3 % vs 2.2 % inside 1 s TTC
* the K = 1 counterfactual is **confounded** by changing the constraint count, so its 87.25 %
  is not a staleness share

The modest age effect is preserved as evidence, not promoted: every pairwise conflict is
cross-age, and removing the oldest row restores feasibility 1.79× more often than removing the
newest.

### 11.6.2 Decision 2 — the remaining stages are reordered, and a diagnostic is inserted first

The super-physical estimated closing-rate result is now the **leading candidate mechanism**
for infeasibility, but it is an **ASSOCIATION, NOT A CAUSAL FINDING**. Accordingly:

**Nothing is implemented yet.** No IMM, no CT tracking, no uncertainty margin, no predictive
barrier, no recovery ladder, no slack, no estimator redesign of any kind.

**Stage 2b — velocity-estimation causal counterfactual — is inserted before any intervention.**
V0 stays completely frozen. Primary question:

> If ONLY the velocity supplied to `∂h/∂t` is changed, how much of the observed infeasibility
> disappears?

Four arms, on identical recorded states, geometry and constraints:

| arm | `∂h/∂t` supplied |
|---|---|
| **A1** frozen estimate | as recorded (baseline) |
| **A2** ground truth | `−∇h·v_true` of the obstacle that produced the buffered point — **the primary causal test** |
| **A3** zero | `0` |
| **A4** physically capped estimate | `max(∂h/∂t_est, −v_phys)` |

Held identical across arms: robot state, LiDAR geometry, sample selection, critical-row
ordering, `h`, control bounds, objective, `α`, `τ`, and the V0 solver.

Reported per arm: infeasibility rate; minimal infeasible subset size; `h_crit`; closing-rate
distribution; super-physical-row rate; feasibility margin and required control; and a fixed vs
randomized breakdown.

### 11.6.3 Interpretation rule, fixed in advance

* If **ground-truth velocity substantially reduces infeasibility** while everything else is
  held fixed, velocity estimation becomes the **leading supported mechanism**.
* If it does **not**, the velocity-overshoot explanation is **explicitly rejected as
  sufficient** and alternative causes are investigated.

The hypothesis is not to be forced to survive. Stage 3 has not started, and no intervention is
implemented until Stage 2b is reviewed.

## 12. Review record

Reviewed and approved 2026-09-06 with eleven clarifications, all applied above:

| # | clarification | where applied |
|---|---|---|
| 1 | staleness is a **leading hypothesis**, not a finding; Stage 2 decides | §0 Finding C, §1.3, §3.2, §5.1 |
| 2 | factorial arms renamed **B0–B4**; B0 vs B1 is the equivalence check, B1–B4 the factorial | §6.1, §7, §10 |
| 3 | R1.5 must additionally be verified for objective/other-term coupling and for its discrete-time invariance claim; **demoted to ablation by default** | §3.3-D2, §3.4, §8, §13 |
| 4 | D3a is a new **`[extension]` predictive barrier construction**, not a correction; it changes `h` and therefore the controller input | §5.1 |
| 5 | Phase 1–6 baseline preserved completely; nothing existing modified | §2.2, header, and `tests/test_frozen_phase1_6.py` |
| 6 | provenance labels kept and distinguished throughout | header, §5.1, and every module docstring |
| 7 | no tuning on `EVAL_SEED_BASE`; selection and sensitivity on dev/validation only | §6.2 |
| 8 | D2 equivalence gates exact; no runtime reported before L1–L5 pass | §4.4, §4.5, G1 |
| 9 | a recovered action is never called "safe"; always report achieved margin and tier | §3.4 |
| 10 | `Sigma_v` assumed uncalibrated; NEES/NIS/coverage precede any safety interpretation | §5.4, G6 |
| 11 | stage-by-stage implementation; Stage 0 first, then stop and report | §11.2, §10 |
| + | contradicting results update the interpretation, never the pre-registered experiment | §11.1 |
| A1 | **Amendment 1 (2026-09-06)**: Stage 1 gate amended after the forensic diagnosis; original failed result at `998eeaa` preserved | §11.4 |
| A2 | **Amendment 2 (2026-09-07)**: P4b → solver-independent KKT certificate, L4 → tier from the certified optimum; no third threshold; stopping rule fixed | §11.5 |
| A3 | **Amendment 3 (2026-09-07)**: G-D1a recorded CONTRADICTED in the stated mechanism; Stage 2b velocity causal counterfactual inserted before any intervention | §11.6 |

## 13. Open questions to settle before Stage 1 begins

1. **Final evaluation size.** §6.4 argues n = 500. That is a change from Phase 6's 200 and
   needs explicit approval, since it extends the reserved seed block to `1_000_499`.
2. **Does B0 (frozen V0 baseline) get a full 500-episode run**, or only the 200 that tie it to
   Phase 6? The cheaper option is 200; the cleaner is 500. Recommendation: 500, since D2 makes
   it nearly free and it removes an asymmetry between arms.
3. **R1.5 (`alpha` relaxation) is an ABLATION by default** and is promoted to a primary rung
   only if all three admission checks of §3.3-D2 pass AND G4 demonstrates its ordering against
   R2 empirically. This was tightened at review; the earlier draft listed it as a primary rung.
4. **`z_beta` for U2.** Recommendation: fixed at 1.0 a priori, curve reported from validation
   seeds only.
5. **Is the S3 arm (`r_W = 0`) acceptable?** It touches a frozen parameter. Recommendation:
   yes, as a single clearly labelled ablation, because the redundancy question cannot be
   answered otherwise — but only with explicit approval.
