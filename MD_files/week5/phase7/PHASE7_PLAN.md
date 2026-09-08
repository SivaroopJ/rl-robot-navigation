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

## 11.7 AMENDMENT 4 — evidence-driven reordering and the Stage 3 design (dated 2026-09-07)

**Linked to:** Stage 2 `df42437`, provenance correction `5415bc7`, Amendment 3 `e2cda71`,
Stage 2b result `c2019ff`. Chain preserved: `998eeaa` → `4c45c08` → `3c85f9d` → `00587e3` →
`4e55219` → `cc692ab` → `df42437` → `5415bc7` → `e2cda71` → `c2019ff`. V0 is not modified.

### 11.7.1 Two mechanisms, kept explicitly separate from here on

Stage 2b established causally that infeasibility has **two distinct mechanisms**, and no text in
this plan may collapse them:

| # | mechanism | measured size | signature |
|---|---|---|---|
| **M1** | **estimator-induced**: super-physical estimated closing rates | 66.1 % of the infeasibility rate; 69.4 % of baseline infeasible steps; **all 884 cardinality-1 subsets** | a single row demands more closing-rate compensation than the action box can deliver |
| **M2** | **genuine multi-constraint geometric conflict** | the residual 792 steps (0.0114) that survive exact ground-truth velocity | **entirely** multi-constraint (738 pairwise, 54 three-way, 0 singletons); 2.05× worse under randomized motion (0.0154 vs 0.0075) |

**Velocity estimation does not explain all infeasibility.** M2 is real, separate, and untouched
by any velocity fix.

**The de-staling intervention (D3a, §5.1) is NOT the explanation or the fix for infeasibility.**
Stage 2 contradicted the staleness hypothesis in its stated mechanism (§11.6.1). D3a remains an
independent `[extension]` if retained, and is to be justified on its own terms — consistency
between the `h` and `∂h/∂t` channels — never as an infeasibility remedy.

### 11.7.2 The evidence-driven stage ordering, replacing §0 and §10

```
Stage 3  velocity-estimation intervention          addresses M1
Stage 4  residual / genuine infeasibility recovery  addresses M2, sized by Stage 3's residual
Stage 5+ uncertainty and advanced estimator work    ONLY if justified by Stage 3's results
```

The causal sequence is fixed: **estimated-velocity problem → address the estimator → measure the
residual genuine infeasibility → then design recovery for what remains.** Recovery must not be
used to compensate for estimator-induced infeasibility before the estimator intervention is
tested, so **Stage 3 runs on the FROZEN `u = 0` fallback**, unchanged.

### 11.7.3 Stage 3 — hypotheses

* **H3-efficacy.** Preventing physically impossible closing-rate estimates reduces the
  infeasibility rate relative to the frozen baseline.
* **H3-safety (stated symmetrically, so the result cannot be spun).** The intervention acts in
  *both* directions and the net sign is unknown:
  * it removes *pessimistic* overshoot → fewer infeasible steps → fewer `u = 0` fallbacks →
    plausibly **fewer** collisions (Phase 6: 86 % / 67 % of collisions follow an infeasible step);
  * it also removes conservatism on rows where a high estimate was warranted → more
    *optimistic* error → plausibly **more** collisions (Phase 4: `P(e>0 | TTC<1s) = 0.142`
    already, and capping can only raise it).
* **H3-residual.** After the intervention, the surviving infeasibility is predominantly M2 —
  multi-constraint, with few or no cardinality-1 subsets.

### 11.7.4 Stage 3 — treatments, with the capping rule fixed BEFORE the experiment

The intervention lives **upstream of the controller**, in `ξ` construction. **The DR-CBF
controller itself is not modified at all**: α, τ, ε, the objective, sample selection and the
`u = 0` fallback are untouched, and V0 remains the solver.

| arm | definition |
|---|---|
| **C0** frozen baseline | the Phase 6 system, unchanged — reproduces the published numbers |
| **T1** projection clamp *(primary)* | `∂h/∂tᵢ := max(∂h/∂tᵢ, −v_cap)` when building `ξ`. Exactly Stage 2b's A4, so its open-loop effect is already measured |
| **T2** velocity-vector clamp *(variant)* | inside a NEW tracker subclass: `v̂ := v̂ · min(1, v_cap/‖v̂‖)`. Repairs the estimate rather than the derived quantity; Stage 2b showed it is at least as effective as T1 on closing rows, but it also alters receding rows where T1 is a no-op, so it is not strictly dominated |
| **D-oracle** *(diagnostic only)* | ground-truth velocity. An upper bound on any estimator. **Never a headline arm**, always labelled |

**`v_cap` IS PRE-REGISTERED HERE AT 0.96, AND IS DERIVED, NOT TUNED.**

A single row is unsatisfiable inside the action box iff `max_v·‖∇h‖₁ < τ − α·h − ∂h/∂t`. With
unit gradients `‖∇h‖₁ ≥ 1`, and with the clamp `∂h/∂t ≥ −v_cap` and `h ≥ 0`:

```
max_v * ||grad h||_1  >=  max_v  =  1.0
tau - alpha*h - dh_dt  <=  tau + v_cap
so  v_cap <= max_v - tau  =  1.0 - 0.04  =  0.96   ==>   NO cardinality-1 subset can exist
```

`v_cap = 0.96` is therefore **the largest cap that provably eliminates the entire M1 singleton
class**, and it is derived from `max_v` and `τ` alone — both controller-internal constants. The
guarantee is conditional on `h ≥ 0`, which held on **69 226 / 69 226** corpus steps.

**Information-ledger note, and a correction to how Stage 2b must be read.** Stage 2b's A4 used
`V_PHYS = 0.675 / 0.75`, the *configured obstacle speed* — environment ground truth. That is
legitimate for a diagnostic but **would break the information ledger as an intervention**: the
controller is not entitled to know how fast obstacles are. `v_cap = 0.96` uses no environment
knowledge whatsoever. Because 0.96 is a **looser** cap than 0.675/0.75, it clamps less, so
**Stage 2b's −52.1 % is an UPPER bound on what T1 at `v_cap = 0.96` can achieve, and a smaller
reduction is expected.** That prediction is pre-registered.

Sensitivity arms, reported from dev/validation seeds only and never used to select a value:
`v_cap ∈ {0.96, 1.0}` (1.0 = the robot's own `MAX_SPEED`, also ledger-clean), plus
`v_cap = V_PHYS` as a clearly-labelled diagnostic that reproduces Stage 2b's A4.

### 11.7.5 Stage 3 — metrics, both sides, mandatory

**Infeasibility side:** infeasibility rate; absolute count; minimal-infeasible-subset cardinality
distribution (the M1/M2 split); super-physical row rate; required slack; fraction of steps where
the clamp actually bound (**intervention frequency**); `u = 0` fallback frequency.

**Safety side:** collision rate, split **dynamic / static / wall**; success; timeout; **minimum
true clearance**; SPL; collisions while feasible vs collisions after an infeasible step.

**Estimator side:** velocity-estimation error `‖v̂ − v‖`; projected error `e = ∂h/∂t_est −
∂h/∂t_true` with `P(e>0)`, p95, p99; **`P(e>0 | TTC < 1 s)`** — the Phase 4 primary, directly
comparable to its 0.142; and the *pessimistic* tail `P(e < −0.05)`, which the clamp targets.

**Protocol:** the frozen environment and evaluation code, the same paired seeds, and the Phase 6
200-paired-episode protocol per condition. Tiering as §6.2: dev 20 → validation 50 → final 200.

### 11.7.6 Stage 3 — gate G-S3, pre-registered

**Safety is a veto, not a trade.** A method that reduces infeasibility but increases collisions
is **not** an acceptable improvement.

1. **Safety veto.** If the paired collision rate increases significantly (exact McNemar
   `p < 0.05`) in either condition, the arm is **rejected regardless of any infeasibility gain**.
2. **Efficacy.** The infeasibility rate must fall relative to C0, paired, with the reduction and
   its CI reported.
3. **Residual characterisation.** The surviving infeasibility must be characterised by MIS
   cardinality, to size M2 for Stage 4.
4. **Honest power statement, pre-registered.** At n = 200 paired episodes the collision test
   resolves roughly 0.08 absolute (Phase 6 could not resolve 0.275 vs 0.335). **A non-significant
   collision result therefore does NOT establish safety — it only fails to detect harm**, and
   must be reported in those words. Because of this, two higher-`n` per-step safety endpoints are
   *also* required and are treated as primary alongside collision:
   * minimum true clearance (n = 400 episodes),
   * `P(e>0 | TTC < 1 s)` (n ≈ 10⁴ steps), which must not worsen materially against Phase 4's 0.142.
5. **Outcome routing.** Accept for Stage 4 only if efficacy holds and no safety endpoint worsens.
   If efficacy holds but a safety endpoint worsens, the arm is rejected and the M1 finding stands
   as *diagnostic only*. If efficacy fails, H3-efficacy is contradicted and Stage 3 reports that
   rather than escalating to richer estimators.

**Only if Stage 3 is accepted** does improved motion-model work (CT / IMM) or uncertainty
propagation become justified; none of it is authorised by this amendment.

### 11.7.8 CLARIFICATION to G-S3 (recorded 2026-09-07, before implementation)

**The efficacy gate is MECHANISM-SPECIFIC. "Total infeasibility decreased" is NOT the success
criterion.** Three quantities are reported separately for every arm and condition:

1. **singleton / M1 infeasibility** — steps whose minimal infeasible subset has cardinality 1
2. **multi-constraint / M2 infeasibility** — steps whose MIS has cardinality ≥ 2
3. **total infeasibility**

**PRE-REGISTERED PREDICTION, recorded before the run:**

> T1 reduces the singleton / M1 infeasibility rate relative to C0, but residual
> multi-constraint / M2 infeasibility remains.

T1 is **not** required or expected to eliminate M2. An arm that removes the singleton class
while leaving M2 substantially intact **satisfies** the efficacy criterion; an arm that reduces
total infeasibility without reducing the singleton class does **not**.

**The `v_cap = 0.96` guarantee, stated precisely and not generalised.**

> **Under the observed corpus conditions `h ≥ 0` and `‖∇h‖₁ = 1`**, any `v_cap ≤ max_v − τ =
> 0.96` eliminates the singleton infeasibility class.

Both conditions are empirical facts of this corpus (`h ≥ 0` on 69 226/69 226 steps; gradients
are unit vectors by construction, and `‖∇h‖₁ = 1` exactly when a gradient is axis-aligned, which
is the worst case — off-axis gradients give `‖∇h‖₁ ∈ (1, √2]` and are strictly easier). **The
guarantee is not claimed beyond those conditions**: it does not hold for `h < 0`, nor for any
configuration with a different `max_v` or `τ`, and the code asserts the preconditions rather
than assuming them.

**Stage 2b's A4 result (−52.1 %) remains an UPPER-BOUND DIAGNOSTIC, not an expected T1 result.**
A4 used the privileged physical obstacle speed (0.675 / 0.75); T1 uses the ledger-clean derived
0.96, which is looser and therefore clamps less. A smaller reduction is expected and predicted.

**Safety, restated.** G-S3 remains a **veto**: a significant paired collision increase rejects
the intervention regardless of infeasibility reduction. **A non-significant collision test at
n = 200 must NOT be described as proof of safety.** The higher-`n` endpoints — minimum true
clearance and `P(e > 0 | TTC < 1 s)` — are retained and are evaluated alongside navigation
outcomes.

**Information ledger.** T1 and T2 use only information available to the controller.
Ground-truth velocity remains **diagnostic only** and never enters a headline arm.

### 11.7.7 Files Stage 3 would add, when approved

New only; nothing frozen is edited. `dr_control/capped_velocity.py` (T1/T2, subclassing the
frozen source and tracker), `dr_control/policy_phase7.py`, `tests/test_dr_control_phase7_stage3.py`,
`experiments/exp7_3_velocity_intervention.py`, `results/week5_phase7/stage3_velocity_intervention/`.
`PHASE7_PATHS` in the manifest generator gains the new `dr_control/` modules — a reviewable act.

## 11.8 AMENDMENT 5 — Stage 3 outcome, T1 accepted, and the Stage 4 design (dated 2026-09-07)

**Linked to:** Stage 3 result `c4ae28b`, its pre-registration `5284025` and clarification
`6566d76`. Chain preserved: `cc692ab` → `df42437` → `5415bc7` → `e2cda71` → `c2019ff` →
`5284025` → `6566d76` → `c4ae28b`. V0 remains frozen and no previous result is overwritten.

### 11.8.1 Stage 3 conclusions, recorded precisely

* **M1 was eliminated exactly.** Singleton (cardinality-1) infeasibility counts went
  **405 → 0** (fixed) and **298 → 0** (randomized); the rate went 0.0148 → 0.0000 and
  0.0114 → 0.0000.
* **Total infeasibility fell 36.4 % (fixed) and 26.3 % (randomized)** — smaller than the
  privileged Stage-2b A4 (−52.1 %), exactly as pre-registered, because `v_cap = 0.96` is looser
  than the ground-truth 0.675/0.75 that A4 used.
* **The residual under T1 is entirely M2 / multi-constraint: 0.0232 (fixed) and 0.0283
  (randomized).**
* **T1 is NOT claimed to improve navigation safety.** Collision differences were not
  significant (0.140 → 0.135, p = 1.000 fixed; 0.275 → 0.280, p = 1.000 randomized), and at
  n = 200 the test resolves only ~0.08 absolute.
* **The small M2 increase is reported as observed and is not explained away.** M2 rose
  0.0216 → 0.0232 (fixed) and 0.0271 → 0.0283 (randomized) under T1. Changed actions imply
  changed visited states; that is noted as a mechanism, not offered as a dismissal.
* **The singleton → fallback → collision conjecture is WEAKENED and must not be presented as
  established.** Eliminating the entire singleton class moved fallback-associated collisions
  only 24 → 22 (fixed) and 37 → 39 (randomized).
* **D-oracle remains diagnostic and ledger-breaking only.** Its randomized improvement
  (0.275 → 0.195, p = 0.0070) demonstrates the value of accurate velocity information but
  **does NOT authorise estimator-accuracy work at Stage 5+**.

### 11.8.2 Decision: T1 accepted, T2 rejected

**T1 (projection clamp, `v_cap = 0.96`) is the accepted Stage-3 configuration and is carried
forward.** `v_cap = 0.96` carries forward unchanged, derived not tuned, under the stated
conditions `h ≥ 0` and `‖∇h‖₁ = 1` only.

**T2 (vector clamp) is rejected in favour of T1** and is not continued unless explicitly
requested later. Basis: T2 clamps more often (0.110–0.130 vs 0.070–0.094) and buys more
infeasibility reduction, but moves three randomized endpoints adversely — collision +0.015,
mean clearance 0.182 → 0.177, worst-case clearance −0.0593 → −0.0712.

**V0 stays frozen and the `u = 0` fallback stays unchanged** everywhere except inside the
Stage-4 treatment arms defined below.

### 11.8.3 Gating rule for estimator work

Estimator-accuracy work — IMM, constant-turn models, covariance propagation, uncertainty
margins — **remains gated on Stage 4 outcomes** and is not authorised by the D-oracle result or
by anything in this amendment.

### 11.8.4 Stage 4 — objective and scope

**Recover the residual genuine multi-constraint infeasibility (M2) and nothing else**, sized
from the measured Stage-3 T1 residual: **2.32 % (fixed), 2.83 % (randomized)**, which is
**100 % multi-constraint** in both conditions.

Out of scope, and not to appear in any Stage-4 arm: governor logic, IMM/CT estimation,
uncertainty modelling, de-staling / predictive barriers, changes to `α`, `τ`, `ε`, `v_cap`, the
objective, sample selection, the planner, the LiDAR pipeline, `robot_env/` or `evaluation/`.

**The Stage-4 baseline is the Stage-3 T1 configuration**, not the frozen C0 — Stage 3 changed
the states visited, so C0's infeasible steps are not the population Stage 4 acts on.

### 11.8.5 Stage 4 — arms

| arm | fallback on an infeasible step |
|---|---|
| **B4** baseline | T1 + the frozen `u = 0` fallback (reproduces Stage 3's T1 row) |
| **R1** drop the Wasserstein margin | re-solve with `τ := 0`; gives up distributional robustness **only** |
| **R2** minimum-violation lexicographic | maximise the worst margin `min_i CBC_i`, then minimise the frozen objective among the maximisers |
| **LADDER** | R0 → R1 → R2, stopping at the first feasible rung — the composed policy |
| *(ablation)* **R1.5** | local `α'` relaxation, **only if** its three admission checks of §3.3-D2 pass; otherwise not run |

Every rung is convex, parameter-free, and applies **only** on steps the frozen problem declares
infeasible. On feasible steps the ladder must be **bit-identical** to B4 — a hard test.

### 11.8.6 Stage 4 — the five-way distinction, mandatory in every record

1. feasible DR solution · 2. infeasible DR problem · 3. recovered action ·
4. **actual CBC violation of the recovered action** · 5. subsequent collision.

Per step: the rung reached; the achieved margin `m = min_i CBC_i` recomputed from the returned
action against the samples the controller saw; the guarantee tier T0 (`m ≥ τ`) / T1
(`0 ≤ m < τ`) / T2 (`m < 0`); and the implied worst-case decay bound for T2. **A recovered
action is never described as "safe"; `m` and its tier are always printed beside it.**

### 11.8.7 Stage 4 — metrics and gate G-S4

Metrics: infeasibility rate and MIS cardinality (must stay ≥ 2 — a Stage-4 arm that creates
singletons is broken); **rung distribution**; **tier distribution and `m`**; rung-switching rate
(chatter); collision split dynamic/static/wall; success; timeout; minimum true clearance; SPL;
collisions while feasible vs after a recovered step; `‖u − u_nom‖`.

**G-S4, pre-registered.**
1. **Safety veto**, unchanged: a significant paired collision increase against B4 rejects the
   arm regardless of any other gain. A non-significant result at n = 200 is reported as
   *failed to detect harm*, never as proof of safety.
2. **Inertness:** on feasible steps the ladder is bit-identical to B4.
3. **Characterisation, not outcome, is the primary claim.** The deliverable is the rung/tier/`m`
   distribution over the M2 population — what guarantee is actually obtained when the problem is
   infeasible.
4. **Pre-registered expectation about collisions.** Stage 3 removed ~40 % of infeasible steps
   and moved fallback-associated collisions by 24 → 22 and 37 → 39, i.e. essentially not at all.
   **Stage 4 is therefore expected to show little or no collision benefit**, and a null result
   is a valid, reportable outcome — not a reason to escalate.

### 11.8.8 Stage 4 — pre-registered predictions

* **P1.** R1 alone (`τ := 0`) resolves only a small minority of M2 steps: Stage 2 measured the
  required uniform slack at median 0.1853 against `τ = 0.04`, with only 315 of 2 337 steps in
  the `slack ≤ τ` tier. Predicted R1 success on M2: **well under 25 %**.
* **P2.** R2 always returns an action, and most recovered actions land in **tier T2**
  (`m < 0`), because the required slack exceeds `τ` on the large majority of steps.
* **P3.** Collision rate is **not** significantly reduced relative to B4, following §11.8.7-4.
* **P4.** The residual MIS cardinality stays ≥ 2 in every arm; no arm creates singletons.

## 11.9 AMENDMENT 6 — Stage 4 outcome, and the Stage 5 design (dated 2026-09-08)

**Linked to:** Stage 4 result `91bb969`, its implementation `0073790` and its pre-registration
`aa66cd7`. Chain preserved: `cc692ab` → `df42437` → `5415bc7` → `e2cda71` → `c2019ff` →
`5284025` → `6566d76` → `c4ae28b` → `aa66cd7` → `0073790` → `91bb969`. V0 remains frozen; no
previous result is overwritten or restated. Nothing in this amendment revises an earlier
conclusion.

### 11.9.1 Stage 4 verdicts, recorded exactly

* **P1 CONFIRMED.** R1 (`τ := 0`) recovers **13.3 % (fixed)** and **12.1 % (randomized)** of the
  infeasible steps it faces — both below the pre-registered 25 % threshold. Dropping the
  Wasserstein margin alone does not resolve the M2 population.
* **P2 CONFIRMED.** R2 always returns an action (377/377 fixed, 580/580 randomized) and the
  returned actions are **predominantly tier T2**.
* **P3 CONTRADICTED.** R2 significantly reduces collisions in both conditions:
  * fixed **0.135 → 0.055**, paired p = **0.0004**
  * randomized **0.280 → 0.175**, paired p **< 0.0001**

  LADDER shows the same effect:
  * fixed **0.135 → 0.065**, p = **0.0013**
  * randomized **0.280 → 0.180**, p = **0.0002**

  P3 continues to **HOLD for R1** (p = 0.18 fixed, 0.73 randomized).
* **P4 HOLDS.** No treatment arm creates MIS cardinality 1; the residual minimal infeasible
  subsets stay ≥ 2 in every arm and both conditions.

### 11.9.2 Mechanistic interpretation, stated with the distinction intact

Stage 3 and Stage 4 intervened on **different objects**, and the conclusions must not be merged.

* **Stage 3 weakened** the singleton → fallback → collision conjecture: eliminating the entire
  singleton (M1) class did not materially change fallback-associated collisions (24 → 22 fixed,
  37 → 39 randomized). That record stands unchanged.
* **Stage 4 changed the action taken after infeasibility** and found collisions **after an
  infeasible step** falling **22 → 4 (fixed)** and **39 → 12 (randomized)**, while collisions
  while feasible rose slightly (5 → 7 and 17 → 23).

**Therefore the broader hypothesis — that the `u = 0` infeasible-step fallback is an important
collision pathway — is now strongly supported.** Stage 3 could not have detected this, because
it removed infeasible *steps* and never changed the *fallback*.

**Explicitly NOT claimed: that singleton infeasibility itself causes collisions.** Stage 3
remains the direct test of that narrower claim, and it weakened it.

**Closed-loop qualification, preserved.** The treatment arms visit different states, so the
lower subsequent infeasibility rate under recovery (0.0232 → 0.0124 fixed, 0.0282 → 0.0189
randomized) is an **outcome of changed trajectories**, not evidence that recovery "removed" the
underlying M2 population. Every per-arm recovery fraction is conditional on that arm's own
trajectory distribution.

### 11.9.3 Safety qualification, mandatory wherever Stage 4 is cited

**Recovered actions are NOT safe guarantees.**

* **No recovered action reached T0** in any arm or condition.
* Most recoveries are **T2 with `m < 0`** (79.8 %/76.7 % under R2; 96.3 %/96.9 % under LADDER),
  meaning the original DR-CBF forward-invariance/robustness guarantee **is lost on those steps**.
  Median `m` −0.070/−0.059, minimum −0.572/−0.638; worst implied decay floors −1.43/−1.59.
* **Worst-case clearance does not uniformly improve.** Fixed −0.0510 → −0.0398 (R2, better) but
  −0.0580 (LADDER, worse); randomized −0.0629 → −0.0702 (R2 and LADDER, worse) and **−0.1451**
  for the R1.5 ablation.
* **R2 introduced small static-collision regressions in randomized** (0 → 2; 0 → 1 for LADDER and
  R1.5) — a category that had been exactly zero since Phase 6.
* **R1.5 remains an ablation and is NOT promoted**, despite passing all three admission checks
  and performing comparably to R2. Its worst-clearance regression is the stated reason.

The collision reduction is an **outcome measurement**, not a guarantee: the ladder trades a hard
constraint for a quantified violation, and in this environment that trade paid.

### 11.9.4 The two directions Stage 4 leaves open

**No estimator intervention is authorised by this amendment.** IMM, constant-turn models,
covariance propagation into the barrier, and uncertainty margins remain **gated** and are NOT
implemented at Stage 5.

| | claim | evidence | what it costs |
|---|---|---|---|
| **A. estimator accuracy** | accurate velocity information can materially improve safety | **D-oracle**, randomized collision **0.275 → 0.195, p = 0.0070** (Stage 3) | nothing yet — D-oracle is **privileged and diagnostic only** |
| **B. recovery** | replacing the `u = 0` fallback materially improves empirical collision outcomes | Stage 4 R2/LADDER, both conditions, §11.9.1 | the DR guarantee itself: no T0, predominantly T2 |

Direction B is **established and closed as an empirical result**. It is not re-litigated at
Stage 5 and it is not re-tuned; R2/LADDER as run at Stage 4 (`91bb969`) is the recovery
configuration, unchanged, and it remains **the current empirical fallback baseline**.

Direction A is **unestablished**. The only evidence for it is a privileged diagnostic.
**D-oracle is an upper bound / target, not proof that a practical estimator reproduces its
benefit**, and no part of the Stage-5 design assumes it will.

### 11.9.5 Provenance of every claim Stage 5 rests on

Stage 5 must not silently inherit a hypothesis as though it were a result. The status of each
claim at the moment Stage 5 is designed:

**Established by Phases 1–4 (frozen, not re-opened):**
* The τ collapse identity: with `εN = 0.5 < 1` the DR constraint is exactly `minᵢ CBCᵢ ≥ τ`,
  `τ = r_W‖ū‖∞/ε = 0.04`, constant. Verified analytically and over 69 226 steps at Stage 0.
* The DRCCP feasible set reduces exactly to the nominal CBF set at `r_W = 0, ε = 1, N = 1`.
* `ξ` ordering, the clearance convention, and the LiDAR radius bookkeeping.
* The frozen estimator's structure: fixed-radius reconstruction, compactness-led static
  rejection, CV Kalman filter with isotropic `R` and hand-set `POINT_COUNT_FACTOR`.
* `P(e > 0 | h < 0.6 & binding) = 0.142` for the CV estimator **in the standalone Phase-4
  harness**. **This number is NOT the Stage-5 reference** — see §11.9.10.

**Established by Stage 2 (diagnosis, observational):**
* Staleness (G-D1a) is **CONTRADICTED** in its stated mechanism: 0/1402 same-track conflicts.
* Estimated closing rates exceeding the physical obstacle-speed bound ("super-physical" rows)
  are strongly associated with infeasibility: `P(≥1 super-physical row | infeasible) = 0.6795`
  vs `0.0450` at base rate (15.1×); `P(infeasible | ≥1 super-physical row) = 0.3454` vs `0.01159`
  (29.8×). **Association, measured against a controlled base rate — not causation.**
* The super-physical test itself uses the configured obstacle-speed bound (0.675 / 0.75) as a
  **metric threshold only**. It is a diagnostic instrument, never an input to any controller.

**Established by Stage 2b (causal counterfactual):**
* Substituting ground-truth obstacle velocity removes **all 884 singleton (cardinality-1)
  infeasible steps** and reduces total infeasibility by **52.1 %**. Velocity-estimation error is
  therefore a **cause** of the M1 class, not merely a correlate. Privileged; diagnostic only.

**Established by Stage 3 (intervention, non-privileged):**
* T1 (projection cap, `v_cap = 0.96`, derived) eliminates M1 exactly: **405 → 0** (fixed),
  **298 → 0** (randomized). Total infeasibility −36.4 % / −26.3 %. Accepted configuration.
* The residual is **100 % M2 / multi-constraint**: 0.0232 (fixed), 0.0283 (randomized).
* T1 does **not** improve navigation safety: collisions 0.140 → 0.135 (p = 1.000) and
  0.275 → 0.280 (p = 1.000).
* The singleton → fallback → collision conjecture is **WEAKENED**: 24 → 22 and 37 → 39.
* D-oracle improves randomized collisions 0.275 → 0.195 (p = 0.0070) — **privileged**.
* In-environment optimistic tail under T1: `P(e > 0 | TTC < 1 s)` = **0.397** (fixed, from C0
  0.405) and **0.498** (randomized, from C0 0.485). Mixed sign; T1 was **worse** under
  randomized motion.

**Established by Stage 4 (intervention, non-privileged):**
* §11.9.1 verdicts; §11.9.2 mechanism; §11.9.3 safety qualification. In particular the `u = 0`
  fallback **is** an important collision pathway, and recovery buys that at the cost of the
  guarantee (no T0; 77–97 % T2).

**Merely HYPOTHESES for Stage 5 — none of these is a result:**
* H-a. A **non-privileged** estimator change can reduce optimistic closing-rate error.
* H-b. Such a change reduces the super-physical row rate **without** making the estimator
  sluggish (see §11.9.8 — the sluggish failure mode makes the super-physical rate trivially
  improvable while making safety worse).
* H-c. Any of that reduces **M2** infeasibility. Stage 3 showed velocity work eliminates M1;
  **nothing establishes that it touches M2 at all**, and M2 is the entire residual.
* H-d. An estimator intervention can leave rescued steps in **T0** — i.e. retain the DR
  guarantee — which no recovery rung has ever achieved.
* H-e. Any of it reduces collisions. D-oracle is the only evidence and it is privileged.

### 11.9.6 Stage 5 — the research question, stated precisely

**The remaining research question is NOT "can we get fewer collisions?".** Stage 4 already
answered that empirically, in the affirmative, by sacrificing the guarantee.

> **Q(Stage 5): Can better velocity estimation reduce the *need* for recovery — i.e. reduce
> infeasibility — while *retaining* the DR-CBF guarantee (T0), rather than trading it away?**

The two sub-questions, kept apart:

* **Q-EST (accuracy).** Does a minimal, non-privileged estimator change reduce optimistic
  closing-rate error and super-physical rows? This is answered on **step-level** endpoints and is
  independent of anything downstream.
* **Q-GUAR (guarantee).** Does that accuracy gain convert into **fewer infeasible steps that
  remain in T0**, rather than into recovered steps in T2? This is the question that distinguishes
  Direction A from Direction B and is the **primary Stage-5 claim**.

Deliberately **not** the primary question: Q-COLL (does it reduce collisions). Collisions are a
**safety-veto and secondary** endpoint at Stage 5, for the reason in §11.9.11: at n = 200 the
paired test cannot resolve the effect sizes an estimator change plausibly produces, whereas the
step-level endpoints rest on 10⁴–10⁵ observations.

### 11.9.7 Stage 5 — baseline, and exactly what changes

**The Stage-5 baseline is the Stage-4 B4 configuration, unchanged in every respect:**

| component | Stage-5 baseline | status |
|---|---|---|
| controller | frozen V0 (CVXPY/SCS reference) | **frozen, never modified** |
| `α`, `r_W`, `ε`, `τ` | 0.4, 0.004, 0.1, 0.04 | unchanged |
| barrier source | T1 `ProjectionCappedSource`, `v_cap = 0.96` | **accepted at Stage 3, carried forward unchanged** |
| velocity estimator | frozen `LidarVelocityTracker` (CV, isotropic `R`, `POINT_COUNT_FACTOR`) | the **only** thing the treatment changes |
| fallback on infeasible | frozen `u = 0` | unchanged in A0/A1; LADDER in A2/A3 |
| recovery ladder | `RecoveryLadder` exactly as run at Stage 4 | **not re-tuned, not re-designed** |
| objective, sample selection, criticality sort, planner, LiDAR pipeline | frozen | unchanged |
| `robot_env/`, `evaluation/` | untouched | unchanged |

**Exactly one thing differs between baseline and treatment:** the measurement-noise covariance
returned by the tracker's `measurement_noise`. Nothing else in the estimator — motion model,
gate, association, confirmation logic, segmentation, static rejection, `r_nominal` — is altered.

**Implementation constraint:** `dr_control/velocity_tracker.py` is **extended by subclass, not
edited**, so the frozen sha256 manifest continues to cover it and A0 remains bit-reproducible.

### 11.9.8 Stage 5 — the first estimator intervention (E1), and the failure it targets

**E1 = geometry-derived reconstruction covariance. Nothing else.** This is deliberately the
smallest interpretable change with a derivation behind it, and it is **not** IMM, not CT, not
covariance-into-the-barrier, and not an uncertainty margin.

**The failure mechanism it targets, traced through the stages.** Stage 2 found that infeasible
steps are dominated by **super-physical estimated closing rates** — the estimator reporting an
obstacle closing faster than any obstacle can move. Stage 2b showed those estimates are a
**cause** of singleton infeasibility, and Stage 3 confirmed it by removing the class entirely
with a cap. But T1 caps the **symptom** at the barrier — it clips `∂h/∂t` after the fact — and
Stage 3 measured that it does **not** improve the optimistic tail (0.485 → 0.498 randomized,
worse). The estimate itself is still wrong; T1 only stops the wrongness from entering in one
direction.

**Where the wrongness comes from, geometrically.** At the ranges where the barrier binds
(`h < 1 m`) an obstacle returns **1–4 LiDAR points** (measured: 4.07 points at 0.6 m, 1.00 at
2.0 m). For a one-point return the fixed-radius reconstruction `c = q + r·u_ray` is exact only
head-on and carries up to `r·sin θ_incidence` of **tangential** error, with near-zero **radial**
error. The frozen filter is told `R = diag(s², s²)` — **isotropic** — with `s` inflated by a
hand-set `POINT_COUNT_FACTOR = {1: 3.0, 2: 1.5}`. So the filter under-states the error in the
one direction where it is large and over-states it in the direction where it is small. Frame to
frame, the reconstructed centre slides tangentially along the obstacle surface, and the Kalman
filter — trusting that displacement too much — converts it into velocity. **That is a
structurally plausible source of super-physical closing rates, and it is the one E1 addresses.**

**What E1 replaces it with.** The surface-point noise is propagated through the reconstruction
geometry, per point count:

* **n = 1:** **[CORRECTED before implementation — see §11.9.18, which supersedes this
  bullet.]** the neighbouring rays returned no hit in this segment, which bounds the disc's
  angular extent and hence the admissible incidence interval. The tangential displacement is
  modelled as bounded on `[−r·sin Δ, +r·sin Δ]` with `Δ = 2π/24` the ray spacing, giving
  `σ_t² = r²sin²Δ / 3`; the radial variance stays `σ_range²`. `R` is that diagonal **rotated
  into the ray frame** — anisotropic, with the large axis tangential.
* **n = 2:** first-order propagation of `σ_range` through the two-root chord construction
  `c = m ± b·n̂`, `b = √(r² − a²)`, which correctly blows up the normal variance as `a → r`
  (the near-degenerate chord the frozen code merely flags).
* **n ≥ 3:** the Gauss-Newton fixed-radius fit's own covariance, `σ_range²(JᵀJ)⁻¹`.

**Degrees of freedom: zero new tuned parameters.** E1 uses only quantities that already exist —
`r_nominal = 0.3`, `SIGMA_R_BASE = 0.02`, the ray spacing `Δ`, and the existing range-growth
factor `(1 + d/3)`, which is **retained unchanged** so that exactly one thing varies. It
**removes** two hand-set constants (`POINT_COUNT_FACTOR`). It is therefore not tunable, and
§11.9.15 forbids making it so.

**No privileged information.** E1 reads ray geometry and the existing design constant
`r_nominal`. No obstacle speed bound, identity, radius, position or velocity enters. In
particular the configured obstacle-speed range (0.6–0.75) — which the Stage-2 *metric* uses as a
threshold and which Stage-2b A4 used as a privileged input — **does not appear anywhere in E1**.
The four frozen leakage layers are re-applied to it (§11.9.13-A3).

**The honest failure mode, pre-registered.** Inflating `R` makes the filter slower. A sluggish
filter reports **smaller** velocities, which mechanically reduces the super-physical rate while
**increasing** optimistic error — it under-states genuine closing. **The super-physical rate is
therefore not admissible as E1's primary endpoint**, and §11.9.13-A4 requires improvement on the
optimistic tail *and* non-increase in the super-physical rate jointly, precisely to make this
failure mode non-passing.

### 11.9.9 Stage 5 — arms, and the relationship to recovery

| arm | estimator | fallback on an infeasible step | role |
|---|---|---|---|
| **A0** | frozen CV + isotropic `R` | frozen `u = 0` | **baseline** (= Stage-4 B4) |
| **A1** | **E1** | frozen `u = 0` | estimator main effect |
| **A2** | frozen CV + isotropic `R` | **LADDER** (R0→R1→R2) | recovery replication (= Stage-4 LADDER) |
| **A3** | **E1** | **LADDER** | interaction |
| **D-oracle** | ground-truth obstacle velocity | frozen `u = 0` | **`IS_DIAGNOSTIC_ONLY`** |

**Why a 2 × 2 factorial is appropriate here**, where the eight-cell structure of §6.1 was
rejected: both factors act on the **same population of steps**, each has exactly two levels
already defined by prior stages, and the **interaction is the scientific question** — *does
recovery still add anything once estimation improves?* No cell is a new configuration invented
for Stage 5.

**Recovery is a fixed factor, not a comparator to be optimised.** A2 reproduces Stage 4's
LADDER **exactly as run** (`91bb969`): same rungs, same order, same admission logic, no
re-tuning. R1.5 is **not** included — it remains an unpromoted ablation (§11.9.3). Stage 5 must
never compare the estimator against a newly configured recovery arm.

**D-oracle is ledger-breaking and appears for exactly one purpose:** to size the remaining
headroom, i.e. the denominator of "what fraction of the achievable gap did E1 close". It is
never a headline arm, never a safety comparator, and never a target to tune toward.

### 11.9.10 Stage 5 — endpoints, kept separate

Seven families, each with its own comparator, unit of observation, and role. **Collision
improvement is explicitly NOT the estimator's success criterion.**

**(1) Velocity-estimation accuracy — PRIMARY for Q-EST.** Unit: step.
* `P(e > 0 | TTC < 1 s)` on the binding subset, `e = dh_dt_est − dh_dt_true`, `e > 0` optimistic.
* **Reference is the Stage-3 T1 in-environment value: 0.397 (fixed), 0.498 (randomized)** — i.e.
  arm A0 measured in this same run, which is the only valid comparator. **Phase 4's 0.142 is a
  different harness and a different subset (`h < 0.6 & binding`, standalone) and must not be
  compared against.** Both are reported side by side with the incomparability stated.
* Secondary: `p95`, `p99`, `max` of `e`; the pessimistic tail `P(e < −0.05)`; all split by range
  band and by point count (1 / 2 / ≥3), since E1's predicted effect is concentrated at 1–2 points.
* Also reported: mean `‖v̂ − v‖` — **reported, never a success criterion**, per §5.4.

**(2) Super-physical rate.** Unit: step. Fraction of steps with ≥1 row whose estimated closing
rate exceeds the configured physical bound (0.675 fixed / 0.75 randomized, metric only).
**Directional guard, not a primary endpoint** (§11.9.8).

**(3) M1 / singleton infeasibility.** Unit: step. Count and rate of cardinality-1 minimal
infeasible subsets. **Expected to be 0 in every arm**, because T1 already eliminates the class;
this is a **regression check**, not an effect to be found. A non-zero count in any arm means E1
broke something.

**(4) Total and M2 infeasibility.** Unit: step. Total infeasibility rate; the M2 /
multi-constraint share; MIS cardinality distribution. **This is where H-c is tested, and nothing
establishes that velocity work touches M2 at all.** Baseline: 0.0232 (fixed), 0.0283
(randomized) under T1 + `u = 0`.

**(5) DR-CBF feasibility and guarantee preservation — PRIMARY for Q-GUAR, and the primary
Stage-5 claim.** Unit: step.
* **T0 share** — steps solved with `m ≥ τ`, the DR guarantee intact.
* T1-tier and T2 shares; the achieved margin `m = minᵢ CBCᵢ` distribution; for recovering arms
  the rung distribution and the implied decay floor for T2.
* The decisive contrast: **does E1 convert infeasible-and-recovered-in-T2 steps into
  feasible-in-T0 steps, or does it merely move them around?** A0 vs A1 answers it without
  recovery in the way; A2 vs A3 answers it with recovery present.
* **A recovered action is never described as "safe"; `m` and its tier are printed beside it**
  (§11.8.6, unchanged).

**(6) True collision outcomes — SAFETY VETO and secondary.** Unit: episode. Collision rate split
dynamic / static / wall; collisions while feasible vs after an infeasible step; success;
timeout. Governed by §11.9.12.

**(7) Clearance.** Unit: episode. Mean minimum true clearance **and worst-case minimum clearance
across episodes**, both reported for every arm. Stage 4 showed these can move adversely while
collisions improve (−0.1451 for R1.5), so worst-case clearance is reported in the headline, never
a footnote.

**Cross-cutting, reported for every arm:** `‖u − u_nom‖`; SPL; solver status distribution; and —
**reported, not gating** — `Σ_v` calibration (NEES, NIS, empirical coverage of the `z_β`-σ
interval). Calibration exists at Stage 5 solely to license or forbid *later* Stage-6 work; **no
Stage-5 claim depends on `Σ_v` being calibrated**, because E1 applies no margin.

### 11.9.11 Stage 5 — paired experimental design

**Conditions.** Both, run separately and never pooled: **fixed** (`obstacle_speed = 1.0`) and
**randomized** (`obstacle_speed_range = U[0.6, 0.75]`), `n_dynamic_obstacles = 6`, `config.json`,
exactly as Stages 3 and 4.

**Episode count and seeds.**
* **Validation tier: 50 episodes per condition**, seeds `DEV_SEED_BASE + 1_000 … +1_049` — the
  same held-out block Stages 3 and 4 used. **All admission checks, all diagnostics and any
  sensitivity curve run here.**
* **Final tier: 200 episodes per condition per arm**, seeds `EVAL_SEED_BASE … EVAL_SEED_BASE +
  199`, matching Phases 3–6 and Stage 4 so the numbers are directly comparable.
* **`EVAL_SEED_BASE` is touched once, after admission passes.** No Stage-5 quantity — no
  constant, no threshold, no arm selection — is chosen using it. E1 has no free parameter to
  tune in any case (§11.9.8).

**Pairing.** Episode `k` uses the same seed in every arm and condition, so `env.reset(seed=...)`
gives identical initial states, obstacle placements and motion streams. Records are matched
**by seed**, and every test below is paired at the episode level.

**Exact comparisons, pre-registered.**

| # | comparison | question | status |
|---|---|---|---|
| C1 | **A1 vs A0** | estimator main effect, recovery absent | **PRIMARY** |
| C2 | A3 vs A2 | estimator effect with recovery present | secondary |
| C3 | A2 vs A0 | recovery replication (must reproduce Stage 4) | **replication check** |
| C4 | A3 vs A0 | combined | secondary |
| C5 | (A3−A2) − (A1−A0) | **interaction**: does recovery still add once estimation improves? | secondary |
| C6 | D-oracle vs A0 | headroom denominator | **diagnostic only** |

**Statistical tests.**
* **Binary episode-level endpoints** (collision, static collision, wall collision, success,
  timeout): **exact two-sided McNemar** on the discordant pairs, as implemented in
  `experiments/exp7_4_recovery.py::mcnemar`. Reported as `n01`, `n10`, `p`.
* **Continuous episode-level endpoints** (mean and worst-case clearance, SPL, `‖u − u_nom‖`):
  **paired percentile bootstrap, 10 000 resamples**, as in
  `experiments/exp7_3_velocity_intervention.py::boot`, resampling **episode pairs**.
* **Step-level rates** (optimistic tail, super-physical rate, infeasibility, tier shares):
  reported as proportions with an **episode-level cluster bootstrap** (10 000 resamples of
  episodes, not steps). Steps within an episode are **not** independent and a naive binomial
  interval would be anti-conservative; this is pre-registered rather than discovered later.
* **Interaction C5:** difference-in-differences on the paired episode-level indicators, tested by
  the same episode-level bootstrap. McNemar does not extend to a DiD statistic and is not used
  for it.

**Confidence intervals.** Every reported effect carries a **two-sided 95 % CI** on the *paired
difference* — including, mandatorily, every **non-significant** one, so that the magnitude of
harm not excluded is visible (§11.9.12).

**Significance level and multiplicity.** α = 0.05, two-sided. **C1 on the endpoint families (1)
and (5) is the single pre-registered primary comparison**; C2–C5 are secondary and are reported
with their CIs as descriptive, without inventing an α-spending scheme after the fact. **The
safety veto is deliberately NOT multiplicity-corrected** — correction there would make harm
harder to detect, which is the wrong direction.

**Minimum resolvable effect, stated up front.** At the observed collision rates (0.135 fixed /
0.280 randomized) and n = 200 paired episodes, exact McNemar resolves roughly **≥ 0.06–0.08
absolute**. Estimator effects plausibly smaller than that are **undetectable by design**, and a
null on collisions is therefore expected and uninformative. The step-level endpoints rest on
**10⁴–10⁵ steps** per arm-condition and resolve far smaller differences. **This asymmetry is the
entire reason the primary claim is guarantee-preservation and not collisions**, and it is
recorded here so that the null cannot later be read as evidence of safety or of no effect.

### 11.9.12 Stage 5 — safety veto, pre-registered

1. **Veto.** A **statistically significant paired increase in collision rate against A0** (for
   A1) or **against A2** (for A3), in **either** condition, at α = 0.05 two-sided uncorrected,
   **rejects the estimator intervention outright** — regardless of any gain on any primary
   endpoint. There is no offsetting argument.
2. **Non-significance is never proof of safety.** A non-significant result is reported verbatim
   as **"failed to detect harm at n = 200; the 95 % CI does not exclude an increase of up to
   X"**, with X stated numerically. The phrases "safe", "no harm", and "safety preserved" are not
   used for a non-significant result anywhere in the Stage-5 report.
3. **Mandatory headline reporting even when non-significant**, with these pre-registered flag
   thresholds:
   * worst-case minimum clearance regressing by **> 0.02 m** against A0 in either condition;
   * **any** new static or wall collision in an arm where the comparator had none;
   * the M1/singleton count being **non-zero** in any arm (§11.9.10-3).
   Each triggers an explicit headline statement, not a footnote. None of them is itself a veto —
   they are disclosure obligations, so that a favourable primary result cannot bury them.
4. **A recovered or clamped action is never called "safe" because it exists.** Achieved margin
   `m` and guarantee tier accompany every such report (standing rule, §11.8.6).

### 11.9.13 Stage 5 — admission, acceptance, escalation and termination gates

**A. Admission — before any final-tier run.** All must pass, in order:

* **A1 Inertness.** With E1 disabled the pipeline is **bit-identical** to A0 over a fixed trace —
  hard test, not statistical.
* **A2 Correctness.** E1's `R` is anisotropic in the derived direction (tangential > radial) for
  a one-point return on a hand-computed numeric case; reduces to the frozen isotropic form in the
  limit of many well-spread points; is symmetric positive-definite everywhere.
* **A3 Leakage.** All four frozen layers re-applied to the new class: signature (no obstacle
  argument), AST import scan, runtime tripwire, and bit-identical action under corruption of the
  ground-truth observation slice `obs[28:52]`. **Non-negotiable.**
* **A4 Validation-tier directional check** (50 held-out episodes per condition): E1 must show
  **both** (i) a lower `P(e > 0 | TTC < 1 s)` than A0 **and** (ii) a **non-increase** in the
  super-physical rate — jointly, in **both** conditions. The conjunction is what makes the
  sluggish-filter failure mode of §11.9.8 non-passing.
* **A5 Regression.** Frozen sha256 manifest PASS; full test suite PASS.

Failing any of A1–A3 or A5 is an **implementation defect**: fix and re-run admission. Failing
**A4** is a **scientific result** — see D2.

**B. Acceptance of E1 (after the final tier).** E1 is **ACCEPTED** iff all hold:
1. No safety veto fired (§11.9.12-1);
2. `P(e > 0 | TTC < 1 s)` is lower than A0 in **both** conditions, with the 95 % CI excluding
   zero in at least one;
3. The super-physical rate does not increase in either condition;
4. M1 / singleton count remains 0 in every arm;
5. **Either** the total infeasibility rate falls, **or** the **T0 share rises**, in at least one
   condition with the CI excluding zero. *(This is the Q-GUAR criterion; either route counts,
   because fewer infeasible steps and more guaranteed steps are both wins for the guarantee.)*

**Acceptance does NOT require a collision improvement.** A collision improvement is welcome, is
reported, and is **not** required; its absence does not reject E1.

**C. Rejection of E1.** Any of: a safety veto; the optimistic tail worsening in either condition
with the CI excluding zero; the super-physical rate rising; a non-zero M1 count.

**D. Termination of the estimator branch.** The branch **terminates — it does not escalate — if**:
* **D1.** E1 is REJECTED under C. A more complex estimator is not the remedy for an intervention
  that made the measurement worse.
* **D2.** E1 fails **A4** at the validation tier. The final tier is **not run**, `EVAL_SEED_BASE`
  is not touched, and **the null is the reported Stage-5 result**.
* **D3.** E1 is accepted on Q-EST (accuracy improves) but shows **no** movement in infeasibility
  or T0 share in either condition. This falsifies **H-c/H-d**: better velocity estimates do not
  reach the M2 population. That is a **substantive finding** — the residual infeasibility is
  genuinely geometric and **recovery is the only lever that acts on it** — and it is reported as
  such, not treated as a reason to try a bigger estimator.

In all three cases: **LADDER remains the established empirical fallback baseline**, Stage 4's
conclusions stand unchanged, and no further estimator work is authorised without a new approval.

**E. Escalation to a richer estimator (IMM / CT) — all three required.**
1. E1 is **ACCEPTED** under B, and significantly improves `P(e > 0 | TTC < 1 s)`, yet closes
   **< 50 %** of the A0 → D-oracle gap on that endpoint.
2. The residual is attributable to **motion-model** mismatch, **measured not assumed**: NIS fails
   its χ² band on track segments whose estimated heading change over the confirmation window
   exceeds a threshold fixed a priori at **15° per 0.5 s**, while **passing** on straight
   segments. A uniform NIS failure across both indicates the *measurement* model — E1's own
   object — and is **not** evidence for CT.
3. Headroom remains: the D-oracle collision advantage over the best non-privileged arm is
   **≥ 0.05 absolute in the randomized condition**.

Any one failing forbids escalation. **A collision improvement under E1 is not, by itself, grounds
to escalate**, and neither is the D-oracle result alone.

**F. Escalation to uncertainty / covariance work (Stage 6, U2 `z_β` margins) — all three
required.**
1. E1 is **ACCEPTED** under B;
2. **`Σ_v` calibration passes on validation seeds**: NEES and NIS inside their χ² bands and
   empirical coverage of the `z_β`-σ interval within **±10 %** of nominal. If calibration fails,
   U2 may still be *proposed*, but only as **"a heuristic tightening of unknown confidence
   level"** — the phrase "chance-constrained" is not used for it, per §5.4;
3. A residual optimistic tail remains that a per-sample margin could plausibly address, i.e.
   `P(e > 0 | TTC < 1 s)` under E1 is still **> 0.25** in either condition.

**U3 (enlarging `r_W`) remains REJECTED as primary** for the three reasons in §5.3, and no gate
here revives it.

### 11.9.14 Stage 5 — decision tree

```
              ┌─ A1/A2/A3/A5 fail ──► implementation defect: fix, re-run admission
ADMISSION ────┤
              └─ A4 fails ──────────► D2: TERMINATE. Final tier NOT run. Report the null.
                                       LADDER remains the fallback baseline.
   │ A4 passes
   ▼
FINAL TIER (200 paired episodes x 2 conditions x 4 arms + D-oracle diagnostic)
   │
   ├─ safety veto fires ────────────► C: REJECT E1. TERMINATE (D1). Report the harm.
   │
   ├─ optimistic tail worse, or super-physical up, or M1 non-zero
   │                        ────────► C: REJECT E1. TERMINATE (D1).
   │
   ├─ accuracy improves, infeasibility and T0 share both unmoved
   │                        ────────► D3: TERMINATE with a POSITIVE finding —
   │                                   the residual M2 is geometric, not estimator-induced;
   │                                   recovery is the only lever that acts on it.
   │
   └─ E1 ACCEPTED under B
          │
          ├─ E-1,2,3 all hold ──────► propose Stage 6a: IMM{CV,CT}. NEW APPROVAL REQUIRED.
          ├─ F-1,2,3 all hold ──────► propose Stage 6b: U2 z_beta margin. NEW APPROVAL REQUIRED.
          └─ neither ───────────────► STOP. E1 is the accepted estimator. Report and close
                                       the estimator branch. No further escalation.
```

**A negative or null estimator result is a valid scientific outcome and terminates the branch.
It does not automatically trigger a more complicated estimator.** Escalation requires the
affirmative, pre-registered evidence in E or F — never the mere absence of success.

### 11.9.15 Stage 5 — no post-hoc optimisation

* **No parameter sweep is authorised.** E1 introduces **no free parameter** (§11.9.8); there is
  nothing to sweep. If implementation reveals that a constant is in fact needed, it must be
  **derived and recorded in this plan before the run**, or the intervention is redesigned — it is
  **not** selected empirically.
* **No arm may be added, dropped, split or re-ordered after data exist.** The five arms of
  §11.9.9 are the complete set.
* **`v_cap = 0.96`, `α`, `r_W`, `ε`, `τ` are not touched.** The pre-registered `v_cap`
  sensitivity arms remain **not run** and are not revived by Stage 5.
* **Recovery is not re-tuned.** A2/A3 use Stage 4's LADDER exactly as run.
* **No threshold in §11.9.12 or §11.9.13 may be amended after seeing the final tier.** Any
  amendment is recorded here **before** the rerun it justifies, with the original result
  preserved — the standing rule since §11.4.
* **Every deviation is classified** `[adaptation]` / `[reference implementation correction]` /
  `[extension]` with provenance, as throughout Phase 7. **E1 is an `[extension]`.**
* Results are written **per condition**, never overwritten, under
  `results/week5_phase7/stage5_estimator/`, labelled **`Week5-Phase7 / Stage 5 / <name>`**.

### 11.9.16 Stage 5 — pre-registered predictions

* **Q-EST.** E1 reduces `P(e > 0 | TTC < 1 s)` against A0, concentrated in the 1–2-point
  returns. Direction predicted; magnitude not predicted.
* **Q-GUAR.** E1 closes only a **minority** of the A0 → D-oracle accuracy gap: a geometry-correct
  covariance improves conditioning, not the CV model's structural mismatch with an OU-steered
  turn.
* **M1.** Stays 0 in every arm — T1 already removed the class. Regression check only.
* **M2.** **Genuinely uncertain, and this is the most informative cell of the design.** Stage 3
  established that velocity work removes M1; **nothing establishes it touches M2**. A null here
  (D3) is a real finding, not a failure.
* **T0.** E1 is expected to retain a **non-zero T0 share on steps it rescues** — the one thing no
  recovery rung achieved. If it does not, Direction A has no advantage over Direction B on its
  own terms, and that is the finding.
* **Collisions.** The estimator main effect is **not expected to reach significance at n = 200**
  (§11.9.11). The recovery main effect (C3) is expected to reproduce Stage 4.
* **Interaction (C5).** Expected **sub-additive**: if E1 removes infeasible steps LADDER would
  otherwise have recovered, the two improvements partly overlap. A significant interaction in
  either direction is reportable, not a problem.

### 11.9.17 Stage 5 — files, when approved

```
dr_control/tracking2.py                 E1 only: geometry-derived reconstruction covariance,
                                        as a SUBCLASS of LidarVelocityTracker (D3b of §5.2).
                                        NOT IMM, NOT CT -- those stay unwritten.
experiments/exp7_5_estimator.py         2x2 factorial + D-oracle diagnostic, paired, per condition
tests/test_dr_control_phase7_stage5.py  inertness, anisotropy, isotropic limit, SPD, 4 leakage layers
MD_files/week5/phase7/STAGE5_REPORT.md
results/week5_phase7/stage5_estimator/  written per condition, never overwritten
```

`dr_control/uncertainty.py` stays **reserved and unwritten** — it is Stage 6 (`Σ_v` margins) and
is gated by §11.9.13-F. `dr_control/velocity_tracker.py` is **extended by subclass, never
edited**; `dr_control/policy_phase7.py` gains an arm name, no behavioural change. Nothing under
`robot_env/`, `evaluation/`, or any Phase 1–6 or Stage 0–4 artefact is modified. V0 stays frozen.

### 11.9.18 PRE-IMPLEMENTATION CORRECTION to §11.9.8 (recorded 2026-09-08, BEFORE any code or run)

Working the `n = 1` derivation through before implementing it showed the formula written in
§11.9.8 to be **wrong**. It is corrected here rather than silently in place, per §11.9.15. The
error was found by derivation, not by data: **no Stage-5 code existed and no experiment had been
run when this was written.** No threshold, gate, endpoint, arm or prediction changes.

**What §11.9.8 said (WRONG).** "the tangential displacement is modelled as bounded on
`[−r·sin Δ, +r·sin Δ]` with `Δ = 2π/24` the ray spacing, giving `σ_t² = r²sin²Δ/3`; the radial
variance stays `σ_range²`."

**Why it is wrong.** Two independent errors.

1. The ray-spacing bound constrains *which* discs give a one-point return (a disc subtending less
   than the ray spacing), **not where on the disc the surviving ray lands**. Given that a ray
   does hit, its offset from the centre bearing is bounded by the disc's own half-width, so the
   tangential offset is bounded by `r`, not by `r·sin Δ`. The old formula understated `σ_t` by a
   factor of `1/sin Δ ≈ 3.9`.
2. "Radial error near zero" holds only for a head-on hit. It grows to `r` at the limb.

**The exact geometry.** Let the ray hit the disc at perpendicular offset `s` from the centre,
`|s| ≤ r`. The reconstruction places the centre **on the ray line** at range `ρ + r`, while the
true centre is off the ray line by `s` and its projection on the ray is `ρ + √(r² − s²)`.
Therefore, **exactly**:

```
e_tangential = −s                     (perpendicular to the ray)
e_radial     = r − √(r² − s²)         (along the ray, always ≥ 0: the estimate sits BEYOND the true centre)
```

With `s ~ U(−r, r)` — the only distribution available for a single return, since nothing
observed says where on the disc the ray landed — the **mean-square** errors are, in closed form:

```
MSE_t = r²/3                      = 0.030000   (r = 0.3)   ->  σ_t = 0.1732
MSE_r = r²·(2 − π/2 − 1/3)        = 0.008628   (r = 0.3)   ->  σ_r = 0.0929
```

Verified against 2×10⁶ Monte-Carlo draws: 0.030007 and 0.008633. The radial term is a **mean
square, not a variance** — `e_radial` has mean `r(1 − π/4) ≈ 0.215 r`, a genuine bias that a
Kalman `R` cannot represent, so the bias is folded in conservatively as `MSE = Var + bias²`.
That is the standard conservative treatment and introduces **no parameter**.

The range-measurement variance adds to the **radial** axis only, since range noise displaces the
point along the ray:

```
R_1pt = (MSE_r + σ_range(d)²)·û ûᵀ  +  MSE_t·t̂ t̂ᵀ ,      σ_range(d) = SIGMA_R_BASE·(1 + d/3)
```

**Consequences, stated plainly.**

* The corrected covariance is **anisotropic with the large axis tangential**, ratio
  `σ_t/σ_r = 1.86` (3.5× in variance) — the direction §11.9.8 predicted, which is what admission
  check A2 tests.
* It is **more conservative than the frozen isotropic value**, not less: frozen 1-point `s` is
  0.0800 at `d = 1` m and 0.1000 at `d = 2` m, against `σ_t = 0.1732`. The *earlier, wrong*
  formula would have given `σ_t = 0.0448`, i.e. it would have made the filter **trust a
  one-point reconstruction more** than the frozen code does — the opposite of the intent, and it
  would plausibly have **increased** super-physical rates. Recording this because it is the kind
  of error the pre-registration exists to catch.
* Still **zero new tuned parameters**: only `r_nominal`, the existing `σ_range` and its retained
  range-growth factor appear. `POINT_COUNT_FACTOR` is still removed.

**`n = 2` and `n ≥ 3` are unchanged in substance** and are stated precisely for implementation:

* **n = 2:** first-order propagation of the two range measurements through the **exact frozen
  two-root map** `c = m ± b·n̂`, `b = √(r² − a²)`: `R = J Σ_ρ Jᵀ`, `Σ_ρ = diag(σ_range(ρ₁)²,
  σ_range(ρ₂)²)`, `J = ∂c/∂(ρ₁, ρ₂)` evaluated by central differences **on that same map** so
  the covariance describes the reconstruction actually used. This reproduces the `a → r`
  blow-up along `n̂` that the frozen code merely flags. If the chord is degenerate (`a ≥ r`) the
  map is not differentiable there and the **1-point form is used instead**, which is the
  conservative choice.
* **n ≥ 3:** first-order propagation through the Gauss-Newton fixed-radius fit.
  `nᵢ = (c − qᵢ)/‖c − qᵢ‖`, residual sensitivity to range `∂fᵢ/∂ρᵢ = −nᵢ·ûᵢ`, giving
  `R = (JᵀJ)⁻¹ Jᵀ Σ_f J (JᵀJ)⁻¹` with `Σ_f = diag(σ_range(ρᵢ)²(nᵢ·ûᵢ)²)` — the correct
  first-order form rather than the cruder `σ²(JᵀJ)⁻¹`.

**One uniform numerical rule, derived not tuned:** the eigenvalues of `R` are floored at
`min_i σ_range(ρᵢ)²`. A reconstruction can never be more certain than the range measurement it
is built from. This also keeps `R` symmetric positive-definite when a fit is ill-conditioned.

**Scope limit, deliberate.** `Track.__init__`'s initial covariance `P` keeps its frozen
`SIGMA_R_BASE²` position block. Replacing it with `R` would be a **second** change, and §11.9.7
allows exactly one. Recorded as a known asymmetry, not an oversight.

**Implementation note.** The frozen `Track.measurement_noise(n_points, range_to_sensor)`
signature cannot carry the ray geometry `R` depends on, so
`GeometricLidarVelocityTracker` overrides `update()` with the frozen algorithm and **one**
substitution — the source of `R`. Admission check A1 is what guarantees no drift: with
`use_geometric_R=False` the subclass must be **bit-identical** to the frozen tracker over a real
scan trace.

**Stage 5 is NOT implemented and NOT run under this amendment. It awaits explicit approval.**

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
| A4 | **Amendment 4 (2026-09-07)**: M1/M2 mechanisms separated; stages reordered to velocity intervention → residual recovery → advanced estimator; Stage 3 designed with `v_cap = 0.96` derived and a safety veto | §11.7 |
| A5 | **Amendment 5 (2026-09-07)**: Stage 3 outcome recorded; T1 accepted and T2 rejected; singleton→fallback→collision conjecture weakened; estimator work gated on Stage 4; Stage 4 designed against M2 only | §11.8 |
| A6 | **Amendment 6 (2026-09-08)**: Stage 4 outcome recorded — P1/P2/P4 confirmed, P3 CONTRADICTED for R2/LADDER/R1.5; the `u = 0` fallback established as a collision pathway while the singleton claim stays weakened; recovered actions never called safe; Stage 5 designed as a 2×2 estimator × fallback factorial (A0–A3 + D-oracle diagnostic) around **one** minimal, parameter-free, geometry-derived estimator change (E1 = D3b reconstruction covariance); primary claim is guarantee preservation (T0), not collisions; safety veto, admission/acceptance/escalation/termination gates and a stopping decision tree pre-registered; IMM/CT and `Σ_v` margins remain gated | §11.9 |

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
