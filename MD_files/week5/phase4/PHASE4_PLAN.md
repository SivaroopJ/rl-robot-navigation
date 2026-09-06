# Phase 4 plan (revised) — replace the oracle velocity channel with a LiDAR-only estimator

**Status: plan for review, revision 2. No code until approved.**

Scope: **velocity estimation only.** The original plan bundled this with sensor noise and the
`r_W` ablation; those are deferred to a later phase (Phase 4b) and are out of scope here.

---

## 0. Fixed constraints (requirement 1)

Unchanged from Phase 3, verbatim: `alpha = 0.4`, `r_W = 0.004`, `eps = 0.1`, the `u = 0`
infeasibility fallback, the CBF formulation, SCS with the problem rebuilt per call, `n_keep = 5`,
`k_scans = 5`, and obstacle **positions** from the 24-ray LiDAR.

**The only control-path change in Phase 4 is the source of `dh_dt`: oracle velocity becomes
estimator output.** No file that Phases 1–3 import is edited. `results/phase{1,2,3}/*.json`
keep their checksums, and the full test suite (123 tests) must still pass.

---

## 1. What the sensor gives the estimator (measured, drives every choice below)

| range d | subtend | returns/obstacle (measured) |
|---|---|---|
| 0.6 m | 60.0° | 4.07 |
| 1.0 m | 34.9° | 2.20 |
| 1.5 m | 23.1° | 1.40 |
| 2.0 m | 17.3° | 1.00 |
| 2.3 m | 15.0° | 1.00 |
| 3.0 m | 11.5° | 0.60 |

Phase 2 detection miss rate: **0% inside 2 m**, 32.3% at 2–3 m, 58.3% at 3–4 m, 84.5% beyond 4 m.
Obstacle displacement 0.060–0.075 m/step; robot up to 0.1 (0.141 diagonal); `dt = 0.1`.

Consequences: clusters are 1–4 points where the barrier binds, so free-radius circle fitting is
out; raw differencing of a 0.02 m centre error yields ~0.28 m/s of velocity noise (4x the true
speed), so cross-frame smoothing is mandatory; and because misses are far-field only, the
dominant risk is **track initialisation latency**, not steady-state accuracy.

---

## 2. Estimator design (`dr_control/velocity_tracker.py`)

### 2.1 Segmentation
Adjacent-ray clustering with wrap-around at ray 23→0; rays join when
`|r_k - r_{k+1}| < C * min(r_k, r_{k+1}) * tan(15°) + eps_seg`. Range-max returns break segments.
Points are lifted to the world frame using the known ego pose.

### 2.2 Static rejection — ordered by discriminative power (requirement 3)

**Compactness is the PRIMARY test, not the line fit.** A disc spans at most `2*r_nominal = 0.6 m`;
a wall or rectangle face spans metres. So a segment is a dynamic candidate only if its maximum
pairwise point distance is `<= 2*r_nominal + margin`. This is a strong, well-conditioned test.

The line/TLS residual test is **secondary**, and its margin is thin enough to state numerically:
a disc of radius 0.3 seen over one 15° ray step deviates from its chord by only
`r*(1 - cos 7.5°) = 0.0026 m`, and over two steps by `r*(1 - cos 15°) = 0.0102 m`. A flat wall
has a residual near float32 zero. So `line_tol` must sit well below 0.01 m or **a real disc at
0.8–1.0 m will be classified static and frozen at `v = 0`** — precisely the failure you flagged.
This is why compactness leads and the line test only supplements it for long segments.

**This is an explicit failure analysis, not an assumption.** Reported:
- **static false-positive rate** — static structure given non-zero velocity;
- **dynamic false-static rate** — a real moving obstacle classified static (broken out by range,
  since the 0.8–1.0 m band is the predicted danger zone);
- the resulting impact on projected velocity error and on collisions, isolated by the
  `no_static_rejection` ablation (§6).

### 2.3 Centre reconstruction (requirements 5, 13)
Track the **centre**, not the visible centroid: the visible patch slides as relative geometry
changes, the centre does not.

- **≥2 points:** fixed-radius least-squares fit, minimising `sum_i (||q_i - c|| - r_nominal)^2`.
- **1 point:** fallback `c = q + r_nominal * u_ray`, exact only for a head-on hit, with error up
  to `r_nominal * sin(theta_incidence)`.

`r_nominal` is a **fixed configured scalar** `[adaptation]`. The production estimator receives it
from config and nothing else — it never infers or selects a radius from obstacle identity, type,
or simulator state (requirement 13). It is a global design constant, of the same kind as the
controller's existing `r_robot = 0.3`.

**Reported separately, never pooled** (requirement 5): centre error and projected velocity error
broken out by **1-point / 2-point / ≥3-point** reconstruction, and **as a function of range**.

### 2.4 Tracking and velocity
Constant-velocity Kalman filter per track, state `[px, py, vx, vy]`, measurement = reconstructed
centre. White-acceleration process noise tuned to smooth hard (true inter-bounce motion is
constant-velocity), accepting bounce lag. Measurement noise `R` scaled by range **and by point
count**, so a 1-point reconstruction is correctly distrusted. Velocity is read from the filtered
state, never from raw frame differencing.

### 2.5 Association and identity (requirement 2, accepted)
*Measurement-to-track* association against the estimator's own hypotheses — never
point-to-ground-truth-obstacle. Mahalanobis gating (chi2 0.99), then global nearest-neighbour
assignment via `scipy.optimize.linear_sum_assignment`, which avoids the greedy identity swaps
that occur when two obstacles pass close. Track management: 3-of-5 confirmation before a
velocity is trusted; coast on prediction for up to 5 frames (0.5 s) through occlusion or
far-field dropout, with inflated covariance; delete after that.

Unconfirmed tracks contribute `dh_dt = 0` by default (matching the reference's implicit
behaviour). A `--unconfirmed-conservative` flag substitutes `-v_assumed`; both are measured,
neither is adopted without approval.

### 2.6 Tracks to `dh_dt`
Each buffered barrier sample's critical surface point is matched to a confirmed track by
proximity to that track's predicted surface. Match → `dh_dt = -grad_h . v_hat`; otherwise, or if
the segment was classified static, `dh_dt = 0`.

---

## 3. Interface

```python
# dr_control/velocity_tracker.py
class LidarVelocityTracker:
    def __init__(self, *, r_nominal=0.3, dt=0.1, k_confirm=3, n_confirm_window=5,
                 n_coast=5, sigma_a=..., line_tol=..., compact_margin=..., ...): ...
    def reset(self) -> None: ...
    def update(self, p_ego, ranges) -> list[Track]:      # ONLY these two inputs, ever
    def velocity_at(self, q_world) -> tuple[np.ndarray, bool]

# dr_control/estimated_cbf.py
class EstimatedLidarBarrierSource(LidarBarrierSource):
    def push(self, p, ranges) -> None
    def samples(self, p) -> (h, grad_h, dh_dt)
```

No parameter exists through which obstacle positions, velocities, identities, or per-object radii
could enter. This is a signature-level guarantee, not a convention.

---

## 4. Sign convention — verified before the experiment runs (requirement 8)

Robot at origin, obstacle at `+x`, so `grad_h = (-1, 0)` (points from obstacle toward robot).
`dh_dt_true = -grad_h . v_true`; obstacle closing at speed `s` gives `v = (-s, 0)` and
`dh_dt = -s < 0`. Define **`e = dh_dt_est - dh_dt_true`**.

| case | `dh_dt_true` | `dh_dt_est` | `e` | controller believes | verdict |
|---|---|---|---|---|---|
| approaching, speed under-estimated | `-s` | `-ŝ`, `ŝ<s` | **> 0** | closing more slowly than reality | **optimistic / DANGEROUS** |
| approaching, speed over-estimated | `-s` | `-ŝ`, `ŝ>s` | < 0 | closing faster than reality | conservative / safe |
| receding, speed under-estimated | `+s` | `+ŝ`, `ŝ<s` | < 0 | receding more slowly | conservative / safe |
| receding, speed over-estimated | `+s` | `+ŝ`, `ŝ>s` | **> 0** | receding faster than reality | **optimistic / DANGEROUS** |

So **`e > 0` ⟺ the controller believes the obstacle is safer (closing less / receding more) than
reality.** All four rows become a hand-constructed unit test that must pass **before** the full
experiment is launched.

---

## 5. Metrics

### 5.1 Projected error is primary (requirement 7)
`e = dh_dt_est - dh_dt_true = -grad_h . (v_hat - v_true)`. Only the gradient-projected component
enters the constraint; reporting `||v_hat - v_true||` alone would overstate the damage, since
tangential error is harmless.

Reported as a **signed distribution with quantiles — median, P90, P95, P99, min, max** — so a
small dangerous tail cannot hide behind a good mean. Raw `||v_hat - v_true||` is reported too
(mean, median, P90, P99, max), as a secondary quantity.

**Headline safety metric: `P(e > 0 | h < 0.6 and CBF binding)`**, plus the P90/P95/P99 of `e`
within that same cell. A tracker with an excellent mean and a fat optimistic tail there is a
worse result than a noisier unbiased one.

### 5.2 Dangerous timing (requirement 8)
`e` binned by: true clearance `h ∈ [0,0.3), [0.3,0.6), [0.6,1.0), [1.0,2.0), ≥2.0`; whether the
CBF was binding (`min CBC` within tolerance of `tau = r_W/eps`); whether the sample was the
critical one; and time-to-collision `h / max(-dh_dt_true, eps)` bucketed, since error at TTC < 1 s
is what kills.

### 5.3 Confirmation latency, measured directly (requirement 6)
Per track: **frames from first detection to confirmation**; **distance the obstacle travelled
during that window**; **whether the CBF was already binding before confirmation**; and the
**collision rate during unconfirmed periods**. This answers whether 3-of-5 is too slow for the
actual hazard geometry — the predicted mode where the controller silently degenerates to the
`zero` arm exactly as an obstacle arrives.

### 5.4 Track-quality record (requirement 11)
For **every dynamic-obstacle interaction step**, one row: track age, number of observations
currently contributing, confirmation state, consecutive missed frames, estimated speed, true
speed, projected error `e`, and whether the CBF was binding. This is what makes it possible to
explain *why* `estimated` differs from `oracle` rather than merely reporting that it does.

Plus: tracks confirmed / lost / coasting per episode, identity-swap count, static false-positive
and dynamic false-static rates (§2.2), centre error by point count and range (§2.3).

### 5.5 Behavioural comparison (requirement 6)
Success rate; **dynamic / static / wall collisions separately**; timeout rate; min true clearance
(mean and worst episode); `min CBC | solved`; `frac_infeasible`; `n_infeasible`;
`mean ||u - u_nom||`; `n_dcp_error`; `frac_h_negative`; solver time.

Phase 3's nearest-neighbour figures (9.8% velocity-relevant misassociation, 0.066 m/s mean,
0.50 m/s max) are the **error budget to beat**, not a candidate estimator.

---

## 6. Arms, pairing, and seeds (requirements 9, 10, 14)

**Strict pairing.** For seeds 0–149 the three main arms see **identical initial conditions and
identical obstacle trajectories**. Obstacles are unaffected by the robot, so this holds by
construction for constant-velocity motion; for OU motion the obstacle noise stream uses a
**dedicated per-episode RNG advanced once per step**, independent of the robot's actions, so the
trajectory is identical across arms even though the robot's path differs.

| arm | `dh_dt` source | role |
|---|---|---|
| `oracle` | ground-truth velocity + identity | upper bound (retained) |
| `estimated` | LiDAR-only tracker | the Phase 4 result |
| `zero` | 0 | lower bound (retained) |

**150 paired episodes** per main arm (`DEV_SEED_BASE + 0..149`); Phase 3 used 0..49 so those
remain directly comparable. `EVAL_SEED_BASE = 1_000_000` stays untouched.

**Both motion models, with the estimator configuration frozen** (requirement 9): (A) Phase 1–3
constant-velocity + elastic bounce, (B) the env's OU `SmoothStochasticMotion`. **The KF is not
retuned between them** — the OU result is the model-mismatch measurement and retuning would
destroy it.

**Ablations** (50 episodes each): `r_nominal ∈ {0.15, 0.30, 0.45}`; radius-free closest-point
variant; and — computation permitting — `no_static_rejection`, `no_confirmation_no_coasting`,
and `greedy_association` (replacing Hungarian). Each isolates whether one architectural
component actually buys anything.

**Reporting (requirement 10):** for `estimated` vs `oracle` and `estimated` vs `zero`, both an
**aggregate confidence interval** on the collision-rate difference and an **exact paired
(McNemar) test**. If the `estimated`–`oracle` gap is unresolved, it will be reported as
unresolved. **"Not statistically significant" will not be written as "equivalent to oracle."**

**`r_nominal` is never tuned on the evaluation set** (requirement 4): the production value stays
0.30 and the 0.15/0.45 results are reported separately as sensitivity, not used to select it.

---

## 7. Expected failure modes

1. **Wall-bounce lag** — instantaneous reflection vs a smoothed CV filter gives **wrong-sign**
   `dh_dt` for several frames. Conditioned on frames-since-bounce (ground truth, metrics only).
2. **Confirmation latency** — track born as the obstacle enters 2 m, 3 frames (~0.2 m of closing)
   to confirm, `dh_dt = 0` throughout. Measured directly per §5.3.
3. **Identity swap** when obstacles pass close.
4. **False dynamic on rectangle corners** (short, curved segments), injecting phantom `dh_dt`.
5. **Dynamic frozen as static** in the 0.8–1.0 m band, per the 0.0102 m sagitta margin in §2.2.
6. **1-point reconstruction bias** at 1.5–2.3 m.
7. **CV mis-specification under OU steering.**

---

## 8. Files

**Created:** `dr_control/velocity_tracker.py`, `dr_control/estimated_cbf.py`,
`tests/test_dr_control_phase4.py`, `experiments/exp4_dynamic_estimated.py`, `results/phase4/`,
this document.

**Modified: none.** No new dependencies (`scipy`, `cvxpy` already installed).

---

## 9. Leakage safeguards (requirement 12)

1. **Signature-level** — `update(p_ego, ranges)` and `push(p, ranges)` take no obstacle argument.
2. **Import-level** — a test parses the two new modules' imports and fails if `ray_owners`,
   `associate_nearest`, `dynamic_cbf`, or `cbf_sources` appear.
3. **Runtime tripwire** — monkeypatch `dr_control.dynamic_cbf.ray_owners` and
   `AnalyticBarrierSource.samples` to raise, then run a full `estimated` episode to completion.
4. **Perturbation, strengthened** — after scan generation, corrupt the simulator's ground-truth
   velocity array and assert that **both the action sequence and the estimator's own outputs
   (tracks, velocities, `dh_dt`) are bit-identical**, not merely close.
5. **Indirect-exposure audit** — verify no environment helper reachable from the control path
   returns obstacle state: `robot_env.lidar_core.cast_rays` returns ranges only; the harness's
   `scan()`, `collision_type()`, and `AnalyticBarrierSource` are confined to the metrics path.
   A test asserts the estimator's inputs per step are exactly `(p_ego, ranges)`.

Ground truth may influence the scan — it must — but nothing else.

---

## 10. Gate (requirement 15)

**Phase 4 passes if:**
- leakage safeguards pass;
- `estimated` significantly outperforms `zero` on paired evaluation;
- the optimistic projected-velocity tail is quantified;
- the `estimated`-vs-`oracle` gap is quantified;
- the dominant causes of that gap are identified.

**Matching oracle performance is NOT a requirement.**
