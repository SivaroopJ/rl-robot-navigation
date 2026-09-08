# Week5-Phase7 / Stage 1 — Direction 2: computational acceleration

**Gate G1 as pre-registered: FAIL on 4 of 6 criteria.** Stage 1 is NOT declared passed.
The measurements below show why every one of those four criteria was mis-specified — each
treats the frozen controller as ground truth, and it is not — but changing a gate after seeing
its result is the user's call, not mine. §7 proposes revised criteria for approval.

Nothing frozen was modified. `dr_control/drccp_controller.py` is untouched and is still the
arbiter for the delegated branch.

---

## 1. Pre-registered gate results, stated first

| criterion | requirement | measured | verdict |
|---|---|---|---|
| L1 exact status agreement, enriched infeasible corpus | 100 % | 0.99786 – 0.99872 | **FAIL** |
| L2 action vs frozen (E1) | ≤ 1e−6 | 1.7e−3 – 2.7e−3 | **FAIL** |
| L2 action vs arbiter (E1′) | ≤ 1e−8 | 4.9e−8 – 6.2e−8 | **FAIL** |
| L3 objective agreement | ≤ 1e−8 | 2.7e−4 – 5.2e−4 | **FAIL** |
| L4 guarantee-tier changes | 0 | **0** | PASS |
| L5 identical outcomes, 200 paired seeds (E2) | 1.000 | 0.990 both conditions | **FAIL** |

Runtime (L6) is reported in §5 **after** L1–L5, as the protocol requires.

---

## 2. L0 — the finding that reframes every other row

The plan pre-registered E1 as `‖u − u_ref‖∞ ≤ 1e−6`, implicitly treating the frozen
controller's output as ground truth. **It is not.** SCS is a first-order solver.

Measured against an ARBITER that belongs to neither implementation — the *frozen formulation,
transcribed unchanged*, solved by CLARABEL at tol 1e−14:

| | max ‖u − u_arbiter‖∞ |
|---|---|
| **frozen controller (V0)** | **2.69e−3** |
| reduced QP + OSQP (V3) | **6.2e−8** |
| parameterized + CLARABEL (V2) | 3.6e−6 |
| parameterized + SCS (V1) | 2.7e−3 |

The accelerated V3 is **~4.5 orders of magnitude nearer the true optimum than the reference it
is being asked to match to 1e−6**. And `L2(V3 vs frozen) = 2.69e−3` equals `L0(frozen vs
arbiter) = 2.69e−3` to three digits: *the entire discrepancy between V3 and the frozen
controller is the frozen controller's own solver error.*

V1 corroborates this from the other side — it uses the same solver as the reference, sits
close to it (5.5e−4), and is equally far from truth (2.7e−3). A threshold of 1e−6 against a
reference carrying 2.7e−3 of noise cannot be met by any implementation, including a perfect one.

The same reasoning applies to L3: the objective differences (2.7e−4 – 5.2e−4) are the frozen
controller's objective evaluated at its own slightly-suboptimal action.

E1′ also fails, but by a factor of ~6 against a threshold I picked *a priori* without data
(1e−8 vs 6.2e−8 observed). That is a badly chosen number, not a defect in V3.

---

## 3. L1 — status disagreements, and what they actually do

3–5 steps out of 2 337 in the enriched infeasible corpus are labelled differently. Every one
was examined individually (`experiments/exp7_1_disagreement.py`), against three predicates:

| predicate | reduced_osqp | reduced_osqp_ws | param_scs | param_clarabel |
|---|---|---|---|---|
| **P1** exact status string (pre-registered) | 0.99786 | 0.99786 | 0.99829 | 0.99872 |
| **P2** coarse `"infeasible" in status` — *what `policy.py` counts* | 0.99957 | 0.99957 | 0.99914 | **1.00000** |
| **P3** the ACTION | **1.00000** | **1.00000** | **1.00000** | **1.00000** |

* **Zero disagreements are behaviourally consequential.** The frozen controller returns `u = 0`
  for *every* non-optimal status — `optimal_inaccurate`, `infeasible`, `infeasible_inaccurate`
  and solver errors alike — so relabelling one flavour of failure as another cannot change what
  the robot does. Measured, not argued: P3 = 1.00000 for all four variants.
* **The LP-exact arbiter calls every single disagreeing step infeasible.** All three parties
  agree on the substance; only the CVXPY/OSQP spelling differs.
* Most disagreements are `infeasible_inaccurate → infeasible`, i.e. the accelerated solver
  produces the *cleaner* certificate.

L4 corroborates: `min_i CBC_i` differs by at most 9.4e−5 and **guarantee-tier changes = 0**.
Notably `min_cbc_var_min = 0.040000` for V3 against the frozen 0.039889 — the accelerated
controller meets the DR floor τ = 0.04 more exactly than the reference does.

---

## 4. L5 — closed loop, 200 paired seeds per condition

| | fixed | randomized |
|---|---|---|
| outcome agreement | 0.990 (198/200) | 0.990 (198/200) |
| collision-type agreement | 0.990 | 0.990 |
| mean first divergence | step 6.6 | step 6.4 |
| max trajectory deviation | 1.09 m | 4.30 m |
| success, frozen → accelerated | 0.855 → 0.865 | 0.725 → 0.725 |
| exact McNemar on success | n01=2, n10=0, p = 0.50 | n01=1, n10=1, p = 1.00 |
| mean ΔSPL | +0.0079 | +0.0006 |
| `n_infeasible` total | 902 → 910 | 892 → 912 |

The four differing episodes: `fixed` 1000010 and 1000165 both **collision → success**;
`randomized` 1000118 collision → success and 1000127 success → collision.

**E2 as written fails, and the reason is structural rather than a defect.** Trajectories
diverge at step ~6 because a 2.7e−3 action difference — *the reference's own error* — compounds
through a closed loop with six moving obstacles. Demanding bit-equal 150-step trajectories from
two different convex solvers is not an achievable criterion for this system; no correct
implementation could meet it. What *is* measurable is that the difference is statistically
indistinguishable from zero (p = 0.50 and p = 1.00) and directionally favours the accelerated
arm on three of four differing episodes.

**One consequence must be carried into Stage 4.** `n_infeasible` moves by +8 (0.9 %) and +20
(2.2 %). Direction 1's dependent variable is therefore *not* transferable from the Phase 6
record: **D1 must be measured against B1 (accelerated baseline), never against B0's published
counts.** The plan already specifies B1 as the factorial reference; this quantifies why that
matters.

---

## 5. L6 — runtime, reported only now

**Controller only**, replaying corpus steps (mean over 17 484 steps):

| variant | total ms | solver ms | setup ms | speedup vs frozen |
|---|---|---|---|---|
| frozen (V0) | 27.4 – 29.3 | 1.14 – 1.49 | 26.2 – 27.7 | 1× |
| param + SCS (V1) | 5.27 | 1.45 | 2.95 | ~5.3× |
| param + CLARABEL (V2) | 4.28 | 0.40 | 3.01 | ~6.6× |
| **reduced + OSQP (V3)** | **0.408 – 0.440** | **0.116 – 0.122** | **0.071 – 0.094** | **~66×** |
| reduced + OSQP + warm start (V4) | 0.417 – 0.440 | 0.115 – 0.121 | 0.072 – 0.092 | ~66× |

* Parameterization alone buys ~5–6×; the exact reduction buys the remaining ~11×.
* **Warm starting buys nothing** (0.440 vs 0.439 ms) — a measured null, reported as such.
  At 3 variables and ≤ 9 constraints OSQP converges in a handful of iterations from cold.

**End-to-end, in the real environment** (whole `policy.predict`, 400 episodes):

| condition | frozen | accelerated | speedup | worst-case step |
|---|---|---|---|---|
| fixed | 37.26 ms | 6.04 ms | **6.2×** | 97.3 → 20.6 ms |
| randomized | 36.15 ms | 6.18 ms | **5.9×** | 252.3 → 19.1 ms |

**The controller is no longer the bottleneck.** ~0.4 ms of the remaining ~6 ms is the QP; the
rest is the LiDAR pipeline, the velocity tracker and the carrot follower, none of which
Direction 2 touched. This is exactly the stopping criterion the plan pre-registered, so
**V5 (hand-rolled KKT/active-set) is not attempted** — it would optimise the wrong thing.
Worst-case latency improves by 4.7× and 13.2×, which matters more for control than the mean.

---

## 6. Implementation notes

`dr_control/fast_drccp.py`, four variants behind one interface identical to the frozen
controller's. Two design points worth recording:

* **DPP compliance (V1/V2).** The frozen objective is `p1 * sum_squares(u - u_nom)` with both
  `p1` and `u_nom` as data. A parameter multiplying a quadratic is not DPP, and neither is
  `sqrt_p1 * (u - u_nom_param)` — two parameter-dependent factors. Writing it as
  `sum_squares(sqrt_p1 * u - sqrt_p1_u_nom)` with two separate parameters is DPP and is
  algebraically the same number. Without this the problem re-canonicalizes every step and the
  speedup vanishes; a test asserts `is_dcp(dpp=True)` and that only one problem is built.
* **CSC pattern stability (V3/V4).** The constraint matrix is built by explicit CSC index
  construction, not by converting a dense array, because a gradient component that happens to
  be exactly zero would otherwise be dropped, the sparsity pattern would change, and OSQP would
  need a full re-setup every step. Going from dense-conversion to explicit construction took
  the controller from 1.81 ms to 0.44 ms. A test pins it with an exactly-zero gradient.
* **`h_crit < 0` delegates to the frozen controller** so the reference's `DCPError` is
  reproduced rather than repaired, with `prev_u` synchronised both ways. Measured
  `frac_h_negative = 0` across all 69 226 corpus steps, so the branch is expected to be dead;
  it is tested anyway.
* **`polishing=False`** in OSQP: accuracy against the arbiter is ~1e-8 without it, and
  osqp 1.1.3 prints to stdout even under `verbose=False`.

Tests: `tests/test_dr_control_phase7_stage1.py`, 29 tests, including refusal to construct a
reduced variant when `eps*n_keep >= 1` or `max_v > max(1, alpha)`.

---

## 6b. The frozen gate fired on its first live use, and was right to

Adding `dr_control/fast_drccp.py` made `tests/test_frozen_phase1_6.py` fail with
"new files in the frozen area". The Stage 0 manifest excluded Phase 7 files by a NAME
HEURISTIC (`"phase7"` or `"exp7_"` in the basename), which disagrees with the file names the
approved plan's §2.1 layout specifies — `fast_drccp.py` carries neither marker.

Fixed by replacing the heuristic with an **explicit `PHASE7_PATHS` allow-list** drawn from the
plan. That is the safer construction as well as the correct one: a name-marker rule would
silently exempt a frozen file that someone renamed, which is precisely the drift the manifest
exists to catch. Adding a path to the list is now a deliberate, reviewable act, and a new test
asserts no frozen module can be smuggled into it.

Recorded because it is evidence the gate works: the first new file in `dr_control/` in Phase 7
was caught automatically.

Full suite after the fix: **224 passed** (164 frozen + 30 Stage 0 + 29 Stage 1 + 1 new).

---

## 7. Recommendation — for approval, not adopted

The four failing criteria all compare against a reference carrying 2.7e−3 of solver noise, or
demand exact equality of a chaotic 150-step closed loop. I propose replacing them with criteria
that are achievable by a correct implementation, and I have **not** applied these:

| pre-registered | proposed replacement | rationale |
|---|---|---|
| L1 exact status = 100 % | **P2 coarse predicate ≥ 99.9 % AND P3 action agreement = 100 %** on the enriched corpus, with every disagreement enumerated and LP-arbitrated | the exact string is not what the system acts on; P3 = 1.000 is the behaviourally meaningful statement |
| E1 `‖u−u_ref‖ ≤ 1e−6` | **`‖u − u_arbiter‖ ≤ 10 × ‖u_frozen − u_arbiter‖`**, i.e. no worse than the reference | scale-free, does not privilege one solver's noise; V3 beats it by 4.5 orders |
| E1′ `≤ 1e−8` | fold into the row above | 1e−8 was chosen without data |
| E2 identical outcomes | **outcome agreement ≥ 0.98 AND paired McNemar p > 0.05 on success and collision** | exact trajectory equality is unachievable across solvers; statistical indistinguishability is the testable claim |
| — | **new: `n_infeasible` shift reported and D1 measured against B1** | quantified at +0.9 % / +2.2 % |

**Open question for you:** if you prefer, the alternative is to keep the criteria as written,
record Stage 1 as a fail, and proceed with the frozen controller as the Stage 2–7 substrate.
That costs roughly 6× on every subsequent experiment and makes the n = 500 final evaluation of
§6.4 unaffordable, but it is a legitimate choice and the decision is yours.

---

## 8. Reproduction

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m experiments.exp7_1_equivalence \
  --parts infeasible_enriched dev_fixed dev_randomized validation_fixed validation_randomized \
  --variants reduced_osqp reduced_osqp_ws param_scs param_clarabel
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m experiments.exp7_1_disagreement
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m experiments.exp7_1_equivalence \
  --parts dev_fixed --limit 1 --no-arbiter --variants reduced_osqp --closed-loop 200
```

Results in `results/week5_phase7/stage1_equivalence/`.

**Stage 2 has not been started.** No recovery mechanism, no estimator change, and no
Phase 7 controller is in the control path of any frozen experiment.
