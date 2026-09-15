# Track A continuation — unlabelled-threat experiment: Stage A–D report

**Verdict: NO-GO for the full evaluation as implemented.** A specific, measured root cause was
found and the smallest next change is specified below. No large batch was launched. No frozen
artefact, canonical controller, or previous result was modified.

Git: branch `Week6`, commit `07d74ca`, working tree clean of tracked modifications (all Week-6 work
is untracked). `PURSUIT_RESULTS.md` md5 `65180116797ea9413e7e071ab887481d`, unchanged. Protected
manifest (466 files) verifies clean.

---

## 1. Research question and hypothesis

Can the protagonist retain useful collision-avoidance and goal-reaching performance if it treats
**every confidently tracked moving obstacle as a possible threat**, instead of receiving the
hunter's privileged identity and state? The purpose is to separate the value of privileged hunter
information from the identity-classification errors that blocked the earlier detector
(`MD_files/week6_tracka/TRACKA_RESULTS_A_P1.md`: precision 0.33–0.37 among six decoys).

## 2. Conditions compared (hunter policy and safety filter identical in all three)

| arm | hunter visible to protagonist LiDAR | oracle hunter row | per-track threat rows |
|---|---|---|---|
| `U0_oracle` (audited baseline, unchanged code path) | **no** | **yes** (true position + velocity) | no |
| `U1_unlabelled` | yes | **no** | **yes** |
| `U2_visible_only` | yes | no | no |

`U2` exists because `U0` and `U1` otherwise differ in **two** ways at once (visibility *and* row
structure); it isolates "the hunter is merely visible as geometry" from "explicit threat rows".

## 3. Information available (VERIFIED BY TESTS)

- `U1`/`U2` receive: 24 LiDAR ranges, ego pose, the static map already used by the frozen A*
  planner, and their own tracker's estimates. **No** hunter position/velocity, **no** identity
  labels, **no** ground-truth obstacle state, **no** array ordering.
- Code-level guarantees: the unlabelled source has no `push_hunter`/`clear_hunter` at all; an AST
  test rejects `robot_env`, `obstacle_positions`, `obstacle_velocities`, `hunter_position/velocity`,
  `env` in the module; a structural test asserts the single `push_hunter` call in the harness is
  inside an explicit `U0_oracle` branch; and a behavioural test shows that **moving the hunter's
  hidden state while holding the observation and pose fixed leaves the U1/U2 action bit-identical**.

## 4. Equations and eligibility

For an eligible track with estimated centre `c`, velocity `v`, disc radius `r_t = 0.3`:

```
h      = ||p - c|| - r_robot - r_t
grad_h = (p - c) / ||p - c||
dh_dt  = -grad_h . v
h_eff  = h - misses * dt * ||v||        (coasted tracks only; never inflates h)
```

The first three lines are **exactly** the oracle row's convention with estimates substituted; the
fourth is the only deviation, and reduces to the oracle form when `misses = 0`.

Eligibility (sensor-legal): tracker-**confirmed** (3 hits in a 5-frame window, which is also what
makes the velocity estimate meaningful — no velocity is fabricated for a fresh detection);
`misses <= 5` (the tracker's own coast limit); estimated speed `>= 0.15 m/s`. One row per track id
per step, so no duplicates. **A stationary hunter is unrepresentable under this rule** and is
covered only by the frozen geometric scan rows.

**Capacity and what is lost.** The frozen controller keeps `n_keep = 5` rows ordered by
`α·h + dh_dt`. That key reads only row values, so the truncation is sensor-legal, but it ignores the
`grad_h·u` term, so a discarded row can still bind for some `u`. **Measured** (8 episodes, 763
steps): 6.5 rows built per step, 4.93 kept, of which **1.46 are track rows and 3.47 scan rows**;
only 2.2 % of steps have the kept set entirely composed of track rows. Displacement is therefore
moderate, not catastrophic — the capacity risk flagged before implementation did not dominate.

## 5. Files added (nothing existing modified)

| file | why |
|---|---|
| `continuation/unlabelled_threat/source.py` | `UnlabelledThreatSource`: frozen scan rows + one row per eligible moving track |
| `continuation/unlabelled_threat/harness.py` | three-arm episode runner; `U0` delegates to the unchanged pursuit builder |
| `experiments/week6_tracka2/u_pilot.py` | Stage C pilot across scenarios and arms |
| `experiments/week6_tracka2/u_coverage.py` | offline coverage + row-displacement diagnostic |
| `experiments/week6_tracka2/u_row_provenance.py` | offline provenance of every built row |
| `tests/test_week6_tracka2.py` | Stage A + Stage B tests |

## 6. Tests (EXECUTED)

`python -m pytest -q tests/test_week6_tracka2.py` → **21 passed**.
`python -m pytest -q tests/test_week6_tracka2.py tests/test_pursuit.py tests/test_week6_tracks.py`
→ **71 passed**. Coverage: information audit (4 tests), no-eligible/one/many tracks, acquisition and
loss, stale/unconfirmed/too-slow exclusion, coasted deflation, ordering permutation invariance,
duplicate prevention, finite row dimensions, zero-distance guard, visible-only arm builds no rows,
per-arm reproducibility, frozen u = 0 fallback under track rows, simultaneous-update invariant,
capture/contact constants.

Unrelated pre-existing failure, untouched: `tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main`
(4 cases), caused by the `main` ref moving under a frozen test.

## 7. Pilot (MEASURED — DEVELOPMENT ONLY, 6 episodes/arm/scenario, seeds 13 200 000–13 200 005)

Fresh block, disjoint from every prior range (navigation 10.0–10.9 M, pursuit 11.0–11.3 M,
Track B 12.0–12.1 M, Track A A-P1 13 000 000–13 000 002).

| scenario | arm | outcomes | rows/step | max rows | infeasible frac | min hunter dist |
|---|---|---|---|---|---|---|
| S4 full stochastic (6 obstacles) | U0 | capture 3, collision 3 | 0 | 0 | 0.140 | 1.08 |
| | **U1** | capture 3, collision 3 | **6.12** | **14** | **0.173** | 0.71 |
| | U2 | capture 5, collision 1 | 0 | 0 | **0.075** | 1.03 |
| S1 no decoys (0 obstacles) | U0 | capture 3, goal 2, collision 1 | 0 | 0 | 0.141 | 0.94 |
| | **U1** | capture 4, collision 2 | 4.94 | 13 | 0.110 | 0.72 |
| | U2 | capture 5, goal 1 | 0 | 0 | 0.034 | 0.91 |
| S3 several movers (3 obstacles) | U0 | goal 3, collision 3 | 0 | 0 | 0.095 | 1.30 |
| | **U1** | capture 3, collision 3 | 6.19 | 15 | 0.167 | 1.18 |
| | U2 | capture 4, goal 2 | 0 | 0 | 0.063 | 1.14 |

**Scripted occlusion (S2) was not implemented as a separate scenario**; occlusion occurs incidentally
in S3/S4 and was instead measured by the coverage diagnostic below. This is a deviation from the
requested pilot list and is recorded as such.

The pilot is 6 episodes per cell and **carries no statistical claim**. What it does show
unambiguously is a mechanism: **U1 builds far more rows than the solver can keep and has the
highest infeasibility of the three arms, and it reached the goal in 0 of 18 episodes.**

### Offline diagnostics (measurement only, ground truth used after the action is fixed)

Coverage, 8 episodes / 763 steps, seeds 13 200 100+:
- hunter has a matching track row **50.3 %** of the steps it is inside LiDAR range,
- but **95.8 %** of the steps it is within 2 m — i.e. coverage is good exactly when it matters;
- 2.1 % of steps have no track rows at all.

Row provenance, 4 episodes / 3 083 rows, seeds 13 200 200+:
- **1 050 rows (34 %) match a real body** (hunter or dynamic obstacle) within 0.6 m;
- **2 033 rows (66 %) match nothing real**, and **98 % of those lie within 0.6 m of mapped static
  geometry** (a rectangle surface or a wall);
- phantom rows carry a substantial spurious speed (mean 0.98 m/s, q90 1.94 m/s) — faster than any
  real obstacle (0.675 m/s) and above the hunter's bound.

## 8. Diagnosis, limitations and failure cases

**Root cause (MEASURED):** the tracker's compactness test rejects long wall-like segments, but a
*fragment* of a rectangle or wall visible between occluders is compact and is admitted as a moving
disc. Because the robot itself is moving, the visible fragment appears to slide, so the constant-
velocity filter assigns it a large velocity. The motion gate (≥ 0.15 m/s) therefore does not remove
it — it *selects* it. Two thirds of the threat constraints the controller solved against were these
artefacts, which over-constrains the QP and explains U1's higher infeasibility and zero goals.

This is a failure of the eligibility rule, **not** of the row equations, the information ledger, or
the capacity handling — all three behaved as designed and are covered by tests.

Other limitations: a stationary or near-stationary hunter is invisible to this rule; U1 and U0
differ in LiDAR visibility as well as row structure (which is why U2 exists); the coasted-track
deflation is a conservative heuristic, not a calibrated uncertainty bound; matching in the offline
diagnostics uses a 0.6 m tolerance, so a badly-estimated centre of a real body counts as "phantom".

## 9. Go/no-go

**NO-GO.** Of the Stage D conditions: the information audit passes, the tests pass, capacity is
manageable, and the oracle baseline is intact — but **the controller does not behave as intended**
(two thirds of its threat rows are static-geometry artefacts), so the results could not be
interpreted as the intended unlabelled-threat comparison. A full evaluation now would measure
tracker artefacts, exactly as the earlier detector experiment would have measured classification
errors.

**Smallest next change (PROPOSED, not implemented):** add a **static-map gate** to eligibility —
reject any candidate track whose estimated centre lies within a margin (start at 0.5 m ≈ r_robot +
tracker tolerance) of a mapped rectangle surface or wall. The static map is already available to the
frozen A* planner, so this uses no new information and is legal under the project's existing ledger.
Expected effect from the provenance data: removes ~98 % of phantom rows, taking rows/step from ~6.5
to ~2.2. It must be validated on a **fresh dev seed block** (propose 13 200 300+) and re-piloted
before any go decision — the current pilot must not be reused as independent evidence.

Secondary options if the gate is insufficient: require a longer minimum track age; require speed
within a plausible band (e.g. 0.2–1.3 m/s); cap threat rows to the k most critical by the frozen key.

## 10. Reproduction commands (exactly what was run)

```bash
cd /home/bt3/23CS10067/projects/rl-robot-navigation
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_week6_tracka2.py
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_week6_tracka2.py tests/test_pursuit.py tests/test_week6_tracks.py
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_pilot --episodes 6
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_coverage --episodes 8
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_row_provenance --episodes 4
.venv/bin/python -m experiments.week6_continuation.protect_manifest
```

Outputs: `results/week6_tracka2/pilot/u_pilot.json`,
`results/week6_tracka2/coverage/u_coverage_8.json`,
`results/week6_tracka2/coverage/row_provenance_4.json`.

## 11. Proposed full-evaluation protocol (NOT run; only if a re-pilot passes)

Paired on environment seeds from a fresh block disjoint from all of the above; identical hunter
policy and safety filter in every arm; the canonical stochastic obstacle model unchanged; arms
`U0_oracle`, `U1_unlabelled` (gated), `U2_visible_only`; hunter speed 1.0 m/s only, kept entirely
separate from any 1.25 speed sweep; terminal outcomes reported under the established priority
(protagonist collision → capture → goal → timeout) **with all event flags logged separately** so
contested steps can be re-analysed offline; static and dynamic protagonist collisions reported
separately; **capture-range termination never described as physical contact**; paired exact McNemar
for binary outcomes with discordant counts, paired bootstrap for continuous metrics, episode-cluster
bootstrap for step-level rates; solver feasibility, infeasibility bursts, fallback use, rows built
vs kept, runtime and exposure in steps all reported; and no equivalence or safety claim unless the
analysis supports it.
