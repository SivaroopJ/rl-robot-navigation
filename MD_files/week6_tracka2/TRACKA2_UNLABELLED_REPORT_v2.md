# Track A continuation — unlabelled-threat experiment, v2: static-map gate and fresh-seed diagnostics

**Supersedes nothing.** `TRACKA2_UNLABELLED_REPORT.md` (md5 `dd4c8233951ac316e08e4640bcf87ffd`) is
preserved unchanged and remains the record of the pre-gate NO-GO. This v2 adds the static-map gate
and three fresh-seed runs.

**Decision: NO-GO / pending.** The gate fixed the failure it targeted, but the gated pilot artifact
exposes a **new failure mode introduced by our own row construction**, verified below. One minimal
fix and one re-pilot are required before a GO.

Git: branch `Week6`, commit `07d74ca`; no tracked file modified; nothing committed.
`PURSUIT_RESULTS.md` md5 `65180116797ea9413e7e071ab887481d`, unchanged. Protected manifest (466
files) verifies clean. Tests: `pytest -q tests/test_week6_tracka2.py tests/test_pursuit.py
tests/test_week6_tracks.py` → **78 passed** (executed 2026-09-13, output inspected).

---

## 1. The static-map gate (implemented)

**Rule.** Reject a candidate track if its estimated centre lies within **0.3 m** of a mapped
rectangle surface or of the arena boundary. Implemented in
`continuation/unlabelled_threat/source.py::UnlabelledThreatSource.near_static_geometry`, applied
inside `eligible_tracks()` after the confirmation, staleness and motion gates.

- It uses the **prior static map already supplied to the frozen A\* planner**
  (`static_obstacles`, `world_size`, passed at construction by the harness), so it opens no new
  information channel.
- It **never inspects live obstacle state**; an AST test rejects `obstacle_positions` /
  `obstacle_velocities` in the module, and a dedicated test asserts the gate reads only the
  constructor-supplied map.
- A non-positive margin disables the gate, so the earlier eligibility rules remain testable alone.

**Margin selection — done on PRE-GATE development data (seeds 13 200 200+), before any fresh run:**

| Margin | Phantom rows surviving | Real rows lost | Hunter rows lost |
| ------ | ---------------------: | -------------: | ---------------: |
| 0.3 m  |                   4.2% |          26.7% |             6.7% |
| 0.5 m  |                   2.7% |          35.0% |            10.6% |
| 0.8 m  |                   1.2% |          53.0% |            34.9% |

**0.3 m was chosen as the smallest tested margin that removes ≈ 96 % of clutter rows**, at the
lowest cost in genuine rows. The predicted cost was explicit and is repeated here: **≈ 27 % of
genuine rows and ≈ 6.7 % of hunter rows were expected to be lost.**

**Tests.** Seven new gate tests: rectangle rejection; open-space retention; boundary-hugging
rejection; exact margin boundary either side; prior-map-only access; gate counter reported in
`row_stats()`; harness passes the real 5-rectangle map and world size into the source.

## 2. Fresh-seed diagnostics (all three on disjoint blocks)

### 2.1 Row provenance — seeds 13 200 500+, artifact `results/week6_tracka2/coverage/row_provenance_4_gated.json`

| Measure                   |    Pre-gate |   Post-gate |
| ------------------------- | ----------: | ----------: |
| Rows matching a real body | 1 050 (34 %) | 1 379 (90 %) |
| Rows matching nothing     | 2 033 (66 %) |  147 (9.6 %) |

Absolute phantom reduction **2 033 → 147, ≈ 93 %**. Of the residual phantoms, **59 % lie near static
geometry, against 98 % before the gate**, so the single dominant cause has been removed and what
remains is a mixture. Post-gate total rows: 1 526.

### 2.2 Coverage — seeds 13 200 400+, 521 steps, artifact `u_coverage_8_gated.json`

| Measure | Pre-gate | Post-gate |
|---|---:|---:|
| hunter covered while in LiDAR range | 50.3 % | **59.0 %** |
| hunter covered within 2 m | 95.8 % | **98.9 %** |
| steps with no track rows | 2.1 % | 15.5 % |
| kept rows (total / track / scan) | 4.93 / 1.46 / 3.47 | **4.87 / 0.66 / 4.21** |
| steps whose kept set is entirely track rows | 2.2 % | **0 %** |

**Measured:** the gate did not reduce hunter coverage in this diagnostic; it rose.
**Interpretation (not a controlled causal result):** with clutter rows removed there is less
competition for the five row slots, so genuine rows survive truncation more often.

### 2.3 Gated pilot — seeds 13 200 300–13 200 305, 6 episodes per arm per scenario, artifact `results/week6_tracka2/pilot/u_pilot_gated.json` (verified by reading the artifact, not the console)

| scenario | arm | outcomes | rows/step (max) | infeasible frac | infeasible / solver-fail | recovery events | min hunter dist | min clearance |
|---|---|---|---|---|---|---|---|---|
| S4 full stochastic | U0 oracle | collision 5 (4 dyn, 1 static), goal 1 | 0 (0) | 0.083 | 39 / **39** | 39 | 1.40 | −0.016 |
| | **U1 gated** | capture 4, collision 1 (dyn), goal 1 | 2.69 (7) | 0.074 | 51 / **82** | 51 | 0.71 | 0.093 |
| | U2 visible-only | capture 3, collision 2 (dyn), goal 1 | 0 (0) | 0.032 | 18 / **18** | 18 | 1.10 | 0.084 |
| S1 no decoys | U0 oracle | goal 4, capture 2 | 0 (0) | 0.066 | 39 / 39 | 39 | 1.69 | 0.323 |
| | **U1 gated** | goal 4, capture 2 | 0.84 (4) | 0.043 | 21 / **38** | 21 | 1.55 | 0.282 |
| | U2 visible-only | goal 4, capture 2 | 0 (0) | 0.036 | 14 / 14 | 14 | 1.53 | 0.348 |
| S3 several movers | U0 oracle | capture 4, goal 2 | 0 (0) | 0.178 | 137 / 137 | 137 | 0.84 | 0.191 |
| | **U1 gated** | goal 4, capture 2 | 1.86 (5) | 0.047 | 31 / **53** | 31 | 1.30 | 0.212 |
| | U2 visible-only | capture 4, goal 2 | 0 (0) | 0.041 | 20 / 20 | 20 | 0.83 | 0.211 |

Gate activity is substantial: 2 828 / 2 247 / 3 286 candidate tracks were rejected by the static-map
gate across the three scenarios.

**Improvements, measured:** rows per step fell from 6.12 → 2.69 (S4), 4.94 → 0.84 (S1), 6.19 → 1.86
(S3), worst-case rows 15 → 7; U1's infeasible fraction fell from the highest of the three arms
(0.173 S4) to at or below the oracle arm in every scenario; and U1 reached the goal in 9 of 18
episodes against **0 of 18** pre-gate.

## 3. NEW FAILURE MODE found in the gated pilot (this is why the decision is NO-GO)

For U0 and U2, `solver_fail == n_infeasible` exactly. **For U1 they diverge**: 82 vs 51, 38 vs 21,
53 vs 31 — i.e. **31 / 17 / 22 extra solver failures that are not infeasibility.**

Root cause, **verified** by a 3-episode instrumented run (seeds 13 200 600+, 148 steps):
statuses `{optimal: 141, infeasible: 5, DCPError: 2}`, with **2 steps at `h_crit < 0` and exactly 2
negative-`h_eff` track rows**. The staleness deflation `h_eff = h − misses·dt·‖v‖` can drive a row's
clearance negative; the frozen objective schedules its proximity weights off `h_crit`, so a negative
value makes the objective non-convex and CVXPY raises `DCPError`.

Consequences: the frozen fallback (u = 0) still applies, so nothing unsafe is executed and the
frozen behaviour is intact — but the audited pursuit work showed a stationary protagonist is exactly
what a pursuer exploits. In S4 U1's total fallback rate (82 / 804 steps ≈ 10.2 %) exceeds the oracle
arm's (39 / 600 ≈ 6.5 %). This failure mode is **introduced by our own row construction**, not by
the environment or the frozen controller.

**Minimal fix (PROPOSED, not implemented):** clamp the deflation so it cannot drive a row below
contact, `h_eff = max(h − misses·dt·‖v‖, 0)` when the raw `h ≥ 0`, leaving genuinely negative raw
clearance to behave exactly as the frozen system already does. This keeps the conservative intent
(deflation still shrinks clearance) while removing the concave-objective trigger. It must be
validated on a **fresh** block (propose 13 200 700+) with a re-pilot; the current pilot cannot serve
as its evidence.

## 4. Limitations and one process issue

- **Pre/post comparisons use different seed blocks** (13 200 000+ vs 13 200 300+ for the pilot, and
  separate blocks for each diagnostic), deliberately, so the fix is not validated on the data that
  motivated it. They are therefore **not paired before/after tests**; episode difficulty differs.
  Row-composition changes are direct properties of the gate; outcome and infeasibility changes are
  suggestive only.
- **The coverage diagnostic is small** (8 episodes, 521 steps) and suggestive, not conclusive.
- The margin table predicted ≈ 27 % genuine-row loss. That **did not** translate into worse hunter
  coverage in the closed-loop diagnostic, but **this does not prove real threats are never missed** —
  it is one small measurement on one seed block.
- Bodies hugging a wall or rectangle are now dropped by the gate and covered only by the frozen
  geometric scan rows, which are **partial** coverage: one row per buffered scan, nearest point only.
- Six episodes per pilot cell carry **no statistical weight**; S3's oracle arm (0.178 infeasible,
  137 recovery events) is plausibly one or two hard episodes.
- **Process issue, recorded.** The first diagnostic reruns were launched without output tags; the
  never-overwrite guard refused them and the chain continued past the failure, leaving two 75–79 byte
  logs and **no new artifacts**. The stale pre-gate file was briefly read back as if current, and was
  caught because its numbers were identical to the pre-gate run. `--tag` was added to both scripts
  and both were rerun. **Only the `_gated` artifacts are authoritative, and no conclusion in this
  report rests on the stale output.**
- **Track B was not touched.** Its conclusion stands as recorded: reduced infeasibility for
  Random-CLF-DR-CBF with **no task-outcome improvement**, which is **not** a safety improvement.

## 5. Decision: NO-GO / pending

Against the decision rule: the pilot artifacts are verified; the eligibility failure is materially
improved (phantoms 66 % → 9.6 %, rows/step 6.1 → 2.7); hunter coverage is acceptable and did not
degrade (98.9 % within 2 m); infeasibility improved. **But the "no new serious failure mode"
condition fails**: U1 introduces DCPError-driven fallbacks at a rate that lifts its total fallback
rate above the oracle arm's in S4, and those fallbacks command u = 0 in the presence of a pursuer.

Required before GO: (1) the clamped-deflation fix; (2) a re-pilot on a fresh block confirming
`solver_fail == n_infeasible` for U1, or a documented, quantified residual; (3) no regression in
rows/step, coverage or infeasibility. Estimated cost: one code change, one test, ~8 minutes of pilot.

## 6. Proposed full-evaluation protocol (only after the above passes; NOT approval to run now)

Paired on a fresh seed block disjoint from every range above; identical hunter policy and safety
filter in all arms; canonical stochastic obstacle model unchanged; arms kept **separate, never
pooled**: `U0_oracle`, `U1_unlabelled_gated`, `U2_visible_only` (and, if wanted as a negative
control, `U1_unlabelled_ungated` reported separately). Hunter speed 1.0 m/s only; any 1.25 sweep
stays a separate artifact. Terminal outcomes under the established priority (protagonist collision →
capture → goal → timeout) **with all event flags logged** for offline re-analysis; static and dynamic
protagonist collisions separated; **capture-range termination never described as physical contact**;
paired exact McNemar with discordant counts for binary outcomes, paired bootstrap for continuous
metrics, episode-cluster bootstrap for step-level rates; rows built vs kept, gate rejections,
infeasibility bursts, fallback split by status (`infeasible` vs `DCPError`), runtime and exposure in
steps all reported. Diagnostics and pilots are **not** pooled into the evaluation. No equivalence or
safety claim unless the analysis supports it.

## 7. Files changed and commands run

Changed (all untracked Week-6 work): `continuation/unlabelled_threat/source.py` (gate + counters +
row-vector logging), `continuation/unlabelled_threat/harness.py` (passes the prior static map),
`tests/test_week6_tracka2.py` (+7 gate tests, fixture now gate-disabled by default),
`experiments/week6_tracka2/{u_pilot,u_coverage,u_row_provenance}.py` (fresh seed blocks, `--tag`).

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_week6_tracka2.py tests/test_pursuit.py tests/test_week6_tracks.py
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_pilot --episodes 6 --tag _gated
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_coverage --episodes 8 --tag _gated
PYTHONPATH=. .venv/bin/python -m experiments.week6_tracka2.u_row_provenance --episodes 4 --tag _gated
.venv/bin/python -m experiments.week6_continuation.protect_manifest
```

---

# 8. UPDATE (2026-09-13): deflation clamp applied and re-piloted

**Decision unchanged: NO-GO / pending.** The clamp fixed its target, but the fresh pilot shows the
failure moved rather than disappeared, and it corrected one attribution made in section 3.

## 8.1 Change

`h_eff = max(h - misses·dt·‖v‖, 0)` when the raw `h ≥ 0`; a genuinely negative raw `h` passes
through untouched, so real contact behaves exactly as in the frozen system. Pinned by two new tests
(clamped coasted track; untouched negative raw clearance). **30 tests pass** in
`tests/test_week6_tracka2.py` (executed, output inspected).

## 8.2 Clamped pilot — seeds 13 200 700–13 200 705, 6 episodes/arm/scenario,
artifact `results/week6_tracka2/pilot/u_pilot_clamped.json`

| scenario | arm | outcomes | rows/step (max) | infeasible frac | infeasible / solver-fail |
|---|---|---|---|---|---|
| S4 | U0 oracle | goal 3, collision 2, capture 1 | 0 (0) | 0.074 | 22 / 22 (match) |
| | **U1 clamped** | goal 3, collision 2, capture 1 | 2.18 (7) | **0.236** | 67 / 68 (**extra 1**) |
| | U2 visible-only | goal 3, capture 2, collision 1 | 0 (0) | 0.023 | 7 / 7 (match) |
| S1 | U0 oracle | goal 3, capture 3 | 0 (0) | 0.057 | 41 / 41 (match) |
| | **U1 clamped** | goal 3, capture 3 | 0.87 (3) | 0.064 | 31 / 31 (**match**) |
| | U2 visible-only | goal 3, capture 3 | 0 (0) | 0.031 | 16 / 21 (**extra 5**) |
| S3 | U0 oracle | goal 4, capture 2 | 0 (0) | 0.101 | 74 / 74 (match) |
| | **U1 clamped** | goal 4, capture 2 | 1.50 (5) | **0.015** | 9 / 10 (extra 1) |
| | U2 visible-only | goal 5, capture 1 | 0 (0) | 0.009 | 5 / 5 (match) |

**Measured:** extra (non-infeasibility) solver failures for U1 fell **31 / 17 / 22 → 1 / 0 / 1**.

## 8.3 Two findings that keep this at NO-GO

**(a) Correction to section 3's attribution.** In S1 the `U2_visible_only` arm — which builds **no
track rows at all** — shows 5 extra solver failures. Negative `h_crit` therefore also arises from
the **frozen scan rows** near walls, and DCPError is **not exclusively caused by our row
construction**, as section 3 stated. Section 3's measurement (U1 diverging while U0/U2 matched on
that block) stands; its causal claim was too strong and is corrected here.

**(b) The failure moved from DCPError to chronic infeasibility.** In S4, U1's infeasible fraction is
**0.236 against the oracle arm's 0.074 on identical seeds**, and it is bimodal per episode —
U1 `0.381, 0.000, 0.469, 0.047, 0.474, 0.048` vs U0 `0.048, 0.000, 0.125, 0.025, 0.222, 0.024`.
Three of six episodes spend nearly half their steps infeasible. **Interpretation (not yet verified
by instrumentation):** a clamped row sits exactly at contact (`h_eff = 0`), which is maximally
tight, so the QP now reports infeasible where it previously raised DCPError. S3 moved the other way
(U1 0.015 vs oracle 0.101), so the effect is scenario-dependent.

Six episodes per cell carry no statistical weight, and the pilot blocks differ between runs, so none
of these are controlled comparisons.

## 8.4 Next step (PROPOSED, not implemented)

Instrument first, then choose: log, on each infeasible step, how many **kept** rows are clamped
(`h_eff == 0`) and how many are track rows, to confirm or refute 8.3(b) before changing anything.
If confirmed, the candidate fixes in increasing order of intrusiveness are: (i) **drop** a track row
whose deflation would drive it below contact, treating it as too uncertain to constrain, rather than
clamping it to a maximally tight row; (ii) cap the deflation at a fraction of the raw clearance
(e.g. `h_eff ≥ 0.5·h`); (iii) widen the staleness limit so heavily-coasted tracks never enter.
Each needs its own fresh block; the current pilots cannot serve as their evidence.
