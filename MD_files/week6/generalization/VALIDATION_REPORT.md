# Week 6 / Generalization Suite — validation report

**Implementation complete. The 118-cell screening run has NOT been started.**
Reporting for approval before launching it, as instructed.

## 1. Scope compliance

| requirement | status |
|---|---|
| only the §8 files added | ✅ `git status` shows **untracked additions only**, no modifications |
| `robot_env/`, `evaluation/`, `dr_control/`, `optional_navigation/`, Phase 1–7 artefacts untouched | ✅ frozen sha256 manifest **PASS**, 60 files |
| F11 / moving goal not implemented | ✅ no code, no manifest entry; test asserts absence |
| F1′ = staircase, interface test, fidelity recorded | ✅ area ratio, Hausdorff, cell count per map |
| F10 without replanning, either arm | ✅ declared in the manifest and asserted |
| `v_cap = 0.96` fixed everywhere | ✅ AST-level test on executable code |
| LiDAR range 5.0 m at every map size | ✅ asserted for all 64 configurations |
| fairness invariants preserved | ✅ asserted per episode; negative controls confirm they fire |
| no ground truth / `obs[28:52]` | ✅ bit-identical actions under corruption, both arms |
| manifest, ordering, counts, selection rule preserved | ✅ pinned by test |

## 2. Tests

**31 new tests, all passing. Full suite: 319 passed** (288 pre-existing + 31 new), no
pre-existing test modified.

Counts asserted rather than trusted: **64** configurations (42/11/4/7 by level), **118**
screening cells, **111** final cells = **101 benchmark @ 200** + **10 stress @ 50**,
**11 800 + 41 400 = 53 200** episodes. Family ordering and the first-three-by-index rule for
F10/F12 are pinned. Seed blocks are asserted disjoint, and the Phase 1–7 block
`1 000 000…+199` is reachable only from the G0 gate.

**Negative controls run**, so the fairness assertions are not vacuous: perturbing start,
horizon or static-geometry count each raises; identical inputs pass.

## 3. G0 — M0 canonical regression gate

200 paired episodes from `EVAL_SEED_BASE = 1 000 000`, `config.json` unchanged.

| arm | motion | success | target | collision | target | |
|---|---|---|---|---|---|---|
| A original | deterministic | 0.855 | 0.855 | 0.140 | 0.140 | **PASS** |
| A original | stochastic | 0.725 | 0.725 | 0.275 | 0.275 | **PASS** |
| B updated | deterministic | 0.925 | 0.925 | 0.065 | 0.065 | **PASS** |
| B updated | stochastic | 0.815 | 0.815 | 0.180 | 0.180 | **PASS** |

**G0 PASS — exact reproduction on all four**, to the tolerance `1e-9`. Arm A reproduces
Phase 6 (`results/phase6/`), arm B reproduces the Phase-7 final comparison
(`results/week5_phase7/final_comparison/`), through a third independently written harness.
Artefact: `results/week6_generalization/g0_regression.json`.

## 4. Smoke tests — one configuration per family, both arms

All 13 families construct, step and complete episodes; every paired fairness assertion held.

| configuration | model | A succ/coll | B succ/coll | s/ep |
|---|---|---|---|---|
| F1_rot15_obs0 | deterministic | 0.50 / 0.50 | 0.00 / 1.00 | 3.93 |
| F2a_scale0.5 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 4.41 |
| F2b_n2 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 2.21 |
| F3_world7 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 2.84 |
| F4_r0.15 | deterministic | 0.50 / 0.50 | 0.50 / 0.50 | 3.84 |
| F5_n2 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 2.91 |
| F6_v0.3 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 2.89 |
| F8a_open_goal | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 2.21 |
| F9_w1.6 | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 4.68 |
| F10a_behind | deterministic | 1.00 / 0.00 | 1.00 / 0.00 | 5.05 |
| F12a_crossing | scripted | 0.50 / 0.50 | 0.50 / 0.50 | 4.12 |
| F7a_r45_v96 | deterministic | 0.00 / 1.00 | 0.00 / 1.00 | 2.52 |
| X1_corridor_blocked | scripted | 0.00 / 0.00 | 0.00 / 0.00 | 19.28 |

**These are 2 episodes per cell and carry no inferential weight whatever.** They exist to prove
the machinery runs. In particular F1_rot15's 0.50 → 0.00 is one episode and means nothing.

Mechanism plumbing confirmed working: the recovery ladder fires and records tiers
(F1_rot15 B: 20 recoveries, all T2; F10a B: 8, all T2), and arm A records infeasible steps
with no recovery, as it should.

## 5. Runtime — recomputed, not carried over

Measured **4.68 s/episode** across the smoke cells, against the proposal's assumed 5.65.

| tier | episodes | at 4.68 s/ep | at 8-way |
|---|---|---|---|
| screening | 11 800 | 15.3 h | **1.9 h** |
| final | 41 400 | 53.9 h | **6.7 h** |
| **total** | **53 200** | **69.2 h** | **8.7 h** |

Down from the 83.5 h / 10.4 h projection. **This is still a weak estimate** — 2 episodes per
cell, one motion model each, and it is dominated by whichever scenarios time out. The runner
recomputes the projection from its own rate after 5, 10 and 20 completed screening cells and
prints the revision; the episode counts are exact and do not move.

## 6. One design observation, reported and NOT acted on

**X1_corridor_blocked produced a timeout with zero infeasible steps**, in both arms: the
controller stops short of the blocked corridor (min clearance 0.083 m) and runs out the
500-step horizon. The DR problem never became infeasible because **a safe action does exist —
stopping** — so the scenario as declared does not remove every admissible control the way X3 is
meant to.

That is a property of the scenario design, visible from two episodes. It is **reported, not
changed**: modifying a family definition in response to observed behaviour is exactly what the
preregistration forbids, and X1 is characterisation, excluded from every benchmark aggregate, so
a timeout is a legitimate recorded outcome rather than a failure. It is also why X1 costs
19.3 s/episode — full-length timeouts.

If you want X1 tightened, that decision belongs **now**, before screening, and should be
recorded as an amendment.

## 7. Ready to run

Everything gated before screening has passed:

* 31 new tests + 288 existing = **319 passed**
* frozen manifest **PASS**
* **G0 PASS**, exact on all four targets
* smoke coverage of all 13 families, fairness assertions live throughout

**Awaiting approval to launch the 118-cell screening run** (11 800 episodes, ~1.9 h at 8-way on
the current estimate). Nothing further will be started without it.
