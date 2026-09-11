# Week 6 Continuation — Implementation Audit (pre-implementation)

Status: **AUDIT ONLY. No controller, runner, or experiment code has been written. No episode run.**
Date: 2026-09-10. Branch `DR_safe` @ `427e589`. Existing suite: **322 passed** (143 s).

Naming used below: *Week 5.5* = the completed multi-map generalization study
(`generalization/`, `experiments/exp9_*`, `results/week6_generalization/`, `MD_files/week6/`).
Experiment stages `RANDOM_EXP_R1/R2/R3` are **not** the LADDER rungs R1/R1.5/R2/R2_lp.

---

## 1. What exists (verified from code, not from docs)

| component | file | status |
|---|---|---|
| Original CLF-DR-CBF policy | `dr_control/policy.py::DRCBFPolicy` | FROZEN, SHA-256 manifest |
| DRCCP QP (SCS, rebuilt per step) | `dr_control/drccp_controller.py::ClfCbfDrccpController` | FROZEN |
| LiDAR barrier source (k-scan buffer) | `dr_control/lidar_cbf.py`, `estimated_cbf.py` | FROZEN |
| LiDAR velocity tracker | `dr_control/velocity_tracker.py` | FROZEN |
| T1 projection cap, v_cap = 0.96 | `dr_control/capped_velocity.py::ProjectionCappedSource` | Phase-7 extension |
| Phase7Policy (arm selector) | `dr_control/policy_phase7.py` | Phase-7 extension |
| LADDER R0→R1→R2→R2_lp | `dr_control/recovery.py::RecoveryLadder` | Phase-7 extension |
| Canonical M0 harness + metrics + stats | `experiments/exp6_ppo_comparison.py` (`make_env`, `episode_record`, `summarise`, `mcnemar`, `paired_ci`, `true_clearance`) | FROZEN |
| Updated-controller final eval | `experiments/exp8_final_comparison.py` | committed (427e589) |
| Frozen-area guard | `tests/test_frozen_phase1_6.py` + `MD_files/week5/phase7/FROZEN_MANIFEST.sha256` (60 files) | active |

**Canonical M0** (`exp6.make_env`): `config.json`, 6 dynamic obstacles, obstacle_speed 0.675, dt 0.1,
max_steps 500, 24-ray LiDAR range 5.0, single integrator, max_speed 1.0, 5 fixed rectangles.
Two motion conditions: **`fixed`** (deterministic billiard) and **`randomized`** (OU heading noise).

### 1.1 Verified reference configuration

| symbol | code name | value | where set | exposed at policy level? |
|---|---|---|---|---|
| α | `cbf_rate` / `rateh` | 0.4 | `DRCBFPolicy(alpha=0.4)` | **yes** |
| λ_V | `clf_rate` / `rateV` | 1.0 | controller default | no (controller ctor only) |
| r_W | `wasserstein_r` | 0.004 | controller default | no |
| ε | `epsilon` | 0.1 | controller default | no |
| k_v | `k_v` = `K_V_REFERENCE` | **0.05** (not 1.0) | controller default | no |
| N (scan buffer = DR sample count) | `k_scans` | 5 | **hard-coded literal** in `DRCBFPolicy.reset` and `Phase7Policy.reset` | no |
| n_keep | `n_keep` | 5 | controller default | no |
| noise_level | — | **not present in the port** (reference JSON only) | — | — |
| v_max | `max_v` | 1.0, **per-axis box** \|u_x\|,\|u_y\| ≤ 1 | env MAX_SPEED | — |
| objective | p0 = 3, p1 = 4·h_crit, p3 = 5·h_crit | fixed | `build_objective` | no |
| carrot lookahead | 1.0 m | `DRCBFPolicy(lookahead=1.0)` | yes |
| v_cap (T1 only) | 0.96 = 1 − r_W/ε | `V_CAP_DERIVED` | yes |
| solver | SCS | controller default | no |
| infeasible fallback | u = 0, `prev_u` reset | `_solve` | — |

Reference numbers (200 eps, 1 000 000 block, already published — **not** reused here):

| cond | arm | success | collision | dyn | static | wall | timeout | n_infeasible | step ms |
|---|---|---|---|---|---|---|---|---|---|
| fixed | original | 0.855 | 0.140 | 0.140 | 0 | 0 | 0.005 | 902 | 36.0 |
| fixed | T1+LADDER | 0.925 | 0.065 | 0.065 | 0 | 0 | 0.010 | 383 | 37.1 |
| randomized | original | 0.725 | 0.275 | 0.275 | 0 | 0 | 0.000 | 892 | 36.4 |
| randomized | T1+LADDER | 0.815 | 0.180 | 0.175 | 0.005 | 0 | 0.005 | 573 | 42.1 |

Note: on M0 essentially **all** collisions are dynamic. Static/wall rates will be reported but
a total-collision improvement on M0 is, de facto, a dynamic-collision improvement.

---

## 2. Findings that affect the specification

**A1 — The C1 buffer "N" and the DR sample count "N" are the same quantity.** Each buffered scan
contributes exactly one sample (its nearest point to the current pose), so `k_scans` = the N of the
Wasserstein/CVaR problem; `n_keep = 5` never truncates. C1 therefore changes the DR sample set, not
only staleness. For N ∈ {1,3,5} with ε = 0.1, εN < 1, so the constraint stays "worst sample ≥ τ";
the maths is unchanged, only the sample set shrinks. Changing N needs a subclass because
`k_scans=5` is a literal in the frozen `reset()`.

**A2 — The (r_W, ε) grid is largely degenerate.** When εN ≤ 1 the DR constraint reduces
exactly to `min_i CBC_i ≥ τ_eff`, so r_W and ε enter **only through τ_eff**. Verified numerically:
(0.002, 0.05), (0.004, 0.1) and (0.008, 0.2) return the same action to within 2·10⁻⁵ (SCS
tolerance). With N = 5, ε ∈ {0.05, 0.1, 0.2} is in the collapsed regime (εN = 0.25/0.5/1.0; at
exactly 1.0 the CVaR sup is still attained at the worst sample). The 6×3 (r_W, ε) cells map to only
**10 distinct τ values** {0, .005, .01, .02, .04, .06, .08, .12, .16, .24}; with r_W = 0 all ε
are identical. So of the 120 H1 cells, 90 form **50 distinct controllers**. The rest are
solver-noise duplicates.

**A3 — ε = 0.3 changes the constraint's mathematical meaning (stop condition 5).** With N = 5,
εN = 1.5 > 1, so the CVaR no longer collapses. The enforced quantity becomes
(2/3)·c₍₁₎ + (1/3)·c₍₂₎ ≥ τ_eff, a tail average of the two worst samples, rather than the worst
sample. `test_cvar_does_not_collapse_when_eps_N_ge_1` already pins this. Knock-on effects:
- LADDER's rung/tier logic assumes the collapsed form.
- The random-recovery score `min_i CBC_i` is no longer the QP's own constraint.

**A4 — For α > 1, τ_eff = α·r_W/ε, not r_W/ε.** The DRCCP bound is componentwise on
|ū| = [1, α, |u_x|, |u_y|], so τ_eff = r_W·max(1, α)/ε. Verified numerically: the achieved
minimum CBC is 0.048 at α = 1.2 and 0.080 at α = 2.0. For α ∈ {1.2, 2.0}:
- `RecoveryLadder` defaults `tau = r_W/ε`, which would be **wrong**. It accepts a `tau=` kwarg,
  so the correct value can be passed without editing frozen or Phase-7 code.
- The T1 cap derivation v_cap = 1 − τ no longer holds either.

**A5 — k_v's real value is 0.05, and the code documents why.** At k_v = 1.0 the CLF slack
dominates the QP and SCS violates the action box by ~10⁻² (`drccp_controller.py:169-175`). The
specified grid {0.5, 1, 2} is 10–40× the reference and inside that documented failure regime.
k_v *is* an independent runtime parameter; algebraically it scales the CLF-slack weight by k_v².
A relative grid {0.5, 1, 2}×0.05 = {0.025, 0.05, 0.1} is the scientifically comparable
version.

**A6 — Unattainable escalation criterion.** The CLF row has an unbounded slack, so QP
infeasibility means exactly m* = max over the box of min_i CBC_i < τ_eff (collapsed regime). Every
random candidate satisfies ‖u‖₂ ≤ 1, so it lies inside the box. Hence **no candidate can ever
reach m ≥ τ_eff on an infeasible step**. An "m ≥ τ" criterion would escalate every event to
1.0 m/s. The only non-trivial criterion is **m ≥ 0** (nominal CBF holds on all samples, tier T1).

**A7 — The action bound is a per-axis box.** Both the env clip and the QP use |u_x|, |u_y| ≤ 1, so
the QP's own actions can reach ‖u‖ = √2. Random candidates with ‖u‖ = v_R ≤ 1.0 are always
admissible, but they cannot use the box corners the QP and LADDER can use. That is a mild handicap
for Random, and it is kept as specified.

**A8 — Only "infeasible" statuses trigger recovery.** `"infeasible" in status` triggers LADDER;
DCPError (h_crit < 0) and other solver failures fall back to frozen u = 0. On the 1M block these are
0–3 of ~900 events per condition. Random recovery should use the **identical trigger** for
parity.

**A9 — Paired fairness is guaranteed structurally.** The obstacle motion model reads only
`np_random`, obstacle positions and velocities, never the agent. So obstacle trajectories are
identical across controllers for every step until termination. The controller RNG must be
separate from `env.np_random` to preserve this.

**A10 — Where new code can live.** The frozen-area test fails on *any new file* under
`dr_control/`, and admitting one requires editing `exp7_0_frozen_manifest.PHASE7_PATHS`, a Week-5
file. New code therefore goes in a new top-level package. Also note that `results/week6/` already
exists and is the Sep-2 PPO+SR study, so the continuation must not use that name.

**A11 — The Week 5.5 artefacts are untracked in git and not in any manifest.** "Untouched" cannot
currently be verified. Plan: snapshot their SHA-256 into a continuation manifest plus a test.
Recommendation: commit them first. I will not commit unless asked.

**A12 — Code-path equivalence can be proven without touching 1 000 000.** Stage-3 validation
(`21_000+0..49`, both conditions, arm `C0_frozen`) and Stage-4 dev (`20_000+0..19`, `LADDER`) have
per-episode records. The new tunable policy at reference parameters must reproduce them
**exactly**, which is the byte/code-path gate.

---

## 3. Proposed architecture (nothing frozen is edited)

```
continuation/                       new package (sibling of generalization/)
  params.py        DRCBFParams dataclass + tau_eff(), collapsed(), validation
  policy.py        TunableDRCBFPolicy(DRCBFPolicy): builds ctrl with all params, source with k_scans
  random_recovery.py  RandomRecovery wrapper (same wrapping pattern as RecoveryLadder)
  harness.py       paired episode runner, per-step logging, manifest rows, worker pool
  seeds.py         the continuation seed table + guard asserts
experiments/week6_continuation/     cont_c0.py, cont_c1.py, cont_h1.py, cont_h2.py,
                                    cont_random_exp_r1.py, _r2.py, _r3.py, cont_final.py, report.py
tests/test_week6_continuation_*.py
results/week6_continuation/<stage>/ never overwritten (exists → abort)
MD_files/week6_continuation/        this audit, SELECTION_RULES.md (committed before H2/R3 selection), reports
```

Metrics and statistics are **imported from exp6 unchanged**: `episode_record`, `summarise`,
`mcnemar` (exact), `paired_ci` (10 000 reps, 95 %). New metrics are added alongside them, never in
place of them. Step-level rates use an episode-cluster bootstrap. Timing uses `perf_counter`
around `predict()` (exp6's definition), with QP time and recovery time logged separately.

---

## 4. Hyperparameter search table (FINAL — decisions D1–D6 applied, §8)

Chain: **C0-A → C1 → H1 → H2 → RANDOM_EXP_R1/R2/R3 → F**. C0-B (T1 + LADDER) is a historical
reference only: it is never tuned and none of its T1 parameters feed the new chain. **No T1 cap in
F-B, F-C, F-D** (D2); all three share the identical tuned base controller.

| stage | parameter | values | notes |
|---|---|---|---|
| C1 | N (`k_scans`) | {1, 3, 5} | diagnostic only; N = 5 kept for H1 unless evidence is strong |
| H1 | α | {0.2, 0.4, 0.8, 1.2, 2.0} | τ_eff = r_W·max(1,α)/ε logged per cell |
| H1 | r_W | {0, .001, .002, .004, .008, .012} | |
| H1 | ε | {0.05, 0.10, 0.20} = **collapsed regime** (main search) | |
| H1 | ε | {0.30} = **non-collapsed regime**, run and reported SEPARATELY | never shortlisted into the main chain (D3) |
| H1 | shortlist | best 5–10 distinct **(α, τ_eff, CVaR regime)** classes, collapsed regime only | each class represented by its lowest-r_W cell; duplicates reported as a degeneracy check |
| H2 | λ_V | {0.25, 0.5, 1.0, 2.0, 4.0} | |
| H2 | k_v | {0.025, 0.05, 0.10}; reference 0.05 | D4. {0.5, 1, 2} is NOT used |
| RANDOM_EXP_R1 | v_R, K, H | 0.2, 8, 1 | best-of-K randomly rotated set |
| RANDOM_EXP_R2 | V_R ladder | {0.2, 0.4, 0.6, 0.8, 1.0} | accept first speed with m ≥ 0; else best over all speeds (D5) |
| RANDOM_EXP_R3 | K × H × speed schedule | {4, 8, 16} × {1, 3, 5} × 5 schedules = **45 configs** | see below |

**R3 speed axis (5 schedules), fixed before R1 is run.** If RANDOM_EXP_R2 is not worse than
RANDOM_EXP_R1 under the §6 rule on the R2 block, the axis is the *escalation start speed*
v₀ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}: the ladder is {v ≥ v₀} with m ≥ 0 acceptance. So v₀ = 0.2 is the
R2 schedule and v₀ = 1.0 is fixed 1.0 m/s. Otherwise the axis is *fixed speed*
v ∈ {0.2, 0.4, 0.6, 0.8, 1.0} with no escalation. Both readings have 5 values, so the budget is the
same either way.

## 5. Seed allocation (all ranges verified unused; code, results JSON and training all < 10 000 000)

| block | base | episodes | purpose |
|---|---|---|---|
| existing dev / val | 20 000 / 21 000 | — | **read-only**, equivalence gate only (A12) |
| existing final | 1 000 000 | — | **never touched** |
| Week 5.5 | 7 000 000 / 8 000 000 / 9 000 000 | — | **never touched** |
| C0 | 10 000 000 | +0..99 | baseline |
| C1 | 10 100 000 | +0..99 | buffer diagnostic |
| H1 screen | 10 200 000 | +0..19 | coarse search |
| H2 validation | 10 300 000 | +0..49 | fine search, final tuned selection |
| RANDOM_EXP_R1 | 10 400 000 | +0..99 | paired with Tuned on same seeds |
| RANDOM_EXP_R2 | 10 500 000 | +0..99 | |
| RANDOM_EXP_R3 screen | 10 600 000 | +0..19 | |
| RANDOM_EXP_R3 validation | 10 700 000 | +0..99 | |
| reserved (nominal+random ablation) | 10 800 000 | — | unused unless requested |
| **FINAL F (sealed)** | **10 900 000** | +0..199 | only `cont_final.py` may read it; a test enforces this |
| controller RNG | `default_rng([0x52434246, episode_seed])` | — | independent of `env.np_random` |

The same seed index is used in both motion conditions, as exp6 did.

## 6. Pre-declared definitions (defaults; written to SELECTION_RULES.md before use)

- **Infeasible step:** frozen-QP status contains "infeasible" (the LADDER trigger, A8).
- **Infeasibility event:** a maximal run of consecutive infeasible steps; its duration is the run length.
- **Random recovery:** the QP still runs every step, so scans and the tracker are always updated.
  - On an infeasible step, K candidates are drawn at angles φ + 2πj/K with φ ~ U(0, 2π). Each is
    scored on the QP's own `xi_kept` by m_j = min_i CBC_i.
  - **Selection:** the argmax candidate is executed for H steps, strict persistence (the QP is
    still solved for logging only).
  - **State:** `prev_u` is set to the executed action, mirroring LADDER.
- **R2 escalation:** try speeds in ascending order and accept the first whose best candidate has
  m ≥ 0. At 1.0 m/s with none acceptable, execute the best candidate over **all** speeds. There is
  no LADDER fallback. The best margin at every speed is logged.
- **Recovery success:** no collision during the persistence window, **and** the QP is feasible on
  the first step after it.
- **P(collision | infeasibility):**
  - (i) episode-level: P(collision | the episode had ≥ 1 infeasible step);
  - (ii) event-level: collision within 10 steps (1 s) of an event's start;
  - (iii) exp6's `collided_after_infeasible`.
- **Selection (H1, H2, R3):** lexicographic on **pooled** (fixed + randomized) metrics, in this order:
  1. collision rate ↓
  2. success rate ↑
  3. timeout rate ↓
  4. mean true min-clearance ↑
  5. infeasibility rate ↓ (H1/H2) or mean step time ↓ (R3)
  - Fixed, randomized and pooled metrics are always reported side by side.
  - A candidate must pass all of the gates below to be ranked at all.
  - **G1 (pooled eligibility):** pooled timeout ≤ 10 % **and** pooled success ≥ reference success
    − 5 pp.
  - **G2 (condition-specific robustness):** in **each** condition separately, success ≥ reference
    − Δ **and** collision ≤ reference + Δ, where Δ is:
    - 10 pp at n ≥ 50 episodes per condition (H2, R3 validation);
    - 15 pp at n = 20 (H1 screen, R3 screen), i.e. 3 episodes, so that one or two episodes of noise
      do not trigger a rejection.
  - **Reference** = the base controller run on **the same seed block**, paired:
    - H1: C0-A, which is the grid's own (0.4, 0.004, 0.1) cell.
    - H2: a C0-A anchor arm on the H2 block.
    - R3 validation: Tuned-CLF-DR-CBF on the R3 validation block.
    - R3 screen: G1's timeout clause only, because no paired reference exists on the screen block.
  - Every rejection is written to the stage report with the gate and the numbers. **A candidate
    that fails G2 is never selected silently**, even if it would win on the pooled ranking.

## 7. Budget (measured ~140 steps × ~37 ms ≈ 5.2 s per episode, single core)

All counts are **arms × episodes × 2 motion conditions**.

| stage | derivation | episodes, both conditions |
|---|---|---|
| C0 | 2 arms (C0-A, C0-B) × 100 × 2 | 400 |
| C1 | 3 N values × 100 × 2 | 600 |
| H1 screen | 120 cells (5 α × 6 r_W × 4 ε) × 20 × 2 | 4 800 |
| H2 | ≤ 10 classes × 5 λ_V × 3 k_v × 50 × 2, plus a C0-A anchor of 50 × 2 | ≤ 15 100 |
| RANDOM_EXP_R1 | 2 arms (Tuned, R1) × 100 × 2 | 400 |
| RANDOM_EXP_R2 | 3 arms (Tuned, R1, R2) × 100 × 2 (R1 is re-run so that R2 vs R1 is paired) | 600 |
| RANDOM_EXP_R3 screen | **45 configs × 20 × 2** | **1 800** |
| RANDOM_EXP_R3 validation | **(4 shortlisted + Tuned reference) × 100 × 2** | **1 000** |
| F | 4 arms × 200 × 2 | 1 600 |
| **Total** | | **≤ 26 300** |

**R3 check:** 1 800 + 1 000 = **2 800 = 1 400 per condition**. This matches the earlier figure,
which was 45 × 20 + 5 × 100 per condition. The shortlist is now 4 rather than 5 so that the paired
Tuned reference fits inside the same 2 800. The R3 screen has no paired reference (see G2).

**Overall:** ≈ 26 300 episodes ≈ 38 core-hours, which is **≈ 5 h wall time on 8 of the 16
cores**. The machine is shared, so the cap is 8 workers. If H1 shortlists fewer than 10 classes, H2
shrinks proportionally. No stage may exceed its row: if a stage would, its screen or validation
size is reduced and the reduction is recorded, rather than the budget being exceeded.

---

## 8. Decisions — RESOLVED (2026-09-10)

| id | decision |
|---|---|
| D1 | APPROVED: both `fixed` and `randomized` at every rollout stage, paired within each condition. Selection is pooled, subject to G2 (§6); fixed, randomized and pooled are all reported |
| D2 | APPROVED: no T1 in F-B, F-C, F-D. C0-B stays the exact historical T1 + LADDER, is never tuned, and feeds nothing forward |
| D3 | APPROVED: the full grid is run; shortlisting dedupes on (α, τ_eff, CVaR regime). ε = 0.30 is a separate non-collapsed regime; the main chain stays in the collapsed regime |
| D4 | APPROVED: k_v ∈ {0.025, 0.05, 0.10}, reference 0.05 |
| D5 | APPROVED: accept at m ≥ 0; at 1.0 m/s, execute the best candidate over all speeds; no LADDER fallback |
| D6 | APPROVED and extended: G1 + G2 (§6) |
| gate | **Mandatory before H1**: the tunable policy at (0.4, 1.0, 0.004, 0.1, 0.05, N = 5) must reproduce the frozen development/validation records exactly (A12). On failure: stop and diagnose |

The option list below is kept for the record.

- **D1 — Motion condition.** Run every stage in both `fixed` and `randomized` (the Phase-6
  canonical pair) and select on pooled results? *(recommended)* Or run `randomized` only?
- **D2 — T1 cap in the tuned arms.** Recommended: no T1 in F-B, F-C or F-D, so the Tuned arm has
  the original structure and C vs D differ **only** in the recovery mechanism. The alternatives
  confound H4:
  - T1 in F-C only (the literal "existing updated" recipe) makes C vs D differ in the barrier
    source too.
  - The 0.96 cap also stops being derived once τ_eff ≠ 0.04 (A4).

  C0-B keeps T1 + LADDER exactly, as the historical reference.
- **D3 — (r_W, ε) degeneracy.** Run the grid as specified (it is cheap, and the duplicates
  empirically confirm A2), but **shortlist over distinct (α, τ_eff) classes** *(recommended)*.
  Report ε = 0.3 as a separate, explicitly non-collapsed regime. The alternative is to exclude
  ε = 0.3.
- **D4 — k_v grid.** Use the relative grid {0.025, 0.05, 0.1} *(recommended)*. The alternatives
  are the literal {0.5, 1, 2}, which A5 shows is numerically unsafe, or omitting k_v.
- **D5 — R2 acceptance criterion.** Accept at m ≥ 0 *(recommended; m ≥ τ is impossible by A6)*.
  The final fallback is the best candidate found.
- **D6 — Eligibility guard** as in §6 *(recommended)*, or pure lexicographic ranking.
