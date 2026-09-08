# Week5-Phase7 / Stage 0 — regression substrate and trace corpus

**Gate G0: PASS.** No Phase 1–6 file was modified. No controller change was made. Stage 0 only
observes; the frozen system is byte-for-byte what it was.

---

## 1. What Stage 0 delivered

| deliverable | path | what it is |
|---|---|---|
| frozen-file manifest | `MD_files/week5/phase7/FROZEN_MANIFEST.sha256` | sha256 of 60 Phase 1–6 files |
| manifest tool | `experiments/exp7_0_frozen_manifest.py` | `--write` / `--verify` |
| regression gate | `tests/test_frozen_phase1_6.py` | 12 tests, run on every `pytest` |
| trace corpus | `results/week5_phase7/stage0_trace_corpus/` | 6 parts, 69 226 steps, 31 MB |
| corpus recorder | `experiments/exp7_0_trace_corpus.py` | wraps the frozen controller, edits nothing |
| corpus gate | `tests/test_dr_control_phase7_stage0.py` | 18 tests |

Full suite: **194 passed** (164 frozen + 30 new), 80 s.

---

## 2. The frozen manifest

60 version-controlled files across `robot_env/`, `evaluation/`, `dr_control/`,
`optional_navigation/`, `experiments/exp0..6*`, `tests/test_dr_control*`,
`tests/test_week4_env.py`, `results/phase1..6/*.json`, and `config.json`.

Excluded deliberately, each for a stated reason: `requirements.txt` (Phase 7 may add a
dependency, and an entry expected to change teaches people to ignore failures); `*.log`
(gitignored by repo convention, so absent from a fresh clone — the JSON results carry the same
numbers and *are* covered); Phase 7's own files, recognised by `phase7` / `exp7_` in the
basename.

The gate has three teeth: no listed file may change, none may go missing, and **no new file may
appear** in the frozen area — Phase 7 output belongs in `results/week5_phase7/` and
`MD_files/week5/phase7/`.

## 3. The trace corpus

Phase 6 configuration — the real `RobotNavEnv` under the frozen `DRCBFPolicy` — in both
obstacle-motion conditions, at three seed tiers.

| part | episodes | steps | solved | infeasible | other | success | collision | timeout | ms/step |
|---|---|---|---|---|---|---|---|---|---|
| dev/fixed | 20 | 2 046 | 1 984 | 62 | 0 | 16 | 4 | 0 | 28.67 |
| dev/randomized | 20 | 1 861 | 1 777 | 84 | 0 | 11 | 9 | 0 | 29.26 |
| validation/fixed | 50 | 5 559 | 5 380 | 179 | 0 | 41 | 9 | 0 | 28.45 |
| validation/randomized | 50 | 5 689 | 5 471 | 218 | 0 | 33 | 17 | 0 | 29.13 |
| final/fixed | 200 | 27 039 | 26 137 | **902** | 0 | 171 | 28 | 1 | 27.39 |
| final/randomized | 200 | 27 032 | 26 138 | **892** | 2 | 145 | 55 | 0 | 28.80 |

Seeds: dev `20 000`, validation `21 000`, final `EVAL_SEED_BASE = 1 000 000`. The final tier is
a *recording* of frozen behaviour on the reserved block, used only for equivalence checking; no
method is selected and no parameter is tuned from it.

Per step the corpus stores the control-path inputs (`p`, `gamma`, full pre-truncation `xi`,
`u_prev`, `u_nom`), the frozen outputs (`u`, status, `delta`, `h_crit`, objective weights,
`box_overshoot`, timings), the kept-sample subset, per-sample provenance (`sample_age`,
`sample_track_id`), and derived `min_cbc_kept` / `min_cbc_all`. Ground truth (`truth_clearance`,
`truth_obs_pos`, `truth_obs_vel`) is recorded in a separately named channel for Stage 5's
estimator metrics and never reaches the controller — a test asserts every array is either a
known control-path key or `truth_*`-prefixed.

**Enriched infeasible corpus:** all 2 337 infeasible steps pooled across tiers and conditions,
so Stage 1's L1 status-agreement gate is tested where it matters rather than where it is easy.

---

## 4. Gate evidence

### 4.1 The corpus reproduces Phase 6 exactly

Not "closely" — exactly, on every headline number, from an independent 200-episode re-run:

| metric | published `results/phase6/` | Stage 0 corpus |
|---|---|---|
| fixed: success | 0.855 | 171/200 = 0.855 |
| fixed: collision | 0.140 | 28/200 = 0.140 |
| fixed: timeout | 0.005 | 1/200 = 0.005 |
| fixed: `n_infeasible` | 902 | 902 |
| randomized: success | 0.725 | 145/200 = 0.725 |
| randomized: collision | 0.275 | 55/200 = 0.275 |
| randomized: `n_infeasible` | 892 | 892 |

This is the strongest available evidence that the recorder observes the frozen system without
perturbing it.

### 4.2 Every recorded step replays bit-identically

1 800 steps across the six parts (300 per part) were re-solved from their recorded inputs alone
— a fresh `ClfCbfDrccpController`, `prev_u` restored, `(p, gamma, xi)` replayed — and the action
matched **bit for bit** in every case. **0 mismatches.** The test repeats this independently on
120 further dev steps.

This is the item Stage 1 depends on: an accelerated controller can only be compared against a
record that reconstructs the frozen one from its own contents.

### 4.3 The `tau` identity holds across 69 226 real steps

`min_i CBC_i` over the kept samples, restricted to `status == "optimal"`:

| part | min CBC \| solved | deviation from `tau = 0.04` |
|---|---|---|
| dev/fixed, dev/randomized | 0.039906 | −9.41e−5 |
| validation/fixed | 0.039909 | −9.07e−5 |
| validation/randomized | 0.039906 | −9.44e−5 |
| final/fixed | 0.039895 | −1.05e−4 |
| final/randomized | 0.039899 | −1.01e−4 |

Worst deviation −1.05e−4, one-sided, 0.26 % of `tau` — SCS first-order tolerance, and the same
range Phases 2–6 already recorded (0.03989–0.03996). Directions 1 and 2 both rest on this
identity and it now has 69 226 empirical confirmations in addition to its proof.

### 4.4 The identity is also tested symbolically, with a negative control

`tests/test_frozen_phase1_6.py` verifies both directions of
`eps*N < 1  =>  DR constraint <=> min_i CBC_i >= r_W||u_bar||_inf/eps` against the CVXPY
feasible set, with per-solver tolerance (SCS 1e−4, CLARABEL 1e−6 — measured worst deviations
1.2e−6 and 1.9e−8, with `t* == tau` and `sum_i s_i == 0` at the optimum, exactly the structure
the derivation predicts). A **negative control** confirms the identity *fails* for `eps*N >= 1`,
so the reduction cannot be silently applied outside its precondition. The test asserts both the
feasible and infeasible branches were exercised, so it cannot become vacuous.

### 4.5 Sample provenance is validated, not assumed

`sample_age == row index` is asserted directly (the frozen buffer is newest-first and emits one
row per scan). The owning track id had to be recovered by re-deriving the nearest-point argmin,
since the frozen method does not return it; that duplication is checked rather than trusted —
the recomputed `h` must equal the frozen `h` exactly, and **`geometry_recompute_mismatch = 0`
across all 69 226 steps**.

---

## 5. Two corrections made during Stage 0, both in Stage 0's own new code

Neither touched the frozen system; both are recorded because the methodology requires that no
fix be silent.

**C1 — my own test tolerance was tighter than the solver's accuracy.**
The first version of the `tau`-identity test asserted agreement to 1e−5 and failed on one seed.
Investigation showed the identity holds and the structure is exactly as derived (`t* = 0.0400 =
tau`, `sum_i s_i ≈ 0`); the residual was −1.2e−6 with SCS and −1.9e−8 with CLARABEL. The fix was
to the **test**, not to the claim: per-solver tolerances, which keep "the identity is wrong"
distinguishable from "SCS is a first-order solver". A single loose tolerance would have hidden
the difference.

**C2 — `min_cbc_solved` was computed over the wrong set.**
The recorder initially defined "solved" as *not infeasible*. `final/randomized` contains 2
`optimal_inaccurate` steps, which the frozen controller treats as failures and answers with
`u = 0`; pooling them with solved steps reported a floor of **−0.371** instead of 0.03990. The
recorded `.npz` arrays were correct throughout — only the derived JSON summary was wrong — so
the corpus was re-summarised (`--refresh-meta`), not re-recorded. "Solved" now means
`status == "optimal"` exactly, matching `EpisodeDiagnostics`, and a third bucket
`n_non_optimal_non_infeasible` is reported so these steps are visible rather than absorbed into
either neighbour. A test asserts the three buckets sum to the step count.

C2 is worth carrying forward: Stage 1's L1 gate compares *feasibility status*, and this shows
the status space has at least four values (`optimal`, `optimal_inaccurate`, `infeasible`,
`infeasible_inaccurate`) that must not be collapsed. Phase 6's own `n_infeasible` counts
`"infeasible" in status`, so it includes `infeasible_inaccurate` — the corpus matches that
convention exactly, which is why 902 and 892 agree.

---

## 6. Scope decision, flagged

The plan said Stage 0 would record "the Phase 5 and Phase 6 configurations". **It records the
Phase 6 configuration only.** Reason: that is the configuration Stage 7 evaluates and Stage 2
diagnoses, so it is the corpus every gate consumes; adding the Phase 5 standalone harness would
roughly double the corpus and the code surface without serving a gate. If a later stage needs
standalone-harness traces they are added then, as a plan amendment under §11.1.4. The plan text
has been updated to say so rather than leaving the discrepancy implicit.

---

## 7. Observations for Stage 1 and Stage 2 (recorded, not acted on)

* **Latency:** 27.4–29.3 ms/step mean, consistent with the 27.6–28.9 ms of Phase 5. Stage 1 will
  break this into canonicalization and solve and report worst-case, not just mean. No
  acceleration work has begun.
* **Infeasibility rate:** 3.0 % (fixed) and 3.3 % (randomized) of steps at the final tier;
  3.0–3.8 % at dev/validation. Enough events for Stage 2's diagnosis.
* **The `optimal_inaccurate` bucket is tiny but non-empty** (2 of 69 226). Stage 1's status
  agreement must treat it as its own category.
* Nothing here bears on the Finding C staleness **hypothesis** either way. Stage 2 tests it; the
  corpus now contains the `sample_age` and `sample_track_id` fields that test needs.

---

## 8. Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m experiments.exp7_0_frozen_manifest --verify
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m experiments.exp7_0_trace_corpus \
    --tiers dev validation final --conditions fixed randomized --replay-checks 300
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q
```

Corpus build: 74 min wall, single-threaded. Every `.npz` carries a sha256 in its `meta_*.json`
and the corpus gate re-verifies them.

**Stage 0 is complete. Stage 1 has not been started: no accelerated controller exists, and no
Phase 7 controller code has been written.**
