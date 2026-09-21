# Week 6 tracks — shared repository audit (both tracks)

**All claims below are VERIFIED FROM CODE/LOGS unless marked otherwise.**

## State at the start of this work

| item | value |
|---|---|
| branch | `Week6` |
| commit | `07d74ca` "Week 6 Continuation: tuned CLF-DR-CBF and Random-CLF-DR-CBF on M0" |
| working tree | clean of tracked modifications; untracked: `MD_files/pursuit/`, `continuation/pursuit/`, `experiments/pursuit/`, `results/pursuit/`, `tests/test_pursuit.py` |
| timestamp | 2026-09-12 |

## Authoritative pursuit documents (actual paths, located not assumed)

- `MD_files/pursuit/PURSUIT_AUDIT.md` — the audit of the pursuit results
- `MD_files/pursuit/PURSUIT_RESULTS_v2.md` — corrected results
- `MD_files/pursuit/PURSUIT_RESULTS.md` — **original, preserved unchanged** (md5 `65180116797ea9413e7e071ab887481d`, recorded here so any later change is detectable)
- `MD_files/pursuit/P0_AUDIT_AND_DESIGN.md` — pursuit design/audit note

## Canonical environment — confirmed against `config.json` + `experiments/exp6_ppo_comparison.make_env`

10×10 world · 5 static rectangles · 6 dynamic circles at 0.675 m/s · 24-ray LiDAR range 5.0 ·
agent radius 0.3 · obstacle radius 0.3 · dt 0.1 · max 500 steps · single integrator ·
per-axis |u| ≤ 1. **Matches the prompt's list with one clarification:** `config.json`'s
`dynamic_obstacles.randomize` is `false`, but the canonical stochastic condition is selected by the
*caller* (`make_env(randomize=True)` / `randomize_dynamic_obstacles=True`), which builds
`SmoothStochasticMotion`. Both Week 6 tracks pass `True`, asserted in code and tests.

## Frozen interfaces new code may call (no frozen file is edited)

| interface | file | used for |
|---|---|---|
| `DRCBFPolicy` / `predict()` | `dr_control/policy.py` | frozen control path, never overridden |
| `ClfCbfDrccpController.generate_controller` | `dr_control/drccp_controller.py` | frozen QP; wrappable by assignment (the pattern `RecoveryLadder` and `RandomRecovery` already use) |
| `EstimatedLidarBarrierSource` | `dr_control/estimated_cbf.py` | barrier rows; subclassable (`ProjectionCappedSource`, `HunterAugmentedSource` precedent) |
| `LidarVelocityTracker` | `dr_control/velocity_tracker.py` | multi-target tracker: `.tracks` list of `Track(id, position, velocity, confirmed, age, hits, misses)` |
| `cast_rays(origin, …)` | `robot_env/lidar_core.py` | arbitrary-origin LiDAR, already used for the hunter's own sensor |
| `TunableDRCBFPolicy`, `build_arm`, `attach_recovery` | `continuation/policy.py` | frozen tuned + random construction |
| `RandomRecovery` | `continuation/random_recovery.py` | frozen recovery, unmodified |
| `PursuitEnv`, `HunterController`, `HunterAugmentedSource` | `continuation/pursuit/` | Week-6 pursuit layer (new, not frozen) |

Frozen configs, hash-verified each run: `results/week6_continuation/H2/tuned_frozen.json`
(α 0.8, r_W 0.012, ε 0.1, k_v 0.10, N 5, τ_eff 0.12) and `.../R3/random_frozen.json`
(K 16, H 1, speeds {0.8, 1.0}, accept m ≥ 0).

## Directory separation for the two tracks (no shared code, no shared metrics)

```
Track A   continuation/perception/   experiments/week6_tracka/   results/week6_tracka/   MD_files/week6_tracka/
Track B   continuation/supervisor/   experiments/week6_trackb/   results/week6_trackb/   MD_files/week6_trackb/
```

Seed blocks, fresh and disjoint from every earlier block (navigation 10 000 000–10 999 999,
pursuit 11 000 000–11 299 999):

| block | range | use |
|---|---|---|
| Track B dev | 12 000 000 + | B0 diagnosis, risk-threshold selection |
| Track B eval | 12 100 000 + | factorial evaluation |
| Track A dev | 13 000 000 + | perception tests, pilot |
| Track A eval | 13 100 000 + | paired evaluation |

## Known pre-existing failure, NOT touched

`tests/test_week4_env.py::test_legacy_path_is_bit_identical_to_main` (4 cases) fails because the
`main` branch ref it replays moved onto the merged code on 2026-09-11. It is inside the Phase-7
frozen manifest and is left alone, per instruction. Any new failure in these tracks is reported
separately.
