# FINAL apples-to-apples evaluation — updated DR-CBF on the Phase-6 canonical harness

**This is an evaluation/reproduction, not a new experiment.** No controller logic was written or
modified, nothing was tuned, no seed or episode count changed, and no experimental branch is
opened from the outcome.

Artefacts: `results/week5_phase7/final_comparison/final_{fixed,randomized}.json`.
Runner: `experiments/exp8_final_comparison.py`. Controller commit basis: `91bb969`.

---

## 1. Provenance verification

The canonical basis is **`experiments/exp6_ppo_comparison.py`** with
`results/phase6/exp6_{fixed,randomized}.json`, which evaluated the initial DR-CBF **and all
twelve PPO checkpoints** inside one harness, on one seed block, through one metric function,
storing per-episode records for every arm. Those records were **read, not recomputed**; **PPO was
not rerun**.

| Item | Phase-6 canonical setup | This evaluation | Match |
|---|---|---|---|
| Map | `config.json`, world 10.0 | same file, same loader | ✅ |
| Static obstacles | 5 rects `[[2,2,.5,1.5],[7,3,1,.5],[4.5,7,.5,1],[2,7.5,1.5,.5],[7.5,7,.5,1.5]]` | identical | ✅ |
| Dynamic obstacles | 6, via `make_env` override | `make_env`, unchanged | ✅ |
| Motion model | fixed = `DeterministicMotion`; randomized = `SmoothStochasticMotion` (OU) | identical | ✅ |
| Dynamic speed | `obstacle_speed = 0.675` scalar, both conditions | identical | ✅ |
| LiDAR | 24 rays, range 5.0, noiseless | identical | ✅ |
| Agent dynamics | single integrator, `max_speed = 1.0`, `dt = 0.1`, action box ±1 | identical | ✅ |
| Episode horizon | `max_steps = 500` | identical | ✅ |
| Seeds | `EVAL_SEED_BASE = 1_000_000 … +199` | identical, asserted at runtime | ✅ |
| Episode count | 200 paired per condition | 200 | ✅ |
| Collision definition | env `info["collision"]`, `collision_type ∈ {dynamic, static, wall}` | same env, untouched | ✅ |
| Success definition | env `info["success"]` | same env, untouched | ✅ |
| Metric computation | `exp6.episode_record` / `exp6.summarise` | **those functions, imported unchanged** | ✅ |
| Statistics | `exp6.mcnemar` (exact) + `exp6.paired_ci` (10 000 bootstrap) | **imported unchanged** | ✅ |
| Pairing | by seed, arm-by-arm | asserted per arm at runtime | ✅ |

**Week-7 provenance note (context only, never a comparator).** `results/week7/SUMMARY.txt`
evaluated the same `_v4` checkpoints with `obstacle_speed_range = [0.6, 0.75]` — speed sampled
per obstacle — not the scalar 0.675 that `exp6` uses. That is the whole reason its aggregate PPO
figures (e.g. PPO fixed→fixed 0.602 ± 0.035 success, 0.395 ± 0.035 collision) differ from the
Phase-6 per-checkpoint numbers below. **No inference in this report uses week7.**

---

## 2. Integrity checks (executed immediately before the run, both conditions)

| check | result |
|---|---|
| frozen sha256 manifest | **PASS**, 60 files |
| full regression suite | **288 passed** |
| `EVAL_SEED_BASE` | **1 000 000**, asserted unchanged |
| barrier source | `ProjectionCappedSource` |
| velocity tracker | `LidarVelocityTracker` (**not** the Stage-5 E1 tracker) |
| `v_cap` | **0.96** |
| recovery mode | `ladder`, α = 0.4, τ = 0.04 |
| planner | `use_planner = True` (A* → carrot → DR-CBF) |
| SB3 / torch imports reachable from `dr_control/` | **none** |
| `models/` path references in `dr_control/` | **none** |
| `obs[28:52]` corruption invariance | **action bit-identical** |
| PPO rerun | **none** — Phase-6 records read from disk |

Controller files are **byte-identical to `91bb969`** (`git diff 91bb969 HEAD` empty for
`drccp_controller.py`, `policy.py`, `capped_velocity.py`, `recovery.py`, `velocity_tracker.py`,
`estimated_cbf.py`, `planner.py`). The only changed file, `policy_phase7.py`, appended the
Stage-5 E1 arm; the `T1_projection_cap` path is unchanged and pinned by
`test_existing_arms_keep_the_frozen_tracker`.

---

## 3. Exact configuration evaluated

```
Phase7Policy(arm="T1_projection_cap", v_cap=0.96, use_planner=True)
  + RecoveryLadder(pol.ctrl, mode="ladder")        # committed Stage-4 LADDER, 91bb969
```

V0 (CVXPY/SCS) solver; α = 0.4, r_W = 0.004, ε = 0.1, τ = 0.04; frozen objective, sample
selection and LiDAR pipeline; `u = 0` fallback unchanged (the ladder acts only on DR-infeasible
steps and is inert otherwise). Policy and ladder constructed per episode, exactly as
`exp7_4_recovery.py` does. **Not used:** E1, IMM/CT, uncertainty, D-oracle, R1-alone, R2-alone,
R1.5, or any privileged ground truth.

---

## 4. Results — full Phase-6 headline metric set

### FIXED obstacles (200 paired episodes from `EVAL_SEED_BASE`)

| metric | **updated DR-CBF** | initial DR-CBF | ppo_fixed s0/s1/s2 | ppo_sr_fixed s0/s1/s2 |
|---|---|---|---|---|
| success | **0.925** | 0.855 | .645/.560/.600 | .565/.535/.545 |
| collision | **0.065** | 0.140 | .355/.440/.390 | .430/.450/.450 |
| dynamic collision | **0.065** | 0.140 | .255/.335/.355 | .310/.345/.355 |
| static collision | **0** | 0 | .095/.100/.035 | .120/.095/.095 |
| wall collision | **0** | 0 | .005/.005/0 | 0/.010/0 |
| timeout | 0.010 | 0.005 | 0/0/.010 | .005/.015/.005 |
| min clearance (mean) | **0.2711** | 0.2452 | .1142/.0825/.1049 | .0832/.0815/.0741 |
| min clearance (worst) | −0.0580 | −0.0556 | −.1465/−.1377/−.1385 | −.1485/−.1501/−.1462 |
| SPL | **0.8041** | 0.7610 | .497/.428/.466 | .465/.417/.439 |
| collisions while feasible | 8 | 4 | 71/88/78 | 86/90/90 |
| collisions after infeasible | **5** | 24 | 0 | 0 |
| infeasible steps | **383** | 902 | – | – |
| solver failures | 384 | 902 | – | – |
| planner failures | 0 | 0 | – | – |
| mean ‖u − u_nom‖ | 0.5736 | 0.5778 | – | – |
| `min_cbc_solved` | 0.03989 | 0.03989 | – | – |
| mean step time (ms) | 37.07 | 35.99 | ≈0.545 | ≈0.549 |

### RANDOMIZED (OU) obstacles

| metric | **updated DR-CBF** | initial DR-CBF | ppo_random s0/s1/s2 | ppo_sr_random s0/s1/s2 |
|---|---|---|---|---|
| success | **0.815** | 0.725 | .565/.635/.600 | .625/.615/.505 |
| collision | **0.180** | 0.275 | .350/.275/.380 | .330/.365/.335 |
| dynamic collision | 0.175 | 0.275 | .215/.180/.220 | .220/.270/.200 |
| static collision | **0.005** | **0** | .115/.080/.125 | .100/.085/.120 |
| wall collision | 0 | 0 | .020/.015/.035 | .010/.010/.015 |
| timeout | 0.005 | 0 | .085/.090/.020 | .045/.020/.160 |
| min clearance (mean) | **0.2043** | 0.1815 | .1092/.1417/.0950 | .1137/.0920/.0902 |
| min clearance (worst) | **−0.0702** | −0.0593 | −.129/−.120/−.173 | −.101/−.162/−.130 |
| SPL | **0.6757** | 0.6257 | .397/.483/.409 | .465/.429/.340 |
| collisions while feasible | 24 | 18 | 70/55/76 | 66/73/67 |
| collisions after infeasible | **12** | 37 | 0 | 0 |
| infeasible steps | **573** | 892 | – | – |
| solver failures | 576 | 894 | – | – |
| planner failures | 0 | 0 | – | – |
| mean ‖u − u_nom‖ | 0.6281 | 0.6177 | – | – |
| `min_cbc_solved` | 0.0399 | 0.0399 | – | – |
| mean step time (ms) | 42.08 | 36.43 | ≈0.536 | ≈0.543 |

### Secondary diagnostics (not Phase-6 headline metrics)

Rungs — fixed: R0 28 116, R1 83, R2 280, R2_lp 20. Randomized: R0 29 693, R1 139, R2 409,
R2_lp 25. Recovery rate **383/383** and **573/573** (100 %).
Guarantee tiers of the recovered actions — fixed **T0 = 0, T1 = 14, T2 = 369**; randomized
**T0 = 0, T1 = 18, T2 = 555**. **No recovered action is described as safe**: 96.3 % and 96.9 %
are T2 with `m < 0`, i.e. the DR forward-invariance guarantee is lost on those steps.

---

## 5. Stage-4 consistency check (not the authoritative result)

The canonical-harness run reproduces Stage 4's LADDER arm **exactly, to every recorded digit**:

| | Stage 4 (`91bb969`) | this run |
|---|---|---|
| fixed success / collision | 0.925 / 0.065 | **0.925 / 0.065** |
| fixed SPL / mean clearance / worst | 0.804124 / 0.271057 / −0.057997 | **identical** |
| fixed infeasible / after-infeasible / while-feasible | 383 / 5 / 8 | **identical** |
| randomized success / collision | 0.815 / 0.180 | **0.815 / 0.180** |
| randomized SPL / mean clearance / worst | 0.675718 / 0.204286 / −0.070172 | **identical** |
| randomized infeasible / after-infeasible / while-feasible | 573 / 12 / 24 | **identical** |
| tiers | fixed 14/369, randomized 18/555 | **identical** |

The pipeline is deterministic given the seed, so this is bit-level reproduction across two
independently written harnesses. **The canonical-harness result above remains the authoritative
final evaluation.**

---

## 6. Paired statistics — Phase-6 methodology, unchanged

Exact McNemar on discordant pairs plus a 10 000-resample paired bootstrap CI, both from `exp6`.
Sign convention is `exp6`'s: **positive diff = updated DR-CBF higher**. Threshold α = 0.05, as
Phase 6 used, fixed before the run.

### 6.1 Initial DR-CBF → updated DR-CBF (the before/after)

| condition | metric | diff | 95 % CI | n10 / n01 | p | |
|---|---|---|---|---|---|---|
| fixed | success | **+0.0700** | [+0.0250, +0.1150] | 18/4 | 0.0043 | **significant** |
| fixed | collision | **−0.0750** | [−0.1200, −0.0300] | 4/19 | 0.0026 | **significant** |
| fixed | dynamic collision | **−0.0750** | [−0.1200, −0.0300] | 4/19 | 0.0026 | **significant** |
| randomized | success | **+0.0900** | [+0.0400, +0.1400] | 23/5 | 0.0009 | **significant** |
| randomized | collision | **−0.0950** | [−0.1500, −0.0450] | 5/24 | 0.0005 | **significant** |
| randomized | dynamic collision | **−0.1000** | [−0.1550, −0.0500] | 5/25 | 0.0003 | **significant** |

### 6.2 Updated DR-CBF vs all twelve PPO checkpoints

**Fixed** — significant on success and collision against **all six** checkpoints
(p < 0.0001 throughout): success +0.280 to +0.390, collision −0.290 to −0.385. Dynamic collision
also significant against **all six**, −0.190 to −0.290.

| arm | success diff [CI] | collision diff [CI] | dyn-coll diff [CI] |
|---|---|---|---|
| ppo_fixed_seed0 | +0.280 [+0.210, +0.355] | −0.290 [−0.365, −0.220] | −0.190 [−0.255, −0.125] |
| ppo_fixed_seed1 | +0.365 [+0.285, +0.445] | −0.375 [−0.455, −0.295] | −0.270 [−0.345, −0.195] |
| ppo_fixed_seed2 | +0.325 [+0.250, +0.405] | −0.325 [−0.400, −0.245] | −0.290 [−0.365, −0.215] |
| ppo_sr_fixed_seed0 | +0.360 [+0.285, +0.435] | −0.365 [−0.440, −0.290] | −0.245 [−0.315, −0.175] |
| ppo_sr_fixed_seed1 | +0.390 [+0.315, +0.465] | −0.385 [−0.460, −0.310] | −0.280 [−0.355, −0.205] |
| ppo_sr_fixed_seed2 | +0.380 [+0.305, +0.455] | −0.385 [−0.460, −0.310] | −0.290 [−0.365, −0.215] |

**Randomized** — significant on success and collision against **all six**; **dynamic collision is
NOT resolved against five of six**.

| arm | success diff [CI], p | collision diff [CI], p | dyn-coll diff [CI], p |
|---|---|---|---|
| ppo_random_seed0 | +0.250 [+0.165, +0.340], <0.0001 | −0.170 [−0.255, −0.085], 0.0002 | −0.040 [−0.115, +0.035], **0.358 not resolved** |
| ppo_random_seed1 | +0.180 [+0.100, +0.260], <0.0001 | −0.095 [−0.175, −0.015], 0.0271 | −0.005 [−0.075, +0.065], **1.000 not resolved** |
| ppo_random_seed2 | +0.215 [+0.130, +0.300], <0.0001 | −0.200 [−0.280, −0.115], <0.0001 | −0.045 [−0.120, +0.030], **0.289 not resolved** |
| ppo_sr_random_seed0 | +0.190 [+0.105, +0.275], <0.0001 | −0.150 [−0.230, −0.065], 0.0006 | −0.045 [−0.120, +0.030], **0.289 not resolved** |
| ppo_sr_random_seed1 | +0.200 [+0.115, +0.280], <0.0001 | −0.185 [−0.265, −0.105], <0.0001 | −0.095 [−0.170, −0.020], 0.0183 significant |
| ppo_sr_random_seed2 | +0.310 [+0.220, +0.400], <0.0001 | −0.155 [−0.240, −0.070], 0.0005 | −0.025 [−0.100, +0.050], **0.609 not resolved** |

**This is the single most important qualification in the report.** Under OU motion the DR-CBF's
overall collision advantage over PPO comes substantially from **static and wall** collisions,
which the DR-CBF essentially never has (0.005 / 0) and PPO has at 0.080–0.125 and 0.010–0.035.
On the **dynamic** obstacles — the ones the DR-CBF's barrier is actually designed for — the
updated controller is **not statistically distinguishable from five of the six randomized PPO
checkpoints** at n = 200. Under fixed motion the dynamic advantage *is* significant against all
six.

---

## 7. Limitations

1. **The information asymmetry runs both ways and is unchanged from Phase 6.** PPO reads
   `obs[28:52]` — exact relative position *and* velocity of the six nearest dynamic obstacles —
   plus 10 M training steps. The DR-CBF reads none of that, but does get ego pose in the world
   frame and a **prior static map for A\***. Neither system is simply better informed. The
   DR-CBF's near-zero static/wall collision rate is plausibly attributable to the prior map, and
   §6.2 shows that is where much of the randomized-condition advantage lives.
2. **A significant collision reduction is not a safety guarantee.** 96.3 % / 96.9 % of recovered
   actions are tier **T2** with `m < 0`; **none reached T0**. The updated controller buys its
   empirical improvement partly by giving up the DR forward-invariance guarantee on infeasible
   steps. That trade-off is the Stage-4 finding and it carries into this evaluation unchanged.
3. **Worst-case clearance did not improve** — fixed −0.0556 → −0.0580, randomized −0.0593 →
   −0.0702, both slightly worse than the initial DR-CBF (still far better than every PPO arm).
   And **one static collision appeared** in randomized where the initial DR-CBF had none.
4. **The infeasibility drop (902 → 383, 892 → 573) is a closed-loop effect**, not proof that the
   underlying constraint conflicts were removed: recovered actions move the robot to states the
   initial controller never visited.
5. **Compute cost is ~70× PPO's** per step (37–42 ms vs ≈0.54 ms). Not a safety metric, but it is
   in the Phase-6 metric set and is reported.
6. **Three seeds per PPO arm.** Each checkpoint is compared individually and paired; no
   across-seed pooled test is claimed.
7. **Week7's aggregate PPO numbers are not comparable** to these, for the obstacle-speed reason
   in §1. They are context, not evidence.

---

## 8. Conclusion

**The updated DR-CBF reproduces the Phase-6 result and improves on it, on the identical
protocol.**

* **Reproduces:** the harness, seeds, environment, metric functions and statistics are Phase-6's
  own, and the initial DR-CBF arm is Phase-6's recorded episode data, not a re-run.
* **Improves over the initial DR-CBF:** success +0.070 / +0.090 and collision −0.075 / −0.095,
  all four **significant** under Phase-6's own paired test (p = 0.0043 / 0.0026 / 0.0009 /
  0.0005). Because the protocol is identical, this is a **clean before/after controller
  comparison**, not a harness artefact. The mechanism is visible and matches Stage 4: collisions
  after an infeasible step fall **24 → 5** and **37 → 12**, while collisions while feasible rise
  slightly (4 → 8, 18 → 24).
* **Still beats every PPO baseline on the headline metrics:** significantly higher success and
  significantly lower total collision rate against **all twelve checkpoints** in both conditions.
* **With one qualification that must travel with the claim:** on **dynamic** collisions under OU
  motion the updated DR-CBF is **not statistically distinguishable from five of the six**
  randomized PPO checkpoints. The overall randomized advantage is driven substantially by static
  and wall collisions, where the DR-CBF benefits from a prior map PPO does not have.

No claim here rests on a raw rate being higher; every claim is tied to the matched protocol and
the paired test. **This is the final apples-to-apples evaluation; no further experimental branch
is opened from it.**
