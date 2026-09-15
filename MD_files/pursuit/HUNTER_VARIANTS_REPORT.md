# Hunter Variants B1 / B1.5 / P2 / P2.5 — Implementation and Evaluation

Branch `Week6`, uncommitted. New report; `PURSUIT_RESULTS.md`, `PURSUIT_AUDIT.md`, `PURSUIT_RESULTS_v2.md`
are unchanged (PURSUIT_RESULTS.md md5 `65180116797ea9413e7e071ab887481d` verified).

## 1. Repository inspection (what exists, verified in code)

| Item | Location | Fact |
|---|---|---|
| Entry point (pursuit) | `continuation/pursuit/harness.py::run_episode`, scripts in `experiments/pursuit/` | one shared pre-step state → both controllers → `PursuitEnv.step_pursuit` |
| Week 4–6 map | `config.json` → `robot_env/robot_nav_env.py`, built by `make_pursuit_env` | canonical M0: 10×10, 5 rectangles, 6 dynamic discs r=0.3 @0.675 m/s `SmoothStochasticMotion`, stratified start/goal, target radius 0.6. The only Week 4–6 navigation map; used unchanged |
| Protag controller | `continuation/policy.py::TunableDRCBFPolicy` (frozen `DRCBFPolicy.predict`) + `RandomRecovery`; source `HunterAugmentedSource` | frozen Tuned CLF-DR-CBF (α=0.8, r_W=0.012, ε=0.1, k_v=0.10, N_keep=5) + Random recovery, A* carrot, known-state hunter row |
| CLF-DR-CBF QP | `dr_control/drccp_controller.py::ClfCbfDrccpController` | V = ½k_v‖p−γ‖², soft `∇V·u + λ_V V ≤ δ`; DRCCP CVaR constraint on ξ=[dh/dt, h, ∇h]; u=0 on any non-optimal solve |
| Existing hunter | `continuation/pursuit/hunter.py::HunterController` | γ = Protag position, same QP + tuned params, no recovery, no planner |
| LiDAR ray-cast | `robot_env/lidar_core.py::cast_rays`; `PursuitEnv.hunter_lidar` | 24 rays, 5 m; hits only the shapes passed in |
| LiDAR → CBF rows | `dr_control/estimated_cbf.py::EstimatedLidarBarrierSource` | 5-scan buffer, one row per scan (nearest surface point), dh/dt from `LidarVelocityTracker` |
| State / dynamics | `RobotNavEnv.step` | **holonomic single integrator**, p ← p + u·dt, per-axis box \|u_i\| ≤ 1.0 m/s, dt = 0.1 s, no heading, no acceleration limit; `agent_velocity` = last applied velocity |
| Update order | `run_episode` | pre-step state → Protag action → hunter action → Protag + obstacles move (frozen step) → hunter moves → outcomes (own-collision → capture → goal → timeout) |
| Capture | `PursuitConfig` | capture d ≤ 0.65 m (0.3+0.3+0.05); contact d < 0.60; hunter crash → disabled, episode continues |
| Evaluation / stats | `experiments/pursuit/p3_main.py` | Wilson CIs, paired bootstrap, exact McNemar |

## 2. Files

Added: `continuation/pursuit/hunter_policies.py`, `experiments/pursuit/hunter_variants.py`,
`tests/test_hunter_variants.py`, this report, `results/pursuit_hunter_variants/{smoke,main,logs}/`.

Modified (opt-in, defaults reproduce the audited code):
- `continuation/pursuit/env.py` — `hunter_lidar(include_protagonist=False)`.
- `continuation/pursuit/hunter.py` — `filter_radius=None` constructor option; `act(..., filter_reference=None)`; empty-buffer guard (u=0, counted; never fired).
- `continuation/pursuit/harness.py` — `run_episode(..., variant=None, trace=False)`; `_hunter_target`, `_variant_fields`.

Not modified: Protag controller, frozen `dr_control/`, `robot_env/`, map, obstacle motion, collision detection, dt, reward, capture semantics.

**Regression proof:** before any edit, two episodes (seeds 11 000 003, 14 999 990) were fingerprinted; after the edits
both `variant=None` and `variant=B1` reproduce every record field exactly (timings excluded). A test pins B1 ≡ audited hunter.

## 3. Direct pursuit CLF

γ_h(t) = p_e(t) (true pre-step Protag position). The frozen CLF gives V_h = ½k_v‖p_h − p_e‖², constraint
k_v(p_h − p_e)·u + λ_V V_h ≤ δ, nominal u_nom = v_max·unit(p_e − p_h) (zero when coincident). Because the robot is holonomic,
this is exactly the plan's form; no dynamics adaptation was needed.

**Moving target (deviation, documented, not a silent change):** the frozen QP has no slot for the −(p_h−p_e)·v_e term and its
CLF is soft. Adding it would change frozen mathematics, so it is not added; the target is refreshed every 0.1 s (receding
reference, as the Protag already does with its moving A* carrot). There is no convergence guarantee; capture is measured.

The hunter's barrier rows never include Protag (B1: not rendered; B1.5/P2.5: rendered then filtered). Tested with a
sensitivity check that an unfiltered source *does* produce the Protag row.

## 4. Predictive pursuit model

p̂_e(t+τ) = p_e + v_e τ + ½ a_g d_g τ², d_g = unit(g_e − p_e) (0 at the goal), v_e = true applied Protag velocity.
Integrated over ⌈τ/dt⌉ sub-steps; predicted velocity clipped per axis to Protag's box after each sub-step, position by the
trapezoid of sub-step velocities (exact closed form when unclipped; never exceeds v_max·τ per axis — a test caught an
initial version that overshot by 2.5 cm). Clipped into the arena; non-finite input → current position (counted; 0 fallbacks in
all runs). Recomputed every step from the pre-step state only. Offline error uses the realised position k=5 steps later.

Assumptions/limits: d_g is frozen over the horizon and points straight at the goal, whereas Protag follows an A* carrot and
avoids obstacles; the prediction does not model Protag's CBF, its reaction to the hunter, or its stop at the goal. Since Protag
has no acceleration limit, a_g only rotates/extends the velocity within the speed box (clipping bound in 41% of predictions).
γ is the predicted point; the filter reference is **not** (see §5).

## 5. LiDAR handling

- **Remove (B1, P2):** hunter scan cast without Protag (the audited scan). Privileged, virtual-only.
- **Filter (B1.5, P2.5):** Protag's body (r=0.3) is rendered in the hunter's scan (second cast, per-ray min; world unchanged,
  Protag's own scan unchanged). Raw ranges go unmodified to the velocity tracker. Surface-point candidates with
  ‖q − p_e‖ ≤ r_f are removed from the buffered scan **at push time**, against the true Protag position at that scan's instant
  (filtering the 5-scan buffer against the current position would leave 4 stale Protag rows). Reference = current true
  position, never the prediction (tested for P2.5).
- False positives: an environmental point within r_f of p_e (i.e. within 5 cm of Protag's surface) is also dropped. Logged via
  simulator identity (logging only): 2 points / 2 episodes (B1.5), 5 points / 3 episodes (P2.5) out of ~600 k; 0 leaked
  Protag points; 0 scans emptied. Occlusion by Protag's body of obstacles behind it is real in these variants and is not undone.

## 6. Commands

```bash
.venv/bin/python -m pytest tests/test_hunter_variants.py -q            # 28 tests
.venv/bin/python -m pytest tests/test_pursuit.py tests/test_week6_tracka2.py tests/test_week6_tracks.py -q   # 80, regression
.venv/bin/python -m experiments.pursuit.hunter_variants --mode smoke                     # all four, 5 seeds
.venv/bin/python -m experiments.pursuit.hunter_variants --mode main --workers 12         # all four, 200 seeds
.venv/bin/python -m experiments.pursuit.hunter_variants --mode smoke --variants B1       # one config (B1|B1.5|P2|P2.5)
.venv/bin/python -m experiments.pursuit.hunter_variants --mode main --variants P2.5 --tag _p25only
```
Per-step debug (target, prediction, nominal and final action, QP status): `run_episode(..., variant=VARIANTS[name], trace=True)`.

## 7. Parameters

| Parameter | Where | Default | Basis |
|---|---|---|---|
| `policy` | `HunterVariant` | direct / predictive | variant |
| `lidar_mode` | `HunterVariant` | remove / filter | variant |
| `prediction_horizon` | `HunterVariant` | 0.5 s (5 steps), bound [0, 2] s | ~one body-length of Protag travel; not tuned |
| `goal_accel` a_g | `HunterVariant` | 1.0 m/s² | MAX_SPEED per second; not tuned |
| `filter_margin` → r_f | `HunterVariant` | 0.05 → r_f = 0.35 m | Protag hits lie on its 0.30 m circle; margin = capture δ |
| capture / contact | `PursuitConfig` (existing) | 0.65 / 0.60 m | unchanged |
| hunter speed, condition | CLI | 1.0 m/s, P-Random | as audited P3 |
| seeds | `VARIANT_BLOCKS` | smoke 14 000 000+5, main 14 100 000+200 | fresh; disjoint test |

## 8. Tests and smoke

28/28 new tests pass (A direct, B predictive, C removal, D filtering, E integration incl. all four variants on M0); 80/80
existing pursuit/Track tests pass. Smoke (5 seeds × 4) ran cleanly; per-episode records confirmed the filter path is active
(it is not a no-op even where outcomes coincide).

## 9. Main evaluation — 200 paired seeds per variant, P-Random protagonist, hunter 1.0 m/s, 712 s wall

Declared before running; no parameter was changed after smoke.

**Terminal outcomes** (exactly one per episode; priority own-collision → capture → goal → timeout; each row sums to 1):

| terminal outcome [95% Wilson for capture] | B1 | B1.5 | P2 | P2.5 |
|---|---|---|---|---|
| capture | **0.310** [0.25,0.38] | 0.275 [0.22,0.34] | 0.265 [0.21,0.33] | 0.255 [0.20,0.32] |
| Protag goal | 0.490 | 0.490 | 0.495 | 0.480 |
| Protag collision | 0.200 | 0.235 | 0.235 | 0.245 |
| timeout | 0 | 0 | 0.005 | 0.020 |
| mean time to capture (terminal captures), s | 12.9 | 12.4 | 11.5 | 12.1 |

**Event flags and controller metrics** (flags can co-occur with another terminal outcome; e.g. B1 capture flag 0.345, of
which 7 episodes ended as a Protag collision on the same step; goal flag 0.495):

| metric | B1 | B1.5 | P2 | P2.5 |
|---|---|---|---|---|
| capture flag | 0.345 | 0.300 | 0.285 | 0.265 |
| hunter collision (hunter disabled; all dynamic obstacles) | 0.200 | 0.210 | 0.205 | 0.210 |
| hunter infeasible-step rate | 0.0194 | 0.0191 | 0.0194 | 0.0193 |
| hunter non-optimal solve (u=0) rate | 0.067 | 0.068 | 0.067 | 0.070 |
| Protag infeasible-step rate | 0.070 | 0.069 | 0.068 | 0.067 |
| mean distance at capture, m | 0.636 | 0.637 | 0.636 | 0.638 |
| prediction error τ=0.5 s, mean [CI] / median | — | — | 0.158 [0.154,0.163] / 0.128 | 0.158 [0.153,0.163] / 0.127 |

Hunter non-optimal solves include `optimal_inaccurate`/DCPError, all mapped to u = 0 by the frozen controller (same as the
audited hunter). The runner's console table and `summary.captured_rate` in the saved JSON (written before this correction)
are the capture *flag*; the terminal rates above were recomputed from the saved per-episode `outcome` field. The runner now
also emits `terminal_*_rate` and terminal paired tests.

Paired comparisons on **terminal outcomes** (diff = first − second, bootstrap 95% CI, discordant counts, exact McNemar):

| pair | capture | Protag collision | goal | timeout |
|---|---|---|---|---|
| B1.5 − B1 | −0.035 [−0.075, 0.000], 4 vs 11, p=0.12 | +0.035 [0.000,+0.070], 10 vs 3, p=0.09 | 0.000, p=1.0 | 0 |
| P2 − B1 | −0.045 [−0.110,+0.020], 17 vs 26, p=0.22 | +0.035 [−0.020,+0.090], p=0.28 | +0.005, p=1.0 | +0.005 |
| P2.5 − B1.5 | −0.020 [−0.085,+0.045], p=0.66 | +0.010, p=0.87 | −0.010, p=0.86 | +0.020 [+0.005,+0.040], 4 vs 0, p=0.125 |
| P2.5 − P2 | −0.010 [−0.050,+0.030], p=0.80 | +0.010, p=0.79 | −0.015, p=0.51 | +0.015, p=0.25 |

(On the capture *flag*, B1.5 − B1 was −0.045, 1 vs 10, p=0.012; that flag difference does not carry over to terminal
captures, because several flag-only captures coincided with Protag collisions.)

### Reading (claims limited to what the data supports)

1. **No variant is shown to differ from B1 in any terminal outcome.** Every paired terminal McNemar p ≥ 0.09 across 4 pairs ×
   4 outcomes, before any multiplicity correction.
2. **Predictive pursuit is not better here.** Point estimates for P2 − B1 (−4.5 pp capture) and P2.5 − B1.5 (−2 pp) are in
   the wrong direction for the hypothesis, CIs include 0. Prediction error (0.16 m mean at τ=0.5 s) is small, so the target
   moves little; the goal-directed model does not capture Protag's obstacle-avoiding and evasive motion.
3. **Keeping Protag visible and filtering** has point estimates of fewer captures and more Protag collisions in the direct
   policy (−3.5 / +3.5 pp), not significant. Filtering itself was clean (0 leaks, ≤5 false-positive points per 200 episodes), so
   any real effect would come from what filtering does not remove: occlusion behind Protag's body and Protag appearing in the
   hunter's velocity tracker. Not diagnosed; would need a larger or targeted study.
4. Hunter safety and feasibility are unchanged across variants (hunter collisions 20–21%, all with dynamic obstacles;
   infeasible 1.9%).
5. B1 (terminal capture 0.310) and audited P3 P-Random (terminal 0.225) differ on different seed blocks with bit-identical code;
   this is between-block variation, so cross-block comparisons are unreliable.

## 10. Limitations and deviations

- No target-velocity term in the hunter CLF (§3); no convergence guarantee to a moving target.
- All four variants use simulator ground truth (Protag position; velocity and goal for prediction). The filter variants
  are a sensing approximation, not deployable: a real hunter would need to estimate p_e to know what to filter.
- Protag's own information model is unchanged (known-state hunter row; hunter not in Protag's LiDAR).
- The prediction ignores Protag's planner, CBF and reaction to the hunter; a_g, τ and r_f are declared defaults, untuned,
  single values — no sensitivity sweep was run.
- r_f filter can drop environmental points within 5 cm of Protag's surface (logged; rare).
- Single hunter speed (1.0) and single protagonist condition (P-Random) evaluated.

## 11. Follow-up: no moving obstacles (static M0)

Same four variants, same code, same seeds (14 100 000–14 100 199), `--static`: the 5 rectangles and walls only, all 6 moving
obstacles removed (construction as `experiments/pursuit/p2_static.py`). P-Random protagonist, hunter 1.0 m/s, 592 s wall.
Output `results/pursuit_hunter_variants/main/HV_main_P-Random_hunter1_static.json`; smoke (5 seeds) run first.

Pairing: the four static variants are fully paired with each other. Against §9, Protag start and goal are identical on all
200 seeds but the hunter start differs on all 200 (the env draws the hunter after the obstacles), so static-vs-dynamic
comparisons are not paired on the hunter and are only indicative.

| terminal outcome | B1 | B1.5 | P2 | P2.5 |
|---|---|---|---|---|
| capture | 0.340 | 0.355 | 0.400 | 0.390 |
| Protag goal | 0.520 | 0.515 | **0.425** | **0.440** |
| Protag collision (all static rectangles) | 0.140 | 0.130 | 0.175 | 0.170 |
| timeout | 0 | 0 | 0 | 0 |
| mean time to capture, s | 11.3 | 11.1 | 12.2 | 11.8 |
| hunter collision | 0 | 0 | 0 | 0 |
| hunter infeasible-step rate | 0.024 | 0.024 | 0.021 | 0.021 |
| Protag infeasible-step rate | 0.096 | 0.098 | 0.102 | 0.096 |
| mean distance at capture, m | 0.640 | 0.639 | 0.637 | 0.637 |
| prediction error τ=0.5 s, mean | — | — | 0.160 | 0.160 |

Filter: 32.6 k / 34.6 k Protag points removed, 0 leaked, 0 / 1 environmental false positive, 0 scans emptied.
Capture flags exceed terminal captures by 4–8 episodes per variant (flag co-occurring with a Protag collision).

Paired, terminal outcomes (diff = first − second, bootstrap 95% CI, discordant counts, exact McNemar):

| pair | capture | Protag goal | Protag collision |
|---|---|---|---|
| B1.5 − B1 | +0.015 [−0.025,+0.060], 11 vs 8, p=0.65 | −0.005, p=1.0 | −0.010, p=0.73 |
| P2 − B1 | +0.060 [0.000,+0.125], 28 vs 16, p=0.096 | **−0.095 [−0.145,−0.045], 4 vs 23, p=0.0003** | +0.035 [−0.020,+0.085], p=0.26 |
| P2.5 − B1.5 | +0.035 [−0.030,+0.100], p=0.37 | **−0.075 [−0.125,−0.030], 5 vs 20, p=0.004** | +0.040 [−0.015,+0.095], p=0.20 |
| P2.5 − P2 | −0.010, p=0.85 | +0.015, p=0.58 | −0.005, p=1.0 |

Reading:
1. **Without moving obstacles, predictive pursuit denies Protag its goal more often.** P2 − B1 goal −9.5 pp (p = 0.0003,
   survives Bonferroni over the 16 terminal tests, threshold 0.003); P2.5 − B1.5 −7.5 pp (p = 0.004, just above it). The
   lost goals split into more captures (+6 / +3.5 pp) and more Protag collisions (+3.5 / +4 pp); neither part is
   significant alone. The hunter's extra pressure is the robust effect, not a capture gain.
2. This reverses the sign seen with moving obstacles (§9, prediction point estimates against capture, not significant). A
   plausible reading, not tested: in traffic the prediction steers the hunter into obstacle interactions (hunters were
   disabled in ~20% of dynamic episodes), whereas on the static map the hunter never collides and the lead point pays off.
3. **LiDAR handling makes no measurable difference on the static map** (B1.5 ≈ B1, P2.5 ≈ P2; all p ≥ 0.58), consistent with
   occlusion by moving obstacles being the main place where rendering Protag could matter.
4. Hunter safety: zero hunter collisions in all 800 episodes. Protag collisions (13–18%) are all with static rectangles; the
   earlier P2 static result and the known-state hunter row pushing Protag toward walls are the relevant context, not
   investigated here.
5. Across runs (indicative only, hunter start unpaired): removing obstacles raised B1 Protag goal 0.49 → 0.52 and lowered
   Protag collisions 0.20 → 0.14, and raised terminal capture most for the predictive hunters (P2 0.265 → 0.400).
