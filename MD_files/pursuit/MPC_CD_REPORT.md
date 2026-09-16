# MPC Pursuit (policies 3 / 3.5) and CD-Random-CLF-DR-CBF — Implementation and Results

Continuation project, branch `Week6`, **uncommitted**. Companion files: `MPC_CD_PREREGISTRATION.md` (design, fixed before
any run), `MPC_CD_DEVIATIONS.md` (gate log + deviations), `MPC_CD_REPORT_DATA.md` (all tables, generated from the saved
records). Historical artifacts (`PURSUIT_RESULTS*.md`, `HUNTER_VARIANTS_REPORT.md`, `results/pursuit_hunter_variants/**`)
are unchanged.

## 1. Architecture

Three layers, independently configurable:

* **Pursuit policy** (hunter target): `direct` (1, 1.5), `predictive` (2, 2.5), `mpc` (3, 3.5) — `hunter_policies.py`,
  `mpc.py`. The policy produces (γ, u_nom) only.
* **Sensing / filtering**: Protag removed from the hunter's LiDAR (1, 2, 3) or rendered and filtered out of the CBF rows
  at push time (1.5, 2.5, 3.5) — unchanged from the audited implementation.
* **Controller**: A Protag Random / hunter tuned (historical), B both Random, C both CD-Random — `ControllerSetup` in
  `harness.py`. The congestion wrapper is installed on `ctrl.generate_controller` **before** `RandomRecovery`, so each step
  runs `RandomRecovery → CongestionWrapper → frozen QP`. The frozen QP filters every executed action; `dr_control/`,
  `robot_env/`, `DRCBFPolicy.predict()` and `random_recovery.py` are untouched (a test checks `git diff` of those paths).

## 2. Files

**Added**: `continuation/pursuit/mpc.py`, `continuation/sensed_obstacles.py`,
`continuation/congestion/{__init__,signals,monitor,detector,wrapper}.py`,
`experiments/pursuit/{policy_matrix,cd_calibrate,policy_matrix_report}.py`,
`experiments/pursuit/run_policy_matrix_chain.sh`, `tests/test_policy_matrix.py`, this report + the three companion files,
`results/pursuit_policy_matrix/**` (65 MB).

**Modified** (opt-in; defaults reproduce the audited episode bit for bit):
`hunter_policies.py` (policy `mpc`, IDs 1–3.5 with the historical names as aliases, `HV_dev` and `HV_check` seed blocks);
`hunter.py` (`act(..., nominal=)` hook called after the scan is pushed);
`harness.py` (controller setups, hunter Random recovery with its own RNG, congestion wrappers, MPC hook, per-step timing
breakdown and step logs, new record fields).

## 3. Equations implemented

**MPC (policies 3, 3.5).** 42 candidates = 7 headings {0,±30,±60,±90}° from the line to Protag × 3 turns {−45,0,+45}° ×
2 speeds {1.0, 0.5} m/s (Euclidean, inside the per-axis box), two 0.5 s segments, N = 10 steps:
p^H_{k+1} = p^H_k + Δt u_k. Protag path p̂^P_k = the policy-2 predictor at τ = kΔt
(p + vτ + ½ a_g d_g τ², per-axis speed clip, arena clip). With d_k = ‖p^H_k − p̂^P_k‖,

    J = 1.0·d_N + 0.5·min_k d_k − 2.0·1[min_k d_k ≤ 0.65] + 0.1·Σ‖u_{k+1} − u_k‖² + 1.0·Σ_k max(0, 0.15 − clr_k)

hard-rejecting any candidate with clr_k < 0. clr from sensed obstacles (LiDAR points −0.3 m; confirmed tracks, constant
velocity, −0.6 m; points bound to confirmed tracks are represented by the track; 3.5 also drops tracks within 0.35 m of
Protag). γ = best candidate's p^H_N, u_nom = its u_0; if all are rejected, direct pursuit (γ = p_P). Replanned every step.

**Congestion monitor.** M(s) = max over a 9×9 control grid on [−1,1]² of min_i CBC_i, using the existing
`random_recovery.margins` on the controller's own 5 kept rows; margin = M − τ_eff, τ_eff = 0.12 (the collapsed-regime
feasibility threshold — **not** 0; see §7). Look-ahead at t ∈ {0, 0.25, 0.5} s along p + t·u_nom, rows from `samples()` of a
source copy whose confirmed-track points are shifted by v·t. Detector: Critical if margin < 0 or predicted clearance < 0.05;
Cautious if margin < c1 or clearance < c2; leaving a state needs the exit condition (+0.05) on 3 consecutive steps.
Caution (config C): u_nom' = k R(a) u_nom, γ' = p + k R(a)(γ − p), k = 0.6 / 0.3, a ∈ {0,±15,±30}° chosen by best predicted
M_look (ties keep 0°). Constraints, control box and recovery are never modified.

## 4. Commands

```bash
.venv/bin/python -m pytest tests/test_policy_matrix.py -q                       # 27 new tests
.venv/bin/python -m experiments.pursuit.policy_matrix --phase repro  --config A --workers 10
.venv/bin/python -m experiments.pursuit.policy_matrix --phase smoke  --config B --workers 12
.venv/bin/python -m experiments.pursuit.policy_matrix --phase main   --config B --workers 11   # Phase I
.venv/bin/python -m experiments.pursuit.policy_matrix --phase dev    --config B --policies 1 3 --workers 4
.venv/bin/python -m experiments.pursuit.cd_calibrate --run results/pursuit_policy_matrix/dev_B
.venv/bin/python -m experiments.pursuit.policy_matrix --phase smoke --config C --thresholds results/pursuit_policy_matrix/calibration/cd_thresholds.json --workers 4
.venv/bin/python -m experiments.pursuit.policy_matrix --phase main  --config C --thresholds .../cd_thresholds.json --workers 12   # Phase III
.venv/bin/python -m experiments.pursuit.policy_matrix --phase latency --config {A,B,C} [--thresholds ...] --workers 1
.venv/bin/python -m experiments.pursuit.policy_matrix_report --out MD_files/pursuit/MPC_CD_REPORT_DATA.md
experiments/pursuit/run_policy_matrix_chain.sh      # calibration -> smoke C -> Phase III -> latency -> report
```

## 5. Runs completed

| run | config | CD | map(s) | policies | seeds | episodes | workers | wall |
|---|---|---|---|---|---|---|---|---|
| repro_A | A | off | both | 1–2.5 | 14 100 000–009 | 80 | 10 | 108 s |
| smoke_B | B | shadow | both | 1–3.5 | 14 000 000–004 | 60 | 12 | 81 s |
| **main_B (Phase I)** | B | shadow | both | 1–3.5 | 14 100 000–199 | **2 400** | 11 | 2 343 s |
| dev_B (calibration) | B | shadow | both | 1, 3 | 14 300 000–049 | 400 | 4 | 509 s |
| smoke_C | C | active | both | 1–3.5 | 14 000 000–004 | 60 | 4 | 227 s |
| **main_C (Phase III)** | C | active | both | 1–3.5 | 14 100 000–199 | **2 400** | 12 | 2 776 s |
| latency_A / B / C | A / B / C | off / shadow / active | both | 1–2.5 / 1–3.5 | 14 000 000–004 | 40 / 60 / 60 | 1 | 389 / 701 / 829 s |
| check_B (§12 follow-up) | B | shadow | both | 3, 3.5 | 14 400 000–099 | 400 | 12 | 386 s |

Total 6 160 episodes (5 760 pre-registered + 400 in the §12 follow-up). Every Phase I / Phase III cell has exactly 200 episodes (check_B: 100 per cell), no unresolved outcomes, and layouts verified
identical across configurations; step logs exist for all 4 800 evaluation episodes. Nothing failed or was left incomplete.

## 6. Results

All tables: `MPC_CD_REPORT_DATA.md`. Metrics and statistics are the previous experiments' (terminal outcomes with Wilson
intervals, event flags kept separate, paired exact McNemar + bootstrap CI). Bonferroni: within-config families α = 0.05/30,
cross-config α = 0.05/20 (B−A) or /30 (C−B); ✱ below means significant after correction.

**6.1 B − A (hunter tuned → hunter Random), policies 1–2.5.** No terminal-outcome difference on either map
(all p ≥ 0.26 static, ≥ 0.30 dynamic). With moving obstacles the hunter crashes less often: policy 1 −6.0 pp
(0.200 → 0.140, 1 vs 13 discordant, p = 0.0018 ✱), policy 1.5 −6.5 pp (p = 0.00098 ✱), 2 and 2.5 −4.0 / −4.5 pp (n.s.).
Mechanism is visible in the logs: the hunter's u = 0 steps fall from 6.7 % to 3.0 % as recovery replaces the frozen zero
fallback on infeasible steps (1.3–1.9 recovery events per episode).

**6.2 Phase I (config B) — does MPC pursuit help?** Static: nothing significant; policy 3.5 has the highest capture
(0.440 vs 0.370 for policy 1, p = 0.06) and the lowest Protag collision (0.075, p = 0.0125 vs 1.5), policy 3 the lowest
capture (0.340). Dynamic: capture is statistically indistinguishable across all six policies; MPC's measurable effect is
**fewer hunter crashes**: 3 − 2 = −9.0 pp (0.165 → 0.075, 4 vs 22, p = 0.00053 ✱) and 3.5 − 2.5 = −8.5 pp (p = 0.0015 ✱);
3 − 1 = −6.5 pp (p = 0.0072, n.s. after correction). MPC also lowers the hunter's own infeasible-step rate (0.014 → 0.011)
and Protag's (0.076 → 0.063). **No evidence that MPC pursuit increases capture.**

**6.3 Phase III (config C) — CD-Random on both agents.** C − B shows **no significant outcome change** anywhere after
correction (smallest p = 0.0025 for policy 3.5 static goal −10.5 pp; 3.5 dynamic capture +11.0 pp, p = 0.008). What does
change is speed: mean time to capture rises 12.6 s → 16.3 s and time to goal 13.6 s → 16.7 s (dynamic, policy 1), and
timeouts appear (0–6 % by cell, 0 in B for policies 1–2.5). Feasibility statistics improve (Protag infeasible/step
0.102 → 0.067 static, 0.076 → 0.049 dynamic; hunter 0.012 → 0.0023 static) — this is a *feasibility* statement, not a safety
claim, and Protag collision rates do not change significantly.
*Within* config C the MPC policies separate from the rest on the dynamic map: 3.5 − 1.5 capture +18.0 pp (45 vs 9,
p = 7.3e-07 ✱), not-escaped +17.0 pp ✱, Protag goal −17.5 pp ✱; 3 − 1 goal −15.0 pp ✱ and not-escaped +13.0 pp ✱ (capture
n.s., Protag collision +10.5 pp n.s.); 3.5 − 2.5 capture +12.5 pp ✱. So when both agents are cautious, MPC pursuit denies
escape more often — but the same MPC advantage is absent in config B, so it is a property of the C pairing, not of MPC alone.

## 7. MPC diagnostics

42 candidates, horizon 10 steps (1.0 s), 2 segments. Per step: **4.7–5.0 ms mean** (single-worker pass 4.5–4.9 ms; max
7.7–8.0 ms, one 32.6 ms outlier under 11-way parallelism). Fallback to direct pursuit 2.6–6.4 % of steps (highest on the
dynamic map, 3.5). Mean 17–21 of 42 candidates hard-rejected per step — the rollout check is doing most of the shaping.
Predicted capture within the horizon on 24–42 % of steps. **Prediction error at the MPC horizon (1.0 s) is 0.48–0.49 m**,
three times the 0.16 m measured at the 0.5 s horizon of policy 2 — the Protag model is the weak element, as expected.

## 8. Congestion diagnostics

**Calibration failed its targets (deviation D1).** No (c1, c2) on the declared grid keeps warnings ≤ 25 % of steps, because
the *fixed* Critical rule alone fires that often; the least-warning pair c1 = 0, c2 = 0.05 was frozen, leaving the Cautious
band empty (every warning is Critical). Achieved on development seeds: warning fraction 0.559, coverage 0.992.
Evidence, mechanism and the exact rule extension are in `MPC_CD_DEVIATIONS.md` §D1.

On evaluation seeds (Phase I shadow replay with the frozen thresholds): warning fraction 0.52–0.75 for Protag and
0.26–0.66 for the hunter (lowest under MPC, whose nominal already avoids obstacles); coverage 0.91–1.00; **precision
0.16–0.25**; mean lead 4.2–6.0 steps (0.42–0.60 s). In Phase III (active) precision falls to 0.05–0.13 at similar coverage.
A detector that is on most of the time trivially precedes most infeasibility onsets; **coverage here is not evidence of
discrimination**. Caution does bite: executed speed drops from 0.78–0.91 m/s in Normal to 0.30–0.53 m/s in Critical, mean
speed scale 0.52–0.87, and the direction search changes heading on 26–101 steps per episode.

## 9. Latency (clean single-worker pass, ms per step, dt budget = 100 ms)

| config | agent | total mean | total p95 | QP | CD | MPC | steps > 100 ms |
|---|---|---|---|---|---|---|---|
| A (mixed) | Protag / hunter | 35.5–37.0 / 34.6–39.0 | 39–45 | 24–28 | — | — | ≤ 0.18 % |
| B (Random both) | Protag / hunter | 37.0–40.7 / 35.8–45.9 | 41–52 | 24–29 | 1.9–2.3 (shadow) | 4.5–4.9 | ≤ 0.26 % |
| C (CD-Random both) | Protag / hunter | 37.5–43.9 / 39.8–46.4 | 45–54 | 23–27 | 2.7–6.8 (active) | 4.5–4.8 | ≤ 0.16 % |

The QP dominates (≈ 25 ms). Random recovery is < 0.05 ms. Congestion detection costs ~2 ms in shadow mode and up to 6.8 ms
active (the ±15/±30° direction search re-evaluates the monitor). MPC costs ~4.8 ms. **No configuration approaches the
100 ms step budget**: worst mean 46.4 ms, worst p95 54.1 ms. Rare >100 ms spikes (max 250 ms) occur in config A as well and
are therefore not attributable to the new code.

## 10. Tests

`tests/test_policy_matrix.py` (27) covers configuration/policy selection, wrapper order (CD inside Random) for both agents,
hunter-RNG independence, frozen-module `git diff`, the sensed-obstacle split and Protag-track exclusion, 3.5 LiDAR
inclusion + row filtering, MPC candidate/rollout/score/fallback and "QP still filters an MPC nominal driven into a wall",
the predictor reuse, margin sign convention, `keep_rows` == the controller's own `xi_kept`, predicted-source shifting,
open/congested/already-infeasible monitor cases, detector hysteresis, alarm definitions, shadow mode passing arguments
through unchanged, caution touching only nominal + CLF target, config-A matrix path reproducing the audited episode, and a
full config-C policy-3 episode. **Final run: 174 passed** (27 new + 147 existing), 3 m 51 s. Three test-construction
errors were found and fixed during development (recorded in the deviations file); no production code was changed for them.

## 11. Gates, deviations, limitations

Gates: tests PASS; config-A exact match on 80 episodes PASS; smoke B and smoke C PASS. The only failed gate is the
congestion **calibration target** (D1), which was continued past on the user's instruction, with the fallback rule
declared before it ran.

Limitations that affect interpretation:
0. **A post-hoc fix ablation (development seeds only, `MPC_CD_DEVIATIONS.md` "D1 follow-up") halves the warning rate
   (0.559 -> 0.278) and nearly doubles precision (0.199 -> 0.348)** by looking ahead along the executed instead of the
   nominal velocity and gating tracks more strictly. A further post-hoc search over all four thresholds plus the hysteresis
   ("D1 follow-up 2") then MET the targets on held-out development data (coverage 0.828, warnings 0.204, precision 0.472),
   and a Phase III rerun with it (config **C2**, `main_C_fix4`, tables in `MPC_CD_REPORT_DATA_v2.md`) reproduced that
   behaviour on evaluation seeds (Protag warnings 0.720 -> 0.260, precision 0.085 -> 0.499). C2 still changes NO outcome
   versus config B after correction; its one significant change versus C is policy 3.5 dynamic capture -0.140 (p = 0.00023),
   reinforcing §12. All of this is exploratory and post-hoc; the pre-registered Phase I / Phase III results below stand as
   produced.
1. **The congestion detector as specified is uninformative at this operating point** — it is Critical on ~50–75 % of Protag
   steps, so Phase III measures "caution nearly always on", not congestion-triggered caution. Causes (measured, not guessed):
   the CBF clearance convention double-insets walls by 0.3 m; the look-ahead follows the straight nominal line and ignores the
   QP's deflection; the frozen tracker holds more confirmed tracks (median 9) than there are obstacles (6).
2. The MPC Protag model is the policy-2 predictor at 1.0 s, error ≈ 0.49 m, and assumes Protag does not react to the hunter.
3. MPC's hard rollout rejection uses the same double-inset wall convention, so it keeps an extra 0.3 m from walls; 17–21 of
   42 candidates are rejected per step.
4. Non-infeasible solver failures (`optimal_inaccurate`, DCPError when h < 0) are **not** covered by Random recovery — the
   frozen trigger is the `infeasible` status only. These are 0–5 % of hunter steps.
5. Between-map comparisons remain unpaired in the hunter's start position (the env draws the hunter after the obstacles).
6. Phase I and the development run executed concurrently (11 + 4 workers), so their in-run latency fields are inflated; §9
   uses the single-worker pass instead.
7. MPC weights, horizon and the caution scales are declared defaults, never tuned; thresholds were calibrated only on the
   development block and never on evaluation seeds.

## 12. Follow-up: why does policy 3.5 differ so much from policy 3?

The two policies differ only in sensing: 3 never renders Protag in the hunter's LiDAR; 3.5 renders it and filters its
returns out of the CBF rows (and its track out of the MPC obstacle set). Yet paired 3.5 − 3 capture differs by
+0.100 (static B, p = 0.0037), +0.045 (dynamic B), +0.085 (static C) and +0.125 (dynamic C, p = 7.0e-05), with Protag
collision moving the opposite way (−0.115, −0.025, −0.030, −0.110) and **not-escaped nearly unchanged** (−0.015…+0.055).
Diagnostics (scripts re-runnable from the saved records; the replication run is `results/pursuit_policy_matrix/check_B`):

**Ruled out.**
1. *Occlusion*: Protag intercepts 0.15 rays/step on average and hides a real return on 0.13 rays/step (25 of 204 steps in a
   replayed episode); the hunter's nearest barrier row was identical in both sensing modes at **every** step of that episode.
2. *Tracker contamination*: 0 of 3 435 barrier rows over 697 steps bound to a track sitting on Protag, so no row inherits
   Protag's velocity (Protag's own points are filtered before binding).
3. *Filter false positives*: 82–108 wrongly filtered environment points out of ~600 000 (14 of 200 episodes) — 4–10x more
   than policies 1.5 / 2.5, but far too rare to move outcomes by 10 pp.

**The actual mechanism: MPC's discrete argmin amplifies a negligible sensing difference.** At the first step where the two
hunters' actions differ, the nominal action jumps by 0.5176 m/s = 2 sin(15°) — exactly one 30° step in the candidate set —
or by exactly 0.5 (the 1.0 -> 0.5 m/s speed switch). One occluded ray reorders the candidate costs and the argmin flips.
Applying the SAME sensing change to each pursuit law shows the amplification is specific to MPC:

| pair | identical outcome | identical step count | mean abs delta steps | capture difference |
|---|---|---|---|---|
| 1.5 vs 1 | 0.87–0.92 | 0.55–0.73 | 4–15 | −0.030 … −0.005 |
| 2.5 vs 2 | 0.83–0.95 | 0.52–0.68 | 4–14 | −0.010 … +0.015 |
| **3.5 vs 3** | **0.75–0.78** | **0.28–0.46** | **17–46** | **+0.045 … +0.125** |

**Replication on fresh seeds (14 400 000–099, config B, 100 seeds per cell, 400 episodes).** The advantage does **not**
reproduce at the evaluated magnitude: capture +0.050 static (15/10 discordant, p = 0.42) and +0.020 dynamic (p = 0.81);
Protag collision −0.070 (p = 0.039) and −0.040 (p = 0.42). The divergence itself does reproduce (identical outcomes
0.74 / 0.78, mean abs delta steps 28.6 / 16.0).

**Conclusion.** Policies 3 and 3.5 are not meaningfully different *pursuit* policies; the sensing change they differ by is
almost inert (a fraction of a ray per step). What the 3.5 − 3 gap mostly measures is the **sensitivity of the sampled MPC to
discrete candidate switching**: an input perturbation far below any modelling tolerance produces a 30° heading or 2x speed
change in the command, and trajectories then diverge (~25 % of episodes end differently). The sign is consistently
capture-up / collision-down across all six cells, so a small genuine bias cannot be excluded, but the evaluated magnitudes
(+0.085…+0.125) are not supported by the fresh sample (+0.02…+0.05, n.s.). **Any 3.5-vs-3 difference in §6 should be read as
run-to-run variation of this mechanism, not as evidence that filtering beats removal**, and the same caution applies to the
significant within-config-C MPC results, which compare MPC against non-MPC policies on the evaluation seeds only.

Design implication for later work: the candidate set is the weak point. Interpolating between candidates, carrying the
previous plan as a candidate (warm start), or adding hysteresis to the argmin would all reduce this sensitivity; none was
tried here because the plan calls for the simplest version first.
