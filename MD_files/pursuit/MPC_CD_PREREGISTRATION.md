# Policy Matrix (MPC pursuit + CD-random-CLF-DR-CBF) — Pre-registration

Written before any development or evaluation run of this experiment. Its SHA-256 is recorded in
`results/pursuit_policy_matrix/prereg.sha256`; the calibration script refuses to run if this file changes.
Decisions are the user's approved answers (continuation project, 16 decisions).

## Fixed setup
- Map: canonical M0 (`config.json`), `dynamic` = 6 stochastic obstacles @0.675 m/s; `static` = same map, 0 moving obstacles.
- Protag: frozen Tuned + Random CLF-DR-CBF, A* carrot, known-state hunter row (true position/velocity). Hunter not in Protag's LiDAR.
- Hunter: radius 0.3, per-axis |u| ≤ 1.0 m/s, no planner, own LiDAR/tracker; capture d ≤ 0.65; crash → disabled, episode continues; 500-step limit.
- Policy IDs: 1 direct/remove, 1.5 direct/filter, 2 predictive/remove, 2.5 predictive/filter, 3 MPC/remove, 3.5 MPC/filter
  (1–2.5 identical to historical B1/B1.5/P2/P2.5; filter radius 0.35 m, τ=0.5 s, a_g=1.0).

## Controller configurations
| config | Protag | Hunter | CD |
|---|---|---|---|
| A (historical) | Random-CLF-DR-CBF | Tuned-CLF-DR-CBF (u=0 on non-optimal) | off |
| B | Random-CLF-DR-CBF | Random-CLF-DR-CBF (frozen spec K=16,H=1,{0.8,1.0},m≥0; own RNG) | shadow (no action change) |
| C | CD-Random-CLF-DR-CBF | CD-Random-CLF-DR-CBF | active |

## MPC (policies 3, 3.5)
42 candidates: segment-0 headings {0,±30,±60,±90}° from the line to Protag's true position, segment-1 turns {−45,0,+45}°,
speed {1.0,0.5} m/s (norm, same both segments), 2×5 steps, N=10 (1.0 s). Protag path = policy-2 predictor at τ=kΔt.
J = 1.0·d_N + 0.5·min d − 2.0·I[min d ≤ 0.65] + 0.1·Σ‖Δu‖² + 1.0·Σ max(0, 0.15 − clr_k); hard reject if any clr_k < 0.
Clearance: LiDAR points −0.3 m, confirmed tracks (constant velocity) −0.6 m; points bound to confirmed tracks are the track.
3.5: also drop confirmed tracks within 0.35 m of Protag's true position. γ = best end-of-horizon position, u_nom = first control.
All rejected → direct pursuit nominal (γ = p_P). QP + Random recovery on every step. Parameters are declared, not tuned.

## Congestion detection
Margin = M_look − τ_eff (τ_eff = 0.12); M = max over 9×9 grid on [−1,1]² of min_i CBC_i on the 5 kept rows (existing `margins`);
look-ahead t ∈ {0, 0.25, 0.5} s along nominal motion; predicted rows = `samples()` on a source copy with confirmed-track points shifted by v·t.
States: Critical if margin < 0 or clr < 0.05; Cautious if margin < c1 or clr < c2; exit when margin ≥ threshold+0.05 and clr ≥ threshold+0.05
for 3 consecutive steps, one level at a time. Initial (provisional, shadow only) c1 = 0.15, c2 = 0.30.
Caution (config C): scale 0.6 (Cautious) / 0.3 (Critical); directions {0,±15,±30}° (best predicted M_look, ties keep 0°);
u_nom' = k R(a) u0, γ' = p + k R(a)(γ − p). Constraints, box and recovery untouched.

## Calibration rule (development seeds only)
Data: shadow mode, config B, policies 1 and 3, both maps, seeds 14 300 000–14 300 049 (400 episodes), both agents, active steps only.
Grid: c1 ∈ {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5}, c2 ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0}.
Pooled over both agents: coverage = 1 − missed/infeasibility onsets (W = 10), warning fraction = steps in state ≥ 1 / active steps.
Select, among pairs with coverage ≥ 0.80 and warning fraction ≤ 0.25, the one with the lowest warning fraction (ties: smaller c1, then c2).
If none qualify: maximise coverage subject to warning fraction ≤ 0.25 (ties: lower warning fraction, smaller c1, c2). Report achieved values.
Thresholds are frozen to `results/pursuit_policy_matrix/calibration/cd_thresholds.json` (+sha256) before any config-C run.

## Alarm definitions
Warning onset: 0 → ≥1. True alarm: infeasible QP step within t+1..t+10. Lead time: first such step − t.
Infeasibility onset: infeasible at t, not at t−1. Missed: onset with no state ≥ 1 in t−10..t−1. Hunter steps after a crash excluded.

## Seeds and runs
Evaluation: 14 100 000–14 100 199 (same as historical A), both maps, 200 episodes per (config, policy, map). Smoke: 14 000 000–004.
Development: 14 300 000–049 (calibration only). Latency pass: smoke seeds, single worker.
Phase I: config B × 6 policies × 2 maps. Phase III: config C × 6 policies × 2 maps.

## Gates (stop and report on failure; never stop on results)
1. Existing + new tests pass. 2. Config A exact-match on 10 seeds × 4 policies × 2 maps against the historical records (all non-timing fields).
3. 5-seed smoke for every cell of a phase before its main run.

## Analysis
Metrics and statistics as in `experiments/pursuit/hunter_variants.py` (terminal outcomes, flags, combined capture+Protag-collision,
hunter collision, times, distances, infeasibility/recovery, latency). Paired exact McNemar + bootstrap 95% CI.
Within config and map: 3−1, 3.5−1.5, 3−2, 3.5−2.5, 1.5−1, 2−1. Across configs per policy/map: B−A (1–2.5), C−B (all six).
Bonferroni reported per family (within-config family per config × map; cross-config family per comparison type × map).
