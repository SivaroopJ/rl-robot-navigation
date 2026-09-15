# Track B — infeasibility-aware navigation: results

Independent of Track A. Canonical M0, stochastic obstacles, frozen controllers, frozen QP.
Commit `07d74ca`, branch `Week6`, run 2026-09-12/13.

Artifacts: diagnosis `results/week6_trackb/B0/{b0_steps_40.jsonl,b0_analysis.json}` ·
pilot `results/week6_trackb/trackB_pilot_20.json` ·
**evaluation `results/week6_trackb/trackB_eval_200.json`** · logs `results/week6_trackb/*.log` ·
code `continuation/supervisor/` · runners `experiments/week6_trackb/` · tests `tests/test_week6_tracks.py`.

## 1. B0 diagnosis (MEASURED; dev seeds 12 000 000–12 000 039, 80 episodes, 10 580 steps)

| | Original CLF-DR-CBF | Random-CLF-DR-CBF |
|---|---|---|
| infeasible steps | 148 / 5 178 = 0.0286 | 146 / 5 402 = 0.0270 |
| infeasible runs (length 1 / mean / max) | 75 (35 / 1.97 / 5) | 80 (51 / 1.82 / 6) |
| what follows a run start | collision ≤ 10 steps **22**, still struggling 38, escaped 15 | recovery used **74**, collision ≤ 10 steps 6 |

**Infeasibility is short-lived and bursty, and its consequences differ sharply by controller** —
without recovery, 29 % of infeasible runs precede a collision within 1 s; with recovery, 8 %.

**Predictive value at a 3-step horizon** (computed on feasible steps only; base rate 0.0396):

| runtime feature | AUC | ground-truth reference | AUC |
|---|---|---|---|
| confirmed tracks within 2 m | **0.75** | true clearance | 0.45 |
| −(margin of the u = 0 action), `−m_u0` | **0.74** | true dynamic clearance | 0.38 |
| −(worst closing speed) | 0.69 | true obstacle count ≤ 2 m | 0.61 |
| corridor width | 0.59–0.64 | | |

Two findings worth stating: **obstacle count does have explanatory value** (it was tested, not
assumed), and **runtime LiDAR/tracker features beat ground-truth clearance** — infeasibility is
driven by how many things are closing on the robot, not by how much room it has.

## 2. Risk score (FROZEN on dev before evaluation)

`risk = (density + margin + closing) / 3`, each component clipped to [0, 1]:
`density = (n_tracks≤2m − 2)/3`, `margin = (0.35 − m_u0)/0.70`, `closing = (−worst_closing − 0.8)/1.0`.
Dev AUC **0.831**. Engage at 0.50, release at 0.35 (hysteresis), ramped speed scale down to 0.40,
50-step engagement cap, 20-step cooldown. All inputs runtime-observable; a test forbids `robot_env`
imports and ground-truth names in the supervisor package.

## 3. Factorial evaluation (MEASURED; eval seeds 12 100 000–12 100 199, 200 paired episodes/arm)

Terminal outcomes, mutually exclusive (goal + collision + timeout = 200 in every arm):

| arm | goal | collision | timeout | dynamic / static / wall collisions |
|---|---|---|---|---|
| O-off | 148/200 = 0.740 | 52/200 = 0.260 | 0 | 0.255 / 0.005 / 0 |
| O-on | 144/200 = 0.720 | 55/200 = 0.275 | 1/200 = 0.005 | 0.275 / 0.000 / 0 |
| R-off | 173/200 = 0.865 | 27/200 = 0.135 | 0 | 0.135 / 0.000 / 0 |
| R-on | **175/200 = 0.875** | **25/200 = 0.125** | 0 | 0.125 / 0.000 / 0 |

Paired, supervisor on vs off (exact McNemar; positive = supervisor higher):

| family | metric | diff | 95 % CI | n10/n01 | p |
|---|---|---|---|---|---|
| Original | goal | −0.020 | [−0.065, +0.025] | 9/13 | 0.52 |
| Original | collision | +0.015 | [−0.030, +0.060] | 12/9 | 0.66 |
| Original | **infeasible-step rate** | −0.0036 (0.0354 → 0.0318) | [−0.0080, +0.0008] | cluster bootstrap | **not significant** |
| Random | goal | +0.010 | [−0.025, +0.045] | 7/5 | 0.77 |
| Random | collision | −0.010 | [−0.045, +0.025] | 5/7 | 0.77 |
| Random | **infeasible-step rate** | **−0.0054 (0.0208 → 0.0154, −26 %)** | **[−0.0082, −0.0026]** | cluster bootstrap | **significant** |

Supervisor behaviour and cost:

| | O-on | R-on |
|---|---|---|
| engaged fraction of steps | 0.289 | 0.258 |
| infeasible steps occurring while engaged | 754 of 860 (**88 %**) | 347 of 397 (**87 %**) |
| mean speed scale when engaged | 0.816 | 0.802 |
| "unnecessary" engagements (risk < 0.5 while engaged, i.e. held by hysteresis) | 0.204 | 0.180 |
| infeasible runs per episode | 2.15 vs 2.10 off | **1.11 vs 1.32 off** |
| mean run length | 1.83 vs 2.02 off | 1.08 vs 1.37 off |
| recovery events | — | 397 vs 524 off |
| min clearance | 0.188 vs 0.192 off | 0.189 vs 0.183 off |
| SPL | 0.626 vs 0.641 off | 0.749 vs 0.743 off |
| steps per episode | 135.3 vs 131.6 off | 128.9 vs 126.1 off |
| step time mean / p95 | 36.96 / 41.70 ms | 36.69 / 41.60 ms |
| collisions per 1 000 steps (descriptive) | 2.03 vs 1.98 off | 0.97 vs 1.07 off |

## 4. Interpretation

1. **The risk score works as a detector.** ~29 % of steps are flagged and they contain ~88 % of all
   infeasible steps, in both families — a ~3× enrichment, matching the dev estimate.
2. **Prevention reduces infeasibility only for Random-CLF-DR-CBF**: −26 % relative, CI excludes 0,
   with fewer and shorter infeasible runs (1.11 vs 1.32 runs/episode, mean length 1.08 vs 1.37) and
   27 % fewer recovery events. For the Original controller the reduction is not significant.
3. **No task-outcome benefit was demonstrated in either family.** Goal and collision differences are
   all within noise (p ≥ 0.52). Per the pre-declared anti-claim, **the supervisor is not called a
   safety improvement**: fewer infeasible QPs did not translate into fewer collisions or more goals.
4. **The cost is small but real**: +3 to +4 steps per episode, SPL −0.015 for Original, and one
   timeout appeared in O-on that does not exist in any other arm. Runtime overhead is negligible
   (step time is if anything slightly lower, i.e. within noise).
5. **Why the asymmetry is plausible (HYPOTHESIS):** slowing down shrinks the closing rates that make
   a row unsatisfiable, which shortens bursts. Random recovery already escapes bursts, so prevention
   and recovery compound (fewer, shorter bursts). Without recovery, the Original controller still
   enters the same geometry and stops there via u = 0, so a milder nominal changes little.

## 5. Limitations

- **One intervention only** (Candidate 1, speed reduction). Wait, retreat and route choice are
  specified in the design note and **not implemented**, on the pre-declared rule that later
  candidates are only built if the first shows value. It did not, for task outcomes.
- Single map, single obstacle-speed setting, 200 episodes per arm: the design can detect ~±6 pp
  outcome differences, not small ones. **"Not significant" here is not "no effect".**
- The risk threshold was frozen on 80 dev episodes; a different operating point trades trigger rate
  against recall and was not explored on evaluation data (deliberately).
- The supervisor only rescales the nominal action. It cannot reroute, and it never overrides the
  frozen QP, so it can prevent only what a slower nominal can prevent.
- `collisions per 1 000 steps` is **descriptive**; episode lengths differ between arms, and no
  formal test was pre-declared for it.
- The frozen infeasible fallback (u = 0) and the frozen Random recovery settings were unchanged, so
  these results say nothing about tuning either.
