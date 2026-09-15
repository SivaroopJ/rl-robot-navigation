# Pursuit–Evasion: known-state baseline results

**KNOWN-STATE / PRIVILEGED-INFORMATION EXPERIMENT.** The protagonist is given the hunter's true
position and velocity through one oracle barrier row (audit §4.3). This is the plan's staged
baseline, not a perception result. LiDAR-based hunter detection is a separate later experiment.

Base commit `07d74ca`, branch `Week6`. Runs 2026-09-12. Environment: canonical M0, **stochastic**
obstacle motion (`SmoothStochasticMotion`), 6 obstacles @ 0.675 m/s, 5 rectangles, dt 0.1,
max 500 steps, |u| ≤ 1 per axis for the protagonist.

Protagonist = frozen **Tuned + Random CLF-DR-CBF** (α 0.8, r_W 0.012, ε 0.1, k_v 0.10, N 5,
τ_eff 0.12; K 16, H 1, speeds {0.8, 1.0}, accept m ≥ 0). Hunter = pursuit nominal + the same tuned
DR-CBF filter, frozen u = 0 fallback. **No LADDER anywhere, for either agent.**
Capture 0.65 m, contact 0.60 m, priority own-collision → capture → goal → timeout.

---

## 1. Stages run

| stage | env | episodes | result |
|---|---|---|---|
| P0 regression | frozen navigation | 20 | **PASS** — 12/12 fields reproduced exactly |
| P1 sanity | no obstacles | 20 | **PASS** — all 4 mechanical checks |
| P2 static | 5 rectangles, no dynamic | 20 | **PASS** — all 3 checks |
| P3 smoke | canonical + stochastic | 10 × 2 | PASS — pairing verified |
| **P3 main (primary)** | canonical + stochastic | **200 × 2** | hunter \|u\| ≤ 1.0 |
| **P3 sweep (labelled extra)** | canonical + stochastic | **200 × 2** | hunter \|u\| ≤ 1.25 |

P1: capture fires exactly at 0.65 (captured ⟺ min distance ≤ 0.65); the hunter closed distance in
100 % of episodes; goal 0.55, capture 0.40. P2: **the hunter never hit a rectangle**; goal 0.70,
capture 0.20; the hunter's nominal action was unsafe on 67 % of steps and it stalled on its own
safety constraints in 9/20 episodes.

## 2. Primary experiment — hunter |u| ≤ 1.0 (200 paired episodes, seeds 11 300 000–11 300 199)

| outcome | P-Random (primary) | P-Tuned (ablation) |
|---|---|---|
| protagonist reaches goal | **0.535** [0.466, 0.603] | 0.425 [0.359, 0.494] |
| hunter captures | **0.240** [0.186, 0.304] | 0.340 [0.278, 0.408] |
| protagonist collision (any) | 0.240 [0.186, 0.304] | 0.235 [0.182, 0.298] |
| — static / dynamic / wall | 0.070 / 0.170 / 0 | 0.005 / 0.230 / 0 |
| hunter collision (any) | 0.190 [0.142, 0.250] | 0.155 [0.111, 0.212] |
| — static / dynamic / wall | 0 / 0.190 / 0 | 0 / 0.155 / 0 |
| agent–agent physical contact | **0.000** [0.000, 0.019] | 0.160 [0.116, 0.217] |
| timeout | 0.000 | 0.000 |

Intervals are Wilson 95 %. Paired comparison (exact McNemar; positive = P-Random higher):

| metric | diff | 95 % CI | discordant | p |
|---|---|---|---|---|
| goal | **+0.110** | [+0.060, +0.165] | 27 / 5 | **0.0001** |
| captured | **−0.100** | [−0.155, −0.045] | 7 / 27 | **0.0008** |
| protagonist collision | +0.005 | [−0.060, +0.070] | 23 / 22 | 1.000 |
| hunter collision | +0.035 | [−0.005, +0.080] | 13 / 6 | 0.167 |
| agent contact | **−0.160** | [−0.215, −0.110] | 0 / 32 | **< 0.0001** |
| timeout | 0.000 | — | 0 / 0 | 1.000 |

Other measures: time to goal 12.93 s [11.78, 14.12] vs 11.65 s [10.56, 12.77]; time to capture
12.05 s [10.45, 13.81] vs **7.21 s** [6.45, 7.98]; min protagonist–hunter distance 1.54 vs 1.59 m;
min obstacle clearance 0.154 vs 0.176 m; SPL 0.452 vs 0.382.

## 3. Speed sweep — hunter |u| ≤ 1.25 (separate, never pooled with §2)

| outcome | P-Random | P-Tuned | paired diff (p) |
|---|---|---|---|
| goal | **0.385** [0.320, 0.454] | 0.310 [0.250, 0.377] | +0.075 (**0.0007**) |
| captured | 0.430 [0.363, 0.499] | 0.475 [0.407, 0.544] | −0.045 (0.163) |
| protagonist collision | 0.180 | 0.220 | −0.040 (0.280) |
| hunter collision | 0.130 | 0.115 | +0.015 (0.508) |
| agent contact | **0.000** | 0.225 | **−0.225 (< 0.0001)** |

A 25 % faster hunter raises capture from 0.24 → 0.43 and cuts the protagonist's goal rate
0.535 → 0.385, so the pursuit pressure behaves as expected. The Random advantage in goal rate and
the zero-contact result both survive.

## 4. Controller and runtime statistics

| | P-Random | P-Tuned |
|---|---|---|
| protagonist infeasible-step rate | 0.069 | 0.061 |
| protagonist recovery fraction (steps) | 0.069 | 0 (by construction) |
| protagonist recovery events | 1 671 | 0 |
| protagonist mean ‖u − u_nom‖ | 0.756 | 0.683 |
| hunter infeasible-step rate | 0.021 | 0.021 |
| hunter u = 0 fallback steps | 1 510 | 1 175 |
| hunter nominal-action-unsafe rate | 0.742 | 0.739 |
| hunter materially filtered (‖u−u_nom‖ > 0.05) | 0.861 | 0.854 |
| protagonist step time mean / p95 | 36.0 / 40.8 ms | 35.2 / 39.7 ms |
| hunter step time mean / p95 | 35.2 / 40.5 ms | 34.7 / 40.0 ms |
| worst per-episode p95 | 48.4 ms (151.9 ms in the sweep) | 48.7 ms |
| episode wall time / steps | 9.1 s / 123 | 7.0 s / 97 |

The hunter's safety filter is doing near-continuous work: its nominal pursuit action violates the DR
constraint on ~74 % of steps, and it falls back to u = 0 on ~2 % of steps.

## 5. Does the protagonist's Random recovery remain useful under active pursuit?

**Yes, and more clearly than in navigation.**

1. **Goal rate: +11.0 pp** (0.535 vs 0.425, p = 0.0001, 27 vs 5 discordant). In the frozen
   navigation study the recovery's benefit was a *collision* reduction; under pursuit it converts
   directly into episodes completed.
2. **Captures: −10.0 pp** (0.240 vs 0.340, p = 0.0008), and captures that do happen take 67 % longer
   (12.05 s vs 7.21 s). The protagonist is materially harder to catch.
3. **Physical contact with the hunter: 0/200 vs 32/200** (p < 0.0001). This is the mechanism. The
   frozen fallback on an infeasible QP is **u = 0** — the protagonist stops dead, and a pursuer
   closes on a stationary target. Random recovery replaces "stop" with a bounded exploratory
   velocity at 0.8–1.0 m/s, so the protagonist keeps moving through exactly the moments when the
   hunter is most able to close. Contact fell to zero in both speed conditions.
4. **The safety trade is a wash, not a win.** Total protagonist collisions are unchanged
   (0.240 vs 0.235, p = 1.0), but the *composition* shifts: dynamic-obstacle collisions fall
   0.230 → 0.170 while static collisions rise 0.005 → 0.070. Escaping at speed sometimes escapes
   into a rectangle. Min obstacle clearance is slightly worse (0.154 vs 0.176 m).
5. **Cost:** +0.8 ms mean and +1.1 ms p95 per step, and longer routes (SPL 0.452 vs 0.382 — both
   arms detour under pursuit, Random less so; time-to-goal is 1.3 s longer because it survives
   harder episodes that the ablation loses).

So under pursuit the recovery buys **goal completion and evasion**, at no net collision cost but with
a redistribution of collision cause toward static geometry.

## 6. Limitations

- **Known-state, not perception.** The protagonist receives the hunter's exact position and velocity
  through an oracle barrier row; the hunter receives the protagonist's true position. Every number
  here is conditional on that. LiDAR-based hunter detection is a separate experiment and would be
  the unprivileged arm.
- The hunter is **not** rendered to the protagonist's LiDAR in this baseline, so the oracle row is
  its only channel; there is no double counting, but also no sensing error, occlusion or track loss.
- **One map (M0), one hunter, one pursuit law** (pure pursuit + safety filter). No interception or
  predictive pursuit, no hunter recovery, no learned pursuer.
- The hunter uses the protagonist's tuned hyperparameters; it was never tuned for pursuit, and its
  nominal action is unsafe ~74 % of the time, so it is a *safe* pursuer rather than a strong one.
- **Hunter-crash rule** (disabled in place, episode continues) is a pre-declared convention; a
  hunter crashed in 13–19 % of episodes, which ends the pursuit pressure early in those episodes.
- Capture at 0.65 m is a proximity criterion, not a physical tag.
- p-values are unadjusted; the primary comparison makes 6 paired tests.
- Both conditions face an identical environment, but the *hunter's* trajectory necessarily differs
  between them, because it reacts to a differently-behaving protagonist. Only the environment seeds
  are paired; the pursuit itself is not.
