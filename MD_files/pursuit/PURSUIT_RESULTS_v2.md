# Pursuit–evasion: known-state baseline results (v2, audited)

Supersedes `PURSUIT_RESULTS.md`, which is preserved unchanged. All numbers here were recomputed
from the raw episode records; see `PURSUIT_AUDIT.md` for the counts behind every rate. No rerun was
needed and none was performed.

**KNOWN-STATE / PRIVILEGED-INFORMATION EXPERIMENT.** The protagonist receives the hunter's true
position and velocity through one oracle barrier row; the hunter receives the protagonist's true
position. This is the plan's staged baseline, **not** a perception result.

Setup: canonical M0, stochastic obstacles (`SmoothStochasticMotion`, 6 @ 0.675 m/s), 5 rectangles,
dt 0.1, max 500 steps. Protagonist = frozen Tuned + Random CLF-DR-CBF (α 0.8, r_W 0.012, ε 0.1,
k_v 0.10, N 5, τ_eff 0.12; K 16, H 1, speeds {0.8, 1.0}). Hunter = pursuit nominal + the same tuned
DR-CBF filter, frozen u = 0 fallback. No LADDER for either agent. Capture 0.65 m, contact 0.60 m,
priority own-collision → capture → goal → timeout. 200 paired episodes per condition,
seeds 11 300 000–11 300 199, disjoint from all navigation blocks.

---

## 1. Terminal outcomes (mutually exclusive, sum to 1.000)

Every episode resolves to exactly one of these under the pre-declared priority.

| terminal outcome | P-Random (primary) | P-Tuned (ablation) | paired diff | n10/n01 | McNemar p |
|---|---|---|---|---|---|
| goal reached | **107/200 = 0.535** | 85/200 = 0.425 | **+0.110** | 27/5 | **0.0001** |
| captured | **45/200 = 0.225** | 68/200 = 0.340 | **−0.115** | 6/29 | **0.0001** |
| protagonist collision | 48/200 = 0.240 | 47/200 = 0.235 | +0.005 | 23/22 | 1.000 |
| timeout | 0/200 | 0/200 | 0 | 0/0 | 1.000 |

Wilson 95 % intervals: goal 0.535 [0.466, 0.603] vs 0.425 [0.359, 0.494]; capture 0.225
[0.172, 0.288] vs 0.340 [0.278, 0.408]; collision 0.240 [0.186, 0.304] vs 0.235 [0.182, 0.298].

## 2. Event flags (overlapping — an episode may set several)

These are *not* terminal outcomes and do **not** sum to 1. The capture flag counts every episode in
which the hunter was within 0.65 m on the terminal step, including 3 P-Random episodes scored as
collisions by priority.

| event flag | P-Random | P-Tuned | paired diff | n10/n01 | p |
|---|---|---|---|---|---|
| capture-range reached | 48/200 = 0.240 | 68/200 = 0.340 | −0.100 | 7/27 | 0.0008 |
| protagonist collision — dynamic | 34/200 = 0.170 | 46/200 = 0.230 | −0.060 | 13/25 | **0.073 (n.s.)** |
| protagonist collision — static | 14/200 = 0.070 | 1/200 = 0.005 | +0.065 | 13/0 | **0.0002** |
| protagonist collision — wall | 0 | 0 | 0 | 0/0 | 1.000 |
| hunter collision (all dynamic) | 38/200 = 0.190 | 31/200 = 0.155 | +0.035 | 13/6 | 0.167 |
| episode ended inside contact distance | **0/200** | 32/200 = 0.160 | −0.160 | 0/32 | **< 0.0001** |

## 3. Capture and contact semantics (see audit §2)

- Capture **terminates** the episode; contact never does.
- Both are evaluated in the same step, after both agents move. Since 0.60 < 0.65, contact implies
  capture in that step unless the hunter has already crashed (a disabled hunter cannot capture).
- The contact figure is an **episode-level flag at the terminal step**, not an event count.
- Of the 32 P-Tuned contacts: **31 at the capture step**, 1 in a collision-terminated episode with an
  already-crashed hunter. None before capture; none without capture.
- The difference is **overshoot at the capture boundary**, not contact avoidance: P-Random captures
  land in [0.602, 0.650] (0 % below 0.60), P-Tuned in [0.557, 0.650] (46 % below 0.60). A stationary
  protagonist lets the hunter close 0.10 m in one step; a moving one does not.

## 4. Other measures

| | P-Random | P-Tuned |
|---|---|---|
| time to goal (n = 107 / 85) | 12.93 s [11.78, 14.12] | 11.65 s [10.56, 12.77] |
| time to capture (n = 45 / 68, terminal captures) | 12.05 s [10.45, 13.81] | 7.21 s [6.45, 7.98] |
| min protagonist–hunter distance | 1.540 m [1.395, 1.694] | 1.589 m [1.432, 1.753] |
| min obstacle clearance | 0.154 m | 0.176 m |
| SPL | 0.452 | 0.382 |
| total steps survived | 24 596 | 19 338 |
| obstacle collisions per 1 000 steps (dyn / static / total) | 1.38 / 0.57 / **1.95** | 2.38 / 0.05 / **2.43** |
| protagonist infeasible-step rate (pooled) | 0.068 | 0.047 |
| protagonist recovery events | 1 671 | 0 |
| hunter infeasible / fallback rate (pooled) | 0.021 / 0.061 | 0.021 / 0.061 |
| hunter nominal-action-unsafe (mean of ratios / pooled) | 0.742 / 0.751 | 0.739 / 0.776 |
| hunter crashes | 38/200 (19 %) | 31/200 (15.5 %) |
| protagonist step time mean / p95 | 36.0 / 40.8 ms | 35.2 / 39.7 ms |
| hunter step time mean / p95 | 35.2 / 40.5 ms | 34.7 / 40.0 ms |

## 5. Speed sweep — hunter |u| ≤ 1.25 (separate, never pooled)

| terminal outcome | P-Random | P-Tuned | paired diff | p |
|---|---|---|---|---|
| goal | 77/200 = 0.385 | 62/200 = 0.310 | +0.075 | **0.0007** |
| captured | 86/200 = 0.430 | 94/200 = 0.470 | −0.040 | 0.230 (n.s.) |
| protagonist collision | 36/200 = 0.180 | 44/200 = 0.220 | −0.040 | 0.280 (n.s.) |
| timeout | 1/200 = 0.005 | 0/200 | +0.005 | 1.000 |

Event flags: dynamic collision 0.135 vs 0.215 (**p = 0.011**), static 0.045 vs 0.005 (p = 0.0078),
ended-inside-contact 0.000 vs 0.225 (p < 0.0001), hunter collision 0.130 vs 0.115 (p = 0.508).
A 25 % faster hunter raises terminal capture 0.225 → 0.430 and cuts goal 0.535 → 0.385, so pursuit
pressure scales as expected. The goal-rate advantage and the zero-overshoot result both survive.

## 6. Does Random recovery remain useful under active pursuit?

**Yes for task completion and evasion; no for total collisions.**

1. **Goal completion +11.0 pp** (0.535 vs 0.425, p = 0.0001, 27 vs 5 discordant). Verified on both
   flag and terminal definitions.
2. **Terminal captures −11.5 pp** (0.225 vs 0.340, p = 0.0001), and the captures that still occur
   take 12.05 s instead of 7.21 s (n = 45 vs 68).
3. **Mechanism, supported by the terminal-distance distributions:** the frozen fallback on an
   infeasible QP is **u = 0**. A stationary protagonist is easy to close on, and 46 % of P-Tuned
   captures end inside the physical contact distance. Random recovery keeps the protagonist moving
   at 0.8–1.0 m/s through exactly those moments, so no P-Random capture ended inside contact
   distance. The episode still ends at capture either way — this is a *margin* result, not
   collision avoidance.
4. **Total protagonist collisions are unchanged**: 48 vs 47 of 200 (+0.005, p = 1.000). What changes
   is composition: static collisions rise significantly (14 vs 1, p = 0.0002), while the
   dynamic-collision fall (34 vs 46) is **not significant at the primary speed** (p = 0.073) though
   it is in the sweep (p = 0.011). Escaping at speed sometimes escapes into a rectangle.
5. **Per unit of time survived**, obstacle collisions are lower (1.95 vs 2.43 per 1 000 steps), but
   P-Random episodes are longer precisely because they survive, so this is descriptive only and was
   not a pre-declared test.
6. **Cost:** +0.8 ms mean / +1.1 ms p95 per step; min obstacle clearance slightly worse
   (0.154 vs 0.176 m).

## 7. Limitations

- **Known-state, privileged information.** The protagonist is handed the hunter's exact position and
  velocity through an oracle barrier row; there is no sensing error, occlusion or track loss. The
  hunter is not rendered to the protagonist's LiDAR. LiDAR-based hunter detection is a separate,
  later experiment and would be the unprivileged arm.
- **Overlapping flags vs terminal outcomes.** §2's flags may co-occur; only §1 sums to 1. Any
  headline rate must state which definition it uses.
- **Contact is an episode-level terminal-step flag**, not an event count. Contact with a *disabled*
  hunter at a non-terminal step is not observable in the current logs. The frozen environment gives
  the hunter no physical collision body, so "protagonist collides with hunter" exists only as this
  proximity flag.
- **Total collisions are unchanged**; no claim of collision reduction is made at the primary speed.
- **Exposure differs between arms** (24 596 vs 19 338 steps), which confounds any per-episode count
  of an event that accumulates with time.
- One map (M0), one hunter, one pursuit law (pure pursuit + filter). No interception, prediction,
  hunter recovery or learned pursuer.
- The hunter uses the protagonist's tuned hyperparameters and was never tuned for pursuit: its
  nominal action is unsafe on ~75 % of steps and it crashes in 11–19 % of episodes, ending pursuit
  pressure early in those episodes (median crash at step 77 of a median 114-step episode).
- Capture at 0.65 m is a proximity criterion, not a physical tag.
- p-values are unadjusted; the primary comparison makes 6 paired tests on flags plus 4 on terminal
  outcomes. The goal and capture results survive Bonferroni at α = 0.005; the static-collision
  result survives at α = 0.005; the dynamic-collision result does not.
- Environment seeds are paired across conditions, but the **hunter's trajectory is not**: it reacts
  to a differently-behaving protagonist, so the two arms face the same world and the same initial
  pursuit geometry, not the same pursuit.
