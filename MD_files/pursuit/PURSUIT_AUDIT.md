# Audit of the pursuit–evasion results (2026-09-12)

Recomputed **from the raw episode records** in `results/pursuit/P3/P3_main_hunter{1,1.25}.json`
with an independent script (own McNemar implementation, own counting). **No rerun was required and
none was performed; no frozen code, log or result file was modified.** The original
`PURSUIT_RESULTS.md` is preserved unchanged; corrections live in `PURSUIT_RESULTS_v2.md`.

## 1. Why the P-Random rates sum to 1.015

**They are overlapping event flags, not mutually exclusive terminal outcomes.** Every rate uses the
same denominator, 200 episodes per condition — no denominator error.

| condition | goal flag | captured flag | prot-collision flag | flag sum | terminal outcomes (exclusive) |
|---|---|---|---|---|---|
| P-Random | 107 | 48 | 48 | 203 → 1.015 | goal 107 · capture **45** · collision 48 = **200** |
| P-Tuned | 85 | 68 | 47 | 200 → 1.000 | goal 85 · capture 68 · collision 47 = **200** |

The excess is exactly **3 P-Random episodes in which the protagonist collided AND the hunter was
within 0.65 m on the same terminal step**. The pre-declared priority (own-collision → capture →
goal → timeout) scores those as collisions, so the terminal capture count is 45, not 48. P-Tuned
happened to have zero such overlaps, which is why its flags summed to 1.000 — coincidence, not a
guarantee of exclusivity. Cross-tabs: goal∧captured = 0 in both arms; goal∧collision = 0 in both
(the frozen env already makes those exclusive); collision∧captured = 3 (P-Random) and 0 (P-Tuned).

**Correction:** the reported 0.240 capture rate for P-Random is the *flag* rate. The *terminal*
capture rate is 45/200 = **0.225**. Every other reported rate is confirmed.

## 2. Capture versus physical contact

**Implementation, verified in `continuation/pursuit/env.py`:**
- Both are evaluated **in the same step**, after both agents and the obstacles have moved.
- `captured = (d <= 0.65) and not hunter_disabled`; `contact = d < 0.60`. Contact never gates
  termination; `terminated = protagonist_collision or captured or goal`.
- So **capture terminates the episode**, and because 0.60 < 0.65, any contact implies capture in the
  same step *unless the hunter has been disabled by its own crash* (a disabled hunter cannot capture
  by construction).

**What the 32 P-Tuned contacts are:** 31 occurred **at the capture step** (the terminating step) and
1 in an episode that terminated as a protagonist collision with an already-crashed hunter nearby.
None occurred before capture, and none in an ordinary episode without capture.

**They are an episode-level flag, not an event count.** The record stores only the final step's
`agent_contact`. Since capture ends the episode at the first crossing of 0.65 m, that is the only
step at which contact can normally be observed.

**The 0-vs-32 difference is an overshoot effect, not contact avoidance.** Both arms end at the first
step where d ≤ 0.65. The terminal distance distribution shows what differs:

| condition | captured episodes | terminal distance min / mean / max | ended below 0.60 |
|---|---|---|---|
| P-Random (1.0) | 48 | 0.602 / 0.636 / 0.650 | **0 (0 %)** |
| P-Tuned (1.0) | 68 | 0.557 / 0.606 / 0.650 | **31 (46 %)** |
| P-Random (1.25) | 86 | 0.610 / 0.638 / 0.650 | 0 (0 %) |
| P-Tuned (1.25) | 95 | 0.526 / 0.600 / 0.650 | 45 (47 %) |

A stationary protagonist (the frozen u = 0 fallback) lets the hunter close a full 0.10 m in one
0.1 s step, so the crossing frequently lands past 0.60. A protagonist still moving at 0.8–1.0 m/s
reduces the closing rate, so the crossing lands inside [0.60, 0.65] every time. The episode ends
either way.

**Limitation that cannot be resolved from the current logs:** contact with a *disabled* hunter at a
non-terminal step would not be recorded, because only the terminal step's flag is stored. Also, the
frozen environment has no physical collision body for the hunter, so "protagonist collides with
hunter" is represented only by this proximity flag.

## 3. Verification of the remaining claims

All denominators are 200 episodes per condition, both speeds; pairing verified (identical seed,
start, goal and hunter start in all 200 pairs, both runs).

| claim (original report) | audited | verdict |
|---|---|---|
| goal +0.110, p = 0.0001 (27/5) | identical for flags **and** terminal outcomes | **confirmed** |
| capture −0.100, p = 0.0008 (7/27) | flag-based; terminal-based is **−0.115, p = 0.0001 (6/29)** | **confirmed, slightly stronger** |
| protagonist collision +0.005, p = 1.000 (23/22) | 48 vs 47 of 200 | **confirmed — unchanged** |
| dynamic 0.170 vs 0.230 | 34 vs 46; paired **p = 0.073** | **not significant** at 1.0 (is significant in the sweep, p = 0.011) |
| static 0.070 vs 0.005 | 14 vs 1; paired **p = 0.0002** | **confirmed, significant** |
| hunter collision 0.190 vs 0.155, p = 0.167 | 38 vs 31, all dynamic | confirmed |
| contact 0 vs 32, p < 0.0001 | confirmed, but see §2 for what it means | **re-interpreted** |
| time to capture 12.05 s vs 7.21 s | means over **terminal-capture** episodes, n = 45 vs 68 | confirmed; denominators differ from the capture *flag* count (48 vs 68) |
| hunter nominal unsafe "~74 %" | mean-of-ratios 0.742 / 0.739; **pooled over steps 0.751 / 0.776** | confirmed as a mean of per-episode ratios |
| sweep goal +0.075, p = 0.0007 | confirmed (17/2) | confirmed |
| sweep capture −0.045, p = 0.163 | flag-based; terminal-based −0.040, p = 0.230 | confirmed, **not significant either way** |
| sweep protagonist collision −0.040, p = 0.280 | 36 vs 44 | confirmed, not significant |

**Exposure confound, newly identified.** P-Random episodes are longer (24 596 vs 19 338 steps at
speed 1.0), because surviving longer is itself the treatment effect. Per 1 000 steps of exposure:

| | dynamic | static | total obstacle | captures |
|---|---|---|---|---|
| P-Random (1.0) | 1.38 | 0.57 | **1.95** | 1.95 |
| P-Tuned (1.0) | 2.38 | 0.05 | **2.43** | 3.52 |

So the equal per-episode collision totals hide a lower collision rate *per unit of time survived*.
This is a descriptive observation, not a pre-declared test.
