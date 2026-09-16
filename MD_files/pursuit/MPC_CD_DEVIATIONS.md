# Policy Matrix — Deviations and Gate Log

Written during execution. `MPC_CD_PREREGISTRATION.md` is unchanged (its hash is still verified by the calibration script);
every departure from it is recorded here with the evidence available at the time, before the affected step ran.
The user instructed: "continue till end even if gate fails" — failures are therefore recorded, not used to stop.

## Gate log
| gate | result | notes |
|---|---|---|
| Existing tests (pursuit, hunter variants, tracks, A2, continuation) | PASS | 147 passed, unchanged from the pre-change baseline |
| New tests `tests/test_policy_matrix.py` | PASS after fixture fixes | 27 passed. Three test-construction errors fixed, no production code changed: (1) the congested-monitor fixture put four points in ONE scan (the source yields one row per scan, so it was not congested); (2) a hysteresis assertion used an interrupt value (0.12) that still satisfied Critical's exit condition, and miscounted the hold; (3) the already-infeasible fixture used h < 0, which hits the frozen objective's DCPError path (not an "infeasible" status, so Random correctly does not act) — replaced by h = 0.01 with opposing closing rows, which is genuinely infeasible |
| Config A exact match (10 seeds × 4 policies × 2 maps) | PASS | every non-timing field identical to the historical records |
| Config B smoke (5 seeds × 6 policies × 2 maps) | PASS | no crash; MPC ≈ 5 ms/step, fallback ≈ 5–6 %, ≈ 20/42 candidates rejected |
| Config C smoke (5 seeds × 6 policies × 2 maps, frozen D1 thresholds) | PASS | no crash; MPC ≈ 5 ms/step; one static timeout (policy 3) consistent with frequent Critical caution |

## D1 — congestion calibration: the 25 % warning cap cannot be met by any (c1, c2)

**Evidence (smoke seeds 14 000 000–004, config B shadow; not evaluation or development data).**
With the least sensitive grid pair (c1 = 0, c2 = 0.05), which coincides with the fixed, non-calibrated Critical rule
(margin < 0 or predicted clearance < 0.05), the detector is already at state ≥ 1 on 62 % of active steps.
Decomposition of active steps:

| agent / map | margin < 0 | predicted clr < 0.05 | either | actual infeasible | grid margin at t=0 < 0 |
|---|---|---|---|---|---|
| Protag / dynamic | 0.387 | 0.642 | 0.680 | 0.078 | 0.115 |
| Protag / static | 0.375 | 0.497 | 0.525 | 0.071 | 0.082 |
| Hunter / dynamic | 0.175 | 0.298 | 0.348 | 0.019 | 0.023 |
| Hunter / static | 0.163 | 0.327 | 0.367 | 0.009 | 0.009 |

A single-episode replay (smoke seed 14 000 001, Protag) ruled out an obstacle-model bug: sensed clearance at the current
position tracks true geometry (median 0.55 m vs 0.66 m true). The low predicted clearances come from properties of the
approved design, not from code errors:
1. the CBF clearance convention subtracts the robot radius from wall readings that the env already insets by 0.3 m, so walls read 0.3 m closer;
2. the look-ahead follows the straight nominal line for 0.5 s and ignores the QP's own deflection (in the replay, steps below 0.05 m rose from 4 % at t = 0 to 16 % at t = 0.5 s);
3. the frozen tracker holds more confirmed tracks than there are moving obstacles (median 9 for 6 obstacles), each adding a 0.6 m clearance disc;
4. the grid margin at predicted states falls below τ far more often than the QP is actually infeasible.

**Deviation.** The pre-registered rule has no branch for "no pair meets the cap". Extension, applied unchanged to the
development data: if no pair has warning fraction ≤ 0.25, select the pair with the **lowest warning fraction**
(ties: higher coverage, then smaller c1, then smaller c2) and report the achieved coverage and warning fraction as a
**failed calibration target**. The detector, look-ahead, clearance convention, Critical rule and caution response are NOT
changed: Phase III tests the approved design as specified. Consequence to expect: caution (speed scale 0.6 / 0.3, CLF
target pulled in) will be active on a large share of steps for both agents; Phase III results must be read as a test of
this detector definition, not of congestion detection in general.

### D1 outcome (development seeds 14 300 000–049, config B shadow, policies 1 and 3, both maps, 400 episodes)
Calibration applied the D1 branch: **c1 = 0.0, c2 = 0.05** (frozen in `results/pursuit_policy_matrix/calibration/cd_thresholds.json`).
Achieved: warning fraction **0.559** (Protag 0.689, Hunter 0.427) against the 0.25 cap; coverage 0.992 (646 of 651
infeasibility onsets warned; Protag 501/501, Hunter 145/150). **Calibration targets not met.**
Consequence: c1 and c2 now equal the Critical thresholds, so the Cautious band is empty — every warning is Critical
(speed scale 0.3, CLF target pulled to 30 %, direction search ±30°). Near-complete coverage here mostly reflects how often the
detector is on, not discrimination.

## Validation and final gate status
| check | result |
|---|---|
| Phase I (main_B) artifacts | VALID — 12 cells × 200 episodes, no unresolved outcomes, 2 400 step logs, layouts paired with config A for policies 1–2.5 |
| Phase III (main_C) artifacts | VALID — 12 cells × 200 episodes, 2 400 step logs, layouts paired with config B, frozen thresholds recorded in the run meta |
| Chain steps (calibrate, smoke C, main C, latency A/B/C, report) | all exit 0 |
| Final test run (new + existing) | 174 passed |
No run failed or was left incomplete. The only failed gate in this experiment is the congestion calibration target (D1).

## Extra diagnostic run (not part of the pre-registered matrix)
`check_B` — fresh seed block `HV_check` (14 400 000–099), config B, policies 3 and 3.5 only, both maps, 400 episodes,
added to answer "why does 3.5 differ from 3?" (report §12). Two lines of code accompany it: the `HV_check` block in
`hunter_policies.py` and the `check` phase in `policy_matrix.py`. No evaluation seed was reused and no threshold or
parameter was changed.

## D1 follow-up: congestion-fix ablation (development seeds only)

Three fixes were implemented behind config flags (defaults reproduce the pre-registered detector exactly; 27 tests pass):
`--cd-clearance rows` (clearance from the controller's own barrier rows), `--cd-lookahead executed` (look ahead along the
last executed, CBF-filtered velocity instead of the raw nominal line), `--cd-tracks strict` (a track counts only if updated
this step and built from >= 2 points). Four shadow runs on the development block (14 300 000–049, policies 1 and 3, both
maps, 400 episodes each) with calibration under the unchanged pre-registered rule:

| variant | clearance | look-ahead | tracks | warning fraction | coverage | precision | Protag warn/prec | hunter warn/prec |
|---|---|---|---|---|---|---|---|---|
| baseline (pre-registered) | sensed | nominal | confirmed | 0.559 | 0.992 | 0.199 | 0.689 / 0.223 | 0.427 / 0.177 |
| **fixall** | rows | executed | strict | **0.278** | 0.876 | **0.348** | 0.379 / 0.440 | 0.176 / 0.216 |
| fix23 (no clearance fix) | sensed | executed | strict | 0.280 | 0.879 | 0.344 | 0.382 / 0.435 | 0.177 / 0.214 |
| fix13 (no look-ahead fix) | rows | nominal | strict | 0.497 | 0.912 | 0.204 | 0.651 / 0.230 | 0.340 / 0.177 |
| fix12 (no track fix) | rows | executed | confirmed | 0.356 | 0.989 | 0.319 | 0.439 / 0.416 | 0.273 / 0.205 |

**Calibration targets are still NOT met** (best 0.278 warnings against the 0.25 cap), so no threshold file was promoted and
Phase III was NOT rerun. The improvement is nonetheless large: warnings halve (0.559 -> 0.278) and precision nearly doubles
(0.199 -> 0.348) at coverage 0.876.

Attribution:
1. **Look-ahead along the executed velocity does almost all the work** (removing it: 0.278 -> 0.497). The pre-registered
   straight-line nominal look-ahead predicted collisions the QP was already steering away from.
2. **Strict track gating helps** (removing it: 0.278 -> 0.356), consistent with the tracker holding ~9 confirmed tracks for
   6 obstacles.
3. **The clearance fix is a NO-OP as implemented** (0.278 vs 0.280). Verified on aligned logs: rows- and sensed-clearance are
   the same number (median 0.337 vs 0.339 m, mean difference -0.0006, 0.041 vs 0.043 of steps below 0.05). Both subtract the
   robot radius from a wall reading the env has already inset by 0.3 m, so the double inset survives in both. Correcting it
   needs a different zero-point (identify wall-like returns, or shift the threshold), which was not attempted.
   The earlier D1 hypothesis that the wall inset was a main driver is therefore **not supported**; the nominal look-ahead was.

Residual drivers after the fixes (share of active steps, fixall): Protag margin < 0 on 0.168 and clearance < 0.05 on 0.057
(only-margin 0.129, only-clearance 0.018); hunter 0.040 / 0.021. Raw triggers therefore fire on ~0.186 (Protag) and ~0.052
(hunter) of steps, while the detector *state* is Cautious-or-worse on 0.379 and 0.176 — the hysteresis dwell (3 clearing
steps at +0.05) multiplies warning time by 2-3x and is now the largest single lever, ahead of either trigger.

Next levers, none attempted: calibrate the Critical thresholds too (they are fixed today, which is why the cap was
unreachable), shorten the hold/hysteresis, or replace the instantaneous test with time-to-infeasibility along the look-ahead.

## D1 follow-up 2: searching the Critical thresholds and hysteresis too (fix 4, exploratory)

The pre-registered rule fixed Critical at (margin < 0, clr < 0.05) and calibrated only (c1, c2) — which is why its cap was
unreachable. `experiments/pursuit/cd_sweep.py` searches all four thresholds plus the hysteresis band and hold length.
No simulation: in shadow mode the detector is a pure function of the logged series, so 1 296 configurations are replayed
offline over the `dev_B_fixall` logs. Development seeds were split in half — configuration SELECTED on 25 seeds, reported on
the held-out 25 — with the rule declared in the script docstring before running (max precision subject to coverage >= 0.80
and warnings <= 0.25).

Selected: **c1 = c2 = crit_margin = crit_clear = −0.05, hysteresis = 0, hold = 3**.

| half | coverage | warning fraction | precision |
|---|---|---|---|
| select (25 seeds) | 0.801 | 0.202 | 0.427 |
| **held-out (25 seeds)** | **0.828** | **0.204** | **0.472** |
| held-out, Protag | 0.876 | 0.267 | 0.620 |
| held-out, hunter | 0.694 | 0.141 | 0.266 |

**Targets MET on held-out data**, and select/held-out agree closely, so the search did not overfit. Versus the
pre-registered detector (0.559 warnings, precision 0.199): warnings fall ~2.7x and precision rises ~2.4x.

Honest caveats: (a) c1/c2 again coincide with the Critical thresholds, so the Cautious band is empty — the gain comes from
relaxing the Critical trigger (margin < −0.05 instead of < 0, clearance < −0.05) and removing the hysteresis band, not from a
graded response; (b) hunter coverage drops to 0.694, i.e. the detector now misses ~31 % of hunter infeasibility onsets — the
warning-rate reduction is partly bought with missed events; (c) this is an exploratory, post-hoc search, not the
pre-registered procedure, and it is reported separately from the Phase I / Phase III results, which stand as produced.

### C2: Phase III rerun with the fixed detector (evaluation seeds, exploratory)

`results/pursuit_policy_matrix/main_C_fix4` — config C with `clearance=rows, lookahead=executed, tracks=strict` and the
fix-4 thresholds (c1 = c2 = crit_margin = crit_clear = −0.05, hysteresis 0, hold 3). 2 400 episodes, same 200 evaluation
seeds, VALID (200 per cell, no unresolved outcomes, 2 400 step logs, layouts paired with B and C). Tables:
`MPC_CD_REPORT_DATA_v2.md`.

**Detector behaviour transfers from development to evaluation seeds** (held-out dev predicted 0.204 warnings / 0.472 precision):

| agent | warnings C → C2 | precision C → C2 | coverage C → C2 |
|---|---|---|---|
| Protag | 0.720 → **0.260** | 0.085 → **0.499** | 0.996 → 0.822 |
| hunter | 0.484 → **0.128** | 0.083 → **0.233** | 0.934 → 0.654 |

So the fixed detector is ~3x less intrusive and ~5x more precise, at the cost of missing 18 % (Protag) and 35 % (hunter) of
infeasibility onsets. Timeouts, which the pre-registered detector introduced (up to 0.060 per cell), nearly vanish in C2
(≤ 0.010). Protag infeasible-step rates land between B and C (e.g. static policy 1: B 0.102, C 0.067, C2 0.091), as expected
from a detector that intervenes less often.

**Outcomes.** C2 − B (fixed detector vs no detector): **nothing significant** after correction on either map (closest:
policy 1 static capture +0.090, p = 0.0064; goal −0.080, p = 0.0037). C2 − C: one significant change, **policy 3.5 dynamic
capture −0.140** (14/42 discordant, p = 0.00023 ✱), i.e. the large MPC-with-filtering capture advantage seen under the
pre-registered detector (C: 0.445) drops back to 0.305 under the fixed one. That is consistent with §12 of the report: the
3.5 advantage in config C was fragile, and it does not survive a change in the detector any more than it survived fresh seeds.

**Conclusion.** Fixing the look-ahead and track gating and calibrating all four thresholds produces a detector that meets the
pre-registered targets on held-out development data and behaves the same way on evaluation seeds — but it still changes no
outcome relative to no detector at all. The honest summary of the congestion work stands: on this map, at this operating
point, congestion-aware caution neither helps nor harms the measured outcomes; the pre-registered version simply also made
both agents slower and occasionally caused timeouts.
