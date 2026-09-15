# Track B — infeasibility-aware navigation: design note

Independent of Track A. Written **before** the B0 diagnostic numbers were inspected, so the analysis
plan is pre-declared; the risk-score form is chosen after B0 and frozen before any evaluation.

## B1 Research question and hypothesis

**Question.** Can a supervisory layer that avoids states likely to produce DR-CBF QP infeasibility
reduce infeasibility *without* trading away goal completion, safety or efficiency?

**Hypothesis (to be falsified).** Infeasible steps are preceded by runtime-observable conditions
(tight local free space and/or fast-closing nearby tracks), so a small interpretable risk score can
flag them a few steps ahead, and a mild intervention (slowing down) reduces infeasibility.

**Explicit anti-claim.** Fewer infeasible QPs is **not** by itself a success. A supervisor that
slows the robot trades time and may increase timeouts, path length and exposure to moving obstacles.
The result is judged on the whole trade-off (feasibility, goal, collisions, efficiency).

## B2 Literature context (LITERATURE-SUPPORTED)

- Wabersich & Zeilinger, *Predictive control barrier functions*, arXiv:2105.10241 (2021/22):
  safety-filter infeasibility is addressed by constraint tightening plus a terminal CBF, i.e. by
  making feasibility a *forward-looking* property rather than a per-step accident.
- Backup-CBF work (e.g. *Uniform Feasibility For Smoothed Backup Control Barrier Functions*,
  arXiv:2511.13499; *OcclusionCBF*, arXiv:2609.06342) obtains recursive feasibility by requiring the
  predicted flow to end in a backup safe set.

Both change the barrier mathematics. **This track deliberately does not**: the frozen QP is
untouched and the mechanism is an external supervisor, which is weaker in theory but auditable and
comparable to the frozen baselines. That is a documented design choice, not a claim of superiority.

## B3 What the supervisor may and may not see

| channel | contents | allowed at runtime |
|---|---|---|
| RUNTIME (`rt_*`) | 24 LiDAR ranges, the frozen controller's own barrier rows (`h_crit`, margin of u = 0), the LiDAR tracker's confirmed tracks, the supervisor's own infeasibility history, the nominal action | **yes** |
| DIAGNOSTIC (`gt_*`) | true clearance, true obstacle positions/velocities, robot coordinates | **offline analysis only** |

Enforced by naming and by a test that the supervisor module never imports `robot_env` and never
receives the env.

## B4 Interventions — one at a time

**Candidate 1 (implemented first): risk-aware speed reduction.** When risk ≥ threshold, scale the
*nominal* action magnitude by a factor, preserving direction. The scaled nominal is handed to the
**unchanged** frozen QP, so the safety filter still has the last word and the frozen infeasible
fallback (u = 0) is preserved. Hysteresis: enter at `risk_on`, leave at `risk_off < risk_on`, with a
minimum dwell time to prevent chatter.

**Candidates 2–4 (wait, retreat, route choice): specified, not implemented in this pass.** The
pursuit experiment already showed a stationary robot is exploitable, and a retreat needs a
look-ahead to avoid moving into a worse region. They are only worth building if Candidate 1 shows
the risk score has predictive value.

## B5 Safety and integration requirements

- The supervisor **only rescales the nominal action**; it never bypasses the QP, never writes the
  executed action, and never touches the barrier rows.
- The frozen u = 0 fallback on infeasibility is preserved exactly.
- Random recovery, when present, is the frozen implementation with frozen settings.
- Logged per step: risk score and each component, trigger state, speed factor, nominal before/after,
  executed action, QP status.
- Hard limits: maximum consecutive intervened steps, after which the supervisor stands down for a
  cooldown, so it cannot stall the robot indefinitely.

## B6 Evaluation design (frozen before running)

Factorial, paired on environment seeds, canonical M0 with stochastic obstacles:

| controller | supervisor off | supervisor on |
|---|---|---|
| Original CLF-DR-CBF (reference params) | O-off | O-on |
| Random-CLF-DR-CBF (tuned + frozen random recovery) | R-off | R-on |

- Dev block 12 000 000 + (B0 diagnosis and threshold selection). Eval block 12 100 000 + (never
  used for tuning).
- Primary metrics: goal / collision (by type) / timeout as **mutually exclusive terminal outcomes**;
  infeasible-step rate and infeasible-run statistics; fallback count; recovery events; path length,
  SPL, time; min clearance; supervision time; runtime overhead; unnecessary-intervention rate.
- Paired exact McNemar for binary outcomes, paired bootstrap for continuous, episode-cluster
  bootstrap for step-level rates. Denominators and discordant counts always reported.
- The two controller families are analysed separately and never pooled.
