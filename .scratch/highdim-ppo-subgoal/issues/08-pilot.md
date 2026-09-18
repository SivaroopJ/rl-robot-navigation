# 08: 1M-step pilot and pilot verdict

**What to build:** The researcher gets a pre-registered pilot verdict on whether PPO-subgoal + Random learns well enough to justify the main run. A slow pilot gets exactly one fallback.

**Blocked by:** 07 (PPO wrapper and short training)

**Status:** ready-for-agent

- [ ] Seed 0 trains for 1M steps at 10 Hz (hold length 1) on the chosen solver
- [ ] Implementation sanity is reported: no NaNs or exceptions, approx_kl mostly below target, all diagnostics logged, throughput recorded
- [ ] The learning curve, goal success, collision breakdown, filter-intervention frequency, QP infeasible rate, Random event rate and subgoal behaviour (γ distance distribution, goal-switch rate) are reported on HD_DEV
- [ ] Pilot rule: the 1M checkpoint's dev success (100 × 2, deterministic) is compared with the floor's dev success from 04
- [ ] If it falls below the floor, one re-pilot runs with hold length 5 and the same comparison
- [ ] A written verdict: viable → proceed with the hold length that passed; NO-GO → tickets 09–10 are marked `wontfix`, and HD_FINAL is never opened
- [ ] If the pilot shows unusually high intervention compared with the ceiling arm, it is noted as a candidate for a separately labelled ablation later, not added to this experiment
