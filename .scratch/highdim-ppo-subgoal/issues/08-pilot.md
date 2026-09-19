# 08: 1M-step pilot and pilot verdict

**What to build:** The researcher gets a pre-registered pilot verdict on whether PPO-subgoal + Random learns well enough to justify the main run. A slow pilot gets exactly one fallback.

**Blocked by:** 07 (PPO wrapper and short training)

**Status:** closed — **FAILURE**. The researcher dropped the experiment during the pilot (2026-09-19); see `MD_files/highdim/HD_EXPERIMENT1_CLOSEOUT.md`

- [~] Seed 0 trains for 1M steps at 10 Hz (hold length 1) on the chosen solver
- [x] Implementation sanity is reported: no NaNs or exceptions, approx_kl mostly below target, all diagnostics logged, throughput recorded
- [~] The learning curve, goal success, collision breakdown, filter-intervention frequency, QP infeasible rate, Random event rate and subgoal behaviour (γ distance distribution, goal-switch rate) are reported on HD_DEV
- [~] Pilot rule: the 1M checkpoint's dev success (100 × 2, deterministic) is compared with the floor's dev success from 04
- [~] If it falls below the floor, one re-pilot runs with hold length 5 and the same comparison
- [x] A written verdict: viable → proceed with the hold length that passed; NO-GO → tickets 09–10 are marked `wontfix`, and HD_FINAL is never opened
- [~] If the pilot shows unusually high intervention compared with the ceiling arm, it is noted as a candidate for a separately labelled ablation later, not added to this experiment

## Comments

- 2026-09-19. Tooling `5b8e74c`: training health log, the pilot rule in `highdim.gates`, and the `experiments/highdim/hd2_pilot.py` evaluation and report. Hold-k control-step budget `90ff8bd`. Suite: 624 passed plus the 4 known legacy failures.
- **Pilot:** run 0 / seed 0, hold 1, frozen SCS, 154 steps/s. In-training dev success was 0.03 at 250k, 0.01 at 500k and 0.00 at 750k, against the floor's 0.62.
  - Sanity was clean over 55 updates: nothing non-finite, KL median 0.0046, no target-KL stops, explained variance about 0.6.
  - The policy learned to head away from the goal and park at the outer wall (mean cosine to the goal direction −0.68 at 750k; 86% of the time within 1.2 m of the wall).
- **The researcher's decision:** drop the experiment and mark it FAILURE. The run was stopped at 901 120 steps, before its 1M checkpoint, so the §9.1 rule was never formally applied and the hold-5 re-pilot never ran. This is recorded as a deviation in the close-out.
- `[~]` marks items left partial by the drop. The subgoal and intervention statistics exist for the 500k and 750k checkpoints only (in the close-out); `hd2_pilot` was never run.
- Tickets 09 and 10 are `wontfix`, and HD_FINAL was never opened for PPO.
