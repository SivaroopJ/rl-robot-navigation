# 10: Sealed evaluation, GO gate and results report

**What to build:** The researcher gets the final answer to HD Experiment 1: whether PPO can replace A* as the navigation component while Random-CLF-DR-CBF stays the safety layer. It is a pre-registered GO/NO-GO on sealed paired data, written up with diagnostics and caveats.

**Blocked by:** 09 (Main training)

**Status:** ready-for-agent

- [ ] The PPO-subgoal + Random arm and the filter-bypass arm are evaluated from each seed's final checkpoint, deterministically, on HD_FINAL (200 × 2), with the frozen SCS controller
- [ ] The floor and ceiling results from 04 are reused on the same seeds (they are deterministic); start and goal equality is asserted
- [ ] GO gate per seed:
  1. McNemar p < 0.05 against the floor on success
  2. (S_PPO − S_floor)/(S_A* − S_floor) ≥ 0.5
  3. collisions not significantly higher than the ceiling
- [ ] GO if the gate holds in ≥ 2 of 3 seeds
- [ ] The results report gives the arm table, paired tests, per-seed gap closure, mean ± sd over seeds, and diagnostic distributions (‖u_exec − u_nom‖, Random event rate, infeasible rate, stuck rate), plus the filter-bypass comparison
- [ ] The report states the three caveats (soft CLF, mild non-Markov filter state, map learned on M0) and lists any deviations separately
- [ ] ADR 0001's status is updated with the outcome
- [ ] The vault is updated: experiment note (result, verdict, commit), journal, Open Threads, Timeline, Reports Index; then the vault sync is run
