# 06: PPO arm and filter-bypass arm through the harness (untrained policy)

**What to build:** The researcher can run the PPO-subgoal + Random arm and the filter-bypass diagnostic arm through the episode harness, driven by any PPO checkpoint, including a randomly initialised one. This shows that a policy-chosen γ = p + L·a flows through the unchanged CLF-DR-CBF and Random recovery.

**Blocked by:** 04 (Baselines and stop check; PPO work starts only if the stop check did not trigger)

**Status:** done (`b9a4d98`)

- [x] A subgoal source is plugged into the frozen policy as its follower, the only hook used to replace A* → carrot → γ. It exposes the same reference/goal behaviour as the carrot follower
- [x] The action is radially clipped to the unit disc; γ = p + L·a with L = 1 m; γ = goal exactly when ‖goal − p‖ < L. No projection and no map access
- [x] The PPO arm is built with planning off and an empty static map; the policy reads obs[0:28] only
- [x] The filter-bypass arm reuses the same checkpoint and executes the saturated nominal velocity towards γ, at evaluation only
- [x] Seam-1 test: the controller receives γ equal to the subgoal source's reference
- [x] Seam-1 test: the recorded V equals ½·0.10·‖p−γ‖²
- [x] Seam-1 test: the parameters hash-match the frozen files
- [x] Seam-1 test: a forced-infeasible sample set produces a Random recovery event
- [x] Seam-1 test: the bypass arm never calls the QP or Random, and executes nominal_action(p, γ, max_v)/max_v
- [x] Seam-1 test: the PPO arm's actions don't change when the static map or the ground-truth obstacle block is perturbed
- [x] Seam-1 test: the same seed and the same checkpoint reproduce an identical episode
- [x] No frozen file is edited, and the manifest test still passes

## Comments

- 2026-09-19. Code and tests `b9a4d98`. Full suite: 595 passed plus the 4 known legacy-path failures. Manifest verifies.
- **The PPO arm (`ppo_random`)**
  - It is the floor-arm construction (planning off, empty map) with a `SubgoalSource` assigned as the follower after reset.
  - `run_episode(..., model=...)` queries the model's mean action on obs[0:28].
  - The trace records `a_raw` (box-clipped) and `a_disc`.
- **The bypass arm (`ppo_unfiltered`)**
  - The controller is replaced by `BypassController`, which returns `nominal_action(p, γ, max_v)`. The frozen predict divides it by max_speed.
  - No Random recovery is attached, and the fast solver is refused.
  - Its record is descriptive only (HD_DESIGN §9, arm 4):
    - status is `bypassed`, so `n_solver_fail` equals the step count and `min_cbc_solved` is NaN;
    - min CBC is taken over every sample, not the QP's kept five;
    - `random_spec` is None.
  - Ticket 10 should present these fields accordingly.
- **Review follow-ups (not done):**
  - Ticket 07's wrapper should refuse `ppo_unfiltered` explicitly (training is filtered only). Today only `run_check_episode` refuses it, and only implicitly, because it takes no model.
  - Arm-kind facts are spread across `KINDS`/`PPO_KINDS`/`has_filter`. A per-arm descriptor would put them in one place.
- **Test seed:** `HD_DEV[8]` gives the untrained seed-0 model Random events within 60 steps in both conditions. It is used for the PPO determinism and invariance tests.
