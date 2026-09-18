# 06: PPO arm and filter-bypass arm through the harness (untrained policy)

**What to build:** The researcher can run the PPO-subgoal + Random arm and the filter-bypass diagnostic arm through the episode harness, driven by any PPO checkpoint, including a randomly initialised one. This shows that a policy-chosen γ = p + L·a flows through the unchanged CLF-DR-CBF and Random recovery.

**Blocked by:** 04 (Baselines and stop check; PPO work starts only if the stop check did not trigger)

**Status:** ready-for-agent

- [ ] A subgoal source is plugged into the frozen policy as its follower, the only hook used to replace A* → carrot → γ. It exposes the same reference/goal behaviour as the carrot follower
- [ ] The action is radially clipped to the unit disc; γ = p + L·a with L = 1 m; γ = goal exactly when ‖goal − p‖ < L. No projection and no map access
- [ ] The PPO arm is built with planning off and an empty static map; the policy reads obs[0:28] only
- [ ] The filter-bypass arm reuses the same checkpoint and executes the saturated nominal velocity towards γ, at evaluation only
- [ ] Seam-1 test: the controller receives γ equal to the subgoal source's reference
- [ ] Seam-1 test: the recorded V equals ½·0.10·‖p−γ‖²
- [ ] Seam-1 test: the parameters hash-match the frozen files
- [ ] Seam-1 test: a forced-infeasible sample set produces a Random recovery event
- [ ] Seam-1 test: the bypass arm never calls the QP or Random, and executes nominal_action(p, γ, max_v)/max_v
- [ ] Seam-1 test: the PPO arm's actions don't change when the static map or the ground-truth obstacle block is perturbed
- [ ] Seam-1 test: the same seed and the same checkpoint reproduce an identical episode
- [ ] No frozen file is edited, and the manifest test still passes
