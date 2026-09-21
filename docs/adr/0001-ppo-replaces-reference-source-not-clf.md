# 0001 — PPO replaces the CLF reference source, not the CLF

- **Status:** accepted, 2026-09-18. The outcome of HD Experiment 1 will be recorded here.
- **Scope:** high-dimensionality axis, HD Experiment 1. Pre-registration: `MD_files/highdim/HD_DESIGN.md`. Spec: `.scratch/highdim-ppo-subgoal/spec.md`.

## Context

In Random-CLF-DR-CBF on M0, A* (`StaticMapPlanner`) plans once per episode on the known static map. `CarrotFollower` then emits a reference point γ 1 m ahead along the planned polyline.

Reading the code showed four things:

1. **A\* supplies exactly one quantity: γ.**
   - `DRCBFPolicy.predict` takes `gamma = follower.reference(p)` if a follower is set, and the goal otherwise.
   - γ enters the CLF `V(p, γ) = ½·k_v·‖p−γ‖²` and the saturated nominal velocity `u_nom = max_v·(γ−p)/‖γ−p‖`.
   - Nothing else uses A\*. The DR-CBF constraint, the LiDAR barrier source, the velocity tracker and Random recovery don't.
2. **The CLF is a soft tracking objective, not a certificate.**
   - Its slack δ is unbounded.
   - The slack weight `5·h_crit` goes to zero near obstacles.
   - So the current system has no formal convergence guarantee that a learned component could break.
3. **Because `u_nom` is always saturated, "local target" and "desired velocity" are the same thing.**
4. **Removing A\* was already measured.** With γ = goal, the Phase 5 standalone harness gave 72% success and 24% stuck, against 91% and 12% with A\* (estimated velocities).

## Decision

**PPO replaces only the source of γ.** The subgoal source replaces the carrot follower through the `follower` hook the frozen policy already has, with planning off and an empty static map:

- PPO outputs a 2-D action, radially clipped to the unit disc.
- γ = p + L·a, with L = 1 m.
- γ = goal when ‖goal − p‖ < L.
- γ is never projected into free space, and PPO gets no map.
- PPO observes obs[0:28]: goal offset, own velocity and 24 LiDAR rays.

Everything downstream of γ is unchanged: the CLF formula and gains, the DR-CCP constraint (collapsed regime, τ_eff = 0.12), Random recovery (K 16, H 1, speeds {0.8, 1.0}), holonomic [vx, vy] dynamics, and the environment reward.

The A\* + Random arm stays available as the baseline.

## Alternatives rejected

- **A. PPO outputs a velocity passed as `u_nom`, with the CLF kept on the goal.** The CLF pulls in a straight line to the goal and fights PPO's detours: p1 = 4h_crit and p3 = 5h_crit are of similar weight. This recreates the Phase 5 goal-direct failure mode inside the QP.
- **C. A learned or value-function CLF.** There is no certificate to preserve (point 2). Changing the CLF would also change the safety controller together with the planner, so the experiment could no longer isolate A\*.
- **D. PPO outputs a short trajectory.** The controller only ever consumes one reference point per step, so a trajectory adds action dimensions without adding capability.

Option B (a local target) was adopted. The "desired velocity" option is the same as B under saturated `u_nom` (point 3).

## Consequences

- The ablation is clean: the arms differ only in where γ comes from, so floor (γ = goal), ceiling (A\*) and PPO can be compared directly.
- V is a soft tracking objective towards a policy-chosen reference. No goal-convergence claim is made.
- The filter keeps hidden state that PPO can't see (the 5-scan buffer, the tracker, the previous action, Random persistence), so the environment PPO sees is mildly non-Markov. This is documented, not engineered away.
- On M0 the static map is fixed, so PPO learns it implicitly. The claim is "PPO replaces the explicit planner", not "PPO navigates unknown maps". Held-out layouts are a later experiment.
- The multi-goal extension (ordered G1 → … → Gn) changes only the task and the observation. The CLF and DR-CBF keep receiving only γ.
