Status: ready-for-agent

# PPO subgoal source replaces A* in Random-CLF-DR-CBF (high-dimensionality axis, Experiment 1)

## Problem Statement

The current safe-navigation stack (Random-CLF-DR-CBF on M0) reaches its goals through an A* planner that runs once per episode on the known static map, followed by a pure-pursuit carrot follower that emits the CLF reference γ one metre ahead along the planned polyline. That design assumes the static map is known in advance. For the high-dimensionality research axis — episodes that eventually contain several sequential navigation goals in environments that are not fully known — the researcher wants the navigation decision itself to be learned from onboard observations instead of being computed from a map.

The researcher needs to know, with a controlled and pre-registered experiment, whether a PPO policy can take over A*'s role without touching the validated safety controller, and how much of A*'s measured contribution it recovers. Code inspection established that A* supplies exactly one quantity to the controller — the CLF reference γ — and that nothing in the CBF, the DR block, the LiDAR barrier source, the velocity tracker or the Random recovery depends on it. The CLF itself, V(p, γ) = ½·k_v·‖p−γ‖², is a soft tracking objective (unbounded slack, weight scheduled to zero near obstacles), not a convergence certificate. Phase 5 measured that replacing the carrot with the raw goal costs roughly 19 percentage points of success (72% vs 91%) mostly through local-minimum stalls behind the static rectangles; that is the gap a learned reference source must close.

## Solution

Replace the source of γ, and nothing else. A PPO policy observes only the goal offset, its own velocity and the 24 LiDAR rays, and outputs a 2-D action a in the unit disc. The reference becomes γ = p + L·a with L = 1 m (the existing lookahead), switching to γ = goal when the goal is closer than L. γ is fed to the unchanged CLF-DR-CBF QP and Random recovery, which produce the executed holonomic velocity. No static map reaches the PPO arm, and γ is never projected into free space.

The experiment runs four arms on a fresh sealed M0 block, paired by seed: A* + Random (the ceiling), goal-only + Random (the floor), PPO-subgoal + Random (the new arm), and the same PPO checkpoint with the safety layer bypassed (diagnostic). A pre-registered gate decides GO / NO-GO. The work is phased so that cheap baseline runs can stop the experiment before any PPO code is written.

## User Stories

1. As the researcher, I want the goal-only + Random floor measured on a fresh sealed M0 block before anything else, so that I know how much A* actually contributes on M0 under the current Random-CLF-DR-CBF.
2. As the researcher, I want the A* + Random ceiling re-run on exactly the same sealed seeds instead of reusing historical F-D numbers, so that every primary comparison is paired.
3. As the researcher, I want the experiment to stop automatically if the ceiling beats the floor by less than 5 percentage points pooled, so that I don't train PPO on a map where A* barely matters.
4. As the researcher, I want the stop verdict computed only after both the floor and the ceiling have run on the sealed block, so that the verdict is paired.
5. As the researcher, I want the A* ceiling arm to be bit-identical to the Week 6 Continuation Random-CLF-DR-CBF arm, so that the baseline is provably the frozen one.
6. As the researcher, I want the goal-only arm to use γ = goal at every step and never call the planner, so that the floor genuinely has no planning.
7. As the researcher, I want the PPO arm and the floor arm constructed with no static map at all, so that no map information can leak into them.
8. As the researcher, I want PPO to observe only the first 28 observation entries (goal offset, own velocity, 24 LiDAR rays), so that PPO never sees the ground-truth dynamic-obstacle block.
9. As the researcher, I want the safety controller to keep receiving the ego pose it receives today, so that the filter behaves identically across arms.
10. As the researcher, I want PPO's action mapped to the unit disc and then to γ = p + L·a, so that the policy chooses both the direction and the distance of a local reference within the existing 1 m lookahead.
11. As the researcher, I want γ set to the goal once the goal is closer than L, so that terminal behaviour matches the A* carrot and the comparison stays symmetric.
12. As the researcher, I want γ never projected into free space, so that PPO is not given map knowledge indirectly.
13. As the researcher, I want the CLF formula, gains and objective left exactly as they are, so that the only independent variable is the source of γ.
14. As the researcher, I want the DR-CBF constraint, the collapsed-regime margin τ_eff = 0.12 and the Random recovery spec (K 16, H 1, speeds {0.8, 1.0}) used from the frozen configuration files, so that the safety layer is the validated one.
15. As the researcher, I want the robot to stay a holonomic single integrator with [vx, vy] actions downstream of the controller, so that dynamics are not a second variable.
16. As the researcher, I want the environment reward left completely unchanged, with no intervention penalty in the primary experiment, so that the task objective matches every earlier PPO run.
17. As the researcher, I want the safety filter treated as part of the environment during PPO training, so that PPO's action is the subgoal action and its log-probability is computed on that action.
18. As the researcher, I want the rollout buffer to hold the observation, the sampled subgoal action, its log-probability, reward, termination/truncation, value estimate and next observation, and never the executed velocity, so that the on-policy update is correct.
19. As the researcher, I want every step to log γ, u_nom, u_exec, ‖u_exec − u_nom‖, QP status, Random-recovery activation and minimum CBC, so that I can diagnose reliance on the filter.
20. As the researcher, I want PPO to choose a new subgoal every control step (10 Hz) with the QP running at its normal rate, so that the reference update rate matches the A* carrot.
21. As the researcher, I want a pre-registered fallback of holding each subgoal for 5 steps, used only if the 1M-step pilot fails to learn, so that a slow pilot has one defined recovery path.
22. As the researcher, I want the existing fast DR-CCP solver checked for training suitability against the frozen SCS controller at the exact Random parameters, so that training can be accelerated only if it is faithful enough.
23. As the researcher, I want that check framed strictly as training suitability, without reopening, amending or reinterpreting the failed Phase 7 Stage 1 KKT gate, so that the historical record stands.
24. As the researcher, I want training to fall back to the frozen SCS controller if the suitability check fails, rather than modifying the fast solver, so that no controller code changes to make a check pass.
25. As the researcher, I want every evaluation and every reported number to use the frozen SCS controller, so that results are comparable with all earlier work.
26. As the researcher, I want a 1M-step single-seed pilot on the development block before the main run, so that implementation bugs and learning failures are found cheaply.
27. As the researcher, I want the pilot to pass only if it shows no NaNs or exceptions, healthy KL, complete diagnostics and a dev success at least equal to the floor's dev success, so that "viable" is defined before the pilot runs.
28. As the researcher, I want the experiment to stop with a NO-GO, without ever opening the sealed block, if the hold-5 re-pilot also fails, so that the sealed data stays untouched.
29. As the researcher, I want the main run to be 3 seeds × 3M steps with 8 parallel environments and the existing PPO hyperparameters and [256, 256] MLP, so that the budget and model are fixed in advance.
30. As the researcher, I want an extension to 5M steps only when at least 2 of 3 seeds gain at least 2 pp of dev success between the 2.0–2.5M and 2.5–3.0M windows, so that extension is decided on dev data by a fixed rule.
31. As the researcher, I want checkpoints every 250k steps plus a final model, with metadata recording commit, solver, L, hold length, frozen-parameter hash, seed and step count, so that every checkpoint is traceable.
32. As the researcher, I want the sealed evaluation to use each seed's final checkpoint, not the best dev checkpoint, so that there is no checkpoint-selection effect.
33. As the researcher, I want the unfiltered diagnostic arm to reuse the same trained checkpoint and execute the saturated nominal velocity towards the PPO-selected γ with the QP and Random bypassed, so that I can see how much the policy depends on the safety layer.
34. As the researcher, I want the gate evaluated per seed and passed when it holds in at least 2 of 3 seeds, so that a single lucky seed cannot pass it.
35. As the researcher, I want the gate to require McNemar p < 0.05 against the floor, gap closure (S_PPO − S_floor)/(S_A* − S_floor) ≥ 0.5, and a collision rate not significantly higher than the re-run A* + Random arm, so that success, planning value and safety are all tested.
36. As the researcher, I want the gate, pilot rule, extension rule and aggregation rule written into a committed design document before any run, so that the experiment is pre-registered.
37. As the researcher, I want new seed blocks (development, sealed final, training) that are disjoint from every existing block and reserved range, so that no earlier data is reused.
38. As the researcher, I want the sealed final block openable only from the baseline and final-evaluation entry points, so that it cannot be touched during development.
39. As the researcher, I want the same seed to reproduce an identical episode for every arm, so that paired comparisons and reruns are exact.
40. As the researcher, I want a hash manifest proving that the planner, the DR-CBF code, the Continuation code and the frozen parameter files are unchanged, so that "frozen" is verifiable.
41. As the researcher, I want the results report to show the arm table, paired tests, per-seed gap closure, mean ± sd over seeds, and diagnostic distributions (filter intervention, Random event rate, stuck rate), so that the result can be interpreted beyond the gate.
42. As the researcher, I want the report to state that the CLF is a soft tracking objective towards a policy-selected reference and not a goal-convergence certificate, so that no stronger claim is implied.
43. As the researcher, I want the report to state that the filter's hidden state (scan buffer, tracker, previous action, recovery persistence) makes the wrapped environment mildly non-Markov for PPO, so that the limitation is on record.
44. As the researcher, I want the report to state that on M0 the static map is learned implicitly by PPO rather than being unknown, so that the claim is "PPO replaces the explicit planner", not "PPO navigates unknown maps".
45. As the researcher, I want an ADR recording that PPO replaces the reference source and not the CLF, with options A (velocity into u_nom), C (learned-value CLF) and D (short trajectory) rejected with reasons, so that the architectural decision survives the experiment.
46. As the researcher, I want the design to extend later to ordered sequential goals G1 → … → Gn by changing only the task and observation, so that the high-dimensionality axis can grow without touching the safety layer.

## Implementation Decisions

**Categories.** Every change is one of: *Frozen* (not edited), *Modified* (behaviour changed by runtime composition only — no existing file is edited), or *New*.

- *Frozen:* CLF V(p, γ) = ½·k_v·‖p−γ‖² and saturated u_nom; DR-CCP/CVaR constraint and ξ construction; LiDAR barrier source and velocity tracker; Random recovery and its frozen specification; tuned parameters (α 0.8, r_W 0.012, ε 0.1, k_v 0.10, 5 scans, τ_eff 0.12, collapsed regime); robot dynamics, collision logic, observation and reward; the A* static-map planner and carrot follower as the baseline; the fast DR-CCP solver as implemented; the Phase 7 Stage 1 verdict.
- *Modified (composition only):* the reference source — the frozen policy already takes γ from its `follower` attribute when one is set and from the goal otherwise; the PPO arm turns planning off, builds the policy with an empty static map, and assigns a PPO subgoal source as the follower after each reset. The training controller may be replaced by the fast solver instance (same parameters) before recovery is attached, only in training and only if the suitability check passes. The evaluation harness is a new module that reuses the existing episode-record, true-clearance, step-recorder and paired-statistics functions.
- *New:* subgoal source; arm builder with four kinds; PPO environment wrapper with diagnostics; PPO trainer; seed blocks; phase entry points; hash manifest; tests; design document, ADR and results report.

**The smallest change replacing A* → carrot → γ with PPO → γ.** Construct the tuned policy with planning off and no static map; reset it (the goal comes from the observation's goal offset, no plan is made); assign the PPO subgoal source as its follower; attach the frozen Random recovery with the per-episode controller RNG. Each step, give the subgoal source the PPO action and the goal, then call the frozen predict. Everything from predict onwards is frozen code.

**Subgoal source.** Duck-types the carrot follower: exposes `reference(p)` and a `goal` property. Holds the latest action. The action is radially clipped to the unit disc (a = a_env / max(1, ‖a_env‖)); `reference(p)` returns the goal when ‖goal − p‖ < L (strict), otherwise p + L·a. L = 1.0 m. Optional hold length k ∈ {1, 5}: with k = 5, γ is frozen in world coordinates between PPO decisions while the QP runs every step.

**Arm builder.** Four kinds: `astar_random` (delegates to the existing Continuation construction of the Random arm plus its recovery attachment, verbatim), `goal_random` (planning off, empty static map, no follower), `ppo_random` (as goal_random plus the subgoal source), `ppo_unfiltered` (evaluation-only; executes nominal_action(p, γ, max_v)/max_v and never calls the QP or Random). Solver selection is "frozen" or "fast"; "fast" is allowed only from the trainer.

**PPO environment wrapper.** Gymnasium wrapper around the canonical M0 environment (6 dynamic obstacles at 0.675 m/s, fixed or randomized motion). Observation space: the first 28 entries of the environment observation. Action space: Box(−1, 1, 2). Reset builds a fresh `ppo_random` policy and Random recovery with a controller RNG drawn from the wrapper's own generator. Step: set the subgoal action, call the frozen predict with the full observation and the environment's ego position, step the environment with the resulting velocity, and return the truncated observation, the unchanged reward and the termination flags. The info dict carries a diagnostics record with γ, u_nom, u_exec, ‖u_exec − u_nom‖, QP status, Random activation, minimum CBC, the raw clipped action and the disc action.

**PPO training.** Stable-Baselines3 PPO with the `config.json` hyperparameters (lr 3e-4 linear, n_steps 2048, batch 256, 10 epochs, γ 0.99, GAE λ 0.95, ent 0.01, target KL 0.02), MLP [256, 256], 8 parallel wrapped environments, no observation normalization (the observation is already in [−1, 1]). SB3's default Gaussian policy stores the raw sample and its log-probability; clipping happens before the environment and the disc mapping inside the wrapper, so the stored action and log-probability are those of the subgoal action. Evaluation uses deterministic (mean) actions. Checkpoints every 250k steps plus a final model and metadata. In-training evaluation: 50 dev seeds × 2 conditions every 250k steps.

**Seed blocks.** Development block at 14 000 000 (100 episodes), sealed final block at 14 100 000 (200 episodes, used for both conditions), training environment seeds from 14 500 000 + 1000·run + env index; PPO seeds 0, 1, 2. Asserted disjoint from all Continuation blocks and forbidden ranges and from the pursuit (11M), Track B (12M) and Track A (13M) blocks. The per-episode Random RNG is the existing controller RNG derived from the episode seed.

**Phases and transitions.**
0. Read-only validation: the baseline suite passes apart from the 4 known `test_legacy_path_is_bit_identical_to_main` failures; both existing frozen manifests verify; the design document and ADR are committed with every pre-registered rule.
1. Floor (goal-only + Random), frozen SCS, dev block then sealed block.
2. Ceiling (A* + Random), frozen SCS, same sealed seeds, then dev block. Stop verdict here: pooled S_A* − S_floor < 0.05 → STOP.
3. Fast-solver training-suitability check at the frozen Random parameters, on 100 dev seeds × 2 conditions driving both baseline arms. (E-a) Shadow per-step: feasibility category agreement on 100% of steps; ‖Δu‖∞ ≤ 1e-3 on ≥ 99% of optimal steps, maximum reported. (E-b) Closed loop, fast vs frozen per arm: McNemar p > 0.05 and success-difference 95% CI within ±0.03. Pass → train on fast; fail → train on frozen SCS. Background evidence (not re-litigated): Stage 1 passed feasible-set equality, guarantee-tier agreement and closed-loop equivalence and failed the strict KKT criterion. The fast solver is documented as an acceleration mechanism, not an equivalent replacement.
4. Implementation of the new modules; all new tests pass.
5. Pilot: seed 0, 1M steps, 10 Hz; sanity and learning criteria as in user stories 27–28, with one hold-5 re-pilot as the only fallback.
6. Main run: 3 seeds × 3M steps; extension rule as in user story 30.
7. Sealed evaluation of the PPO arm and its unfiltered diagnostic, 200 episodes per condition on the sealed seeds; floor and ceiling are already on those seeds and deterministic. Gate as in user stories 34–35.

**Git.** New work goes on a new `highdim` branch from `Week6`. Directory names use `highdim`, not `week7` (a `week7` results folder already exists and holds unrelated Week 4 SR re-runs).

**Documents.** ADR 0001 (PPO replaces the reference source, not the CLF); pre-registration design document and results report under the repo's report folder in a `highdim` subfolder; per-phase tickets in this `.scratch` feature folder; the vault experiment note `hd-ppo-subgoal`.

## Testing Decisions

A good test checks behaviour visible at a seam — what an episode or an environment step returns and records — not how the modules are wired inside. Two seams are used, plus the existing seed-block pattern.

**Seam 1 — the episode harness** (arm, condition, seed, optional model → episode record with per-step γ, V, u_nom, u_exec, QP status, Random events, minimum CBC):
- Ceiling arm trajectories are bit-identical to the Continuation harness's Random arm on 5 existing validation seeds.
- Floor arm: γ equals the goal at every step; the planner's path method, patched to raise, is never called.
- Floor and PPO arms: replacing the static map given to the policy leaves every action identical.
- CLF unchanged: the controller receives γ equal to the subgoal source's reference; recorded V equals ½·0.10·‖p−γ‖²; parameters hash-match the frozen files; a forced-infeasible sample set produces a Random event.
- Unfiltered arm: QP and Random are never called; the executed action equals the saturated nominal velocity towards γ.
- Determinism: the same seed (and checkpoint) reproduces an identical episode for every arm.
- Evaluation always uses the frozen controller class; the fast solver is refused.
- Hash manifest: planner, DR-CBF code, Continuation code and frozen parameter files unchanged.

**Seam 2 — the PPO environment wrapper** (reset / step):
- Observation is 28-d; perturbing the ground-truth obstacle block of the underlying observation leaves the next observation, γ and u_exec bit-identical.
- γ rule: the disc action has norm ≤ 1; γ = p + L·a; γ = goal exactly when the goal is closer than L.
- The diagnostics record contains every required key.
- A 16-step SB3 rollout: re-evaluating the buffered actions reproduces the buffered log-probabilities; the clipped buffered actions equal the wrapper's recorded raw actions; the executed velocity never appears in the buffer.

**Seed blocks:** disjoint from all existing blocks and reserved ranges; the sealed block raises unless opened from the two permitted entry points.

**Prior art:** the Continuation tests (frozen equivalence gate, sealed FINAL block, protected manifest), the Phase 6 test asserting that the ground-truth obstacle block never changes the action, and the Phase 7 frozen-manifest tests.

## Out of Scope

- Multi-agent settings, and any combination of the multi-agent and high-dimensionality axes.
- Sequential multi-goal episodes (design only: ordered, externally supplied goals; observation adds the next goal's offset; +10 per goal; CLF and DR-CBF unchanged; no task sequencing by PPO).
- Held-out static layouts / generalization maps (planned as a later Experiment 1b).
- Any change to the CLF, the CBF/DR formulation, Random recovery, robot dynamics, reward or observation of the underlying environment; learned or value-function CLFs.
- Modifying the fast solver, or reopening or amending the Phase 7 Stage 1 gate.
- Frame stacking, recurrent PPO, SAC/TD3, hierarchical RL beyond the current PPO-plus-QP split.
- Intervention penalties (only as a later, separately labelled ablation if the pilot shows unusually high intervention).
- Free-space projection of γ or any map access for PPO.

## Further Notes

- Phase 5 evidence (estimated velocities): goal-direct 72% success / 24% stuck vs A* carrot 91% / 12% — measured in the standalone Phase 5 harness, not on M0 with Random recovery, which is why the floor is re-measured in Phase 1.
- Cost estimate: frozen SCS is ~28–37 ms per step, ~95% of it CVXPY canonicalization; 3 × 3M steps on frozen SCS is ~90 CPU-hours (~6 h on 16 cores). Torch is CPU-only on this server.
- The historical F-D result (pooled success 0.910, collision 0.090, all collisions dynamic) is context only; the primary comparison uses the re-run ceiling.
- Q20 is still open: whether to commit the untracked `CLAUDE.md` and `docs/agents/` on `Week6` before branching.
