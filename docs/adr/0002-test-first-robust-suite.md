# 0002: Test the frozen A*-Random stack first; improve planning and perception, never the safety formulation

- **Status:** accepted, 2026-09-19. The outcome of RS Experiment 1 will be recorded here.
- **Scope:** RS Experiment 1. Pre-registration: `MD_files/robustsuite/RS_DESIGN.md`. Spec: `.scratch/rs-robust-astar-suite/spec.md`.

## Context

A\* + Random-CLF-DR-CBF is the validated stack: 0.9025 success on M0, where A\* is worth +0.295 over goal-only. It has only been measured on M0, one fixed map whose dynamic obstacles pass through walls.

Several attempts to improve it on M0 changed no outcome:
- tuning (H1/H2; the in-sample gain did not hold on fresh seeds);
- the Random perturbation and persistence;
- the Track B slow-down supervisor;
- congestion detection.

Each was chosen without a failure analysis in varied environments.

HD Experiment 1 tried to replace A\* with a learned subgoal source and failed during the pilot: the policy learned to park at the outer wall.

Two weaknesses are known from code inspection:
- A\* plans once per episode and never replans.
- Random recovery rarely finds a positive-margin direction.

## Decision

1. **Test first.** Characterise the frozen stack on a fresh procedural suite before choosing any improvement:
   - five layout families × four obstacle conditions, including region-triggered static blocks and pedestrian spawns;
   - two pedestrian motion conditions.

   Candidates (at most three) are chosen from the observed failure breakdown and approved before they are built. They are selected on a separate seed block and compared with the baseline once, on a sealed block.
2. **The safety formulation is frozen.** Candidates may change:
   - planning: replanning, Wait/Retreat behaviours, lookahead;
   - parameters, within a pre-registered budget;
   - perception and prediction built on the frozen LiDAR pipeline.

   They may never change the CLF/CBF/DR-CCP formulation, the robot dynamics, the reward or the environment.
3. **Onboard information only.** Candidates see what the frozen policy sees, plus the known static map without triggered blocks and their own history. They never see ground-truth obstacle state or the scenario's trigger and spawn information.
4. **Fresh, feasible, sensible maps.** The suite is new, not the Week 6 generalization suite. Every episode passes feasibility rules in code: a clear path before and after any block, no unavoidable spawn. Dynamic obstacles are pedestrians that walk routes through free space, because the canonical motion models pass through walls.

## Alternatives rejected

- **Improve first on M0, then test on the suite.** This repeats the pattern that produced null results. Candidates would target M0's failures (early dynamic collisions) and not whatever the new obstacle types expose.
- **Change the safety formulation** (e.g. a prediction-aware constraint form, or a different chance-constraint regime). The formulation is the validated contribution, and changing it would require re-validating the Phase 1–7 results. Prediction may enter only through the barrier source's inputs.
- **Reuse the Week 6 generalization suite.** It was never run with the A\* + Random arm and has no final-tier inference. Its dynamic obstacles pass through walls, and its maps are perturbations of M0 rather than structured layouts. The researcher asked for fresh maps.
- **Hand-designed fixed maps.** They are easier to interpret, but the sealed block would reuse layouts the candidates were chosen on. Procedural families give the sealed block unseen layouts.
- **Learned candidates (RL).** HD Experiment 1 closed that direction for now.

## Consequences

- The experiment always produces a result: the baseline's characterisation of the suite on sealed seeds, even if no candidate qualifies.
- The scenario environment, generator and pedestrian model are new code, isolated in their own modules. The canonical environment is hash-frozen, and M0 through the new path must be bit-identical to the canonical environment.
- Candidates that need new information (e.g. ground-truth velocities) are out of scope by construction.
