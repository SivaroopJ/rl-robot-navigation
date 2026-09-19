# HD Experiment 1 — close-out: FAILURE (dropped during the pilot)

Pre-registration: [[HD_DESIGN]]. Baselines: [[HD0_BASELINES_REPORT]]. Fast-solver check:
[[HD1_SOLVER_CHECK_REPORT]]. Date: 2026-09-19.

## Verdict

**FAILURE.** The researcher dropped the experiment during the phase 5 pilot, because PPO-subgoal
+ Random was not learning and was very unlikely to pass. Phases 6–7 (main training, sealed
evaluation) will not run. Tickets 09 and 10 are `wontfix`. **HD_FINAL was never opened for
PPO.**

## Deviations

This is not the §9.1 verdict. The researcher stopped the run before the rule could apply, and
that decision is itself the deviation:

- The pilot was stopped at 901 120 steps, before its 1M checkpoint existed. The §9.1 learning
  rule (the 1M checkpoint's HD_DEV success on 200 episodes ≥ the floor's 124/200) was
  therefore never evaluated.
- The single hold-5 re-pilot was not run.
- The evidence for the decision is the in-training dev evaluations below, which are not a
  pre-registered gate.
- No other rule was changed.

## Pilot record

Run 0, seed 0, hold 1, frozen SCS, 8 environments (4 fixed + 4 randomized), `config.json`
hyperparameters, MLP [256, 256]. Tooling commit `5b8e74c`. The checkpoints are in
`models/highdim/pilot_s0_h1/` (gitignored).

**Dev success** (in-training evaluation, deterministic, 50 HD_DEV seeds × 2 conditions):

| mark | success | fixed | randomized | collision | timeout |
|---|---|---|---|---|---|
| 250k | 0.03 | 0.02 | 0.04 | 0.44 | 0.53 |
| 500k | 0.01 | 0.00 | 0.02 | 0.44 | 0.55 |
| 750k | 0.00 | 0.00 | 0.00 | 0.46 | 0.54 |

For reference, the floor (goal-only + Random) scores 0.62 on HD_DEV and the ceiling
(A\* + Random) scores 0.88.

**Implementation sanity was clean over all 55 updates.** It was a learning failure, not a bug:
- No update produced a non-finite loss, value, action or parameter.
- approx_kl had a median of 0.0046 and a maximum of 0.0066; the target-KL stop never fired.
- Explained variance rose from 0.00 to about 0.6–0.7.
- The policy's action spread (std) increased from 1.00 to 1.18 instead of narrowing.

## What the policy learned

These figures come from the evaluation traces of the 500k and 750k checkpoints (100 episodes
each).

- **It points away from the goal.**
  - The mean cosine between the subgoal direction and the goal direction was −0.58 at 500k
    and −0.68 at 750k.
  - 87% and 93% of subgoals point more than 90° away from the goal.
  - Most subgoals are at the full 1 m (62% at 500k).
- **It parks at the outer wall.**
  - Within 1.2 m of the outer wall for 77% (500k) and 86% (750k) of each episode.
  - The safety filter holds it there, so the mean executed speed is 0.18 m/s.
  - Timeouts are parked episodes. Collisions are almost all dynamic (45 of 46 at 750k), most of
    them at the wall.

**Hypothesis (not tested before the drop):** edge-parking is a local optimum of the unchanged
environment reward under near-random exploration.

- Dynamic obstacles steer away from the outer walls (boundary margin ≈ 2.2 m).
- A timeout costs 500 × −0.02 = −10, the same as the collision penalty. There is no timeout
  penalty.
- Moving away from the goal costs only 0.3 per metre.
- With std > 1 on a clipped action box, sampled subgoals are mostly full-length random
  directions, so the +10 goal reward is rarely seen.
- The trainer did not log training-episode returns, so the return of edge-parking against
  heading for the goal was never measured.

**A structural observation for any later design:**
- The floor is already a strong policy (γ = goal).
- The PPO arm's action frame starts from a = 0 (γ = p), which carries no goal direction, so
  PPO has to rediscover "go toward the goal" from the observation.
- A goal-relative action frame (a = 0 means "head to the goal") would start PPO at the floor.
- That, a timeout penalty, or a lower entropy bonus / squashed policy are the candidate
  changes. Each one is a new pre-registration, not an amendment of this experiment.

## What stands

- **Phases 1–2 hold as a result in their own right.** On M0, A\* + Random beats goal-only +
  Random by +0.295 pooled success on HD_FINAL (0.9025 vs 0.6075, 400 paired episodes). Floor
  failures are mostly dynamic collisions after stalls behind the rectangles.
- **Phase 3 holds.** The existing fast DR-CCP solver is not suitable for training at the
  frozen Random parameters: feasibility-category agreement was 99.38% against 100%, caused by
  OSQP's iteration cap. Stage 1 was not reopened.
- **The HD code stays in the repo, tested** (`highdim/`, `experiments/highdim/`,
  `tests/test_highdim.py`):
  - the PPO wrapper, trainer, health logging and pilot tooling;
  - the hold-k control-step budget (`90ff8bd`).
- **Known SB3 behaviour, for any reuse.** With a linear LR schedule, the last update runs at
  a slightly negative learning rate, because the final rollout overshoots the budget:
  −4.7e−6 at 1M steps and hold 1.
