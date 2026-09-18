# 07: PPO environment wrapper and a short training run

**What to build:** The researcher can train PPO against the safety-filtered environment. A short run of about 20k steps completes on the solver chosen in 05 and writes a checkpoint with metadata. The rollout is shown to store the subgoal action and its log-probability, never the executed velocity.

**Blocked by:** 05 (Fast-solver suitability), 06 (PPO arm, untrained)

**Status:** done (`99ec97a`)

- [x] The wrapper exposes a 28-d observation and a 2-d action box. Each reset builds a fresh PPO-arm policy with Random recovery and a controller RNG drawn from the wrapper's own generator
- [x] The env reward and termination are unchanged; the reward is computed on the executed velocity
- [x] Each step's info holds a diagnostics record: γ, u_nom, u_exec, ‖u_exec − u_nom‖, QP status, Random activation, min CBC, the raw clipped action and the disc action
- [x] The subgoal-hold length is configurable (1 by default; 5 is the pre-registered fallback)
- [x] Training uses the fast solver only if 05 passed, otherwise frozen SCS; evaluation always uses frozen SCS
- [x] SB3 PPO uses the existing config hyperparameters, an MLP [256, 256], 8 environments and no observation normalisation. Checkpoints are saved every 250k steps plus a final model, with metadata recording commit, solver, L, hold length, frozen-parameter hash, seed and step count
- [x] Training env seeds come from the HD_TRAIN block; in-training evaluation uses 50 HD_DEV seeds × 2 conditions and never touches HD_FINAL
- [x] Seam-2 test: the observation is 28-d, and perturbing the ground-truth obstacle block leaves the next observation, γ and u_exec bit-identical
- [x] Seam-2 test: the γ rule holds (disc norm ≤ 1; goal switch strictly inside L)
- [x] Seam-2 test: every diagnostic key is present
- [x] Seam-2 test: in a 16-step SB3 rollout, re-evaluating the buffered actions reproduces the buffered log-probabilities, the clipped buffered actions equal the wrapper's recorded raw actions, and u_exec never appears in the buffer
- [x] A roughly 20k-step smoke run completes, its checkpoint loads and runs through the 06 harness, and its throughput (steps per second) is reported

## Comments

- 2026-09-19. Code `99ec97a`. Full suite: 612 passed plus the 4 known legacy failures. Manifest verifies.
- **The researcher's decision:** the training envs are 4 fixed + 4 randomized. It is recorded in HD_DESIGN §13 before any training, along with the hold-k semantics and the checkpoint marks.
- **Smoke run** (`python -m experiments.highdim.hd_train --smoke`, from the clean `99ec97a`, HD_TRAIN run 99; `models/highdim/smoke_99ec97a/`, gitignored, not a result):
  - 32 768 steps (the first two full 16 384-step rollouts, the minimum above 20k) on frozen SCS;
  - **154 control steps/s** with 8 envs; the checkpoint evaluations took 44 s on top;
  - both checkpoints and the final model loaded and ran through the 06 harness.
- **Projected cost at that rate:**
  - about 1.8 h of training for the 1M-step pilot;
  - about 5.4 h per 3M-step seed;
  - plus every checkpoint evaluation (100 episodes on frozen SCS, ~1–4 min at 8 workers, depending on episode length).
- **Review follow-ups:**
  - **Fixed:** a spec-review bug. `PPO(seed=s)` re-seeds a vec env with s + i, so the training envs never reset on their HD_TRAIN seeds. `build_model` now restores them, and an SB3-level test checks it.
  - **Fixed:** checkpoints are now saved at rollout boundaries (always a freshly updated policy).
  - **Fixed:** under hold k the budget and marks are counted in control steps.
  - **Fixed:** the stronger buffer-leak assertion, one shared hold helper, the stale `run_block` docstring, and `SMOKE_RUN` reserved in `highdim/seeds.py`.
  - **Left as judgement calls:** the `blocks` job tuple could become a NamedTuple, and the sha256 helpers are repeated.
