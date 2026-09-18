# 07: PPO environment wrapper and a short training run

**What to build:** The researcher can train PPO against the safety-filtered environment. A short run of about 20k steps completes on the solver chosen in 05 and writes a checkpoint with metadata. The rollout is shown to store the subgoal action and its log-probability, never the executed velocity.

**Blocked by:** 05 (Fast-solver suitability), 06 (PPO arm, untrained)

**Status:** ready-for-agent

- [ ] The wrapper exposes a 28-d observation and a 2-d action box. Each reset builds a fresh PPO-arm policy with Random recovery and a controller RNG drawn from the wrapper's own generator
- [ ] The env reward and termination are unchanged; the reward is computed on the executed velocity
- [ ] Each step's info holds a diagnostics record: γ, u_nom, u_exec, ‖u_exec − u_nom‖, QP status, Random activation, min CBC, the raw clipped action and the disc action
- [ ] The subgoal-hold length is configurable (1 by default; 5 is the pre-registered fallback)
- [ ] Training uses the fast solver only if 05 passed, otherwise frozen SCS; evaluation always uses frozen SCS
- [ ] SB3 PPO uses the existing config hyperparameters, an MLP [256, 256], 8 environments and no observation normalisation. Checkpoints are saved every 250k steps plus a final model, with metadata recording commit, solver, L, hold length, frozen-parameter hash, seed and step count
- [ ] Training env seeds come from the HD_TRAIN block; in-training evaluation uses 50 HD_DEV seeds × 2 conditions and never touches HD_FINAL
- [ ] Seam-2 test: the observation is 28-d, and perturbing the ground-truth obstacle block leaves the next observation, γ and u_exec bit-identical
- [ ] Seam-2 test: the γ rule holds (disc norm ≤ 1; goal switch strictly inside L)
- [ ] Seam-2 test: every diagnostic key is present
- [ ] Seam-2 test: in a 16-step SB3 rollout, re-evaluating the buffered actions reproduces the buffered log-probabilities, the clipped buffered actions equal the wrapper's recorded raw actions, and u_exec never appears in the buffer
- [ ] A roughly 20k-step smoke run completes, its checkpoint loads and runs through the 06 harness, and its throughput (steps per second) is reported
