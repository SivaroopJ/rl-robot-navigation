# Experiment 3 — PPO + Sibling Rivalry trained on the same randomized obstacles

**Status: implementation complete, smoke test complete, full training NOT run.**

## Question

Does PPO+SR outperform PPO when **both are trained** on the same stochastic
dynamic-obstacle environment? This is Week 4's RQ3 and the primary comparison.

## Fair comparison

Experiment 3 shares with Experiment 2, by construction rather than by convention:

* the same environment, obstacles and randomization parameters;
* the same maze geometry and start/goal distribution;
* the same reward, unchanged from `main`;
* the same observation — no LiDAR added, none removed;
* the same PPO hyperparameters, network widths, `n_envs`, and training budget;
* the same evaluation seeds, so every episode is matched.

Neither arm was tuned separately. The SR hyperparameters are `delta = 0.6`, fixed to the
environment's own `target_radius`, and `epsilon`, set from a calibration run rather than
searched for a result.

Two things genuinely differ, and both are Sibling Rivalry:

1. **The critic is anti-goal-conditioned.** `V(s, g, gbar)` reads 36 inputs where PPO's
   reads 34. The **actor reads the same 34 in both arms** and has identical widths, so the
   comparison is not confounded with network capacity. Conditioning the policy is not an
   option even in principle: the anti-goal is a property of a sibling pair and does not
   exist at action-selection time.
2. **Rollouts are paired complete episodes, filtered by the acceptance rule.**
   `tau_f` is always retained; `tau_c` only when `rho < epsilon` or `d_c < delta`.

The SR terminal payout `min(0, -d(s_T, g) + d(s_T, gbar))` is **added to** the environment's
reward on the final transition only. The per-step reward is byte-identical to the PPO arm's.

## Sibling pairing, and the design question it settles

Sub-envs `2i` and `2i+1` are reset with the same seed. Because the environment derives
everything from `self.np_random` — start, goal, obstacle spawns, initial headings, and the
whole OU noise stream — the two siblings face a byte-identical world and diverge **only**
through actions sampled from the stochastic policy.

This is the one genuinely open design question in combining SR with stochastic obstacles,
and it is settled the paper-faithful way. Had siblings been allowed different obstacle
trajectories, `rho` would have mixed policy variance with environment variance, and
`epsilon` — calibrated against the `rho` distribution — would have been calibrated against
the wrong quantity. `_assert_pairs_match` verifies it on every rollout rather than trusting
it.

## Accounting

`num_timesteps` counts **every** environment step taken, including steps belonging to
episodes the acceptance rule later discarded. Both arms therefore spend the same
environment budget, which is the x-axis of every learning curve. SR trains on fewer of those
steps; that is the algorithm, not a handicap, and it is stated in the report rather than
buried.

## Diagnostics that make a null result interpretable

Logged every iteration to TensorBoard under `sr/`:

`n_pairs`, `n_episodes`, `n_retained`, `accept_rate_closer`, `rho_mean`, `rho_p50`,
`rho_p90`, `epsilon`, `success_rate`.

These exist because the most likely uninteresting outcome is that SR silently degenerates
into PPO — which happens when `rho` collapses (siblings stop diverging) or `epsilon` is set
so high that every pair is accepted. In the smoke run an untrained policy gave `rho` mean
1.11 / median 0.82, and a placeholder `epsilon = 2.0` accepted **94%** of pairs, very nearly
the degenerate case. Without these logs that would be indistinguishable from "SR does not
help", so any reported null must be accompanied by an `accept_rate_closer` well away from
both 0 and 1.

## Commands

```bash
python -m experiments.week4 train    --arms ppo_sr_random --seeds 0 1 2 \
    --timesteps 2000000 --epsilon <calibrated>
python -m experiments.week4 evaluate --arms ppo_sr_random --seeds 0 1 2 --episodes 200
python -m experiments.week4 table
```
