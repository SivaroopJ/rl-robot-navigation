"""Vectorized-environment plumbing for Sibling Rivalry.

Two things stock Stable-Baselines3 does not provide:

1. AN OBSERVATION WIDE ENOUGH TO CARRY THE ANTI-GOAL.
   The anti-goal reaches the critic through the observation's last two columns. It is not
   known during collection -- it is the SIBLING's terminal state, which does not exist
   until both episodes finish -- so those columns are zero while acting and are filled in
   afterwards, before any value is computed. `AntiGoalObsWrapper` widens the space and
   supplies the placeholder. The actor slices these columns off (see
   `sibling_rivalry/policy.py`), so the zeros never influence a sampled action.

2. PER-ENVIRONMENT SEEDS THAT ARE NOT ALL DIFFERENT.
   `VecEnv.seed(s)` deliberately hands sub-env i the seed `s + i`, which is right for
   ordinary training and exactly wrong here: siblings must face a byte-identical world, so
   sub-envs 2i and 2i+1 need the SAME seed. `set_pair_seeds` writes the per-env seed list
   straight onto the base vector env, which is where both SubprocVecEnv.reset and
   DummyVecEnv.reset read it from.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from sibling_rivalry.policy import widen_observation_space


class AntiGoalObsWrapper(VecEnvWrapper):
    """Appends anti-goal placeholder columns to every observation.

    The wrapper never fills the columns with anything but zeros. Filling them is the
    trainer's job, once `relabel_and_select` has decided what each episode's anti-goal is.
    """

    def __init__(self, venv, anti_goal_dim: int = 2):
        self.anti_goal_dim = int(anti_goal_dim)
        self.policy_obs_dim = int(venv.observation_space.shape[0])
        super().__init__(venv, observation_space=widen_observation_space(
            venv.observation_space, self.anti_goal_dim))

    def _widen(self, obs):
        pad = np.zeros((obs.shape[0], self.anti_goal_dim), dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=1)

    def reset(self):
        return self._widen(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # `terminal_observation` is the real final observation of a finished episode,
        # which the VecEnv hides in infos because `obs` has already been replaced by the
        # auto-reset. It has to be widened too: it is fed to the critic for the truncation
        # bootstrap, and a narrow one would reach a network expecting the anti-goal
        # columns. Widening only `obs` left infos silently inconsistent.
        for info in infos:
            terminal = info.get("terminal_observation")
            if terminal is not None:
                info["terminal_observation"] = np.concatenate(
                    [terminal, np.zeros(self.anti_goal_dim, dtype=terminal.dtype)])
        return self._widen(obs), rewards, dones, infos


def base_vec_env(env):
    """Walk down the wrapper chain to the vector env that actually owns `_seeds`."""
    while isinstance(env, VecEnvWrapper):
        env = env.venv
    return env


def set_pair_seeds(env, per_env_seeds):
    """Seed sub-envs individually, so paired sub-envs can share a seed.

    `VecEnv.seed` cannot express this: it derives every sub-env's seed by incrementing one
    base value. Writing `_seeds` directly is the documented consumption point for both
    SubprocVecEnv and DummyVecEnv, and it is cleared automatically after the next reset.
    """
    base = base_vec_env(env)
    if len(per_env_seeds) != base.num_envs:
        raise ValueError(
            f"expected {base.num_envs} seeds, got {len(per_env_seeds)}")
    base._seeds = [int(s) for s in per_env_seeds]
    return base._seeds
