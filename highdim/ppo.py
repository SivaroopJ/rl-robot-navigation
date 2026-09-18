"""PPO model side of HD Experiment 1: the policy's spaces and an untrained checkpoint.

PPO observes obs[0:28] only (goal offset / 10, own velocity / max speed, 24 LiDAR rays / 5), never
the ground-truth obstacle block obs[28:52]. Its action is a 2-D subgoal action in [-1, 1]^2,
mapped to gamma by highdim.subgoal. MLP [256, 256] (HD_DESIGN.md section 9).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from highdim.subgoal import OBS_DIM, policy_obs  # noqa: F401  (re-exported)

NET_ARCH = [256, 256]


def observation_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)


def action_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class _Spaces(gym.Env):
    """Carries the spaces only, so a model can be built without an environment."""

    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = action_space()

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        raise NotImplementedError


def new_model(seed):
    """A randomly initialised PPO checkpoint with the experiment's architecture."""
    return PPO("MlpPolicy", _Spaces(), policy_kwargs={"net_arch": NET_ARCH}, seed=int(seed),
               device="cpu", verbose=0)
