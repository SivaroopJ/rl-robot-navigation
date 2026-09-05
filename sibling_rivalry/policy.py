"""The anti-goal-conditioned critic, V(s, g, gbar).

WHAT IS ASYMMETRIC AND WHY IT MATTERS FOR THE COMPARISON
--------------------------------------------------------
Sibling Rivalry conditions the VALUE function on the anti-goal, not the policy. The reason
is structural: the anti-goal is a property of a sibling PAIR, known only after both
episodes finish, so a policy conditioned on it could not be executed online at all -- at
action-selection time there is no anti-goal yet. The critic has no such problem, because it
is only ever evaluated after the fact, during the update.

Keeping the asymmetry buys something the experiment needs. The ACTOR of the PPO+SR arm is
architecturally identical to the actor of the PPO arm -- same input width, same layer
sizes, same initialisation given the same seed. So when the two arms are compared, they
differ by Sibling Rivalry's machinery and by nothing else. Had the anti-goal been appended
to the shared observation instead, the SR actor would have had two extra inputs and the
comparison would have confounded "SR helps" with "a slightly bigger network helps".

HOW IT IS PLUMBED INTO STABLE-BASELINES3
----------------------------------------
SB3's `ActorCriticPolicy` runs one `MlpExtractor` over a single feature width. To give the
critic a wider input, three things are overridden:

  `_build_mlp_extractor`  builds `AsymmetricMlpExtractor`, whose actor trunk takes the
                          34-dim observation and whose critic trunk takes 36.
  `extract_features`      returns the (actor, critic) pair of views instead of one tensor:
                          `obs[:, :34]` and the full `obs`. `share_features_extractor` is
                          forced False so SB3 takes its two-tensor branch.
  `predict_values`        overridden because the base implementation bypasses
                          `extract_features` and would hand the critic a 34-wide tensor.

The last two dimensions of the observation carry the anti-goal. During rollout collection
they are ZERO -- the anti-goal is not known yet -- and they are overwritten with the real
anti-goal after `relabel_and_select` runs, before any value is computed. Because the actor
slices them off, those placeholder zeros never reach the policy and cannot affect the
actions that were actually taken.
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


def _mlp(input_dim, net_arch, activation_fn):
    layers, last = [], input_dim
    for width in net_arch:
        layers.append(nn.Linear(last, width))
        layers.append(activation_fn())
        last = width
    return nn.Sequential(*layers), last


class AsymmetricMlpExtractor(nn.Module):
    """Two independent trunks of identical shape but different input width.

    SB3's own `MlpExtractor` is not reused because it derives both trunks from a single
    `feature_dim`. The layer widths and activation are taken from the same `net_arch` the
    PPO arm uses, so the only difference between the arms' critics is the two extra inputs.
    """

    def __init__(self, pi_dim, vf_dim, net_arch, activation_fn, device="auto"):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = net_arch.get("pi", [64, 64])
            vf_arch = net_arch.get("vf", [64, 64])
        else:
            pi_arch = vf_arch = list(net_arch)
        self.policy_net, self.latent_dim_pi = _mlp(pi_dim, pi_arch, activation_fn)
        self.value_net, self.latent_dim_vf = _mlp(vf_dim, vf_arch, activation_fn)

    def forward(self, features):
        # Only reached if share_features_extractor is True, which this policy forbids.
        raise RuntimeError(
            "AsymmetricMlpExtractor requires share_features_extractor=False; the actor and "
            "critic read different slices of the observation and cannot share one tensor."
        )

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class AntiGoalActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose critic additionally sees the anti-goal.

    The observation space is `policy_obs_dim + anti_goal_dim` wide; the actor reads only
    the first `policy_obs_dim` entries.
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 policy_obs_dim=None, anti_goal_dim=2, **kwargs):
        total = int(observation_space.shape[0])
        self.anti_goal_dim = int(anti_goal_dim)
        self.policy_obs_dim = int(policy_obs_dim if policy_obs_dim is not None
                                  else total - self.anti_goal_dim)
        if self.policy_obs_dim + self.anti_goal_dim != total:
            raise ValueError(
                f"observation width {total} does not equal policy_obs_dim "
                f"{self.policy_obs_dim} + anti_goal_dim {self.anti_goal_dim}"
            )
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            pi_dim=self.policy_obs_dim,
            vf_dim=self.policy_obs_dim + self.anti_goal_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def extract_features(self, obs, features_extractor=None):
        """(actor view, critic view). The actor never sees the anti-goal columns."""
        obs = obs.float()
        return obs[:, :self.policy_obs_dim], obs

    def predict_values(self, obs):
        """Overridden: the base implementation would hand the critic the actor's view."""
        return self.value_net(self.mlp_extractor.forward_critic(obs.float()))

    def get_distribution(self, obs):
        """Overridden so the actor always receives exactly its own slice.

        SB3's implementation reaches for `pi_features_extractor` directly rather than going
        through `extract_features`, so it would hand the 36-wide observation to the 34-wide
        actor trunk. That path is what `model.predict()` uses, so without this override
        every SAVED SR model was unloadable for evaluation even though training worked --
        the failure only appears at evaluation time.

        Accepting either width makes the call site's job unambiguous: callers may pass the
        full observation and the policy takes what it needs.
        """
        obs = obs.float()
        if obs.shape[-1] > self.policy_obs_dim:
            obs = obs[:, :self.policy_obs_dim]
        return self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(obs))


def widen_observation_space(space, anti_goal_dim=2):
    """The env's Box widened by the anti-goal columns.

    The anti-goal is a world-frame position normalized by world size, so it shares the
    env's [-1, 1] bounds and the widened space stays homogeneous.
    """
    low = np.concatenate([space.low, np.full(anti_goal_dim, -1.0, dtype=space.dtype)])
    high = np.concatenate([space.high, np.full(anti_goal_dim, 1.0, dtype=space.dtype)])
    return spaces.Box(low=low, high=high, dtype=space.dtype)
