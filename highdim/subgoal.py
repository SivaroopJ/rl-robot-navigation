"""The PPO subgoal source: the replacement for A* -> carrot -> gamma (HD_DESIGN.md section 4).

Duck-types optional_navigation.planner.CarrotFollower, which the frozen DRCBFPolicy.predict uses
through exactly two members: `reference(p)` and `goal`. Assigned as the policy's `follower` after
each reset; that assignment is the entire replacement.

    a     = a_env / max(1, |a_env|)        a_env: the (box-clipped) PPO action
    gamma = goal          if |goal - p| < L (strict)
          = p + L * a     otherwise,       L = 1.0 m, the carrot's lookahead

gamma is never projected into free space and the source holds no map.

HOLD (HD_DESIGN.md section 9.1, the pre-registered hold-5 fallback): a decision made with
`set_action(a, p=p)` freezes gamma = p + L a in world coordinates until the next decision; the
goal switch is still checked at every step. `set_action(a)` without p (hold 1) lets gamma follow
the current position, which gives the same gamma when the decision and the step share p.

PPO observes obs[0:28] only (goal offset / 10, own velocity / max speed, 24 LiDAR rays / 5),
never the ground-truth obstacle block obs[28:52]: `policy_obs`.
"""
from __future__ import annotations

import numpy as np

L = 1.0            # m; the carrot follower's lookahead (DRCBFPolicy default)


def check_hold(hold):
    if int(hold) != hold or hold < 1:
        raise ValueError(f"hold must be a positive integer, got {hold}")
    return int(hold)
OBS_DIM = 28


def policy_obs(obs):
    """The part of the environment observation PPO sees."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)[:OBS_DIM]


class SubgoalSource:
    def __init__(self, goal, *, lookahead=L):
        self._goal = np.asarray(goal, dtype=float).reshape(2).copy()
        self.lookahead = float(lookahead)
        self.a_raw = np.zeros(2)
        self.a_disc = np.zeros(2)
        self._frozen = None               # world-frame gamma of a held decision, else None

    @property
    def goal(self):
        return self._goal.copy()

    def set_action(self, a_env, p=None):
        """Store the latest PPO action. SB3 already clips to the Box; clipping again is a no-op
        for SB3 and keeps any other caller inside the action space. With `p`, gamma is frozen
        at p + L a in world coordinates (hold > 1)."""
        raw = np.clip(np.asarray(a_env, dtype=float).reshape(2), -1.0, 1.0)
        self.a_raw = raw
        self.a_disc = raw / max(1.0, float(np.linalg.norm(raw)))
        self._frozen = (None if p is None else
                        np.asarray(p, dtype=float).reshape(2) + self.lookahead * self.a_disc)

    def decide(self, a_env, p, hold):
        """A PPO decision at position p: gamma frozen in world coordinates when hold > 1."""
        self.set_action(a_env, p=p if hold > 1 else None)

    def reference(self, p):
        p = np.asarray(p, dtype=float).reshape(2)
        if float(np.linalg.norm(self._goal - p)) < self.lookahead:
            return self._goal.copy()
        if self._frozen is not None:
            return self._frozen.copy()
        return p + self.lookahead * self.a_disc
