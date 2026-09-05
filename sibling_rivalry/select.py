"""Sibling Rivalry's relabeling and acceptance rule -- Trott et al. Algorithm 1.

WHY THIS IS A SEPARATE, PURE MODULE
-----------------------------------
Nothing here touches the environment, PyTorch or Stable-Baselines3. That is deliberate:
the rule is the part of Sibling Rivalry most easily got subtly wrong, and keeping it pure
means it can be tested directly against the paper's pseudocode rather than inferred from
training curves. It is a transcription of `ppo-nav/ppo_sr/collect.py::relabel_and_select`,
which was validated against the reference implementation.

THE RULE
--------
Two sibling episodes, tau_a and tau_b, start from an IDENTICAL (s0, g) and differ only in
the actions sampled from the current stochastic policy. Let s_T^a, s_T^b be their terminal
states and d(x) = ||x - g||.

  * Each sibling's ANTI-GOAL is the other's terminal state (mutual relabeling).
  * tau_c is whichever ended closer to the goal; tau_f is the other.
  * tau_f is ALWAYS retained.
  * tau_c is retained only if rho = ||s_T^c - s_T^f|| < epsilon, or d(s_T^c) < delta.

WHAT THE ACCEPTANCE RULE IS FOR
-------------------------------
It is the "self-balancing" half of the paper's title. Retaining only the farther sibling
pushes the policy away from whatever attractor both siblings fell into; retaining the
closer one as well, but only when the pair agree (rho small) or when it actually succeeded
(d < delta), stops that push from destroying a policy that has genuinely found the goal.
Dropping the rule -- accepting everything -- reduces SR to naive distance shaping, which is
the baseline it is supposed to beat.

WHAT IT COSTS
-------------
The SR arm trains on FEWER transitions than the PPO arm from the same number of environment
steps. That is the algorithm, not a handicap: both arms consume the same environment steps,
which is the x-axis of every learning curve in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def euclid(a, b) -> float:
    """L2 distance. SR uses Euclidean distance throughout, as the paper does."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


@dataclass
class Episode:
    """One completed episode, with everything the SR rule and the buffer need.

    `positions` holds the agent's world-frame (x, y) at each step. SR's distances are
    measured in world coordinates, never in the normalized observation, because the
    anti-goal must be comparable to the goal and to `delta`.
    """
    env_index: int
    pair_index: int
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    goal: np.ndarray | None = None
    success: bool = False
    collision_type: str | None = None
    truncated: bool = False
    #: The observation at the step limit, kept ONLY for truncated episodes so that the
    #: value bootstrap can be applied the way stock SB3 PPO applies it. A truncated
    #: episode did not actually end -- the clock ran out -- so treating it as terminal
    #: charges the policy for a future it never got to collect. Stock PPO adds
    #: gamma * V(terminal_obs); SR must do the same or the two arms are not comparable.
    terminal_observation: np.ndarray | None = None
    #: Filled in by `relabel_and_select`; None until then. `apply_sr_reward` refuses to
    #: run without it rather than silently degrading SR to distance shaping.
    anti_goal: np.ndarray | None = None
    role: str | None = None

    def terminal(self) -> np.ndarray:
        """World-frame position where the episode ended."""
        return np.asarray(self.positions[-1], dtype=float)

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class PairStats:
    """Per-pair diagnostics, written to sr.csv.

    `rho` and `kept_closer` are the two numbers that say whether SR is doing anything at
    all: if rho collapses toward zero the siblings are not diverging and the acceptance
    rule never binds, which is exactly how SR degenerates into PPO. Logging them makes that
    failure visible instead of leaving it to be inferred from a null result.
    """
    pair_index: int
    d_a: float
    d_b: float
    closer: str
    rho: float
    anti_goal_c: tuple
    anti_goal_f: tuple
    kept_closer: bool
    kept_farther: bool
    success_a: bool
    success_b: bool


def relabel_and_select(episodes, *, epsilon: float, delta: float):
    """Trott et al. Algorithm 1, lines 5-13.

    Parameters
    episodes: list of Episode
        Completed episodes, exactly two per `pair_index`.
    epsilon: float
        Sibling-agreement threshold. Calibrated from the measured rho distribution by
        `sibling_rivalry/calibrate.py`, never transplanted from another environment.
    delta: float
        Success radius. Set to the env's own TARGET_RADIUS so SR's notion of success and
        the environment's termination test cannot disagree.

    Returns
    kept: list of Episode
        Retained episodes, each carrying its `anti_goal`.
    stats: list of PairStats
    """
    kept, stats = [], []
    by_pair = {}
    for episode in episodes:
        by_pair.setdefault(episode.pair_index, []).append(episode)

    for pair_index in sorted(by_pair):
        members = by_pair[pair_index]
        if len(members) != 2:
            raise ValueError(
                f"pair {pair_index} has {len(members)} members, expected exactly 2; "
                "sibling pairing has broken and rho would be meaningless"
            )
        a, b = sorted(members, key=lambda e: e.env_index)
        goal = a.goal
        d_a, d_b = euclid(a.terminal(), goal), euclid(b.terminal(), goal)
        if d_a <= d_b:
            closer, farther, d_c, tag = a, b, d_a, "a"
        else:
            closer, farther, d_c, tag = b, a, d_b, "b"

        # Mutual relabeling: each sibling's anti-goal is the OTHER's terminal state.
        closer.anti_goal = farther.terminal().copy()
        farther.anti_goal = closer.terminal().copy()
        closer.role, farther.role = "closer", "farther"

        rho = euclid(closer.terminal(), farther.terminal())
        keep_closer = bool(rho < epsilon or d_c < delta)

        kept.append(farther)              # tau_f is always retained
        if keep_closer:
            kept.append(closer)

        stats.append(PairStats(
            pair_index=pair_index, d_a=d_a, d_b=d_b, closer=tag, rho=rho,
            anti_goal_c=(float(closer.anti_goal[0]), float(closer.anti_goal[1])),
            anti_goal_f=(float(farther.anti_goal[0]), float(farther.anti_goal[1])),
            kept_closer=keep_closer, kept_farther=True,
            success_a=a.success, success_b=b.success,
        ))

    return kept, stats
