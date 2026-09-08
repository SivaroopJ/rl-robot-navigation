"""The Sibling Rivalry reward term, added to -- not substituted for -- the env's reward.

WHY THIS IS ADDITIVE HERE, WHEN THE PAPER'S ARM IS TERMINAL-ONLY
----------------------------------------------------------------
Trott et al.'s own arm replaces the reward entirely with a terminal payout (their Eq. 3).
Week 4's brief instead requires the existing Euclidean-shaped reward to stay EXACTLY as it
is, so that PPO and PPO+SR differ only by Sibling Rivalry. Both constraints are satisfiable
at once, and ppo-nav already established the pattern: its `composite_sr` mode bolts SR's
machinery onto a dense navigation reward and changes nothing else.

That precedent transfers directly, because ppo-nav's `CompositeTerms` is *literally this
environment's* reward -- +10 goal, -10 collision, +0.3 * delta_d progress, -0.02 per step,
a proximity ramp and an action-smoothing term. So the per-step reward is untouched and the
SR payout is added to the final transition only:

    r[T-1] += 0                                       if the episode reached the goal
    r[T-1] += min(0, -d(s_T, g) + d(s_T, gbar))        otherwise

The clamp at zero is the paper's, and byte-for-byte the reference implementation's
`clamp(drg - dra, -inf, 0)`. It matters: without it a sibling that ended far from its
anti-goal would be PAID for it, turning the anti-goal into a second attractor rather than a
repulsor.

Successful episodes receive no SR term at all. The environment's +10 already dominates, and
adding a penalty on top would make success worth less than it is worth to the PPO arm --
which would be a reward difference between the arms, i.e. the confound this design exists
to avoid.
"""
from __future__ import annotations

import numpy as np

from sibling_rivalry.select import euclid


def sr_terminal_bonus(terminal_position, goal, anti_goal, *, reached: bool) -> float:
    """The SR payout for one episode's final transition.

    Parameters
    terminal_position: array-like (2,)
        World-frame position where the episode ended.
    goal, anti_goal: array-like (2,)
        The goal, and the sibling's terminal state.
    reached: bool
        Whether the episode ended in success.

    Returns
    float
        Zero on success; otherwise `min(0, -d(s_T, g) + d(s_T, gbar))`, which is <= 0.
    """
    if reached:
        return 0.0
    if anti_goal is None:
        raise ValueError(
            "sr_terminal_bonus requires an anti-goal (the sibling's terminal state). "
            "Passing None would silently reduce Sibling Rivalry to naive distance shaping."
        )
    return float(min(0.0, -euclid(terminal_position, goal) + euclid(terminal_position, anti_goal)))


def apply_sr_reward(episode, *, terminal_mode: str = "add",
                    terminal_baseline: float = 0.0) -> float:
    """Apply the SR payout to an episode's final reward, in place.

    Two modes, because the two experiments need different ones and the difference is the
    paper's own distinction between adding SR to a shaped reward and being the reward.

    `terminal_mode="add"` (Weeks 4-5, the default, unchanged behaviour)
        `r[T-1] += bonus`. The environment's dense reward stays exactly as it is and SR's
        machinery is bolted on -- ppo-nav's `composite_sr` pattern. Required by the Week 4
        brief, which fixes the Euclidean reward so PPO and PPO+SR differ only by SR.

    `terminal_mode="replace"` (Experiment 4)
        `r[T-1] = r[T-1] - terminal_baseline + bonus`, i.e. the env's terminal distance
        term is removed and the SR term put in its place. This is the paper's Eq. 3
        SUBSTITUTING for Eq. 2's `-d(s_T, g)`, which is what the SR arm actually is:

            Eq. 2 (PPO)     r[T-1] = 1 if reached else -d(s_T, g)
            Eq. 3 (PPO+SR)  r[T-1] = 1 if reached else min(0, -d(s_T,g) + d(s_T,gbar))

        Adding instead of substituting would give `-d + min(0, -d + d_gbar)`, which is
        neither equation and double-counts the distance term.

    `terminal_baseline` is what the env already paid on the final transition and must be
    removed -- for Experiment 4 that is `-d(s_T, g)`, and it is passed by the caller rather
    than recomputed here so this module never needs to know the env's reward.

    Returns
    float
        The bonus applied, for logging.
    """
    if not episode.rewards:
        return 0.0
    if terminal_mode not in ("add", "replace"):
        raise ValueError(f"terminal_mode must be 'add' or 'replace', got {terminal_mode!r}")

    bonus = sr_terminal_bonus(
        episode.terminal(), episode.goal, episode.anti_goal, reached=episode.success)

    final = float(episode.rewards[-1])
    if terminal_mode == "replace" and not episode.success:
        # Successful episodes keep the env's +1 untouched: both equations agree there.
        final -= float(terminal_baseline)
    episode.rewards[-1] = final + bonus
    return bonus
