"""Arm construction for HD Experiment 1, by composition only (HD_DESIGN.md sections 3-4).

Every arm is the frozen TunableDRCBFPolicy with the frozen tuned parameters and the frozen Random
recovery; arms differ ONLY in where the CLF reference gamma comes from. The frozen
DRCBFPolicy.predict takes `gamma = follower.reference(p)` when a follower is set and the goal
otherwise, so no frozen file is edited.

    goal_random   planning off, no static map, no follower  -> gamma = goal   (the floor)
    astar_random  continuation build_arm("random") verbatim -> gamma = A* carrot (the ceiling,
                  i.e. the Week 6 F-D arm re-run)
"""
from __future__ import annotations

from continuation.policy import TunableDRCBFPolicy, attach_recovery, build_arm, policy_kwargs
from continuation.seeds import controller_rng
from experiments.week6_continuation.common import load_random, load_tuned

KINDS = ("goal_random", "astar_random")

#: The static map handed to map-free arms. Empty by design: the planner is never called, and
#: tests/test_highdim.py swaps in the real map to prove the actions do not depend on it.
POLICY_MAP = ()


def frozen_params():
    """(DRCBFParams, RandomSpec), each hash-checked against its frozen file."""
    tuned, _ = load_tuned()
    spec, _ = load_random()
    return tuned, spec


def build_hd_arm(kind, env, *, params):
    """The policy for one episode, before reset(). Recovery is attached by `attach_hd_recovery`
    AFTER pol.reset(), the order the Continuation harness uses."""
    if kind not in KINDS:
        raise ValueError(f"unknown arm {kind!r}; expected one of {KINDS}")
    if kind == "astar_random":
        return build_arm("random", env, params=params)
    kw = {**policy_kwargs(env), "use_planner": False, "static_obstacles": list(POLICY_MAP)}
    return TunableDRCBFPolicy(params=params, **kw)


def attach_hd_recovery(pol, episode_seed, *, params, spec):
    return attach_recovery("random", pol, params=params, random_spec=spec,
                           rng=controller_rng(episode_seed))
