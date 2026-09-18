"""Arm construction for HD Experiment 1, by composition only (HD_DESIGN.md sections 3-4).

Every arm is the frozen TunableDRCBFPolicy with the frozen tuned parameters and the frozen Random
recovery; arms differ ONLY in where the CLF reference gamma comes from. The frozen
DRCBFPolicy.predict takes `gamma = follower.reference(p)` when a follower is set and the goal
otherwise, so no frozen file is edited.

    goal_random   planning off, no static map, no follower  -> gamma = goal   (the floor)
    astar_random  continuation build_arm("random") verbatim -> gamma = A* carrot (the ceiling,
                  i.e. the Week 6 F-D arm re-run)

SOLVER. "frozen" is the policy's own ClfCbfDrccpController (SCS), used for every evaluation.
"fast" swaps in the unmodified FastClfCbfDrccpController (variant reduced_osqp) with the same
parameters, before reset() and before recovery is attached (HD_DESIGN.md section 8). It is for
the training-suitability check and, if that passes, for training; highdim.harness refuses it in
evaluation.
"""
from __future__ import annotations

from continuation.policy import TunableDRCBFPolicy, attach_recovery, build_arm, policy_kwargs
from continuation.seeds import controller_rng
from dr_control.fast_drccp import FastClfCbfDrccpController
from experiments.week6_continuation.common import load_random, load_tuned

KINDS = ("goal_random", "astar_random")
SOLVERS = ("frozen", "fast")
FAST_VARIANT = "reduced_osqp"

#: The static map handed to map-free arms. Empty by design: the planner is never called, and
#: tests/test_highdim.py swaps in the real map to prove the actions do not depend on it.
POLICY_MAP = ()


def frozen_params():
    """(DRCBFParams, RandomSpec), each hash-checked against its frozen file."""
    tuned, _ = load_tuned()
    spec, _ = load_random()
    return tuned, spec


def fast_controller(ctrl):
    """The unmodified fast solver with exactly the parameters of the frozen controller `ctrl`."""
    return FastClfCbfDrccpController(
        variant=FAST_VARIANT, clf_rate=ctrl.rateV, cbf_rate=ctrl.rateh,
        wasserstein_r=ctrl.wasserstein_r, epsilon=ctrl.epsilon, max_v=ctrl.max_v, k_v=ctrl.k_v,
        p0=ctrl.p0, n_keep=ctrl.n_keep, solver=ctrl.solver)


def build_hd_arm(kind, env, *, params, solver="frozen"):
    """The policy for one episode, before reset(). Recovery is attached by `attach_hd_recovery`
    AFTER pol.reset(), the order the Continuation harness uses."""
    if kind not in KINDS:
        raise ValueError(f"unknown arm {kind!r}; expected one of {KINDS}")
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if kind == "astar_random":
        pol = build_arm("random", env, params=params)
    else:
        kw = {**policy_kwargs(env), "use_planner": False, "static_obstacles": list(POLICY_MAP)}
        pol = TunableDRCBFPolicy(params=params, **kw)
    if solver == "fast":
        pol.ctrl = fast_controller(pol.ctrl)
    return pol


def attach_hd_recovery(pol, episode_seed, *, params, spec):
    return attach_recovery("random", pol, params=params, random_spec=spec,
                           rng=controller_rng(episode_seed))
