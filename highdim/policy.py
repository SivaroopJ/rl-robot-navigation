"""Arm construction for HD Experiment 1, by composition only (HD_DESIGN.md sections 3-4).

Every arm is the frozen TunableDRCBFPolicy with the frozen tuned parameters and the frozen Random
recovery; arms differ ONLY in where the CLF reference gamma comes from. The frozen
DRCBFPolicy.predict takes `gamma = follower.reference(p)` when a follower is set and the goal
otherwise, so no frozen file is edited.

    goal_random   planning off, no static map, no follower  -> gamma = goal   (the floor)
    astar_random  continuation build_arm("random") verbatim -> gamma = A* carrot (the ceiling,
                  i.e. the Week 6 F-D arm re-run)
    ppo_random    as goal_random, plus a SubgoalSource assigned as the follower after reset
                  (`attach_subgoal`)                   -> gamma = p + L a, or the goal inside L
    ppo_unfiltered  the same policy and subgoal source, with the controller replaced by
                  BypassController: u = nominal_action(p, gamma, max_v); no QP, no Random.
                  Evaluation-only diagnostic (arm 4)

SOLVER. "frozen" is the policy's own ClfCbfDrccpController (SCS), used for every evaluation.
"fast" swaps in the unmodified FastClfCbfDrccpController (variant reduced_osqp) with the same
parameters, before reset() and before recovery is attached (HD_DESIGN.md section 8). It is for
the training-suitability check and, if that passes, for training; highdim.harness refuses it in
evaluation.
"""
from __future__ import annotations

import numpy as np

from continuation.policy import TunableDRCBFPolicy, attach_recovery, build_arm, policy_kwargs
from continuation.seeds import controller_rng
from dr_control.drccp_controller import clf_terms, nominal_action
from dr_control.fast_drccp import FastClfCbfDrccpController
from experiments.week6_continuation.common import load_random, load_tuned
from highdim.subgoal import SubgoalSource

KINDS = ("goal_random", "astar_random", "ppo_random", "ppo_unfiltered")
PPO_KINDS = ("ppo_random", "ppo_unfiltered")
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


class BypassController:
    """Arm 4: the filter bypassed. Executes the saturated nominal velocity towards gamma.

    Stands in for the controller so the frozen predict() (LiDAR pipeline, follower hook, action
    scaling) is reused unchanged; it never solves a QP and no recovery is attached. It returns
    the velocity nominal_action(p, gamma, max_v); the frozen predict() divides by max_speed, so
    the env action is nominal_action(...)/max_v as HD_DESIGN.md section 4 states. The record
    carries V and u_nom like the frozen one, with status "bypassed" and all samples as xi_kept
    (descriptive: min CBC over every sample, not the QP's kept five).
    """

    name = "bypass"

    def __init__(self, ctrl):
        self.rateh = ctrl.rateh
        self.max_v = ctrl.max_v
        self.k_v = ctrl.k_v
        self.prev_u = np.zeros(2)
        self.solve_fail = False

    def reset(self):
        self.prev_u = np.zeros(2)

    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        u = nominal_action(p, gamma, self.max_v) if u_nom is None else u_nom
        u = np.asarray(u, dtype=float).reshape(2)
        if record is not None:
            V, _ = clf_terms(p, gamma, self.k_v)
            record.update({"status": "bypassed", "V": V, "u_nom": u.copy(),
                           "xi_kept": np.atleast_2d(np.asarray(xi, dtype=float))})
        self.prev_u = u.copy()
        return u


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
    if solver == "fast" and kind == "ppo_unfiltered":
        raise ValueError("the bypass arm has no solver to swap")
    if kind == "astar_random":
        pol = build_arm("random", env, params=params)
    else:
        kw = {**policy_kwargs(env), "use_planner": False, "static_obstacles": list(POLICY_MAP)}
        pol = TunableDRCBFPolicy(params=params, **kw)
    if solver == "fast":
        pol.ctrl = fast_controller(pol.ctrl)
    if kind == "ppo_unfiltered":
        pol.ctrl = BypassController(pol.ctrl)
    return pol


def attach_subgoal(pol):
    """After pol.reset(): the PPO subgoal source becomes the follower (goal from obs[0:2])."""
    pol.follower = SubgoalSource(pol.goal)
    return pol.follower


def has_filter(kind):
    """False only for arm 4, whose QP and Random recovery are bypassed."""
    return kind != "ppo_unfiltered"


def attach_hd_recovery(pol, episode_seed, *, params, spec):
    return attach_recovery("random", pol, params=params, random_spec=spec,
                           rng=controller_rng(episode_seed))
