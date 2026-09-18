"""Seam 1: one paired episode of an HD arm on the canonical M0 environment.

Reuses, unchanged: exp6.make_env / episode_record / true_clearance, and the Continuation
StepRecorder / continuation_metrics. The episode loop is continuation.harness.run_episode's loop
with the policy built by `highdim.policy`.

EVALUATION IS FROZEN SCS ONLY. `run_episode` raises TypeError if the controller is anything but
the frozen ClfCbfDrccpController (HD_DESIGN.md section 8: the fast solver is for training only).

The record adds a per-step `trace` (gamma, V, u_nom, u_exec, QP status, Random activation,
min CBC of the executed action) and the `trajectory` the controller saw. Ground truth enters
only through true_clearance and the env's outcome flags, both measurement.
"""
from __future__ import annotations

import time

import numpy as np

from continuation.harness import ArmSpec, StepRecorder, continuation_metrics
from dr_control.drccp_controller import ClfCbfDrccpController
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import episode_record, make_env, true_clearance
from highdim import policy as HP


class TraceRecorder:
    """Outermost, metrics-only wrapper around the controller. Never changes the action.

    Captures what StepRecorder does not: the reference gamma, V, u_nom and u_exec. The min CBC of
    the executed action is StepRecorder's `m` and is merged in by `run_episode`, not recomputed.
    """

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.steps = []
        self._inner = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._inner

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u = self._inner(p, gamma, xi, u_nom=u_nom, record=rec)
        uu = np.asarray(u, float).reshape(2)
        V, nom = rec.get("V"), rec.get("u_nom")
        self.steps.append({
            "gamma": [float(x) for x in np.asarray(gamma, float).reshape(2)],
            "V": None if V is None else float(V),
            "u_nom": None if nom is None else [float(x) for x in np.asarray(nom).reshape(2)],
            "u_exec": [float(x) for x in uu],
            "qp_status": str(rec.get("status")),
            "random_event": rec.get("mode") == "event",
        })
        return u


def run_episode(kind, condition, seed, *, env=None, oracle=None):
    if condition not in ("fixed", "randomized"):
        raise ValueError(condition)
    own_env = env is None
    env = env if env is not None else make_env(condition == "randomized")
    if env.randomize_dynamic_obstacles != (condition == "randomized"):
        raise ValueError(f"env motion model does not match condition {condition!r}")
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    params, spec = HP.frozen_params()
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = HP.build_hd_arm(kind, env, params=params)
    if type(pol.ctrl) is not ClfCbfDrccpController:
        raise TypeError(f"evaluation requires the frozen SCS controller, got {type(pol.ctrl)}")
    pol.reset(obs, env.agent_position)
    wrap = HP.attach_hd_recovery(pol, seed, params=params, spec=spec)
    recd = StepRecorder(pol.ctrl, pol.ctrl.rateh)
    trace = TraceRecorder(pol.ctrl)

    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    infeasible_before_end = False
    while not (term or trunc):
        t0 = time.perf_counter()
        action, _ = pol.predict(obs, env.agent_position, deterministic=True)
        times.append(time.perf_counter() - t0)
        infeasible_before_end = pol.last_step_infeasible
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
    ep_time = time.perf_counter() - t_ep
    trace.detach()
    recd.detach()
    wrap.detach()
    if own_env:
        env.close()
    for s, r in zip(trace.steps, recd.steps, strict=True):
        s["min_cbc"] = r["m"]                  # min_i CBC_i of the executed action

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    extra = {"clearances": clear, "step_times": times,
             "planner_failed": int(pol.planner_failed),
             "n_infeasible": pol.n_infeasible, "n_solver_fail": pol.n_solver_fail,
             "collided_after_infeasible": int(outcome == "collision" and infeasible_before_end),
             "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
             "min_cbc_solved": (float(pol.min_cbc_solved)
                                if np.isfinite(pol.min_cbc_solved) else float("nan"))}
    rec = episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info, extra=extra)
    metrics_arm = ArmSpec(kind, "random", params=params, random=spec)
    rec.update(continuation_metrics(metrics_arm, recd.steps, wrap, outcome, steps, times, ep_time))
    rec.update({"arm": kind, "condition": condition, "controller": pol.ctrl.name,
                "start": [float(x) for x in start], "goal": [float(x) for x in goal],
                "params": params.as_dict(), "random_spec": spec.as_dict(),
                "trace": trace.steps,
                "trajectory": [[float(x) for x in q] for q in traj]})
    return rec


def run_paired(kinds, condition, seed, *, max_steps=None):
    """Run each arm on the same (condition, seed), each in a fresh env; assert pairing.

    The obstacle motion model reads only env.np_random and obstacle state, never the agent, so
    arms sharing a seed face the same start, goal, spawn and obstacle trajectory. Seed, start and
    goal equality are checked here, as continuation.harness.run_block does; a mismatch raises
    RuntimeError (not `assert`, which `python -O` would strip).
    """
    recs = {}
    for kind in kinds:
        env = make_env(condition == "randomized")
        try:
            if max_steps is not None:
                env.MAX_STEPS = int(max_steps)
            recs[kind] = run_episode(kind, condition, seed, env=env)
        finally:
            env.close()
    first = recs[kinds[0]]
    for kind in kinds[1:]:
        r = recs[kind]
        if not (r["seed"] == first["seed"] == seed and r["start"] == first["start"]
                and r["goal"] == first["goal"]):
            raise RuntimeError(f"pairing broken for seed {seed}: {kind} vs {kinds[0]}")
    return recs
