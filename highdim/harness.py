"""Seam 1: one paired episode of an HD arm on the canonical M0 environment.

Reuses, unchanged: exp6.make_env / episode_record / true_clearance, and the Continuation
StepRecorder / continuation_metrics. The episode loop is continuation.harness.run_episode's loop
with the policy built by `highdim.policy`.

EVALUATION IS FROZEN SCS ONLY. `run_episode` raises TypeError if the controller is anything but
the frozen ClfCbfDrccpController (HD_DESIGN.md section 8: the fast solver is for training only).
The one exception is arm 4 (ppo_unfiltered), whose controller is the BypassController and which
runs no QP and no recovery at all.

PPO ARMS (ppo_random, ppo_unfiltered) are driven by `model`, anything with an SB3-shaped
predict(obs[0:28], deterministic=True); each step its action is handed to the subgoal source
before the frozen predict. The trace adds the raw (box-clipped) and disc actions.
`run_check_episode` is the training-suitability check's path (section 8), the only one here that
may run the fast solver; its records are never evaluation results.

The record adds a per-step `trace` (gamma, V, u_nom, u_exec, QP status, Random activation,
min CBC of the executed action) and the `trajectory` the controller saw. Ground truth enters
only through true_clearance and the env's outcome flags, both measurement.
"""
from __future__ import annotations

import time

import numpy as np

from continuation.harness import ArmSpec, StepRecorder, continuation_metrics
from dr_control.drccp_controller import ClfCbfDrccpController
from dr_control.fast_drccp import FastClfCbfDrccpController
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import episode_record, make_env, true_clearance
from highdim import policy as HP
from highdim.subgoal import policy_obs
from highdim.gates import DU_TOL

#: Phase 5's stuck definition, reproduced exactly (experiments/exp5_astar_waypoint.py, the loop
#: `for t: ... traj.append(p); if collision/success: break; if t >= 50 and
#: |p - traj[t - 50]| < 0.25: stuck = True`). With traj[0] the start, the position after step t
#: is traj[t + 1], so each check spans 51 steps, and the final step of an episode that ends in
#: success or collision is never checked.
STUCK_WINDOW, STUCK_DIST = 50, 0.25


def stuck(traj, outcome="timeout"):
    """True if some checked step t >= STUCK_WINDOW has |traj[t+1] - traj[t-STUCK_WINDOW]| < DIST."""
    if outcome not in ("timeout", "success", "collision"):
        raise ValueError(f"unknown outcome {outcome!r}")
    p = np.asarray(traj, float).reshape(-1, 2)
    last = len(p) - 1 if outcome == "timeout" else len(p) - 2     # last checked index t + 1
    for k in range(STUCK_WINDOW + 1, last + 1):
        if float(np.linalg.norm(p[k] - p[k - STUCK_WINDOW - 1])) < STUCK_DIST:
            return True
    return False


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


def feasibility_category(status):
    """What the policy does with a QP status: "optimal" -> the action is used, "infeasible" ->
    Random recovery (its trigger is `"infeasible" in status`), anything else -> the frozen u = 0.
    The E-a agreement is over these three categories."""
    status = str(status)
    if status == "optimal":
        return "optimal"
    return "infeasible" if "infeasible" in status else "other"


#: Decade bins of |u_fast - u_frozen|_inf, for the reported distribution (upper edges; the last
#: bin is everything above 1e-1).
DU_BINS = [10.0 ** -k for k in range(12, 0, -1)]


def _du_bin(du):
    for edge in DU_BINS:
        if du <= edge:
            return f"<={edge:.0e}"
    return f">{DU_BINS[-1]:.0e}"


class ShadowSolver:
    """E-a: innermost, metrics-only. The fast solver solves the frozen solver's inputs in shadow.

    Installed on the frozen controller BEFORE Random recovery, so it sees exactly the frozen
    call: (p, gamma, xi, u_nom) and ctrl.prev_u at call time, which Random may have overwritten
    with the previously executed action. The fast solver is given that prev_u before every solve
    and its output is never used; the frozen call and its return value pass through untouched.
    """

    def __init__(self, ctrl, fast):
        self.ctrl = ctrl
        self.fast = fast
        self.steps = []
        self._inner = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._inner

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u_prev = np.array(self.ctrl.prev_u, dtype=float)
        u = self._inner(p, gamma, xi, u_nom=u_nom, record=rec)
        self.fast.prev_u = u_prev.copy()
        frec = {}
        uf = self.fast.generate_controller(p, gamma, xi, u_nom=u_nom, record=frec)
        u0 = [float(x) for x in np.asarray(u, float).reshape(2)]
        u1 = [float(x) for x in np.asarray(uf, float).reshape(2)]
        row = {"status_frozen": str(rec.get("status")), "status_fast": str(frec.get("status")),
               "u_frozen": u0, "u_fast": u1}
        row["cat_frozen"] = feasibility_category(row["status_frozen"])
        row["cat_fast"] = feasibility_category(row["status_fast"])
        both = row["cat_frozen"] == row["cat_fast"] == "optimal"
        row["du_inf"] = max(abs(a - b) for a, b in zip(u1, u0)) if both else None
        row["delegated"] = bool(frec.get("delegated", False))     # h_crit < 0: frozen inside
        self.steps.append(row)
        return u

    def summary(self):
        cross, hist = {}, {}
        du = [s["du_inf"] for s in self.steps if s["du_inf"] is not None]
        for s in self.steps:
            if s["cat_frozen"] != s["cat_fast"]:
                k = f"{s['cat_frozen']}->{s['cat_fast']}"
                cross[k] = cross.get(k, 0) + 1
        for d in du:
            hist[_du_bin(d)] = hist.get(_du_bin(d), 0) + 1
        return {"steps": len(self.steps),
                "agree": sum(s["cat_frozen"] == s["cat_fast"] for s in self.steps),
                "delegated": sum(s["delegated"] for s in self.steps),
                "crosstab": cross, "both_optimal": len(du),
                "du_le_tol": sum(d <= DU_TOL for d in du),
                "du_max": max(du) if du else float("nan"), "du_hist": hist}


def run_episode(kind, condition, seed, *, env=None, oracle=None, model=None):
    """One evaluation episode: frozen SCS (arm 4: no solver). The PPO arms need `model`, which
    is queried with the mean action."""
    return _episode(kind, condition, seed, env=env, oracle=oracle, solver="frozen",
                    model=model)


def run_check_episode(kind, condition, seed, *, env=None, oracle=None, solver="frozen",
                      shadow=False):
    """One episode of the fast-solver training-suitability check (HD_DESIGN.md section 8).

    solver="frozen", shadow=True: E-a (the frozen episode, with the fast solver in shadow).
    solver="fast": E-b's fast side (the fast solver drives the closed loop).
    """
    if shadow and solver != "frozen":
        raise ValueError("the shadow compares against the frozen solver driving the loop")
    return _episode(kind, condition, seed, env=env, oracle=oracle, solver=solver,
                    shadow=shadow)


def _require_solver(ctrl, solver, kind):
    if kind == "ppo_unfiltered":
        if type(ctrl) is not HP.BypassController:
            raise TypeError(f"the bypass arm requires BypassController, got {type(ctrl)}")
        return
    if solver == "frozen" and type(ctrl) is not ClfCbfDrccpController:
        raise TypeError(f"evaluation requires the frozen SCS controller, got {type(ctrl)}")
    if solver == "fast" and not (type(ctrl) is FastClfCbfDrccpController
                                 and ctrl.variant == HP.FAST_VARIANT):
        raise TypeError(f"the fast check requires the {HP.FAST_VARIANT} solver, got {type(ctrl)}")


def _episode(kind, condition, seed, *, env, oracle, solver, shadow=False, model=None):
    if condition not in ("fixed", "randomized"):
        raise ValueError(condition)
    if (kind in HP.PPO_KINDS) != (model is not None):
        raise ValueError(f"arm {kind!r}: a model is required for the PPO arms and only for them")
    own_env = env is None
    env = env if env is not None else make_env(condition == "randomized")
    if env.randomize_dynamic_obstacles != (condition == "randomized"):
        raise ValueError(f"env motion model does not match condition {condition!r}")
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    params, spec = HP.frozen_params()
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = HP.build_hd_arm(kind, env, params=params, solver=solver)
    _require_solver(pol.ctrl, solver, kind)
    pol.reset(obs, env.agent_position)
    subgoal = HP.attach_subgoal(pol) if model is not None else None
    shade = ShadowSolver(pol.ctrl, HP.fast_controller(pol.ctrl)) if shadow else None
    wrap = (HP.attach_hd_recovery(pol, seed, params=params, spec=spec)
            if HP.has_filter(kind) else None)
    recd = StepRecorder(pol.ctrl, pol.ctrl.rateh)
    trace = TraceRecorder(pol.ctrl)

    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    infeasible_before_end = False
    actions = []
    while not (term or trunc):
        t0 = time.perf_counter()
        if subgoal is not None:
            a_env, _ = model.predict(policy_obs(obs), deterministic=True)
            subgoal.set_action(a_env)
            actions.append({"a_raw": [float(x) for x in subgoal.a_raw],
                            "a_disc": [float(x) for x in subgoal.a_disc]})
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
    if wrap is not None:
        wrap.detach()
    if shade is not None:
        shade.detach()
    if own_env:
        env.close()
    for s, r in zip(trace.steps, recd.steps, strict=True):
        s["min_cbc"] = r["m"]                  # min_i CBC_i of the executed action
    if shade is not None:
        for s, r in zip(trace.steps, shade.steps, strict=True):
            s["shadow"] = r
    if subgoal is not None:
        for s, a in zip(trace.steps, actions, strict=True):
            s.update(a)

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    extra = {"clearances": clear, "step_times": times,
             "planner_failed": int(pol.planner_failed),
             "n_infeasible": pol.n_infeasible, "n_solver_fail": pol.n_solver_fail,
             "collided_after_infeasible": int(outcome == "collision" and infeasible_before_end),
             "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
             "min_cbc_solved": (float(pol.min_cbc_solved)
                                if np.isfinite(pol.min_cbc_solved) else float("nan")),
             "stuck": int(stuck(traj, outcome))}
    rec = episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info, extra=extra)
    metrics_arm = (ArmSpec(kind, "tuned", params=params) if wrap is None else
                   ArmSpec(kind, "random", params=params, random=spec))
    rec.update(continuation_metrics(metrics_arm, recd.steps, wrap, outcome, steps, times, ep_time))
    rec.update({"arm": kind, "condition": condition, "controller": pol.ctrl.name,
                "solver": solver,
                "start": [float(x) for x in start], "goal": [float(x) for x in goal],
                "params": params.as_dict(),
                "random_spec": spec.as_dict() if wrap is not None else None,
                "trace": trace.steps,
                **({"shadow": shade.summary()} if shade is not None else {}),
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
