"""Seam 1: one RS episode. (arm, cell, motion condition, seed) -> episode record.

A NEW module: highdim/harness.py is frozen (RS_DESIGN.md section 3). The loop below is
highdim.harness._episode's evaluation path (frozen solver, no PPO model, no shadow), composed
from the same frozen pieces, unchanged:
    highdim.policy        build_hd_arm, attach_hd_recovery, frozen_params
    highdim.harness       TraceRecorder, stuck
    continuation.harness  StepRecorder, ArmSpec, continuation_metrics
    exp6                  episode_record, true_clearance
and adds the cell: the scenario environment gets the cell's EpisodeSpec (or none, for the M0
anchor) before reset. The M0 anchor through this path is bit-identical to highdim's
`astar_random` (tests/test_robustsuite.py).

EVALUATION IS FROZEN SCS ONLY: `run_episode` raises TypeError for any other controller.

ARMS. Every arm is the frozen `astar_random` stack. A candidate (robustsuite.candidates, phase 4)
adds its behaviour layer as the policy's follower after reset and is named by `config`; its
`see(obs)` runs inside the timed part of the step, so its cost is in the step times. The baseline
takes no configuration and its record is unchanged apart from `config` and `candidate`, both None.

The record adds to highdim's: the cell, the motion condition, the spec (None for the anchor), the
generator version, the scenario-event log, whether and on which step the trigger fired
(`trigger_fired`, `trigger_step`; False and None when it never fired), and the dynamic obstacles'
positions after the last step (`final_obstacle_positions`, in slot order: a spawned pedestrian is
the slot after the regular ones), so the diagnostic can tell which one a dynamic collision hit. Step times are
summarised by continuation_metrics (mean, median, p95, max), as in highdim. Ground truth enters
only through true_clearance and the env's outcome flags, both measurement.
"""
from __future__ import annotations

import time

import numpy as np

from continuation.harness import ArmSpec, StepRecorder, continuation_metrics
from dr_control.drccp_controller import ClfCbfDrccpController
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import episode_record, true_clearance
from highdim import policy as HP
from highdim.harness import TraceRecorder, stuck
from robustsuite import candidates as CA
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite.scenario_env import RSScenarioEnv

#: The baseline and the phase 4 candidates (robustsuite.candidates, RS_DESIGN 7.4).
ARMS = ("astar_random",) + CA.ARMS


def cell_spec(cell, motion, seed):
    """The EpisodeSpec for a generated cell; None for the M0 anchor."""
    return None if cell == RS.ANCHOR else SC.generate(cell, motion, seed)


def check_cell(cell):
    if cell != RS.ANCHOR and cell not in RS.CELLS:
        raise ValueError(f"unknown cell {cell!r}")


def check_config(arm, config):
    """The candidate configuration for this arm, or None for the baseline."""
    if arm not in CA.ARMS:
        if config is not None:
            raise ValueError(f"{arm!r} has no configurations; got config={config!r}")
        return None
    return CA.configure(arm, config)


def run_episode(arm, cell, motion, seed, *, env=None, oracle=None, config=None,
                keep_step_times=False):
    """One evaluation episode on the frozen SCS controller.

    `env` is an RSScenarioEnv of this motion condition (a fresh one, closed afterwards, if None).
    The spec is always the generated one for (cell, motion, seed), so the record's cell is the
    map it ran on. `oracle` (SPL only: the static map inflated by the robot radius) defaults to
    one built on the episode's layout; the block runner passes a cached one for the anchor.
    `config` names a candidate's configuration (None: its pre-registered default).
    `keep_step_times` adds the raw per-step times to the record, which the frozen `episode_record`
    otherwise summarises and drops; the timing sample of phase 4 needs them pooled over episodes.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    cfg = check_config(arm, config)
    check_cell(cell)
    RS.check_motion(motion)
    if env is not None and env.motion != motion:
        raise ValueError(f"env motion {env.motion!r} does not match {motion!r}")
    own_env = env is None
    env = RSScenarioEnv(motion) if own_env else env
    try:
        return _episode(arm, cell, motion, seed, env, oracle, cfg, keep_step_times)
    finally:
        if own_env:
            env.close()


def _episode(arm, cell, motion, seed, env, oracle, cfg, keep_step_times):
    spec = cell_spec(cell, motion, seed)
    env.install(spec)
    params, rspec = HP.frozen_params()
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    # every arm is the frozen astar_random stack; a candidate adds its behaviour layer below
    pol = HP.build_hd_arm("astar_random", env, params=params, solver="frozen")
    if type(pol.ctrl) is not ClfCbfDrccpController:
        raise TypeError(f"evaluation requires the frozen SCS controller, got {type(pol.ctrl)}")
    pol.reset(obs, env.agent_position)
    layer = None if cfg is None else CA.attach(arm, pol, cfg)
    wrap = HP.attach_hd_recovery(pol, seed, params=params, spec=rspec)
    recd = StepRecorder(pol.ctrl, pol.ctrl.rateh)
    trace = TraceRecorder(pol.ctrl)

    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    infeasible_before_end = False
    try:
        while not (term or trunc):
            t0 = time.perf_counter()            # the candidate's own work is inside the step
            if layer is not None:
                layer.see(obs)
            action, _ = pol.predict(obs, env.agent_position, deterministic=True)
            times.append(time.perf_counter() - t0)
            infeasible_before_end = pol.last_step_infeasible
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            traj.append(env.agent_position.copy())
            clear.append(true_clearance(env))
    finally:
        trace.detach()
        recd.detach()
        wrap.detach()
    ep_time = time.perf_counter() - t_ep
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
                                if np.isfinite(pol.min_cbc_solved) else float("nan")),
             "stuck": int(stuck(traj, outcome))}
    rec = episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info, extra=extra)
    rec.update(continuation_metrics(ArmSpec(arm, "random", params=params, random=rspec),
                                    recd.steps, wrap, outcome, steps, times, ep_time))
    candidate = None if cfg is None else {
        "params": cfg.as_dict(), **(CA.NO_LAYER if layer is None else layer.summary())}
    rec.update({"arm": arm, "config": None if cfg is None else cfg.name,
                "candidate": candidate, "cell": list(cell), "family": cell.family,
                "obstacles": cell.obstacles, "motion": motion, "controller": pol.ctrl.name,
                "solver": "frozen", "hold": 1,
                "start": [float(x) for x in start], "goal": [float(x) for x in goal],
                "params": params.as_dict(), "random_spec": rspec.as_dict(),
                "generator_version": None if spec is None else spec.generator_version,
                "spec": None if spec is None else spec.as_dict(),
                "scenario_events": list(env.scenario_events),
                "trigger_fired": env.trigger_step is not None,
                "trigger_step": env.trigger_step,
                "final_obstacle_positions": [[float(x) for x in q]
                                             for q in env.obstacle_positions],
                "trace": trace.steps,
                "trajectory": [[float(x) for x in q] for q in traj]})
    if keep_step_times:
        rec["step_times"] = [float(t) for t in times]
    return rec
