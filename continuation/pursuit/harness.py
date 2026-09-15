"""Pursuit episode harness: one shared pre-step state, two controllers, simultaneous application.

PROTAGONIST is the frozen Tuned + Random CLF-DR-CBF, built from the frozen config files and with
its source swapped for `HunterAugmentedSource` (the known-state hunter row). No LADDER anywhere.
HUNTER is `HunterController`: pursuit nominal action + the same DR-CBF filter, frozen u = 0 fallback.

Per step, in this order and no other:
    1. read ONE pre-step state from the env
    2. protagonist action from that state (hunter row pushed from the same state)
    3. hunter action from that state
    4. env.step_pursuit applies both, then the dynamic obstacles move
    5. outcomes evaluated
"""
from __future__ import annotations

import time

import numpy as np

from continuation.policy import TunableDRCBFPolicy, policy_kwargs
from continuation.random_recovery import RandomRecovery
from continuation.seeds import controller_rng
from dr_control.velocity_tracker import LidarVelocityTracker
from evaluation.shortest_path import ShortestPathOracle

from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.hunter_policies import direct_target, predict_position
from continuation.pursuit.source import HunterAugmentedSource

MAP_ID = "M0_pursuit(config.json, 5 rects, 6 dyn obstacles @0.675, + 1 hunter)"


def make_pursuit_env(cfg: PursuitConfig, *, n_dynamic_obstacles=6, config_path="config.json"):
    return PursuitEnv(pursuit=cfg, config_path=config_path,
                      n_dynamic_obstacles=n_dynamic_obstacles, obstacle_speed=0.675,
                      render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=(cfg.motion == "randomized"))


def build_protagonist(env, obs, cfg, tuned, spec, seed):
    """Frozen policy, frozen recovery; only the barrier source is swapped (audit section 4.3)."""
    pol = TunableDRCBFPolicy(params=tuned, **policy_kwargs(env))
    pol.reset(obs, env.agent_position)
    tracker = LidarVelocityTracker(r_nominal=pol.r_nominal, dt=pol.dt, n_rays=pol.n_rays,
                                   lidar_range=pol.lidar_range)
    pol.src = HunterAugmentedSource(hunter_radius=cfg.hunter_radius,
                                    r_robot=pol.agent_radius, k_scans=tuned.k_scans,
                                    n_rays=pol.n_rays, lidar_range=pol.lidar_range,
                                    tracker=tracker)
    wrap = None
    if cfg.condition == "P-Random":
        wrap = RandomRecovery(pol.ctrl, spec=spec, rng=controller_rng(seed),
                              tau=tuned.tau_eff())
    return pol, wrap


def run_episode(cfg: PursuitConfig, seed, *, tuned, spec, env=None, oracle=None, variant=None,
                trace=False):
    """`variant=None` is the audited P3 episode, bit for bit. A `HunterVariant` (hunter_policies)
    selects the pursuit target and the Protag-LiDAR handling; the protagonist is unchanged."""
    own = env is None
    env = env or make_pursuit_env(cfg)
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    hunter_start = env.hunter_position.copy()
    pol, wrap = build_protagonist(env, obs, cfg, tuned, spec, seed)
    filtering = variant is not None and variant.lidar_mode == "filter"
    hunter = HunterController(params=tuned, max_speed=cfg.hunter_max_speed,
                              radius=cfg.hunter_radius, n_rays=env.N_LIDAR_RAYS,
                              lidar_range=env.LIDAR_RANGE, dt=env.dt,
                              filter_radius=(variant.filter_radius(env.AGENT_RADIUS)
                                             if filtering else None),
                              protag_radius=env.AGENT_RADIUS)
    horizon_steps = (int(round(variant.prediction_horizon / env.dt))
                     if variant is not None and variant.policy == "predictive" else 0)

    traj, clear_obs, d_hunter, t_prot, t_hunt = [start.copy()], [], [], [], []
    preds, n_pred_fallback, n_pred_clipped, steps_trace = [], 0, 0, []
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        pre = env.pre_step_state()                              # 1. ONE shared state
        env_obs = obs
        t0 = time.perf_counter()
        pol.src.push_hunter(pre["p_hunter"], pre["v_hunter"])   # known-state channel
        a_prot, _ = pol.predict(env_obs, pre["p_prot"], deterministic=True)
        t_prot.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        if variant is None:
            u_hunt, h_info = hunter.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
            p_target = pre["p_prot"]
        else:
            p_target, pinfo = _hunter_target(variant, pre, env)
            if variant.policy == "predictive":
                preds.append(p_target.copy())
                n_pred_fallback += int(pinfo["fallback"])
                n_pred_clipped += int(pinfo["clipped"])
            ranges = env.hunter_lidar(include_protagonist=variant.protag_in_hunter_lidar)
            u_hunt, h_info = hunter.act(pre["p_hunter"], ranges, p_target,
                                        filter_reference=pre["p_prot"] if filtering else None)
        t_hunt.append(time.perf_counter() - t1)

        obs, _, term, trunc, info = env.step_pursuit(a_prot, u_hunt)   # 4. simultaneous
        steps += 1
        traj.append(env.agent_position.copy())
        d_hunter.append(info["hunter_distance"])
        clear_obs.append(_true_clearance(env))
        if trace:
            steps_trace.append({
                "step": pre["step"], "p_prot": pre["p_prot"].tolist(),
                "v_prot": pre["v_prot"].tolist(), "goal": goal.tolist(),
                "horizon_s": variant.prediction_horizon if horizon_steps else 0.0,
                "p_target": np.asarray(p_target, float).tolist(),
                "p_hunter": pre["p_hunter"].tolist(),
                "u_hunter_nominal": np.asarray(h_info.get("u_nom"), float).tolist(),
                "u_hunter": np.asarray(u_hunt, float).tolist(),
                "hunter_qp_status": str(h_info.get("status")),
                "outcome": info.get("outcome")})
    ep_time = time.perf_counter() - t_ep
    if wrap is not None:
        wrap.detach()

    outcome = info.get("outcome") or ("timeout" if trunc else "unresolved")
    traj = np.asarray(traj)
    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    sp = oracle.path_length(start, goal)
    tp, th = np.asarray(t_prot) * 1e3, np.asarray(t_hunt) * 1e3
    rec = {
        "seed": seed, "condition": cfg.condition, "outcome": outcome, "steps": steps,
        "goal": int(bool(info.get("goal"))), "captured": int(bool(info.get("captured"))),
        "agent_contact": int(bool(info.get("agent_contact"))),
        "protagonist_collision_type": info.get("protagonist_collision_type"),
        "protagonist_collision": int(info.get("protagonist_collision_type") is not None),
        "hunter_collision_type": env.hunter_collision_type,
        "hunter_collision": int(env.hunter_collision_type is not None),
        "hunter_collision_step": env.hunter_collision_step,
        "timeout": int(outcome == "timeout"),
        "time_to_goal_s": steps * env.dt if outcome == "goal" else None,
        "time_to_capture_s": steps * env.dt if outcome == "capture" else None,
        "path_length": plen, "shortest_path": float(sp) if sp is not None else None,
        "spl": (float(sp / max(plen, sp, 1e-9)) if (outcome == "goal" and sp) else 0.0),
        "min_hunter_distance": float(np.min(d_hunter)) if d_hunter else float("nan"),
        "min_clearance": float(np.min(clear_obs)) if clear_obs else float("nan"),
        "start": [float(x) for x in start], "goal_xy": [float(x) for x in goal],
        "hunter_start": [float(x) for x in hunter_start],
        "hunter_start_distance": float(np.linalg.norm(hunter_start - start)),
        # protagonist controller
        "prot_n_infeasible": pol.n_infeasible, "prot_n_solver_fail": pol.n_solver_fail,
        "prot_frac_infeasible": pol.n_infeasible / max(steps, 1),
        "prot_mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
        "prot_n_recovery_events": len(wrap.events) if wrap is not None else 0,
        "prot_recovery_fraction": ((sum(1 for r in wrap.records if r["mode"] in ("event", "persist"))
                                    / max(steps, 1)) if wrap is not None else 0.0),
        "prot_hunter_rows": int(pol.src.n_hunter_rows),
        # hunter controller
        "hunter_n_infeasible": hunter.n_infeasible, "hunter_n_fallback": hunter.n_fallback,
        "hunter_frac_infeasible": hunter.n_infeasible / max(steps, 1),
        "hunter_n_nominal_unsafe": hunter.n_nominal_unsafe,
        "hunter_frac_nominal_unsafe": hunter.n_nominal_unsafe / max(steps, 1),
        "hunter_n_materially_filtered": hunter.n_materially_filtered,
        "hunter_frac_materially_filtered": hunter.n_materially_filtered / max(steps, 1),
        "hunter_mean_u_dev": float(np.mean(hunter.u_dev)) if hunter.u_dev else float("nan"),
        # timing
        "prot_ms_mean": float(tp.mean()), "prot_ms_p95": float(np.percentile(tp, 95)),
        "hunter_ms_mean": float(th.mean()), "hunter_ms_p95": float(np.percentile(th, 95)),
        "episode_time_s": ep_time,
    }
    if variant is not None:
        rec.update(_variant_fields(variant, info, outcome, hunter, preds, traj, steps,
                                   horizon_steps, n_pred_fallback, n_pred_clipped))
        if trace:
            rec["trace"] = steps_trace
    if own:
        env.close()
    return rec


def _hunter_target(variant, pre, env):
    """(p_target, info) from the SHARED PRE-STEP state only; never from any future state."""
    if variant.policy == "direct":
        return direct_target(pre["p_prot"]), {"fallback": False, "clipped": False}
    q, pinfo = predict_position(pre["p_prot"], pre["v_prot"], env.target_position,
                                horizon=variant.prediction_horizon,
                                goal_accel=variant.goal_accel, dt=env.dt, v_max=env.MAX_SPEED,
                                world_size=env.WORLD_SIZE, body_radius=env.AGENT_RADIUS)
    if q is None:
        raise FloatingPointError("Protag state is non-finite; cannot form a pursuit target")
    return q, pinfo


def _variant_fields(variant, info, outcome, hunter, preds, traj, steps, k, n_fb, n_clip):
    out = {
        "hunter_variant": variant.as_dict(),
        "capture_distance_at_capture": (float(info["hunter_distance"])
                                        if outcome == "capture" else None),
        "hunter_n_no_barrier_rows": hunter.n_no_barrier_rows,
        "hunter_n_solver_fail": hunter.n_solver_fail,
    }
    if variant.policy == "predictive":
        # OFFLINE evaluation only: p_hat made at step t is compared with the realised Protag
        # position k steps later. Nothing here reaches the online hunter.
        err = [float(np.linalg.norm(preds[t] - traj[t + k])) for t in range(len(preds))
               if t + k <= steps]
        out.update({
            "prediction_horizon_steps": k,
            "prediction_n_evaluable": len(err),
            "prediction_error_mean": float(np.mean(err)) if err else None,
            "prediction_error_median": float(np.median(err)) if err else None,
            "prediction_error_p90": float(np.percentile(err, 90)) if err else None,
            "prediction_n_fallback": n_fb, "prediction_n_speed_clipped": n_clip,
        })
    if variant.lidar_mode == "filter":
        out["filter_radius"] = float(hunter.src.filter_radius)
        out["filter_stats"] = hunter.src.filter_stats()
    return out


def _true_clearance(env):
    """Ground-truth clearance to STATIC and DYNAMIC obstacles and walls. Measurement only."""
    p = env.agent_position
    vals = [p[0] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[0],
            p[1] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[1]]
    for cx, cy, hw, hh in env.static_obstacles:
        q = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
        vals.append(float(np.linalg.norm(p - q)) - env.AGENT_RADIUS)
    if len(env.obstacle_positions):
        d = np.linalg.norm(env.obstacle_positions - p[None, :], axis=1)
        vals.append(float(d.min()) - env.AGENT_RADIUS - env.OBSTACLE_RADIUS)
    return float(min(vals))
