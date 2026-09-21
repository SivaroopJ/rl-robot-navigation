"""Pursuit episode harness: one shared pre-step state, two controllers, simultaneous application.

PROTAGONIST is the frozen Tuned + Random CLF-DR-CBF, built from the frozen config files and with
its source swapped for `HunterAugmentedSource` (the known-state hunter row). No LADDER anywhere.
HUNTER is `HunterController`: pursuit nominal action + the same DR-CBF filter.

CONTROLLER CONFIGURATIONS (`ControllerSetup`, policy-matrix experiment)
    A  Protag Random-CLF-DR-CBF, Hunter tuned CLF-DR-CBF (u = 0 on non-optimal)  -- historical
    B  Protag Random-CLF-DR-CBF, Hunter Random-CLF-DR-CBF (frozen RandomRecovery, own RNG)
    C  Protag CD-Random-CLF-DR-CBF, Hunter CD-Random-CLF-DR-CBF
`cd_mode` "shadow" installs the congestion wrapper without changing any action (B diagnostics).
Wrapper order for a CD agent: CongestionWrapper first, RandomRecovery attached after it.
`controllers=None` (every earlier caller) is the audited episode, bit for bit.

Per step, in this order and no other:
    1. read ONE pre-step state from the env
    2. protagonist action from that state (hunter row pushed from the same state)
    3. hunter action from that state
    4. env.step_pursuit applies both, then the dynamic obstacles move
    5. outcomes evaluated
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from continuation.policy import TunableDRCBFPolicy, policy_kwargs
from continuation.random_recovery import RandomRecovery
from continuation.seeds import CONTROLLER_RNG_TAG, controller_rng
from dr_control.velocity_tracker import LidarVelocityTracker
from evaluation.shortest_path import ShortestPathOracle

from continuation.congestion.detector import alarm_stats
from continuation.congestion.wrapper import CDConfig, CongestionWrapper
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.hunter_policies import direct_target, predict_position
from continuation.pursuit.mpc import MPCConfig, plan as mpc_plan
from continuation.pursuit.source import HunterAugmentedSource
from continuation.sensed_obstacles import SensedObstacles

MAP_ID = "M0_pursuit(config.json, 5 rects, 6 dyn obstacles @0.675, + 1 hunter)"
#: Mixed into the hunter's recovery RNG so it can never share the protagonist's stream.
HUNTER_RNG_TAG = 0x48554E54                     # "HUNT"
STEP_BUDGET_MS = 100.0                          # dt = 0.1 s


@dataclass(frozen=True)
class ControllerSetup:
    name: str
    hunter_recovery: bool
    cd_mode: str = "off"                        # off | shadow | active

    def __post_init__(self):
        if self.cd_mode not in ("off", "shadow", "active"):
            raise ValueError("cd_mode must be off, shadow or active")
        if self.cd_mode == "active" and not self.hunter_recovery:
            raise ValueError("CD-random requires Random recovery on the hunter")

    @property
    def protag_controller(self):
        return "CD-Random-CLF-DR-CBF" if self.cd_mode == "active" else "Random-CLF-DR-CBF"

    @property
    def hunter_controller(self):
        if self.cd_mode == "active":
            return "CD-Random-CLF-DR-CBF"
        return "Random-CLF-DR-CBF" if self.hunter_recovery else "Tuned-CLF-DR-CBF"

    def as_dict(self):
        d = asdict(self)
        d.update(protag_controller=self.protag_controller,
                 hunter_controller=self.hunter_controller)
        return d


CONTROLLER_SETUPS = {
    "A": ControllerSetup("A", hunter_recovery=False, cd_mode="off"),
    "B": ControllerSetup("B", hunter_recovery=True, cd_mode="shadow"),
    "C": ControllerSetup("C", hunter_recovery=True, cd_mode="active"),
}


def hunter_rng(episode_seed):
    return np.random.default_rng([CONTROLLER_RNG_TAG, HUNTER_RNG_TAG, int(episode_seed)])


def attach_hunter_controllers(hunter, controllers, cd_run, *, spec, seed, tuned, env, cfg):
    """(congestion wrapper or None, RandomRecovery or None), attached in the required order:
    the congestion wrapper first, the frozen RandomRecovery after it (so Random wraps CD)."""
    hcd = hwrap = None
    if cd_run is not None:
        hcd = CongestionWrapper(hunter.ctrl, source_fn=lambda: hunter.src, cfg=cd_run,
                                tau=tuned.tau_eff(), n_rays=env.N_LIDAR_RAYS,
                                lidar_range=env.LIDAR_RANGE, r_robot=cfg.hunter_radius,
                                r_obstacle=env.OBSTACLE_RADIUS)
    if controllers.hunter_recovery:
        hwrap = RandomRecovery(hunter.ctrl, spec=spec, rng=hunter_rng(seed), tau=tuned.tau_eff())
    if hcd is not None:
        hcd.recovery = hwrap
    return hcd, hwrap


def make_pursuit_env(cfg: PursuitConfig, *, n_dynamic_obstacles=6, config_path="config.json"):
    return PursuitEnv(pursuit=cfg, config_path=config_path,
                      n_dynamic_obstacles=n_dynamic_obstacles, obstacle_speed=0.675,
                      render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=(cfg.motion == "randomized"))


def build_protagonist(env, obs, cfg, tuned, spec, seed, *, cd=None):
    """Frozen policy, frozen recovery; only the barrier source is swapped (audit section 4.3).

    `cd` (CDConfig): install the congestion wrapper on the controller BEFORE Random recovery;
    it is exposed as `pol.cd` (None otherwise)."""
    pol = TunableDRCBFPolicy(params=tuned, **policy_kwargs(env))
    pol.reset(obs, env.agent_position)
    tracker = LidarVelocityTracker(r_nominal=pol.r_nominal, dt=pol.dt, n_rays=pol.n_rays,
                                   lidar_range=pol.lidar_range)
    pol.src = HunterAugmentedSource(hunter_radius=cfg.hunter_radius,
                                    r_robot=pol.agent_radius, k_scans=tuned.k_scans,
                                    n_rays=pol.n_rays, lidar_range=pol.lidar_range,
                                    tracker=tracker)
    pol.cd = None
    if cd is not None:
        pol.cd = CongestionWrapper(pol.ctrl, source_fn=lambda: pol.src, cfg=cd,
                                   tau=tuned.tau_eff(), n_rays=pol.n_rays,
                                   lidar_range=pol.lidar_range, r_robot=pol.agent_radius,
                                   r_obstacle=env.OBSTACLE_RADIUS)
    wrap = None
    if cfg.condition == "P-Random":
        wrap = RandomRecovery(pol.ctrl, spec=spec, rng=controller_rng(seed),
                              tau=tuned.tau_eff())
    if pol.cd is not None:
        pol.cd.recovery = wrap
    return pol, wrap


def run_episode(cfg: PursuitConfig, seed, *, tuned, spec, env=None, oracle=None, variant=None,
                trace=False, controllers=None, mpc_cfg=None, cd_cfg=None, step_log=False):
    """`variant=None` is the audited P3 episode, bit for bit. A `HunterVariant` (hunter_policies)
    selects the pursuit target and the Protag-LiDAR handling; the protagonist is unchanged.
    `controllers` (ControllerSetup) enables the policy-matrix configurations and diagnostics."""
    own = env is None
    env = env or make_pursuit_env(cfg)
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    matrix = controllers is not None
    if matrix and variant is None:
        raise ValueError("the policy matrix needs a policy (variant)")
    if variant is not None and variant.policy == "mpc" and not matrix:
        raise ValueError("MPC pursuit runs only inside the policy matrix (pass controllers)")
    mpc_cfg = mpc_cfg or MPCConfig()
    cd_on = matrix and controllers.cd_mode != "off"
    cd_run = replace(cd_cfg or CDConfig(), mode=controllers.cd_mode) if cd_on else None

    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    hunter_start = env.hunter_position.copy()
    pol, wrap = build_protagonist(env, obs, cfg, tuned, spec, seed, cd=cd_run)
    filtering = variant is not None and variant.lidar_mode == "filter"
    hunter = HunterController(params=tuned, max_speed=cfg.hunter_max_speed,
                              radius=cfg.hunter_radius, n_rays=env.N_LIDAR_RAYS,
                              lidar_range=env.LIDAR_RANGE, dt=env.dt,
                              filter_radius=(variant.filter_radius(env.AGENT_RADIUS)
                                             if filtering else None),
                              protag_radius=env.AGENT_RADIUS)
    hcd, hwrap = (attach_hunter_controllers(hunter, controllers, cd_run, spec=spec, seed=seed,
                                            tuned=tuned, env=env, cfg=cfg)
                  if matrix else (None, None))
    horizon_steps = (int(round(variant.prediction_horizon / env.dt))
                     if variant is not None and variant.policy == "predictive" else 0)
    is_mpc = variant is not None and variant.policy == "mpc"

    traj, clear_obs, d_hunter, t_prot, t_hunt = [start.copy()], [], [], [], []
    preds, n_pred_fallback, n_pred_clipped, steps_trace = [], 0, 0, []
    S = {k: [] for k in ("p_qp", "p_rec", "p_cd", "p_inf", "p_event", "p_u0", "p_speed",
                         "h_qp", "h_rec", "h_cd", "h_mpc", "h_inf", "h_event", "h_u0",
                         "h_speed", "h_active", "h_cd_ran", "mpc_fallback", "mpc_rejected",
                         "mpc_pred_capture")} if matrix else None
    mpc_preds = []
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        pre = env.pre_step_state()                              # 1. ONE shared state
        env_obs = obs
        hunter_active = not env.hunter_disabled
        t0 = time.perf_counter()
        pol.src.push_hunter(pre["p_hunter"], pre["v_hunter"])   # known-state channel
        if pol.cd is not None:
            pol.cd.begin_step(ranges=np.asarray(env_obs, float)[4:4 + pol.n_rays] * pol.lidar_range)
        n_solve = len(pol.solve_times)
        a_prot, _ = pol.predict(env_obs, pre["p_prot"], deterministic=True)
        t_prot.append(time.perf_counter() - t0)
        if matrix:
            _log_protag(S, pol, wrap, n_solve, a_prot, env)
        t1 = time.perf_counter()
        if variant is None:
            u_hunt, h_info = hunter.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
            p_target = pre["p_prot"]
        else:
            ranges = env.hunter_lidar(include_protagonist=variant.protag_in_hunter_lidar)
            nominal = None
            if is_mpc:
                p_target = None
                nominal = _mpc_nominal(pre, env, cfg, hunter, variant, mpc_cfg, filtering)
            else:
                p_target, pinfo = _hunter_target(variant, pre, env)
                if variant.policy == "predictive":
                    preds.append(p_target.copy())
                    n_pred_fallback += int(pinfo["fallback"])
                    n_pred_clipped += int(pinfo["clipped"])
            if hcd is not None:
                hcd.begin_step(ranges=ranges, exclude_near=pre["p_prot"] if filtering else None,
                               active=hunter_active)
            n_cd = len(hcd.steps) if hcd is not None else 0
            u_hunt, h_info = hunter.act(pre["p_hunter"], ranges, p_target,
                                        filter_reference=pre["p_prot"] if filtering else None,
                                        nominal=nominal)
            if is_mpc and "nominal_info" in h_info:
                p_target = h_info["gamma"]
                mpc_preds.append(h_info["nominal_info"]["pred_protag_end"])
        t_hunt.append(time.perf_counter() - t1)
        if matrix:
            _log_hunter(S, h_info, hcd, n_cd if variant is not None else 0, u_hunt, cfg,
                        hunter_active)

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
    if pol.cd is not None:
        pol.cd.detach()
    if hwrap is not None:
        hwrap.detach()
    if hcd is not None:
        hcd.detach()

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
    if matrix:
        rec.update(_matrix_fields(controllers, variant, S, tp, th, pol, hcd, hwrap, mpc_cfg,
                                  mpc_preds, traj, steps))
        if step_log:
            rec["step_log"] = _step_log(S, pol.cd, hcd)
    if own:
        env.close()
    return rec


# --------------------------------------------------------------------------- per-step logging
def _log_protag(S, pol, wrap, n_solve, a_prot, env):
    r = wrap.records[-1] if wrap is not None and wrap.records else {}
    S["p_qp"].append(pol.solve_times[-1] * 1e3 if len(pol.solve_times) > n_solve else np.nan)
    S["p_rec"].append(float(r.get("t_recovery", 0.0)) * 1e3)
    S["p_cd"].append(pol.cd.steps[-1]["cd_ms"] if pol.cd is not None and pol.cd.steps else 0.0)
    S["p_inf"].append(int(bool(pol.last_step_infeasible)))
    S["p_event"].append(int(r.get("mode") == "event"))
    status = r.get("status") if r else None
    S["p_u0"].append(int(status not in (None, "optimal") and r.get("mode") != "event")
                     if r else int(bool(pol.last_step_infeasible)))
    S["p_speed"].append(float(np.linalg.norm(np.clip(a_prot, -1, 1))) * env.MAX_SPEED)


def _log_hunter(S, h_info, hcd, n_cd, u_hunt, cfg, active):
    status = str(h_info.get("status"))
    t = h_info.get("total_time")
    S["h_qp"].append(float(t) * 1e3 if t is not None and t == t else np.nan)
    S["h_rec"].append(float(h_info.get("t_recovery", 0.0)) * 1e3)
    ran = hcd is not None and len(hcd.steps) > n_cd
    S["h_cd"].append(hcd.steps[-1]["cd_ms"] if ran else 0.0)
    S["h_cd_ran"].append(int(ran))
    S["h_inf"].append(int("infeasible" in status))
    ev = h_info.get("mode") == "event"
    S["h_event"].append(int(ev))
    S["h_u0"].append(int(status != "optimal" and not ev))
    v = np.clip(np.asarray(u_hunt, float), -cfg.hunter_max_speed, cfg.hunter_max_speed)
    S["h_speed"].append(float(np.linalg.norm(v)) if active else 0.0)
    S["h_active"].append(int(active))
    ni = h_info.get("nominal_info")
    S["h_mpc"].append(ni["mpc_ms"] if ni else 0.0)
    S["mpc_fallback"].append(int(bool(ni and ni["fallback"])))
    S["mpc_rejected"].append(int(ni["n_rejected"]) if ni else 0)
    S["mpc_pred_capture"].append(int(bool(ni and ni["pred_capture"])))


def _stats(x):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def _cd_fields(prefix, wrapper, infeasible, speeds):
    st = wrapper.steps
    states = [s["state"] for s in st]
    active = [s["active"] for s in st]
    al = alarm_stats(states, infeasible[:len(st)], active)
    sp = np.asarray(speeds[:len(st)], float)
    stt = np.asarray(states, int)
    return {
        f"{prefix}_cd_mode": wrapper.cfg.mode,
        f"{prefix}_cd_thresholds": wrapper.cfg.detector.as_dict(),
        f"{prefix}_cd_alarms": {k: v for k, v in al.items() if k != "lead_steps"},
        f"{prefix}_cd_lead_steps": al["lead_steps"],
        f"{prefix}_cd_frac_cautious_or_worse": (al["warning_steps"] / max(al["active_steps"], 1)),
        f"{prefix}_cd_min_margin": float(min(s["margin"] for s in st)) if st else float("nan"),
        f"{prefix}_cd_min_clr": float(min(s["clr_look"] for s in st)) if st else float("nan"),
        f"{prefix}_cd_mean_scale": float(np.mean([s["scale"] for s in st])) if st else 1.0,
        f"{prefix}_cd_n_direction_changes": int(sum(1 for s in st if s["angle"] != 0.0)),
        f"{prefix}_cd_speed_by_state": [float(sp[stt == k].mean()) if (stt == k).any() else None
                                        for k in (0, 1, 2)],
    }


def _matrix_fields(controllers, variant, S, tp, th, pol, hcd, hwrap, mpc_cfg, mpc_preds, traj,
                   steps):
    out = {"policy_id": variant.name, "controller_setup": controllers.as_dict()}
    for agent, total, keys in (("prot", tp, ("p_qp", "p_rec", "p_cd")),
                               ("hunter", th, ("h_qp", "h_rec", "h_cd", "h_mpc"))):
        out[f"{agent}_latency_ms"] = {"total": _stats(total)}
        for k in keys:
            out[f"{agent}_latency_ms"][{"qp": "qp", "rec": "recovery", "cd": "cd",
                                       "mpc": "mpc"}[k.split("_")[1]]] = _stats(S[k])
        out[f"{agent}_frac_steps_over_budget"] = float(np.mean(np.asarray(total) > STEP_BUDGET_MS))
    out["prot_n_u0_steps"] = int(sum(S["p_u0"]))
    out["hunter_n_u0_steps"] = int(sum(S["h_u0"]))
    out["hunter_n_recovery_events"] = len(hwrap.events) if hwrap is not None else 0
    out["hunter_recovery_fraction"] = (sum(S["h_event"]) / max(steps, 1))
    out["hunter_active_steps"] = int(sum(S["h_active"]))
    if variant.policy == "mpc":
        k = mpc_cfg.horizon_steps
        err = [float(np.linalg.norm(mpc_preds[t] - traj[t + k])) for t in range(len(mpc_preds))
               if t + k <= steps]
        ms = np.asarray(S["h_mpc"], float)
        out.update({
            "mpc_config": mpc_cfg.as_dict(),
            "mpc_n_steps": len(mpc_preds),
            "mpc_n_fallback": int(sum(S["mpc_fallback"])),
            "mpc_fallback_rate": sum(S["mpc_fallback"]) / max(len(mpc_preds), 1),
            "mpc_mean_rejected": float(np.mean(S["mpc_rejected"][:len(mpc_preds)]))
            if mpc_preds else float("nan"),
            "mpc_n_pred_capture": int(sum(S["mpc_pred_capture"])),
            "mpc_ms": _stats(ms[ms > 0]),
            "mpc_pred_err_1s_mean": float(np.mean(err)) if err else None,
        })
    if pol.cd is not None:
        out.update(_cd_fields("prot", pol.cd, S["p_inf"], S["p_speed"]))
    if hcd is not None:
        # a hunter CD entry exists only on steps where the QP ran; align on exactly those steps
        ran = [i for i, r in enumerate(S["h_cd_ran"]) if r]
        out.update(_cd_fields("hunter", hcd, [S["h_inf"][i] for i in ran],
                              [S["h_speed"][i] for i in ran]))
    return out


def _step_log(S, pcd, hcd):
    r4 = lambda xs: [None if (x is None or (isinstance(x, float) and not np.isfinite(x)))
                     else round(float(x), 4) for x in xs]
    out = {k: (v if k.endswith(("inf", "event", "u0", "active", "fallback", "rejected",
                                "capture", "ran")) else r4(v)) for k, v in S.items()}
    for name, w in (("p_cd_log", pcd), ("h_cd_log", hcd)):
        if w is None:
            continue
        out[name] = {k: ([s[k] for s in w.steps] if k in ("state", "n_near", "active",
                                                          "recent_recovery", "excluded_tracks")
                         else r4([s[k] for s in w.steps]))
                     for k in ("state", "margin", "M0", "M_look", "clr_look", "gap", "n_near",
                               "max_closing", "min_ttc", "recent_recovery", "scale", "angle",
                               "excluded_tracks", "active")}
    return out


# --------------------------------------------------------------------------- pursuit targets
def _mpc_nominal(pre, env, cfg, hunter, variant, mpc_cfg, filtering):
    def nominal(p, src):
        obst = SensedObstacles.from_source(src, r_robot=cfg.hunter_radius,
                                           r_obstacle=env.OBSTACLE_RADIUS,
                                           exclude_near=pre["p_prot"] if filtering else None,
                                           exclude_radius=mpc_cfg.exclude_radius)
        gamma, u_nom, minfo = mpc_plan(p, pre["p_prot"], pre["v_prot"], env.target_position,
                                       obst, mpc_cfg, dt=env.dt, max_v=hunter.max_speed,
                                       goal_accel=variant.goal_accel, protag_v_max=env.MAX_SPEED,
                                       world_size=env.WORLD_SIZE, body_radius=env.AGENT_RADIUS)
        minfo["n_excluded_tracks"] = obst.n_excluded_tracks
        return gamma, u_nom, minfo
    return nominal


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
