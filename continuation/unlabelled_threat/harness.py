"""Track A continuation: oracle vs unlabelled-threat vs visible-only, on the pursuit environment.

ARMS (the hunter's policy and safety filter are IDENTICAL in all three)
    U0_oracle       audited baseline: hunter INVISIBLE to the protagonist's LiDAR, one oracle row
                    built from the hunter's true position and velocity. Unchanged code path.
    U1_unlabelled   hunter VISIBLE to LiDAR, NO oracle row, one row per eligible moving track.
    U2_visible_only hunter VISIBLE to LiDAR, NO oracle row, NO track rows -- the frozen navigation
                    controller with the hunter present merely as geometry. This arm exists to
                    separate "the protagonist can see the hunter at all" from "the protagonist adds
                    explicit threat constraints", which otherwise differ together between U0 and U1.

Everything else -- map generation, episode termination, capture/contact rules, stochastic obstacle
model, simultaneous update, protagonist controller family and its frozen Random recovery -- is the
existing pursuit machinery, imported unchanged.
"""
from __future__ import annotations

import time

import numpy as np

from continuation.policy import TunableDRCBFPolicy, policy_kwargs
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.harness import _true_clearance, build_protagonist
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.source import HunterAugmentedSource
from continuation.random_recovery import RandomRecovery
from continuation.seeds import controller_rng
from continuation.unlabelled_threat.source import UnlabelledThreatSource
from dr_control.velocity_tracker import LidarVelocityTracker
from evaluation.shortest_path import ShortestPathOracle

ARMS = ("U0_oracle", "U1_unlabelled", "U2_visible_only")


def arm_config(arm, *, hunter_speed=1.0):
    """PursuitConfig per arm. U0 keeps the audited known-state model verbatim."""
    if arm == "U0_oracle":
        return PursuitConfig(condition="P-Random", info_model="known_state",
                             hunter_max_speed=hunter_speed)
    if arm in ("U1_unlabelled", "U2_visible_only"):
        return PursuitConfig(condition="P-Random", info_model="lidar_estimate",
                             hunter_max_speed=hunter_speed)
    raise ValueError(arm)


def make_env(cfg, *, n_dynamic_obstacles=6, config_path="config.json"):
    return PursuitEnv(pursuit=cfg, config_path=config_path,
                      n_dynamic_obstacles=n_dynamic_obstacles, obstacle_speed=0.675,
                      render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=(cfg.motion == "randomized"))


def build_protagonist_arm(arm, env, obs, cfg, tuned, spec, seed):
    """(policy, recovery_wrapper). U0 delegates to the unchanged pursuit builder."""
    if arm == "U0_oracle":
        return build_protagonist(env, obs, cfg, tuned, spec, seed)
    pol = TunableDRCBFPolicy(params=tuned, **policy_kwargs(env))
    pol.reset(obs, env.agent_position)
    tracker = LidarVelocityTracker(r_nominal=pol.r_nominal, dt=pol.dt, n_rays=pol.n_rays,
                                   lidar_range=pol.lidar_range)
    pol.src = UnlabelledThreatSource(
        track_radius=cfg.hunter_radius, dt=pol.dt,
        enable_track_rows=(arm == "U1_unlabelled"),
        # the SAME prior static map the frozen A* planner already receives (no new channel)
        static_obstacles=env.static_obstacles, world_size=env.WORLD_SIZE,
        r_robot=pol.agent_radius, k_scans=tuned.k_scans, n_rays=pol.n_rays,
        lidar_range=pol.lidar_range, tracker=tracker)
    wrap = RandomRecovery(pol.ctrl, spec=spec, rng=controller_rng(seed), tau=tuned.tau_eff())
    return pol, wrap


def run_episode(arm, seed, *, tuned, spec, hunter_speed=1.0, env=None, oracle=None):
    cfg = arm_config(arm, hunter_speed=hunter_speed)
    own = env is None
    env = env or make_env(cfg)
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    hunter_start = env.hunter_position.copy()
    pol, wrap = build_protagonist_arm(arm, env, obs, cfg, tuned, spec, seed)
    hunter = HunterController(params=tuned, max_speed=cfg.hunter_max_speed,
                              radius=cfg.hunter_radius, n_rays=env.N_LIDAR_RAYS,
                              lidar_range=env.LIDAR_RANGE, dt=env.dt)
    n_rows, kept_track_rows, d_hunter, clear, times = [], [], [], [], []
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        pre = env.pre_step_state()
        t0 = time.perf_counter()
        if arm == "U0_oracle":
            pol.src.push_hunter(pre["p_hunter"], pre["v_hunter"])     # privileged, U0 only
        a_prot, _ = pol.predict(obs, pre["p_prot"], deterministic=True)
        times.append(time.perf_counter() - t0)
        u_hunt, _ = hunter.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
        # row bookkeeping (diagnostic only; not fed back to the controller)
        if isinstance(pol.src, UnlabelledThreatSource):
            n_rows.append(len(pol.src.last_rows))
        obs, _, term, trunc, info = env.step_pursuit(a_prot, u_hunt)
        steps += 1
        d_hunter.append(info["hunter_distance"])
        clear.append(_true_clearance(env))
    ep_time = time.perf_counter() - t_ep
    if wrap is not None:
        wrap.detach()

    outcome = info.get("outcome") or ("timeout" if trunc else "unresolved")
    t = np.asarray(times) * 1e3
    rec = {
        "arm": arm, "seed": seed, "outcome": outcome, "steps": steps,
        "goal": int(bool(info.get("goal"))), "captured": int(bool(info.get("captured"))),
        "agent_contact": int(bool(info.get("agent_contact"))),
        "protagonist_collision": int(info.get("protagonist_collision_type") is not None),
        "protagonist_collision_type": info.get("protagonist_collision_type"),
        "hunter_collision": int(env.hunter_collision_type is not None),
        "hunter_collision_type": env.hunter_collision_type,
        "timeout": int(outcome == "timeout"),
        "min_hunter_distance": float(np.min(d_hunter)) if d_hunter else float("nan"),
        "min_clearance": float(np.min(clear)) if clear else float("nan"),
        "start": [float(x) for x in start], "goal_xy": [float(x) for x in goal],
        "hunter_start": [float(x) for x in hunter_start],
        "prot_n_infeasible": pol.n_infeasible, "prot_n_solver_fail": pol.n_solver_fail,
        "prot_frac_infeasible": pol.n_infeasible / max(steps, 1),
        "prot_n_recovery_events": len(wrap.events) if wrap is not None else 0,
        "track_rows_mean": float(np.mean(n_rows)) if n_rows else 0.0,
        "track_rows_max": int(max(n_rows)) if n_rows else 0,
        "steps_with_track_rows": int(sum(1 for x in n_rows if x > 0)),
        "prot_ms_mean": float(t.mean()), "prot_ms_p95": float(np.percentile(t, 95)),
        "episode_time_s": ep_time,
    }
    if isinstance(pol.src, UnlabelledThreatSource):
        rec.update(pol.src.row_stats())
    if own:
        env.close()
    return rec
