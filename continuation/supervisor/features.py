"""Per-step features for the Track B infeasibility diagnosis and, later, the risk score.

TWO STRICTLY SEPARATED CHANNELS.
    RUNTIME  -- computable by a deployed supervisor from what the frozen controller already sees:
                the 24 LiDAR ranges (obs[4:28]), the controller's own barrier rows, the LiDAR
                velocity tracker's confirmed tracks, and the supervisor's own history.
    DIAGNOSTIC -- ground truth from the environment (true clearance, true obstacle geometry,
                positions). OFFLINE ANALYSIS ONLY. Never an input to a deployed supervisor.
Every feature name is prefixed rt_ or gt_ so the two can never be confused in the analysis.
"""
from __future__ import annotations

import numpy as np

from robot_env.lidar_core import ray_angles


def runtime_features(ranges, *, n_rays, lidar_range, u_nom, tracks, p, hist_infeasible,
                     h_crit=None, m_u0=None, speed=None):
    """Features a deployed supervisor may use. `tracks` are the tracker's CONFIRMED tracks."""
    r = np.asarray(ranges, float).reshape(-1)
    ang = ray_angles(n_rays)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    hit = r < lidar_range - 1e-6
    f = {}
    f["rt_min_range"] = float(r.min())
    f["rt_mean_range"] = float(r.mean())
    f["rt_n_rays_below_1m"] = int(np.sum(r < 1.0))
    f["rt_n_rays_below_2m"] = int(np.sum(r < 2.0))
    f["rt_frac_rays_hit"] = float(np.mean(hit))
    # free-space width along the nominal direction: ranges within +-30 deg of u_nom
    un = np.asarray(u_nom, float).reshape(2)
    nn = float(np.linalg.norm(un))
    if nn > 1e-9:
        cosang = dirs @ (un / nn)
        sector = cosang >= np.cos(np.deg2rad(30))
        f["rt_range_along_nominal"] = float(r[sector].min()) if sector.any() else float(r.max())
        # widest free direction and how far the nominal is from it
        best = int(np.argmax(r))
        f["rt_best_free_range"] = float(r[best])
        f["rt_nominal_vs_best_deg"] = float(np.degrees(np.arccos(
            np.clip(float(dirs[best] @ (un / nn)), -1, 1))))
    else:
        f["rt_range_along_nominal"] = float(r.min())
        f["rt_best_free_range"] = float(r.max())
        f["rt_nominal_vs_best_deg"] = 0.0
    # corridor width: sum of opposing ray pairs, smallest pair = tightest passage through ego
    half = n_rays // 2
    pair = r[:half] + r[half:]
    f["rt_min_corridor_width"] = float(pair.min())
    # confirmed dynamic tracks near the robot (runtime: from the LiDAR tracker, not ground truth)
    p = np.asarray(p, float).reshape(2)
    d_tracks = sorted(float(np.linalg.norm(np.asarray(t.position, float) - p)) for t in tracks)
    f["rt_n_tracks_within_2m"] = int(sum(1 for d in d_tracks if d <= 2.0))
    f["rt_n_tracks_within_3m"] = int(sum(1 for d in d_tracks if d <= 3.0))
    f["rt_nearest_track_dist"] = float(d_tracks[0]) if d_tracks else float("nan")
    # closing speed of the most threatening confirmed track
    worst = 0.0
    for t in tracks:
        q = np.asarray(t.position, float) - p
        dq = float(np.linalg.norm(q))
        if dq > 1e-6:
            worst = min(worst, float(np.asarray(t.velocity, float) @ (q / dq)))
    f["rt_worst_closing_speed"] = float(worst)
    f["rt_hist_infeasible_5"] = int(sum(hist_infeasible[-5:]))
    f["rt_hist_infeasible_20"] = int(sum(hist_infeasible[-20:]))
    f["rt_h_crit"] = float(h_crit) if h_crit is not None else float("nan")
    f["rt_m_u0"] = float(m_u0) if m_u0 is not None else float("nan")
    f["rt_speed"] = float(speed) if speed is not None else float("nan")
    return f


def diagnostic_features(env):
    """GROUND TRUTH. Offline analysis only -- never an input to a deployed supervisor."""
    p = np.asarray(env.agent_position, float)
    f = {"gt_x": float(p[0]), "gt_y": float(p[1])}
    walls = [p[0] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[0],
             p[1] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[1]]
    f["gt_wall_clearance"] = float(min(walls))
    rects = []
    for cx, cy, hw, hh in env.static_obstacles:
        q = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
        rects.append(float(np.linalg.norm(p - q)) - env.AGENT_RADIUS)
    f["gt_static_clearance"] = float(min(rects)) if rects else float("inf")
    if len(env.obstacle_positions):
        d = np.linalg.norm(np.asarray(env.obstacle_positions, float) - p[None, :], axis=1)
        f["gt_dyn_clearance"] = float(d.min()) - env.AGENT_RADIUS - env.OBSTACLE_RADIUS
        f["gt_n_dyn_within_2m"] = int(np.sum(d <= 2.0))
        f["gt_n_dyn_within_3m"] = int(np.sum(d <= 3.0))
        v = np.asarray(env.obstacle_velocities, float)
        rel = p[None, :] - np.asarray(env.obstacle_positions, float)
        nrm = np.linalg.norm(rel, axis=1, keepdims=True)
        close = np.sum(v * (rel / np.maximum(nrm, 1e-9)), axis=1)
        f["gt_worst_true_closing"] = float(-close.max()) if len(close) else 0.0
    else:
        f.update(gt_dyn_clearance=float("inf"), gt_n_dyn_within_2m=0, gt_n_dyn_within_3m=0,
                 gt_worst_true_closing=0.0)
    f["gt_true_clearance"] = float(min(f["gt_wall_clearance"], f["gt_static_clearance"],
                                       f["gt_dyn_clearance"]))
    return f
