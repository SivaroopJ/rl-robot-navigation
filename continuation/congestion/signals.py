"""Supporting congestion signals. Diagnostics only: they are logged, they do not set the state.

FREE GAP (in the intended direction). Rays with range > pass_distance are "open"; a ray that hits
nothing returns exactly lidar_range and therefore counts as open. Starting from the ray nearest the
intended direction, the contiguous open sector is grown both ways; its angular width dtheta
(multiples of the 15 deg ray spacing) gives the chord width 2 * pass_distance * sin(dtheta / 2)
at pass_distance. 0 if the intended ray itself is blocked; 2 * pass_distance if every ray is open;
NaN with no intended direction. Angular resolution limits it to ~0.26 m at 1 m.

TRACK PROXIMITY / CLOSING MOTION, from the tracker's state x = [px, py, vx, vy] (plus the
protagonist's known hunter disc). rel = c - p, w = v_obs - v_agent,
closing = -(rel . w) / ||rel|| (> 0 approaching), ttc = max(||rel|| - r_clear, 0) / closing.
v_agent is the agent's last executed velocity (ctrl.prev_u).
"""
from __future__ import annotations

import numpy as np

from robot_env.lidar_core import ray_angles

TTC_CAP = 99.0


def free_gap(p, ranges, direction, *, n_rays, lidar_range, pass_distance=1.0):
    ranges = np.asarray(ranges, float).reshape(-1)
    d = np.asarray(direction, float).reshape(2)
    if np.linalg.norm(d) < 1e-9:
        return float("nan")
    ang = ray_angles(n_rays)
    target = np.arctan2(d[1], d[0]) % (2 * np.pi)
    j0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - target))))))
    open_ = ranges > pass_distance
    if not open_[j0]:
        return 0.0
    if open_.all():
        return 2.0 * pass_distance
    n = 1
    for step in (1, -1):
        j = (j0 + step) % n_rays
        while open_[j]:
            n += 1
            j = (j + step) % n_rays
    dtheta = min(n * 2 * np.pi / n_rays, np.pi)
    return float(2.0 * pass_distance * np.sin(dtheta / 2.0))


def track_signals(p, v_agent, obstacles, *, radius=2.0):
    pos, vel, rad = obstacles.movers()
    if len(pos) == 0:
        return {"n_near": 0, "max_closing": 0.0, "min_ttc": TTC_CAP}
    p = np.asarray(p, float).reshape(2)
    rel = pos - p[None, :]
    dist = np.linalg.norm(rel, axis=1)
    w = vel - np.asarray(v_agent, float).reshape(2)[None, :]
    closing = -np.sum(rel * w, axis=1) / np.maximum(dist, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > 1e-9, np.maximum(dist - rad, 0.0) / closing, TTC_CAP)
    return {"n_near": int((dist <= radius).sum()),
            "max_closing": float(max(closing.max(), 0.0)),
            "min_ttc": float(min(ttc.min(), TTC_CAP))}
