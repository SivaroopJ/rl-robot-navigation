"""Analytic sources of (h, grad_h, dh_dt) -- Phase 1 only.

NO LIDAR IS INVOLVED ANYWHERE IN THIS FILE. Phase 1 hands the controller perfect barrier
information on purpose, so that a failure can only be in the controller mathematics. The
LiDAR source (Phase 2) is a separate module and is not written yet.

Barrier convention: `h` is the robot-body clearance, so contact is exactly h = 0. See
dr_control/__init__.py. The radius is folded in here, at the source, and never again.

Geometry matches RobotNavEnv._check_collision_type so that "h <= 0" and "the environment
reports a collision" are the same event:
    dynamic circle : ||p - o|| < AGENT_RADIUS + OBSTACLE_RADIUS
    static rect    : ||p - closest_point_on_rect|| < AGENT_RADIUS
    wall           : p outside [AGENT_RADIUS, WORLD_SIZE - AGENT_RADIUS]
"""
from __future__ import annotations

import numpy as np


def circle_barrier(p, centre, r_obs, r_robot, v_obs=(0.0, 0.0)):
    """Clearance to a circular obstacle, its gradient, and its explicit time derivative.

        h        = ||p - o|| - r_obs - r_robot
        grad_h   = (p - o) / ||p - o||
        dh_dt    = -grad_h . v_obs

    Returns (h, grad_h (2,), dh_dt).
    """
    p = np.asarray(p, dtype=float).reshape(2)
    centre = np.asarray(centre, dtype=float).reshape(2)
    d = p - centre
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        # Degenerate: robot exactly on the obstacle centre. Gradient undefined; pick +x and
        # let h be maximally negative so the caller sees an unambiguous violation.
        return -(r_obs + r_robot), np.array([1.0, 0.0]), 0.0
    grad = d / n
    return n - r_obs - r_robot, grad, -float(grad @ np.asarray(v_obs, dtype=float).reshape(2))


def rect_barrier(p, rect, r_robot):
    """Clearance to an axis-aligned rectangle given as (cx, cy, half_w, half_h).

    Uses the closest point on the rectangle, matching RobotNavEnv._check_collision_type. The
    gradient is exact outside the rectangle and is discontinuous at the corner diagonals --
    a property of the true distance function, not an approximation.

    Inside the rectangle the closest point degenerates; we return the outward direction along
    the least-penetrated axis, which is the standard signed-distance continuation.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    cx, cy, hw, hh = (float(v) for v in rect)
    centre = np.array([cx, cy])
    half = np.array([hw, hh])
    d = p - centre
    outside = np.abs(d) - half
    if np.any(outside > 0):
        q = np.maximum(outside, 0.0) * np.sign(d)
        n = float(np.linalg.norm(q))
        return n - r_robot, q / n, 0.0
    # Inside: negative distance to the nearest face.
    axis = int(np.argmax(outside))
    grad = np.zeros(2)
    grad[axis] = np.sign(d[axis]) if d[axis] != 0 else 1.0
    return float(outside[axis]) - r_robot, grad, 0.0


def wall_barriers(p, world_size, r_robot):
    """Clearance to each of the four walls.

    The environment's wall collision test is on the agent CENTRE leaving
    [r_robot, world_size - r_robot], so clearance to the left wall is (p_x - r_robot) and to
    the right wall is (world_size - r_robot - p_x). Yields (h, grad_h, dh_dt) per wall.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    lo, hi = r_robot, world_size - r_robot
    return [
        (p[0] - lo, np.array([1.0, 0.0]), 0.0),
        (hi - p[0], np.array([-1.0, 0.0]), 0.0),
        (p[1] - lo, np.array([0.0, 1.0]), 0.0),
        (hi - p[1], np.array([0.0, -1.0]), 0.0),
    ]


class AnalyticBarrierSource:
    """Assembles the full sample set from ground-truth geometry.

    Phase 1 only. Every quantity here is exact; no sensing, no estimation, no noise.
    """

    def __init__(self, *, world_size, r_robot, static_rects=(), include_walls=True):
        self.world_size = float(world_size)
        self.r_robot = float(r_robot)
        self.static_rects = [tuple(float(v) for v in r) for r in static_rects]
        self.include_walls = include_walls

    def samples(self, p, circles=(), circle_radius=0.0, circle_velocities=None):
        """Return (h, grad_h (2,N), dh_dt) over walls, rectangles and circles.

        circles: (M, 2) centres; circle_velocities: (M, 2) or None for static.
        """
        h_list, g_list, dt_list = [], [], []

        if self.include_walls:
            for h, g, dd in wall_barriers(p, self.world_size, self.r_robot):
                h_list.append(h); g_list.append(g); dt_list.append(dd)

        for rect in self.static_rects:
            h, g, dd = rect_barrier(p, rect, self.r_robot)
            h_list.append(h); g_list.append(g); dt_list.append(dd)

        circles = np.asarray(circles, dtype=float).reshape(-1, 2)
        if len(circles):
            vels = (np.zeros_like(circles) if circle_velocities is None
                    else np.asarray(circle_velocities, dtype=float).reshape(-1, 2))
            for centre, v in zip(circles, vels):
                h, g, dd = circle_barrier(p, centre, circle_radius, self.r_robot, v)
                h_list.append(h); g_list.append(g); dt_list.append(dd)

        if not h_list:
            raise ValueError("no barrier samples: need walls, rectangles or circles")
        return np.array(h_list), np.column_stack(g_list), np.array(dt_list)

    def true_clearance(self, p, circles=(), circle_radius=0.0):
        """Minimum true geometric clearance -- the ground-truth safety metric."""
        h, _, _ = self.samples(p, circles, circle_radius)
        return float(np.min(h))
