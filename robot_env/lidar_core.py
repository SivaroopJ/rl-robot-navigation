"""The vectorized ray caster, shared by the dynamic-obstacle env and the Point Maze.

WHY THIS MODULE EXISTS
----------------------
Experiment 4 needs the SAME 24-ray LiDAR on a different environment. Two implementations
of "distance to the nearest wall" would be two things to keep in agreement, and the whole
point of the experiment is that the LiDAR is a controlled, unchanged input. So the body of
`RobotNavEnv._cast_lidar_rays` moved here verbatim and both environments call it.

This is a pure extraction: no arithmetic changed, no operation reordered. Two existing
tests guard that claim from opposite directions and BOTH must still pass untouched --
`tests/test_week4_env.py::test_vectorized_lidar_matches_the_scalar_reference` (agreement
with the scalar reference) and `::test_legacy_path_is_bit_identical_to_main` (agreement
with the pre-Week-4 environment). If either breaks, the extraction is wrong.

THE FLOAT32 DISCIPLINE, WHICH IS LOad-BEARING
---------------------------------------------
`origin` is kept float32 and several intermediates are forced to float32 on purpose. The
scalar reference this was derived from computes `python_float - float32` throughout, which
under NumPy 2's weak promotion evaluates in float32; and it stores into a float32 array, so
it rounds at every write. Computing in float64 and casting once at the end is "more
accurate" and differs by ~5e-7 -- enough to diverge a trajectory and break bit-identity.
Matching the reference matters more here than matching the real numbers. Do not "clean
this up".
"""
from __future__ import annotations

import numpy as np


def ray_angles(n_rays):
    """The fixed ray bearings, evenly spaced over the full circle starting at +x."""
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def cast_rays(origin, *, n_rays, lidar_range, world_size, agent_radius,
              rects=(), circles=(), circle_radius=0.0):
    """Distance to the nearest obstacle or boundary along each of `n_rays` directions.

    Parameters
    origin: array-like (2,)
        Ray origin in world coordinates. Cast to float32 -- see the module docstring.
    n_rays: int
        Number of rays, evenly spaced from +x.
    lidar_range: float
        Maximum reported distance; rays that hit nothing return exactly this.
    world_size: float
        Side of the square arena. The boundary is at `agent_radius` and
        `world_size - agent_radius`.
    agent_radius: float
        Inset of the boundary. Pass 0.0 for a point agent (the Point Maze).
    rects: sequence of (centre_x, centre_y, half_width, half_height)
        Axis-aligned rectangles, tested by the slab method. Maze walls are thin rectangles.
    circles: array-like (N, 2)
        Circle centres, tested by the quadratic. Empty for the Point Maze.
    circle_radius: float
        Shared radius of every circle.

    Returns
    readings: np.ndarray (n_rays,) float32
        Distance along each ray, in [0, lidar_range].
    """
    origin = np.asarray(origin, dtype=np.float32)
    angles = ray_angles(n_rays)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)      # (R, 2)
    readings = np.full(n_rays, lidar_range, dtype=np.float32)

    # ---- world boundary. Smallest positive t at which the ray leaves the arena.
    wall_t = np.full(n_rays, np.inf)
    for dim in range(2):
        component = directions[:, dim]
        usable = np.abs(component) > 1e-9
        bound = np.where(component > 0,
                         world_size - agent_radius,
                         agent_radius).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (bound - origin[dim]) / component
        candidate = np.where(usable & (t > 0), t, np.inf)
        wall_t = np.minimum(wall_t, candidate)
    readings = np.minimum(
        readings, np.where(np.isinf(wall_t), lidar_range, wall_t).astype(np.float32))

    # ---- axis-aligned rectangles, by the slab method over both axes at once.
    if len(rects):
        rect_array = np.asarray(rects, dtype=np.float64)                 # (M, 4)
        centre, half = rect_array[:, :2], rect_array[:, 2:]
        low = (centre - half)[None, :, :].astype(np.float32)             # (1, M, 2)
        high = (centre + half)[None, :, :].astype(np.float32)
        d = directions[:, None, :]                                       # (R, 1, 2)
        o = origin[None, None, :]

        parallel = np.abs(d) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (low - o) / d
            t2 = (high - o) / d
        slab_enter = np.where(parallel, -np.inf, np.minimum(t1, t2))
        slab_exit = np.where(parallel, np.inf, np.maximum(t1, t2))
        # A ray parallel to a slab misses entirely unless the origin lies inside it.
        outside = parallel & ((o < low) | (o > high))
        miss = outside.any(axis=2)

        t_enter = slab_enter.max(axis=2)
        t_exit = slab_exit.min(axis=2)
        hit = (~miss) & (t_enter <= t_exit) & (t_exit > 0)
        t = np.where(t_enter > 0, t_enter, t_exit)
        t = np.clip(t, 0, lidar_range)
        readings = np.minimum(
            readings, np.where(hit, t, lidar_range).min(axis=1).astype(np.float32))

    # ---- circles, by the quadratic, for every (ray, circle) pair.
    if len(circles):
        centres = np.asarray(circles, dtype=np.float32)                  # (N, 2)
        oc = origin[None, :] - centres                                   # (N, 2) f32
        b = 2.0 * (oc[None, :, :] * directions[:, None, :]).sum(axis=2)  # (R, N) f64
        # `c` stays float32: the reference computes np.dot(f32, f32) then subtracts a
        # Python float, both of which remain float32 under weak promotion.
        c = ((oc * oc).sum(axis=1)[None, :] - np.float32(circle_radius ** 2))
        disc = b * b - 4.0 * c
        with np.errstate(invalid="ignore"):
            root = np.sqrt(np.where(disc < 0, 0.0, disc))
        t1 = (-b - root) / 2.0
        t2 = (-b + root) / 2.0
        nearest = np.where(t1 > 0, t1, t2)
        valid = (disc >= 0) & (nearest > 0)
        t = np.clip(nearest, 0, lidar_range)
        readings = np.minimum(
            readings, np.where(valid, t, lidar_range).min(axis=1).astype(np.float32))

    return readings
