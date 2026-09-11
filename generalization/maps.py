"""Week 6 / map generation. Pure functions: family -> (config dict, metadata).

A "map" here is a config dict of exactly the shape robot_env.load_config returns, so the
canonical RobotNavEnv consumes it unchanged. NOTHING in robot_env/ is modified; the entire
static-geometry axis of the suite rides on the config file, which the smoke test in
MAP_SUITE_SPEC.md 1.1 verified end to end.

FIXED ACROSS EVERY MAP, per the approved spec:
    lidar_range 5.0     never scaled with world size (decision 2) -- the changing
                        sensor-horizon-to-world ratio IS part of the intended shift
    max_speed   1.0     agent
    dt          0.1
    n_lidar_rays 24
    agent_radius 0.3

SCALED WITH WORLD SIZE, identically for both arms (spec 3, F3):
    max_steps        ceil(500 * world_size / 10)
    distance_bins    canonical bins * world_size / 10   -- forced: a 7x7 arena raises
                     ValueError on the canonical [7.0, 8.5] and [8.5, 10.5] bins
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

CANONICAL_CONFIG = "config.json"
#: Cell size of the F1' staircase raster, metres. Declared here, not tuned.
STAIRCASE_CELL = 0.1


def canonical_config(path=CANONICAL_CONFIG):
    """The Phase-6 configuration, untouched. This is M0."""
    return json.loads(Path(path).read_text())


def _scaled(cfg, world_size):
    """Apply the world-size-dependent rules. Called by every generator that resizes."""
    c = copy.deepcopy(cfg)
    s = float(world_size) / 10.0
    c["environment"]["world_size"] = float(world_size)
    c["environment"]["max_steps"] = int(math.ceil(500 * s))
    c["environment"]["static_obstacles"] = [
        [x * s, y * s, hw * s, hh * s] for x, y, hw, hh in c["environment"]["static_obstacles"]]
    sg = c.setdefault("start_goal", {})
    if sg.get("distance_bins"):
        sg["distance_bins"] = [[lo * s, hi * s] for lo, hi in sg["distance_bins"]]
    sg["min_separation"] = float(sg.get("min_separation", 4.0)) * s
    return c


# --------------------------------------------------------------------------- F1'
def rotate_rect_staircase(cx, cy, hw, hh, degrees, cell=STAIRCASE_CELL, world_size=10.0):
    """Approximate a ROTATED rectangle by a union of axis-aligned cells.

    INTERFACE / GENERALIZATION TEST, NOT LITERAL ROTATED-RECTANGLE SUPPORT (decision 1).
    The environment still knows only axis-aligned rectangles: this is a construction layered
    on top of that primitive, not a new one. `robot_env/lidar_core.cast_rays`,
    `RobotNavEnv._check_collision_type` and `ShortestPathOracle._blocked` all remain the
    axis-aligned code they were, which is what keeps the frozen manifest green.

    A cell is occupied when its CENTRE lies inside the true rotated rectangle. Occupied cells
    are then run-length merged along x, so a 90-degree rotation collapses back to a single
    rectangle rather than hundreds of cells.

    Returns (rects, meta) with meta carrying the measured fidelity: area ratio, Hausdorff
    distance to the true rotated rectangle, and cell count. These are reported beside every
    F1' result and are never used to select or tune anything.
    """
    th = math.radians(float(degrees))
    ct, st = math.cos(th), math.sin(th)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    R = np.array([[ct, -st], [st, ct]])
    poly = corners @ R.T + np.array([cx, cy])

    lo = np.maximum(poly.min(axis=0) - cell, 0.0)
    hi = np.minimum(poly.max(axis=0) + cell, float(world_size))
    nx = max(1, int(math.ceil((hi[0] - lo[0]) / cell)))
    ny = max(1, int(math.ceil((hi[1] - lo[1]) / cell)))

    def inside(px, py):
        dx, dy = px - cx, py - cy
        u = dx * ct + dy * st          # into the rectangle's own frame
        v = -dx * st + dy * ct
        return (abs(u) <= hw) and (abs(v) <= hh)

    occ = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        py = lo[1] + (iy + 0.5) * cell
        for ix in range(nx):
            px = lo[0] + (ix + 0.5) * cell
            occ[iy, ix] = inside(px, py)

    rects = []
    for iy in range(ny):
        ix = 0
        while ix < nx:
            if not occ[iy, ix]:
                ix += 1
                continue
            j = ix
            while j + 1 < nx and occ[iy, j + 1]:
                j += 1
            x0 = lo[0] + ix * cell
            x1 = lo[0] + (j + 1) * cell
            y0 = lo[1] + iy * cell
            y1 = lo[1] + (iy + 1) * cell
            rects.append([0.5 * (x0 + x1), 0.5 * (y0 + y1),
                          0.5 * (x1 - x0), 0.5 * (y1 - y0)])
            ix = j + 1

    true_area = 4.0 * hw * hh
    approx_area = float(occ.sum()) * cell * cell
    meta = {
        "degrees": float(degrees), "cell": float(cell),
        "n_cells_occupied": int(occ.sum()), "n_rects_after_merge": len(rects),
        "true_area": true_area, "approx_area": approx_area,
        "area_ratio": (approx_area / true_area) if true_area > 0 else float("nan"),
        "hausdorff": _hausdorff(poly, rects, cell),
    }
    return rects, meta


def _hausdorff(poly, rects, cell, n=4000, seed=0):
    """Symmetric Hausdorff distance between the true rotated rect and the staircase union.

    Measured by dense sampling rather than asserted from the cell size, so the number in the
    manifest is an observation. Bounded above by cell*sqrt(2) for a centre-inclusion raster.
    """
    if not rects:
        return float("inf")
    rng = np.random.default_rng(seed)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    pts = rng.uniform(lo - cell, hi + cell, size=(n, 2))

    c = poly.mean(axis=0)
    e0 = poly[1] - poly[0]
    e1 = poly[3] - poly[0]
    u = e0 / np.linalg.norm(e0)
    v = e1 / np.linalg.norm(e1)
    hw = np.linalg.norm(e0) / 2.0
    hh = np.linalg.norm(e1) / 2.0
    d = pts - c
    in_true = (np.abs(d @ u) <= hw) & (np.abs(d @ v) <= hh)

    ra = np.asarray(rects, dtype=float)
    in_appr = np.zeros(len(pts), dtype=bool)
    for rx, ry, rhw, rhh in ra:
        in_appr |= (np.abs(pts[:, 0] - rx) <= rhw) & (np.abs(pts[:, 1] - ry) <= rhh)

    def _max_gap(a_mask, b_mask):
        """Largest distance from a point in A to the nearest sampled point of B."""
        A, B = pts[a_mask], pts[b_mask]
        if len(A) == 0:
            return 0.0
        if len(B) == 0:
            return float("inf")
        out = 0.0
        for k in range(0, len(A), 512):
            chunk = A[k:k + 512]
            dd = np.linalg.norm(chunk[:, None, :] - B[None, :, :], axis=2).min(axis=1)
            out = max(out, float(dd.max()))
        return out

    return max(_max_gap(in_true & ~in_appr, in_appr), _max_gap(in_appr & ~in_true, in_true))


def f1_rotated(base, which, degrees):
    """`which` is a list of static-obstacle indices to rotate; the rest stay axis-aligned."""
    c = copy.deepcopy(base)
    rects, metas = [], []
    for i, (x, y, hw, hh) in enumerate(c["environment"]["static_obstacles"]):
        if i in which:
            rr, m = rotate_rect_staircase(x, y, hw, hh, degrees,
                                          world_size=c["environment"]["world_size"])
            m["obstacle_index"] = i
            rects.extend(rr)
            metas.append(m)
        else:
            rects.append([x, y, hw, hh])
    c["environment"]["static_obstacles"] = rects
    area_num = sum(m["approx_area"] for m in metas)
    area_den = sum(m["true_area"] for m in metas)
    return c, {"rotated_indices": sorted(which), "degrees": float(degrees),
               "per_obstacle": metas,
               "area_ratio_overall": (area_num / area_den) if area_den else float("nan"),
               "hausdorff_max": max((m["hausdorff"] for m in metas), default=0.0),
               "n_rects_total": len(rects),
               "cells_total": sum(m["n_cells_occupied"] for m in metas)}


# --------------------------------------------------------------------------- F2-F7
def f2a_static_scale(base, factor):
    c = copy.deepcopy(base)
    c["environment"]["static_obstacles"] = [
        [x, y, hw * factor, hh * factor]
        for x, y, hw, hh in c["environment"]["static_obstacles"]]
    return c, {"static_scale": float(factor)}


#: Two extra rectangles for D+, placed in canonical free space. Declared, not sampled.
EXTRA_RECTS = [[5.0, 4.5, 0.6, 0.4], [8.5, 5.5, 0.4, 0.8], [1.5, 5.0, 0.4, 0.6],
               [5.5, 9.0, 0.7, 0.4]]


def f2b_static_count(base, n_rects):
    """Canonical five, truncated for D- or extended from the declared EXTRA_RECTS for D+."""
    c = copy.deepcopy(base)
    canon = [list(r) for r in c["environment"]["static_obstacles"]]
    if n_rects <= len(canon):
        rects = canon[:n_rects]
    else:
        rects = canon + [list(r) for r in EXTRA_RECTS[:n_rects - len(canon)]]
    c["environment"]["static_obstacles"] = rects
    return c, {"n_static": len(rects)}


def f3_world(base, world_size):
    c = _scaled(base, world_size)
    return c, {"world_size": float(world_size),
               "max_steps": c["environment"]["max_steps"],
               "lidar_range": c["environment"]["lidar_range"],
               "lidar_to_world_ratio": c["environment"]["lidar_range"] / float(world_size)}


def f4_obstacle_radius(base, radius):
    c = copy.deepcopy(base)
    c["environment"]["obstacle_radius"] = float(radius)
    span = 2.0 * float(radius)
    return c, {"obstacle_radius": float(radius), "obstacle_span": span,
               "tracker_compactness_limit": 0.75,
               "tracker_rejects_as_static": bool(span > 0.75)}


def f4_meta_only(radius):
    return f4_obstacle_radius(canonical_config(), radius)[1]


def f5_obstacle_count(base, n):
    c = copy.deepcopy(base)
    c["environment"]["n_dynamic_obstacles"] = int(n)
    return c, {"n_dynamic_obstacles": int(n), "obs_slots": 6,
               "truncated_in_obs_block": bool(n > 6)}


def f6_speed(base, speed):
    c = copy.deepcopy(base)
    c["environment"]["obstacle_speed"] = float(speed)
    return c, {"obstacle_speed": float(speed), "v_cap": 0.96,
               "above_v_cap": bool(speed > 0.96),
               "above_agent_max_speed": bool(speed > c["environment"]["max_speed"])}


def f7_combo(base, *, radius=None, n=None, speed=None):
    c = copy.deepcopy(base)
    meta = {}
    if radius is not None:
        c, m = f4_obstacle_radius(c, radius); meta.update(m)
    if n is not None:
        c, m = f5_obstacle_count(c, n); meta.update(m)
    if speed is not None:
        c, m = f6_speed(c, speed); meta.update(m)
    return c, meta


# --------------------------------------------------------------------------- F9
def f9_corridor(base, width, *, world_size=10.0):
    """A single vertical barrier at x = world/2 with a gap of exactly `width` metres.

    Free clearance per side is (width - 2*agent_radius)/2; at 0.75 m that is 0.075 m, below
    the tau-margin the DR constraint asks for, which is why infeasibility is expected to become
    the normal state here rather than the exception.
    """
    c = copy.deepcopy(base)
    w = float(width)
    mid = world_size / 2.0
    gap_lo, gap_hi = mid - w / 2.0, mid + w / 2.0
    thick = 0.2
    lower_h = gap_lo / 2.0
    upper_h = (world_size - gap_hi) / 2.0
    c["environment"]["static_obstacles"] = [
        [mid, lower_h, thick, lower_h],
        [mid, gap_hi + upper_h, thick, upper_h],
    ]
    r = c["environment"]["agent_radius"]
    return c, {"corridor_width": w, "gap_lo": gap_lo, "gap_hi": gap_hi,
               "clearance_per_side": (w - 2 * r) / 2.0,
               "robot_diameter": 2 * r}
