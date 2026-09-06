"""Phase 4 -- LiDAR-only obstacle velocity estimation. NO GROUND TRUTH REACHES THIS MODULE.

The public entry point is `LidarVelocityTracker.update(p_ego, ranges)`. There is deliberately
no parameter through which obstacle positions, velocities, identities, or per-object radii
could enter: the absence of ground truth is a signature-level guarantee, not a convention.
`r_nominal` is a fixed configured scalar (a global design constant of the same kind as the
controller's r_robot = 0.3); it is never inferred or selected from obstacle identity, obstacle
type, or simulator state.

WHY THE DESIGN LOOKS LIKE THIS -- MEASURED SENSOR FACTS
-------------------------------------------------------
Returns per obstacle, measured against robot_env.lidar_core.cast_rays:

    range   0.6   1.0   1.5   2.0   2.3   3.0  m
    points  4.07  2.20  1.40  1.00  1.00  0.60

So where the barrier actually binds (h < 1 m) an obstacle is 1-4 points. Free-radius circle
fitting is ill-posed at that size, hence the FIXED-radius reconstruction below. And a 0.02 m
centre error differenced over dt = 0.1 would give ~0.28 m/s of velocity noise -- four times the
true obstacle speed -- so velocity is read from a Kalman filter, never from frame differencing.

STATIC REJECTION: COMPACTNESS LEADS, THE LINE TEST ONLY SUPPLEMENTS
-------------------------------------------------------------------
A disc of radius r seen over one 15-degree ray step deviates from its chord by only
r*(1 - cos 7.5deg) = 0.0026 m, and over two steps by r*(1 - cos 15deg) = 0.0102 m, whereas a
flat wall's residual is ~0. A line/TLS test alone therefore has a very thin margin and would
freeze a real disc at v = 0 in the 0.8-1.0 m band if `line_tol` were set loosely.

Compactness is the well-conditioned discriminator instead: a disc spans at most 2*r_nominal
= 0.6 m, a wall or rectangle face spans metres. The line test is kept only as a supplement for
segments that pass compactness but are still clearly flat. Both false-classification rates are
measured, not assumed -- see experiments/exp4_dynamic_estimated.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from robot_env.lidar_core import ray_angles

# --------------------------------------------------------------------------------------
# FIXED CONFIGURATION CONSTANTS.
# Set from geometry and sensor reasoning (above), NOT tuned on the evaluation set. The
# evaluation seed block EVAL_SEED_BASE = 1_000_000 is never used by Phase 4 at all, and the
# r_nominal sensitivity arms are reported separately rather than used to select a value.
# --------------------------------------------------------------------------------------
SEG_C = 1.0             # adaptive segmentation gain on min(r_k, r_k+1) * tan(delta)
SEG_EPS = 0.05          # additive segmentation slack, metres
COMPACT_MARGIN = 0.15   # a dynamic candidate spans at most 2*r_nominal + this
LINE_TOL = 0.004        # TLS residual below which a >=3-point segment is called flat.
                        # 2.5x below the 0.0102 m sagitta a real disc shows over two ray
                        # steps, and far above float32 scan noise (~1e-5).
LINE_MIN_PTS = 3
SIGMA_A = 0.5           # white-acceleration process noise, m/s^2
SIGMA_R_BASE = 0.02     # centre measurement noise at close range, metres
SIGMA_R_RANGE_REF = 3.0 # measurement noise grows as (1 + d / this)
POINT_COUNT_FACTOR = {1: 3.0, 2: 1.5}   # >=3 points -> 1.0
INIT_VEL_STD = 1.0      # initial velocity uncertainty, m/s (obstacles run 0.6-0.75)
GATE_CHI2 = 9.21        # chi-square, 2 dof, 0.99
SURFACE_MATCH_TOL = 0.15  # |‖q - c‖ - r_nominal| below which a point belongs to a track


def segment_scan(ranges, *, n_rays, lidar_range, seg_c=SEG_C, seg_eps=SEG_EPS):
    """Split a scan into connected segments of ray indices.

    Adjacent rays join when the range jump is below an adaptive threshold; range-max returns
    are misses and break segments. Wraps around from the last ray to the first, because the
    sensor is a full 360-degree scan and an obstacle straddling bearing 0 is otherwise cut in
    two.
    """
    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    hit = ranges < lidar_range - 1e-6
    delta = 2 * np.pi / n_rays
    idx = [k for k in range(n_rays) if hit[k]]
    if not idx:
        return []

    def joins(a, b):
        return abs(ranges[a] - ranges[b]) < seg_c * min(ranges[a], ranges[b]) * np.tan(delta) + seg_eps

    segs, cur = [], [idx[0]]
    for prev, k in zip(idx, idx[1:]):
        if k == prev + 1 and joins(prev, k):
            cur.append(k)
        else:
            segs.append(cur)
            cur = [k]
    segs.append(cur)

    # Wrap-around merge: ray n-1 adjacent to ray 0.
    if len(segs) > 1 and segs[0][0] == 0 and segs[-1][-1] == n_rays - 1 \
            and joins(n_rays - 1, 0):
        segs[0] = segs[-1] + segs[0]
        segs.pop()
    return segs


def reconstruct_centre(points, dirs, p_ego, r_nominal):
    """Estimate a disc centre from its visible surface points, with the radius FIXED.

    Returns (centre, n_points_used, degenerate_flag).

    ONE POINT
        c = q + r * u_ray. Exact only for a head-on hit; error grows as
        r * sin(theta_incidence). Flagged so its bias is reported separately.

    TWO POINTS -- THE CENTRE IS AMBIGUOUS, AND THE RULE IS PHYSICAL, NOT ARBITRARY
        Two points on a circle of known radius admit TWO centres, mirrored across the chord:
        c = m +- b * n_hat, with m the chord midpoint, a = |q2 - q1| / 2, b = sqrt(r^2 - a^2),
        and n_hat a unit normal to the chord. We always take the centre FARTHER FROM THE
        SENSOR. That is not a tie-break: a LiDAR ray terminates on the NEAR surface of an
        opaque body, so every observed point lies on the hemisphere facing the sensor and the
        centre must lie beyond it. Choosing the nearer root would place the body between the
        sensor and its own surface, which is unphysical. `test_two_point_centre_is_the_far
        _root` pins this.

        Degenerate a >= r: the two points are farther apart than the disc's diameter, so they
        cannot lie on one disc of this radius. b is clamped to 0 (centre = midpoint) and the
        result is flagged; the compactness test normally rejects such segments first.

    THREE OR MORE POINTS
        Fixed-radius least squares, minimising sum_i (‖q_i - c‖ - r)^2, initialised from the
        centroid pushed r along the sensor-to-centroid direction and refined by Gauss-Newton.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 2)
    p_ego = np.asarray(p_ego, dtype=float).reshape(2)
    n = len(pts)

    if n == 1:
        return pts[0] + r_nominal * dirs[0], 1, False

    if n == 2:
        chord = pts[1] - pts[0]
        L = float(np.linalg.norm(chord))
        if L < 1e-9:
            return pts[0] + r_nominal * dirs[0], 1, True
        m = 0.5 * (pts[0] + pts[1])
        a = 0.5 * L
        degenerate = a >= r_nominal
        b = 0.0 if degenerate else float(np.sqrt(max(r_nominal ** 2 - a ** 2, 0.0)))
        n_hat = np.array([-chord[1], chord[0]]) / L
        c_plus, c_minus = m + b * n_hat, m - b * n_hat
        # Physical rule: the centre lies BEYOND the visible surface, i.e. farther from the
        # sensor. Deterministic; ties (b = 0) collapse to the midpoint anyway.
        c = c_plus if np.linalg.norm(c_plus - p_ego) >= np.linalg.norm(c_minus - p_ego) else c_minus
        return c, 2, degenerate

    centroid = pts.mean(axis=0)
    to_c = centroid - p_ego
    nrm = float(np.linalg.norm(to_c))
    c = centroid + r_nominal * (to_c / nrm) if nrm > 1e-9 else centroid.copy()
    for _ in range(5):                      # Gauss-Newton on the fixed-radius residual
        d = c[None, :] - pts
        rho = np.linalg.norm(d, axis=1)
        ok = rho > 1e-9
        if not np.any(ok):
            break
        J = d[ok] / rho[ok, None]
        res = rho[ok] - r_nominal
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
        if float(np.linalg.norm(step)) < 1e-10:
            break
    return c, n, False


def segment_is_compact(points, r_nominal, margin=COMPACT_MARGIN):
    """PRIMARY static/dynamic test: a disc spans at most 2*r_nominal; a wall spans metres."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return True, 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    extent = float(d.max())
    return extent <= 2.0 * r_nominal + margin, extent


def line_residual(points):
    """Total-least-squares residual of a straight-line fit (0 for a perfectly flat surface)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    q = pts - pts.mean(axis=0)
    s = np.linalg.svd(q, compute_uv=False)
    return float(s[-1] / np.sqrt(len(pts)))


class Track:
    """One constant-velocity Kalman track. State x = [px, py, vx, vy]."""

    _next_id = 0

    def __init__(self, centre, dt, *, n_points):
        self.id = Track._next_id
        Track._next_id += 1
        self.dt = float(dt)
        self.x = np.array([centre[0], centre[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([SIGMA_R_BASE ** 2, SIGMA_R_BASE ** 2,
                          INIT_VEL_STD ** 2, INIT_VEL_STD ** 2])
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.recent = [True]
        self.confirmed = False
        self.n_points_last = int(n_points)
        self.first_seen_age = 0
        self.frames_to_confirm = None

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:].copy()

    def predict(self):
        dt = self.dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = SIGMA_A ** 2
        Q = q * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def measurement_noise(self, n_points, range_to_sensor):
        """R grows with range and shrinks with point count -- a 1-point reconstruction is
        much less certain than a 4-point fit, and the filter must be told so."""
        f = POINT_COUNT_FACTOR.get(int(n_points), 1.0)
        s = SIGMA_R_BASE * f * (1.0 + range_to_sensor / SIGMA_R_RANGE_REF)
        return np.diag([s ** 2, s ** 2])

    def innovation(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        return np.asarray(z, dtype=float) - H @ self.x

    def gate_distance(self, z, R):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        S = H @ self.P @ H.T + R
        y = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z, R, *, n_points, k_confirm, n_window):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ self.innovation(z)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0
        self.n_points_last = int(n_points)
        self._mark(True, k_confirm, n_window)

    def mark_missed(self, k_confirm, n_window):
        self.misses += 1
        self._mark(False, k_confirm, n_window)

    def _mark(self, hit, k_confirm, n_window):
        self.recent.append(bool(hit))
        if len(self.recent) > n_window:
            self.recent.pop(0)
        if not self.confirmed and sum(self.recent) >= k_confirm:
            self.confirmed = True
            self.frames_to_confirm = self.age


class LidarVelocityTracker:
    """Segment -> classify -> reconstruct centre -> associate -> Kalman filter.

    update() takes ONLY the ego pose and the range vector.
    """

    def __init__(self, *, r_nominal=0.3, dt=0.1, n_rays=24, lidar_range=5.0,
                 k_confirm=3, n_confirm_window=5, n_coast=5,
                 use_static_rejection=True, use_confirmation=True,
                 association="hungarian", radius_free=False):
        self.r_nominal = float(r_nominal)
        self.dt = float(dt)
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.k_confirm = int(k_confirm)
        self.n_confirm_window = int(n_confirm_window)
        self.n_coast = int(n_coast)
        self.use_static_rejection = bool(use_static_rejection)
        self.use_confirmation = bool(use_confirmation)
        if association not in ("hungarian", "greedy"):
            raise ValueError(f"unknown association {association!r}")
        self.association = association
        self.radius_free = bool(radius_free)

        self.tracks = []
        self.dirs = np.stack([np.cos(ray_angles(self.n_rays)),
                              np.sin(ray_angles(self.n_rays))], axis=1)
        # Diagnostics, metrics-only; never fed back into estimation.
        self.n_static_segments = 0
        self.n_dynamic_segments = 0
        self.last_detections = []

    def reset(self):
        self.tracks = []
        self.n_static_segments = 0
        self.n_dynamic_segments = 0
        self.last_detections = []

    # ------------------------------------------------------------------ detection
    def detect(self, p_ego, ranges):
        """Scan -> list of dynamic-candidate detections. Pure function of (p_ego, ranges)."""
        p_ego = np.asarray(p_ego, dtype=float).reshape(2)
        ranges = np.asarray(ranges, dtype=float).reshape(-1)
        dets = []
        for seg in segment_scan(ranges, n_rays=self.n_rays, lidar_range=self.lidar_range):
            pts = p_ego[None, :] + ranges[seg, None] * self.dirs[seg]
            compact, extent = segment_is_compact(pts, self.r_nominal)
            flat = (len(seg) >= LINE_MIN_PTS and line_residual(pts) < LINE_TOL)

            is_static = (not compact) or flat if self.use_static_rejection else False
            if is_static:
                self.n_static_segments += 1
                continue
            self.n_dynamic_segments += 1

            if self.radius_free:
                j = int(np.argmin(ranges[seg]))
                centre, npts, degen = pts[j].copy(), len(seg), False
            else:
                centre, npts, degen = reconstruct_centre(
                    pts, self.dirs[seg], p_ego, self.r_nominal)
            dets.append({"centre": centre, "n_points": len(seg), "extent": extent,
                         "degenerate": degen, "rays": list(seg),
                         "range": float(np.min(ranges[seg]))})
        self.last_detections = dets
        return dets

    # ------------------------------------------------------------------ tracking
    def update(self, p_ego, ranges):
        """Advance all tracks by one frame. ONLY (p_ego, ranges) enter here."""
        p_ego = np.asarray(p_ego, dtype=float).reshape(2)
        dets = self.detect(p_ego, ranges)

        for t in self.tracks:
            t.predict()

        assigned = {}
        if self.tracks and dets:
            n_t, n_d = len(self.tracks), len(dets)
            cost = np.full((n_t, n_d), 1e6)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    R = t.measurement_noise(d["n_points"], d["range"])
                    g = t.gate_distance(d["centre"], R)
                    if g <= GATE_CHI2:
                        cost[i, j] = g
            if self.association == "hungarian":
                rows, cols = linear_sum_assignment(cost)
                pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
            else:                                    # greedy nearest, for the ablation
                pairs, used_t, used_d = [], set(), set()
                for i, j in sorted(((i, j) for i in range(n_t) for j in range(n_d)),
                                   key=lambda ij: cost[ij[0], ij[1]]):
                    if cost[i, j] >= 1e6 or i in used_t or j in used_d:
                        continue
                    pairs.append((i, j))
                    used_t.add(i)
                    used_d.add(j)
            assigned = dict(pairs)

        matched_dets = set(assigned.values())
        for i, t in enumerate(self.tracks):
            if i in assigned:
                d = dets[assigned[i]]
                t.update(d["centre"], t.measurement_noise(d["n_points"], d["range"]),
                         n_points=d["n_points"], k_confirm=self.k_confirm,
                         n_window=self.n_confirm_window)
            else:
                t.mark_missed(self.k_confirm, self.n_confirm_window)

        for j, d in enumerate(dets):
            if j not in matched_dets:
                self.tracks.append(Track(d["centre"], self.dt, n_points=d["n_points"]))

        self.tracks = [t for t in self.tracks if t.misses <= self.n_coast]
        return self.tracks

    # ------------------------------------------------------------------ query
    def velocity_at(self, q_world):
        """Filtered velocity of the track owning world point q, and whether it is trusted.

        Returns (v, trusted, track_id). A point matches a track when it lies on that track's
        predicted surface. Unconfirmed tracks return v = 0 unless confirmation is disabled.
        """
        q = np.asarray(q_world, dtype=float).reshape(2)
        best, best_t = SURFACE_MATCH_TOL, None
        for t in self.tracks:
            resid = abs(float(np.linalg.norm(q - t.position)) - self.r_nominal)
            if self.radius_free:
                resid = float(np.linalg.norm(q - t.position))
            if resid < best:
                best, best_t = resid, t
        if best_t is None:
            return np.zeros(2), False, -1
        trusted = best_t.confirmed or not self.use_confirmation
        return (best_t.velocity if trusted else np.zeros(2)), trusted, best_t.id

    def track_by_id(self, tid):
        for t in self.tracks:
            if t.id == tid:
                return t
        return None
