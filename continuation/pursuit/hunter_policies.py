"""Hunter pursuit policies and Protag-LiDAR handling: the four hunter variants.

    variant  | pursuit target               | Protag in the hunter's LiDAR
    ---------+------------------------------+-------------------------------------------------
    B1       | direct:     p_e(t)           | remove: Protag not rendered in the hunter's scan
    B1.5     | direct:     p_e(t)           | filter: rendered; nearby candidates dropped from CBF
    P2       | predictive: p_hat_e(t + tau) | remove
    P2.5     | predictive: p_hat_e(t + tau) | filter (reference = CURRENT true p_e, never p_hat)

The two axes are independent by construction: the policy only produces the CLF reference
`gamma` (and the nominal direction derived from it by the frozen `nominal_action`); the LiDAR mode
only decides what enters the hunter's barrier rows. The CLF-DR-CBF QP is the frozen
`ClfCbfDrccpController`, unchanged, in every variant.

ENVIRONMENT FACTS THIS MODULE IS BUILT ON (inspected, not assumed)
    dynamics   holonomic single integrator p_dot = u, per-axis box |u_i| <= max_v, dt = 0.1 s;
               velocity is set instantaneously each step, so there is NO acceleration limit.
    CLF        V = 0.5 k_v ||p - gamma||^2 with the soft constraint dV.u + rate_V V <= delta.
               gamma is treated as a fixed point at each solve; there is no target-velocity term.
    LiDAR      24 rays, 5 m, ray-cast against exactly the shapes passed in (`cast_rays`).
    CBF rows   one row per buffered scan (5 scans): the scan's nearest surface point.

MOVING-TARGET CLF (plan 4.2 / 7.5). The frozen CLF has no slot for dV/dt = -(p_h - p_target).v_target
and the CLF constraint is soft (slack delta). Adding the term would change the frozen controller
mathematics, so it is NOT added: every variant re-solves with the latest target each step, which is
the established receding-reference use of this controller (the navigation protagonist tracks a
moving A* carrot the same way). Consequence, stated rather than hidden: there is no convergence
guarantee to a moving target; capture is measured, not guaranteed.

PREDICTION MODEL (plan 7.3), adapted to the dynamics above
    p_hat(t + tau) = p_e + v_e tau + 0.5 a_g d_g tau^2,   d_g = unit(g_e - p_e)
    integrated over n = ceil(tau / dt) sub-steps: the velocity is clipped per axis to Protag's own
    box |v_i| <= MAX_SPEED after every sub-step and position advances by the trapezoid of the
    sub-step's start and end velocities. With no clipping this equals the closed form exactly;
    with clipping the predicted displacement never exceeds MAX_SPEED * tau per axis. d_g is frozen at time t (it is the plan's model,
    not a replica of the A*-carrot-following Protag controller). The result is clipped into the
    arena. Any non-finite input or output falls back to the current Protag position and is counted.

FILTERING (plan 6). Applied per scan AT PUSH TIME, against Protag's true position at the moment
that scan was cast. Filtering at samples() time against the current position would leave Protag's
points from the 4 older buffered scans (up to 0.4 s, ~0.4-0.6 m old) in the barrier set as phantom
rows. The raw ranges are untouched and still feed the velocity tracker, exactly as a real sensor's
would; only the surface-point candidates that become CBF rows are filtered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.lidar_cbf import surface_points

POLICIES = ("direct", "predictive", "mpc")   # "mpc": continuation/pursuit/mpc.py (policies 3/3.5)
LIDAR_MODES = ("remove", "filter")
MAX_PREDICTION_HORIZON = 2.0     # s; hard bound, the plan forbids arbitrarily long horizons

#: Fresh, disjoint from every navigation (10_xxx_xxx), pursuit (11_xxx_xxx), Week 5.5
#: (12_xxx_xxx) and Track A/A2/B (13_xxx_xxx) block. A repo-wide grep found no 14_xxx_xxx use.
VARIANT_BLOCKS = {
    "HV_smoke": (14_000_000, 5),
    "HV_main": (14_100_000, 200),
    #: Development block for congestion-threshold calibration ONLY (decision 11). Never evaluated.
    "HV_dev": (14_300_000, 50),
    #: Fresh block for the 3.5-vs-3 replication check (diagnostic, disjoint from every other block).
    "HV_check": (14_400_000, 100),
}


def variant_seeds(name, n=None):
    base, size = VARIANT_BLOCKS[name]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"block {name} holds {size} episodes, requested {n}")
    return [base + i for i in range(n)]


@dataclass(frozen=True)
class HunterVariant:
    name: str
    policy: str = "direct"
    lidar_mode: str = "remove"
    #: Short horizon: 0.5 s = 5 control steps, about one Protag body-length of travel at 1 m/s.
    #: Protag re-plans around obstacles on that scale, so longer horizons extrapolate paths it
    #: will not take. Declared before any run; not tuned.
    prediction_horizon: float = 0.5
    #: Goal-directed acceleration [m/s^2]. 1.0 = Protag's MAX_SPEED per second: turns the predicted
    #: velocity toward the goal within ~1 s. Protag has no acceleration limit (single integrator),
    #: so the bound that matters is the per-axis speed clip applied in `predict_position`.
    goal_accel: float = 1.0
    #: Filter radius r_f = Protag radius + margin. Protag's LiDAR hits lie exactly on its 0.30 m
    #: circle (same-instant position, float32 casting error ~1e-6), so the margin only needs to
    #: absorb rounding. Any environmental point within r_f of p_e is itself within `margin` of
    #: Protag's surface, i.e. nearly touching Protag: that is the false-positive band. 0.05 m
    #: matches the capture tolerance delta.
    filter_margin: float = 0.05

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.lidar_mode not in LIDAR_MODES:
            raise ValueError(f"lidar_mode must be one of {LIDAR_MODES}")
        if not 0.0 <= self.prediction_horizon <= MAX_PREDICTION_HORIZON:
            raise ValueError(f"prediction_horizon must be in [0, {MAX_PREDICTION_HORIZON}] s")
        if not np.isfinite(self.goal_accel) or self.goal_accel < 0:
            raise ValueError("goal_accel must be finite and >= 0")
        if not np.isfinite(self.filter_margin) or self.filter_margin < 0:
            raise ValueError("filter_margin must be finite and >= 0")

    @property
    def protag_in_hunter_lidar(self):
        return self.lidar_mode == "filter"

    def filter_radius(self, protag_radius):
        return float(protag_radius) + float(self.filter_margin)

    def as_dict(self):
        return asdict(self)


VARIANTS = {
    "B1": HunterVariant("B1", policy="direct", lidar_mode="remove"),
    "B1.5": HunterVariant("B1.5", policy="direct", lidar_mode="filter"),
    "P2": HunterVariant("P2", policy="predictive", lidar_mode="remove"),
    "P2.5": HunterVariant("P2.5", policy="predictive", lidar_mode="filter"),
}

#: Policy identifiers used by the policy matrix (configs, logs, tables). 1-2.5 are the variants
#: above with identical parameters; only `name` differs. 3 / 3.5 are MPC pursuit.
POLICY_IDS = {
    "1": HunterVariant("1", policy="direct", lidar_mode="remove"),
    "1.5": HunterVariant("1.5", policy="direct", lidar_mode="filter"),
    "2": HunterVariant("2", policy="predictive", lidar_mode="remove"),
    "2.5": HunterVariant("2.5", policy="predictive", lidar_mode="filter"),
    "3": HunterVariant("3", policy="mpc", lidar_mode="remove"),
    "3.5": HunterVariant("3.5", policy="mpc", lidar_mode="filter"),
}
#: Historical names (results/pursuit_hunter_variants) -> policy id.
ALIASES = {"B1": "1", "B1.5": "1.5", "P2": "2", "P2.5": "2.5"}


def policy_by_id(pid):
    pid = ALIASES.get(str(pid), str(pid))
    if pid not in POLICY_IDS:
        raise KeyError(f"unknown policy id {pid!r}; expected one of {list(POLICY_IDS)}")
    return POLICY_IDS[pid]


# --------------------------------------------------------------------------- pursuit targets
def direct_target(p_e):
    """Direct pursuit: the CLF reference is Protag's current true position."""
    return np.asarray(p_e, dtype=float).reshape(2).copy()


def goal_direction(p_e, g_e, eps=1e-9):
    """unit(g_e - p_e); the zero vector when Protag is (numerically) at its goal."""
    d = np.asarray(g_e, dtype=float).reshape(2) - np.asarray(p_e, dtype=float).reshape(2)
    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n < eps:
        return np.zeros(2)
    return d / n


def predict_position(p_e, v_e, g_e, *, horizon, goal_accel, dt, v_max, world_size,
                     body_radius):
    """(p_hat(t + horizon), info). Pure function of the CURRENT state; see module docstring."""
    p = np.asarray(p_e, dtype=float).reshape(2)
    v = np.asarray(v_e, dtype=float).reshape(2)
    g = np.asarray(g_e, dtype=float).reshape(2)
    info = {"fallback": False, "clipped": False, "d_g": None}
    if not 0.0 <= horizon <= MAX_PREDICTION_HORIZON:
        raise ValueError("horizon outside the declared bound")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v)) and np.all(np.isfinite(g))):
        info["fallback"] = True
        return (p.copy() if np.all(np.isfinite(p)) else None), info
    d_g = goal_direction(p, g)
    info["d_g"] = d_g
    if horizon == 0.0:
        return p.copy(), info
    n = max(1, int(np.ceil(horizon / dt - 1e-9)))
    h = horizon / n
    a = goal_accel * d_g
    q, w = p.copy(), v.copy()
    for _ in range(n):
        w_free = w + a * h
        w_new = np.clip(w_free, -v_max, v_max)
        info["clipped"] |= bool(np.any(w_new != w_free))
        # Trapezoid of the sub-step's end velocities: equals w h + a h^2 / 2 (the exact constant-
        # acceleration step) when no clip binds, and never exceeds v_max h per axis when one does.
        q = q + 0.5 * (w + w_new) * h
        w = w_new
    q = np.clip(q, body_radius, world_size - body_radius)
    if not np.all(np.isfinite(q)):
        info["fallback"] = True
        return p.copy(), info
    return q, info


# --------------------------------------------------------------------------- CBF filtering
class ProtagFilteredSource(EstimatedLidarBarrierSource):
    """Hunter barrier source: raw scan to the tracker, Protag-near candidates removed from rows.

    `push(p, ranges, exclude_center=p_e)` drops the surface points with ||q - p_e|| <= r_f from
    the buffered scan. Everything else is the frozen `EstimatedLidarBarrierSource.push`, line for
    line (samples() is inherited unchanged).

    Diagnostic counters use simulator identity ONLY for logging (they never change what is
    filtered): a filtered point lying on Protag's circle is counted as a true Protag return, any
    other filtered point as an environmental false positive; an unfiltered point on Protag's circle
    is a leak.
    """

    ON_BODY_TOL = 1e-3

    def __init__(self, *, filter_radius, protag_radius=0.3, **kwargs):
        super().__init__(**kwargs)
        self.filter_radius = float(filter_radius)
        self.protag_radius = float(protag_radius)
        self._reset_counters()

    def _reset_counters(self):
        self.n_points_total = 0
        self.n_filtered_protag = 0
        self.n_filtered_env = 0
        self.n_leaked_protag = 0
        self.n_scans_with_env_filtered = 0
        self.n_scans_emptied_by_filter = 0

    def reset(self):
        super().reset()
        self._reset_counters()

    def push(self, p, ranges, *, exclude_center=None):
        self.tracker.update(p, ranges)                         # raw scan: Protag still sensed
        pts, _ = surface_points(p, ranges, n_rays=self.n_rays, lidar_range=self.lidar_range)
        self.n_points_total += len(pts)
        if exclude_center is not None and len(pts):
            c = np.asarray(exclude_center, dtype=float).reshape(2)
            if np.all(np.isfinite(c)):
                d = np.linalg.norm(pts - c[None, :], axis=1)
                drop = d <= self.filter_radius
                on_body = np.abs(d - self.protag_radius) <= self.ON_BODY_TOL
                self.n_filtered_protag += int((drop & on_body).sum())
                n_env = int((drop & ~on_body).sum())
                self.n_filtered_env += n_env
                self.n_scans_with_env_filtered += int(n_env > 0)
                self.n_leaked_protag += int((~drop & on_body).sum())
                if drop.all():
                    self.n_scans_emptied_by_filter += 1
                pts = pts[~drop]
        if len(pts) == 0:
            self.n_empty_scans += 1
            return
        # ---- verbatim from EstimatedLidarBarrierSource.push
        ids = np.empty(len(pts), dtype=int)
        conf = np.empty(len(pts), dtype=bool)
        for i, q in enumerate(pts):
            _, trusted, tid = self.tracker.velocity_at(q)
            ids[i], conf[i] = tid, trusted
        self.buffer.insert(0, pts)
        self.track_buffer.insert(0, (ids, conf))
        if len(self.buffer) > self.k_scans:
            self.buffer.pop()
            self.track_buffer.pop()

    def filter_stats(self):
        return {"points_total": self.n_points_total,
                "filtered_protag": self.n_filtered_protag,
                "filtered_env_false_positive": self.n_filtered_env,
                "leaked_protag": self.n_leaked_protag,
                "scans_with_env_filtered": self.n_scans_with_env_filtered,
                "scans_emptied_by_filter": self.n_scans_emptied_by_filter}
