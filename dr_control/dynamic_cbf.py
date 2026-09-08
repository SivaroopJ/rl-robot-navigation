"""Phase 3 -- dynamic obstacles with an ORACLE velocity channel.

WHAT THIS PHASE DELIBERATELY BREAKS, AND WHY
--------------------------------------------
Phase 2 was strictly LiDAR-only. Phase 3 breaks that in ONE named channel: obstacle
POSITIONS still come from the 24 rays, but obstacle VELOCITY is taken from simulator ground
truth. The point is to test whether the DR-CBF handles dh_dt correctly, in isolation, before
a LiDAR velocity estimator (Phase 4, an [extension]) can confound the result.

The oracle channel is explicit everywhere: `velocity_source` is a constructor argument, is
written into every result record, and no code path silently falls back to it.

ASSOCIATION IS BY GROUND-TRUTH IDENTITY, NOT NEAREST NEIGHBOUR
--------------------------------------------------------------
Binding a LiDAR return to a velocity by "which obstacle is nearest to this point" would fold
association error into the dh_dt measurement -- exactly the confound this phase exists to
avoid. Instead `ray_owners` recovers, per ray, WHICH object actually produced the reading, by
re-casting each object alone and taking the arg-min. That is exact by construction: the
combined scan's value equals the minimum over per-object casts, which
`tests/test_dr_control_phase3.py::test_ray_ownership_reproduces_the_combined_scan` asserts.

Nearest-neighbour association is still implemented (`associate_nearest`) but only as a
SECONDARY measurement, so its error rate can be quantified without ever entering the control
path.

Note that Phase 2's structural finding carries over: each buffered scan contributes only its
single nearest point, so the K samples usually describe ONE obstacle at K instants (measured:
78.9% of static steps had a single distinct constraint direction). With moving obstacles the
identity of that nearest obstacle changes over the buffer, which is why per-sample ownership
must be tracked per scan rather than assumed constant.
"""
from __future__ import annotations

import numpy as np

from dr_control.lidar_cbf import LidarBarrierSource, surface_points
from robot_env.lidar_core import cast_rays

WALL = -1          # owner id for the arena boundary (and for rays that hit nothing)


def ray_owners(p, *, rects, circles, circle_radius, n_rays=24, lidar_range=5.0,
               world_size=10.0, agent_radius=0.3):
    """Per-ray identity of the object that produced each reading.  [oracle channel]

    Returns (owners, ranges) where owners[k] is
        WALL (-1)      the arena boundary, or a ray that hit nothing
        j >= 0         static rectangle index j
        1000 + i       dynamic circle index i

    Exact by construction: the combined scan is the pointwise minimum over the per-object
    casts, so the arg-min identifies the true owner. Ties are measure-zero.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    empty = np.zeros((0, 2))
    kw = dict(n_rays=n_rays, lidar_range=lidar_range, world_size=world_size,
              agent_radius=agent_radius, circle_radius=circle_radius)

    best = cast_rays(p, rects=(), circles=empty, **kw).astype(float)   # walls only
    owners = np.full(n_rays, WALL, dtype=int)

    for j, r in enumerate(rects):
        rr = cast_rays(p, rects=[r], circles=empty, **kw).astype(float)
        m = rr < best - 1e-9
        owners[m], best[m] = j, rr[m]

    circles = np.asarray(circles, dtype=float).reshape(-1, 2)
    for i, c in enumerate(circles):
        rc = cast_rays(p, rects=(), circles=c.reshape(1, 2), **kw).astype(float)
        m = rc < best - 1e-9
        owners[m], best[m] = 1000 + i, rc[m]

    return owners, best


def associate_nearest(q, circles, circle_radius):
    """SECONDARY measurement only -- never used in the control path.

    Nearest-obstacle association of a surface point: returns the circle index whose surface
    is closest to q, or WALL if no circle surface is within a tolerance of it.
    """
    circles = np.asarray(circles, dtype=float).reshape(-1, 2)
    if len(circles) == 0:
        return WALL
    d = np.abs(np.linalg.norm(circles - np.asarray(q, float)[None, :], axis=1) - circle_radius)
    j = int(np.argmin(d))
    return 1000 + j if d[j] < 0.15 else WALL


class DynamicLidarBarrierSource(LidarBarrierSource):
    """LiDAR barrier source that also carries per-point ownership, for the dh_dt channel.

    Subclasses rather than modifies `LidarBarrierSource` so the Phase 2 gate stays
    bit-reproducible. `dh_dt_mode` selects the arm being compared:

        "oracle"        dh_dt = -grad_h . v_true(owner)      (ground-truth velocity)
        "zero"          dh_dt = 0                            (blind to obstacle motion)
        "conservative"  dh_dt = -v_assumed                   (worst case, no identity used)

    The "conservative" arm applies the worst case to EVERY sample, because a controller with
    no velocity information cannot tell which returns are moving -- that is the honest
    no-information baseline.
    """

    def __init__(self, *, dh_dt_mode="oracle", v_assumed=0.75, **kwargs):
        super().__init__(**kwargs)
        if dh_dt_mode not in ("oracle", "zero", "conservative"):
            raise ValueError(f"unknown dh_dt_mode {dh_dt_mode!r}")
        self.dh_dt_mode = dh_dt_mode
        self.v_assumed = float(v_assumed)
        self.owner_buffer = []

    def reset(self):
        super().reset()
        self.owner_buffer = []

    def push(self, p, ranges, owners=None):
        """Buffer one scan's surface points together with their per-ray owner ids."""
        pts, hit = surface_points(p, ranges, n_rays=self.n_rays, lidar_range=self.lidar_range)
        if len(pts) == 0:
            self.n_empty_scans += 1
            return
        self.buffer.insert(0, pts)
        own = np.full(len(pts), WALL, dtype=int) if owners is None \
            else np.asarray(owners, dtype=int)[hit]
        self.owner_buffer.insert(0, own)
        if len(self.buffer) > self.k_scans:
            self.buffer.pop()
            self.owner_buffer.pop()

    def samples(self, p, obstacle_velocities=None):
        """(h, grad_h, dh_dt) with dh_dt set by `dh_dt_mode`.

        h and grad_h are LiDAR-derived exactly as in Phase 2; only dh_dt differs between
        arms, so a difference between arms is attributable to the dh_dt channel alone.
        """
        p = np.asarray(p, dtype=float).reshape(2)
        vels = (np.zeros((0, 2)) if obstacle_velocities is None
                else np.asarray(obstacle_velocities, dtype=float).reshape(-1, 2))

        h, grads, dh_dt = [], [], []
        for pts, own in zip(self.buffer, self.owner_buffer):
            d = np.linalg.norm(pts - p[None, :], axis=1)
            j = int(np.argmin(d))
            dist = float(d[j])
            if dist < 1e-12:
                h.append(-self.r_robot)
                grads.append(np.array([1.0, 0.0]))
                dh_dt.append(0.0)
                continue
            g = (p - pts[j]) / dist
            h.append(dist - self.r_robot)
            grads.append(g)

            if self.dh_dt_mode == "zero":
                dh_dt.append(0.0)
            elif self.dh_dt_mode == "conservative":
                dh_dt.append(-self.v_assumed)
            else:                                  # oracle
                oid = int(own[j])
                v = vels[oid - 1000] if oid >= 1000 and len(vels) else np.zeros(2)
                dh_dt.append(-float(g @ v))

        if not h:
            raise ValueError("no buffered scans; call push() before samples()")
        return np.array(h), np.column_stack(grads), np.array(dh_dt)

    def critical_owner(self, p):
        """Owner id of the most critical sample -- for association-error measurement."""
        best, oid = np.inf, WALL
        for pts, own in zip(self.buffer, self.owner_buffer):
            d = np.linalg.norm(pts - np.asarray(p, float)[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] < best:
                best, oid = float(d[j]), int(own[j])
        return oid
