"""Phase 4 barrier source: LiDAR positions AND LiDAR-estimated velocity. No ground truth.

Mirrors `dr_control/dynamic_cbf.DynamicLidarBarrierSource` in structure so the two are
directly comparable, but replaces the oracle velocity channel with `LidarVelocityTracker`.
This is the ONLY control-path difference between the Phase 3 `oracle` arm and the Phase 4
`estimated` arm: h and grad_h are computed by identical code.

Deliberately does NOT import dr_control.dynamic_cbf or dr_control.cbf_sources -- those reach
ground-truth geometry, and `tests/test_dr_control_phase4.py` parses this module's imports to
enforce that.

TRACK BINDING ACROSS THE SCAN BUFFER
------------------------------------
The barrier buffer holds K past scans and measures from the CURRENT robot position to each
buffered point cloud (the reference's scheme, kept from Phase 2). A buffered point is old, so
matching it against current track surfaces would fail as the obstacle moves. Instead each
surface point is bound to a track id AT PUSH TIME, and the track's CURRENT filtered velocity is
read at samples() time -- the exact analogue of Phase 3, which stored the ground-truth owner at
push time and read that obstacle's velocity at samples() time. If the track has since died or
is unconfirmed, dh_dt falls back to 0.
"""
from __future__ import annotations

import numpy as np

from dr_control.lidar_cbf import LidarBarrierSource, surface_points
from dr_control.velocity_tracker import LidarVelocityTracker


class EstimatedLidarBarrierSource(LidarBarrierSource):
    """h, grad_h from LiDAR (as Phase 2); dh_dt from a LiDAR-only velocity tracker."""

    def __init__(self, *, tracker=None, unconfirmed_conservative=False, v_assumed=0.75,
                 **kwargs):
        r_robot = kwargs.get("r_robot", 0.3)
        super().__init__(**kwargs)
        self.tracker = tracker if tracker is not None else LidarVelocityTracker(r_nominal=r_robot)
        self.unconfirmed_conservative = bool(unconfirmed_conservative)
        self.v_assumed = float(v_assumed)
        self.track_buffer = []
        # metrics-only counters
        self.n_points_no_track = 0
        self.n_points_unconfirmed = 0

    def reset(self):
        super().reset()
        self.tracker.reset()
        self.track_buffer = []
        self.n_points_no_track = 0
        self.n_points_unconfirmed = 0

    def push(self, p, ranges):
        """Advance the tracker, then buffer the scan with a track id per surface point.

        Signature takes ONLY (p, ranges) -- there is no obstacle argument to misuse.
        """
        self.tracker.update(p, ranges)
        pts, hit = surface_points(p, ranges, n_rays=self.n_rays, lidar_range=self.lidar_range)
        if len(pts) == 0:
            self.n_empty_scans += 1
            return
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

    def samples(self, p):
        """(h, grad_h, dh_dt); h and grad_h identical to Phase 2, dh_dt from the tracker."""
        p = np.asarray(p, dtype=float).reshape(2)
        h, grads, dh_dt = [], [], []
        for pts, (ids, conf) in zip(self.buffer, self.track_buffer):
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

            tid = int(ids[j])
            track = self.tracker.track_by_id(tid) if tid >= 0 else None
            if track is None:
                self.n_points_no_track += 1
                dh_dt.append(-self.v_assumed if self.unconfirmed_conservative else 0.0)
            elif not (track.confirmed or not self.tracker.use_confirmation):
                self.n_points_unconfirmed += 1
                dh_dt.append(-self.v_assumed if self.unconfirmed_conservative else 0.0)
            else:
                dh_dt.append(-float(g @ track.velocity))

        if not h:
            raise ValueError("no buffered scans; call push() before samples()")
        return np.array(h), np.column_stack(grads), np.array(dh_dt)

    def critical_track(self, p):
        """(track_id, track) owning the most critical sample -- for the track-quality record."""
        best, tid = np.inf, -1
        for pts, (ids, _) in zip(self.buffer, self.track_buffer):
            d = np.linalg.norm(pts - np.asarray(p, float)[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] < best:
                best, tid = float(d[j]), int(ids[j])
        return tid, (self.tracker.track_by_id(tid) if tid >= 0 else None)
