"""Sensor-derived obstacle model shared by MPC pursuit and congestion detection.

INFORMATION LEDGER. Built ONLY from an agent's own barrier source: the buffered LiDAR surface
points (with the track id each point was bound to at push time) and its LidarVelocityTracker.
The single privileged input is the protagonist's known-state hunter row (HunterAugmentedSource),
which the protagonist's QP already uses; it enters here as a moving disc. No env state, no map.

CONVENTIONS, identical to the CBF rows
    static point      clearance = ||p - q|| - r_robot                          (0.3 m)
    confirmed track   clearance = ||p - c(t)|| - r_robot - r_obstacle           (0.6 m)
    known disc        clearance = ||p - c(t)|| - r_robot - r_other              (0.6 m)
with constant-velocity extrapolation c(t) = c + v t. A buffered point bound to a CONFIRMED track is
represented by that track, not also as a static point; points bound to no track or to an
unconfirmed track are static (the same confirmation test the source uses for dh/dt).
Wall readings inherit the env's inset convention, so walls appear 0.3 m closer than the body
geometry, exactly as they do in the CBF rows.
"""
from __future__ import annotations

import numpy as np

#: Reported when nothing is sensed (the LiDAR range).
NO_OBSTACLE_CLEARANCE = 5.0


def confirmed_tracks(tracker, strict=False):
    """Tracks that count as obstacles. `strict` (congestion fix 3) additionally requires the track
    to have been updated on this scan and to rest on at least two segment points, which removes the
    coasting / single-point duplicates the tracker holds (median 9 tracks for 6 real obstacles)."""
    out = [tr for tr in tracker.tracks if tr.confirmed or not tracker.use_confirmation]
    if strict:
        out = [tr for tr in out
               if getattr(tr, "misses", 0) == 0 and getattr(tr, "n_points_last", 2) >= 2]
    return out


class SensedObstacles:
    def __init__(self, *, static_points, track_pos, track_vel, disc_pos, disc_vel, disc_clear,
                 r_robot, track_clear, n_excluded_tracks=0):
        self.static_points = np.asarray(static_points, float).reshape(-1, 2)
        self.track_pos = np.asarray(track_pos, float).reshape(-1, 2)
        self.track_vel = np.asarray(track_vel, float).reshape(-1, 2)
        self.disc_pos = np.asarray(disc_pos, float).reshape(-1, 2)
        self.disc_vel = np.asarray(disc_vel, float).reshape(-1, 2)
        self.disc_clear = np.asarray(disc_clear, float).reshape(-1)
        self.r_robot = float(r_robot)
        self.track_clear = float(track_clear)
        self.n_excluded_tracks = int(n_excluded_tracks)

    @classmethod
    def from_source(cls, src, *, r_robot=0.3, r_obstacle=0.3, exclude_near=None,
                    exclude_radius=0.35, strict_tracks=False):
        """`exclude_near` (policy x.5 only): Protag's true position; confirmed tracks whose centre
        lies within `exclude_radius` of it are Protag, not environment, and are dropped."""
        tracks = confirmed_tracks(src.tracker, strict=strict_tracks)
        conf_ids = np.array([tr.id for tr in tracks], dtype=int)
        keep, n_excl = [], 0
        c = None if exclude_near is None else np.asarray(exclude_near, float).reshape(2)
        for tr in tracks:
            if c is not None and np.linalg.norm(tr.position - c) <= exclude_radius:
                n_excl += 1
                continue
            keep.append(tr)
        static = [pts[~np.isin(np.asarray(ids, int), conf_ids)]
                  for pts, (ids, _) in zip(src.buffer, src.track_buffer)]
        static = np.vstack(static) if static else np.zeros((0, 2))
        disc_pos, disc_vel, disc_clear = [], [], []
        hp = getattr(src, "_hunter_pos", None)
        if hp is not None:
            disc_pos.append(hp)
            disc_vel.append(src._hunter_vel)
            disc_clear.append(src.r_robot + src.hunter_radius)
        return cls(static_points=static,
                   track_pos=[tr.position for tr in keep], track_vel=[tr.velocity for tr in keep],
                   disc_pos=disc_pos, disc_vel=disc_vel, disc_clear=disc_clear,
                   r_robot=r_robot, track_clear=r_robot + r_obstacle, n_excluded_tracks=n_excl)

    def movers(self):
        """(positions, velocities, clearance radii) of every moving obstacle (tracks + discs)."""
        pos = np.vstack([self.track_pos, self.disc_pos])
        vel = np.vstack([self.track_vel, self.disc_vel])
        rad = np.concatenate([np.full(len(self.track_pos), self.track_clear), self.disc_clear])
        return pos, vel, rad

    def clearance(self, P, times):
        """Clearance at positions P (..., N, 2) reached at times `times` (N,) seconds from now."""
        P = np.asarray(P, float)
        t = np.asarray(times, float).reshape(-1)
        if P.shape[-2] != len(t):
            raise ValueError("one time per position along the second-to-last axis")
        out = np.full(P.shape[:-1], NO_OBSTACLE_CLEARANCE)
        if len(self.static_points):
            d = np.linalg.norm(P[..., None, :] - self.static_points, axis=-1).min(-1)
            out = np.minimum(out, d - self.r_robot)
        pos, vel, rad = self.movers()
        if len(pos):
            C = pos[None, :, :] + t[:, None, None] * vel[None, :, :]          # (N, T, 2)
            d = np.linalg.norm(P[..., :, None, :] - C, axis=-1) - rad            # (..., N, T)
            out = np.minimum(out, d.min(-1))
        return out
