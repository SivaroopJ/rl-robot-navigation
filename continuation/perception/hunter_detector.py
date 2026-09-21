"""Track A: identify the hunter among same-radius dynamic obstacles from geometry-only LiDAR.

WHY A DETECTOR IS NEEDED AT ALL (verified in the Track A sensor audit)
    The scan returns 24 bare ranges. The hunter and all six dynamic obstacles are discs of radius
    0.3, so they are geometrically indistinguishable. Using the environment's object ordering would
    be hidden ground truth. The only honest discriminator is MOTION: the hunter is the single disc
    whose velocity persistently points at the robot.

INFORMATION LEDGER
    Inputs are the robot's own LiDAR ranges and its own pose. This module imports nothing from
    robot_env, never receives the env, and never sees an obstacle list, an object id or a true
    velocity. `tests/test_week6_tracka.py` parses its AST to enforce that.

METHOD
    1. A dedicated LidarVelocityTracker (the frozen class, a SEPARATE instance from the barrier
       source's, so no frozen behaviour changes) turns the scan into confirmed disc tracks.
    2. Each confirmed track gets a pursuit score, averaged over a sliding window:
           a_t = v̂_track · û(ego − track)      (+1 = steering straight at the robot)
       combined with closing speed so that a fast approacher outranks a slow drifter:
           s_t = a_t * min(1, ‖v_track‖ / v_ref)
    3. The hunter estimate is the argmax track whose mean score exceeds `score_threshold` for
       `min_consecutive` frames (hysteresis in, and a separate lower `drop_threshold` out).
    4. If no track qualifies, the detector reports NO DETECTION and the caller appends no hunter
       row, so the protagonist behaves exactly like the frozen navigation controller.
    5. A track the tracker is coasting (misses > 0) is still usable for up to `max_stale` frames and
       is flagged `stale`.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from dr_control.velocity_tracker import LidarVelocityTracker


class HunterDetector:
    def __init__(self, *, dt=0.1, n_rays=24, lidar_range=5.0, r_nominal=0.3,
                 window=5, score_threshold=0.45, drop_threshold=0.25, min_consecutive=3,
                 max_stale=5, v_ref=0.6):
        self.tracker = LidarVelocityTracker(r_nominal=r_nominal, dt=dt, n_rays=n_rays,
                                            lidar_range=lidar_range)
        self.window = int(window)
        self.score_threshold = float(score_threshold)
        self.drop_threshold = float(drop_threshold)
        self.min_consecutive = int(min_consecutive)
        self.max_stale = int(max_stale)
        self.v_ref = float(v_ref)
        self.reset()

    def reset(self):
        self.tracker.reset()
        self._hist = {}            # track id -> deque of per-frame scores
        self._locked = None        # track id currently believed to be the hunter
        self._above = 0            # consecutive frames the best candidate cleared the threshold
        self.n_detections = 0
        self.n_no_detection = 0
        return self

    @staticmethod
    def _score(track, p, v_ref):
        q = np.asarray(track.position, float) - np.asarray(p, float).reshape(2)
        d = float(np.linalg.norm(q))
        v = np.asarray(track.velocity, float)
        sp = float(np.linalg.norm(v))
        if d < 1e-6 or sp < 1e-6:
            return 0.0, d, sp
        align = float((v / sp) @ (-q / d))            # +1 = moving straight at the robot
        return align * min(1.0, sp / v_ref), d, sp

    def update(self, p, ranges):
        """(estimate | None, record). `estimate` = dict(position, velocity, track_id, stale)."""
        p = np.asarray(p, float).reshape(2)
        self.tracker.update(p, np.asarray(ranges, float).reshape(-1))
        live = [t for t in self.tracker.tracks
                if (t.confirmed or not self.tracker.use_confirmation)]
        seen = set()
        cands = []
        for t in live:
            s, d, sp = self._score(t, p, self.v_ref)
            self._hist.setdefault(t.id, deque(maxlen=self.window)).append(s)
            seen.add(t.id)
            cands.append({"id": t.id, "mean_score": float(np.mean(self._hist[t.id])),
                          "score": s, "dist": d, "speed": sp, "misses": int(t.misses),
                          "age": int(t.age), "track": t})
        for tid in list(self._hist):                  # forget tracks the tracker dropped
            if tid not in seen:
                del self._hist[tid]
        rec = {"n_tracks": len(live),
               "candidates": [{k: c[k] for k in ("id", "mean_score", "dist", "speed", "misses")}
                              for c in cands]}
        if not cands:
            self._locked, self._above = None, 0
            self.n_no_detection += 1
            rec.update(detected=False, reason="no_confirmed_tracks")
            return None, rec

        best = max(cands, key=lambda c: c["mean_score"])
        keep = next((c for c in cands if c["id"] == self._locked), None)
        # hysteresis: an already-locked track is kept while it stays above the lower threshold
        if keep is not None and keep["mean_score"] >= self.drop_threshold:
            chosen = keep if keep["mean_score"] >= best["mean_score"] - 1e-12 else best
            if chosen is not keep:
                self._above = 1
        else:
            chosen = best
            self._above = self._above + 1 if best["mean_score"] >= self.score_threshold else 0

        ok = (chosen["id"] == self._locked and chosen["mean_score"] >= self.drop_threshold) or \
             (chosen["mean_score"] >= self.score_threshold and self._above >= self.min_consecutive)
        if not ok or chosen["misses"] > self.max_stale:
            self._locked = None
            self.n_no_detection += 1
            rec.update(detected=False, reason="below_threshold" if not ok else "stale",
                       best_id=chosen["id"], best_score=chosen["mean_score"])
            return None, rec

        self._locked = chosen["id"]
        self.n_detections += 1
        t = chosen["track"]
        est = {"position": np.asarray(t.position, float).copy(),
               "velocity": np.asarray(t.velocity, float).copy(),
               "track_id": int(t.id), "stale": bool(chosen["misses"] > 0),
               "misses": int(chosen["misses"]), "mean_score": chosen["mean_score"]}
        rec.update(detected=True, track_id=int(t.id), mean_score=chosen["mean_score"],
                   stale=est["stale"], reason="locked")
        return est, rec
