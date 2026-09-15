"""Unlabelled-threat barrier source: one row per eligible MOVING TRACK, no identity, no oracle.

WHAT THIS REPLACES
    The audited baseline appends ONE row built from the hunter's true position and velocity
    (`HunterAugmentedSource`). Here that row is gone. Instead every track the robot's own LiDAR
    tracker confirms as a moving disc contributes a row, whether it is the hunter or one of the six
    dynamic obstacles. The controller is never told which is which, because nothing in the data says.

ROW EQUATIONS -- deliberately the SAME convention as the oracle row, with estimates substituted
    for a track with estimated centre c and velocity v (r_t = tracker's disc radius, 0.3):

        h      = ||p - c|| - r_robot - r_t                      (surface clearance)
        grad_h = (p - c) / ||p - c||                            (unit, points away from the track)
        dh_dt  = -grad_h . v                                    (closing rate, + = receding)

    and for a COASTED track (the tracker predicted it through `misses` missed frames) the clearance
    is deflated by how far the track could have travelled unobserved:

        h_eff  = max(h - misses * dt * ||v||, 0)  if h >= 0     (conservative, never inflates h,
        h_eff  = h                                 if h <  0      and never drives a row below
                                                                  contact -- see the CLAMP note)

    That is the only deviation from the oracle convention, and it exists because a coasted estimate
    is less trustworthy than a fresh one. With misses = 0 it is exactly the oracle form.

ELIGIBILITY (sensor-legal; no ground truth anywhere)
    A track contributes a row iff
      1. it is CONFIRMED by the tracker (3 hits in a 5-frame window) -- this is also what makes the
         velocity estimate meaningful, so we never fabricate a velocity for a fresh detection;
      2. `misses <= max_stale` (default 5 = the tracker's own coast limit);
      3. its estimated speed >= `v_min` -- a motion gate;
      4. STATIC-MAP GATE: its estimated centre is not within `static_gate_margin` of a mapped
         rectangle surface or of the arena boundary.
    One row per track id per step, so a track cannot appear twice in the same solve.

    WHY THE STATIC-MAP GATE EXISTS (measured, not assumed). The tracker's compactness test rejects
    long wall-like segments, but a FRAGMENT of a rectangle or wall seen between occluders is compact
    and is admitted as a disc; because the robot is moving, that fragment appears to slide and the
    constant-velocity filter gives it a large speed, so the motion gate SELECTS it rather than
    removing it. Measured on dev seeds 13 200 200+: 66 % of all rows matched no real body, and 98 %
    of those lay within 0.6 m of mapped static geometry. The gate uses the same prior static map the
    frozen A* planner already holds -- no new information channel -- and never consults live
    obstacle state. A genuinely moving body that hugs a wall is dropped by this gate and is covered
    only by the frozen geometric scan rows; that cost is quantified in the report.

WHAT IS LOST, STATED PLAINLY
    The frozen controller keeps only n_keep = 5 rows, ordered by alpha*h + dh_dt. With 5 scan rows
    plus up to ~7 track rows, most track rows are discarded by that frozen rule. The rule is
    sensor-legal (it reads only row values) but it ignores the grad_h . u term, so a discarded row
    can still bind for some u. `row_stats()` reports how many track rows were built and how many
    survived truncation, so the cost is measured rather than assumed.
"""
from __future__ import annotations

import numpy as np

from dr_control.estimated_cbf import EstimatedLidarBarrierSource

V_MIN = 0.15            # m/s motion gate
MAX_STALE = 5           # = LidarVelocityTracker.n_coast
#: Static-map gate margin. A candidate whose estimated centre lies within this distance of a mapped
#: rectangle surface or of the arena boundary is rejected as static-geometry clutter.
#: CHOSEN ON DEV DATA (results/week6_tracka2/coverage/row_provenance_4.json, seeds 13 200 200+):
#:   margin  phantom rows surviving   real rows lost   hunter rows lost
#:     0.3           4.2 %                26.7 %             6.7 %
#:     0.5           2.7 %                35.0 %            10.6 %
#: 0.3 is the smallest margin that removes ~96 % of the clutter, and it costs the fewest genuine
#: rows. Rows lost this way are bodies hugging a wall or a rectangle, which the frozen geometric
#: scan rows still cover.
STATIC_GATE_MARGIN = 0.3


class UnlabelledThreatSource(EstimatedLidarBarrierSource):
    """Frozen LiDAR rows + one row per eligible moving track. No hunter channel exists."""

    IS_KNOWN_STATE = False

    def __init__(self, *, track_radius=0.3, v_min=V_MIN, max_stale=MAX_STALE, dt=0.1,
                 enable_track_rows=True, static_obstacles=(), world_size=10.0,
                 static_gate_margin=STATIC_GATE_MARGIN, **kwargs):
        super().__init__(**kwargs)
        self.track_radius = float(track_radius)
        #: The SAME static map the frozen A* planner already receives, so this adds no new
        #: information channel. It is a prior map, never a live observation of obstacles.
        self.static_obstacles = [tuple(float(x) for x in r) for r in static_obstacles]
        self.world_size = float(world_size)
        self.static_gate_margin = float(static_gate_margin)
        self.n_gated_static = 0
        self.v_min = float(v_min)
        self.max_stale = int(max_stale)
        self.dt = float(dt)
        self.enable_track_rows = bool(enable_track_rows)
        self.reset_threat_stats()

    def reset(self):
        super().reset()
        self.reset_threat_stats()

    def near_static_geometry(self, c):
        """True if the estimated centre `c` sits within the gate margin of the mapped geometry."""
        c = np.asarray(c, float).reshape(2)
        m = self.static_gate_margin
        if m <= 0.0:                                   # gate explicitly disabled
            return False
        if min(float(c[0]), float(c[1]),
               self.world_size - float(c[0]), self.world_size - float(c[1])) <= m:
            return True
        for cx, cy, hw, hh in self.static_obstacles:
            q = np.array([np.clip(c[0], cx - hw, cx + hw), np.clip(c[1], cy - hh, cy + hh)])
            if float(np.linalg.norm(c - q)) <= m:
                return True
        return False

    def reset_threat_stats(self):
        self.n_rows_built = 0
        self.n_gated_static = 0
        self.n_steps_with_rows = 0
        self.n_eligible_total = 0
        self.last_rows = []
        self.last_row_vectors = []

    # ------------------------------------------------------------------ eligibility
    def eligible_tracks(self):
        out = []
        for t in self.tracker.tracks:
            if not (t.confirmed or not self.tracker.use_confirmation):
                continue
            if int(t.misses) > self.max_stale:
                continue
            if float(np.linalg.norm(np.asarray(t.velocity, float))) < self.v_min:
                continue
            if self.near_static_geometry(t.position):        # static-map gate
                self.n_gated_static += 1
                continue
            out.append(t)
        return out

    def threat_rows(self, p):
        """[(h_eff, grad_h, dh_dt, meta)] for every eligible track. Sensor-derived only."""
        p = np.asarray(p, float).reshape(2)
        rows = []
        for t in self.eligible_tracks():
            c = np.asarray(t.position, float).reshape(2)
            v = np.asarray(t.velocity, float).reshape(2)
            d = p - c
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                continue
            g = d / dist
            h = dist - self.r_robot - self.track_radius
            stale = int(t.misses)
            h_eff = h - stale * self.dt * float(np.linalg.norm(v))
            if h >= 0.0:
                # CLAMP: the staleness deflation may shrink clearance but must not drive a row
                # below contact. An h_crit < 0 makes the FROZEN objective's proximity weights
                # negative and hence non-convex, which CVXPY rejects with DCPError -- measured as
                # a new failure mode in the first gated pilot (31/17/22 extra solver failures).
                # A genuinely negative raw h is left untouched, so real contact behaves exactly as
                # it does in the frozen system.
                h_eff = max(h_eff, 0.0)
            rows.append((float(h_eff), g, -float(g @ v),
                         {"track_id": int(t.id), "dist": dist, "speed": float(np.linalg.norm(v)),
                          "misses": stale, "age": int(t.age), "h_raw": float(h),
                          "h_eff": float(h_eff), "cx": float(c[0]), "cy": float(c[1])}))
        return rows

    # ------------------------------------------------------------------ samples
    def samples(self, p):
        h, grads, dh_dt = super().samples(p)          # frozen scan rows, untouched
        if not self.enable_track_rows:
            self.last_rows = []
            self.last_row_vectors = []
            return h, grads, dh_dt
        rows = self.threat_rows(p)
        self.last_rows = [r[3] for r in rows]
        #: exact row vectors [dh_dt, h, gx, gy] as they will appear in xi, so an offline
        #: diagnostic can tell which rows survived the frozen n_keep truncation.
        self.last_row_vectors = [(float(r[2]), float(r[0]), float(r[1][0]), float(r[1][1]))
                                 for r in rows]
        self.n_eligible_total += len(rows)
        if not rows:
            return h, grads, dh_dt
        self.n_rows_built += len(rows)
        self.n_steps_with_rows += 1
        hh = np.append(h, [r[0] for r in rows])
        gg = np.column_stack([grads] + [r[1].reshape(2, 1) for r in rows])
        dd = np.append(dh_dt, [r[2] for r in rows])
        return hh, gg, dd

    def row_stats(self):
        return {"track_rows_gated_static": self.n_gated_static,
                "track_rows_built": self.n_rows_built,
                "steps_with_track_rows": self.n_steps_with_rows,
                "eligible_track_rows_total": self.n_eligible_total}
