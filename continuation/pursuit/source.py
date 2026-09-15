"""Protagonist barrier source augmented with ONE known-state row for the hunter.

KNOWN-STATE / PRIVILEGED BY CONSTRUCTION. The baseline information model (plan section 4) hands
the protagonist the hunter's true position and velocity. This class is the only place that enters
the protagonist's control path, and it mirrors the already-audited `OracleVelocitySource` pattern:
the ground truth arrives through an explicitly separate `push_hunter()` channel, never through the
observation.

WHY THIS AND NOT AN EDIT TO THE CONTROLLER
    `DRCBFPolicy.predict()` stays the frozen code path; only the injected `src` object differs,
    which is exactly how `Phase7Policy` swaps barrier sources. The frozen QP then sorts all rows by
    criticality and keeps n_keep = 5, so eps * N_keep = 0.5 and tau_eff = 0.12 are UNCHANGED, and
    the Random recovery scores candidates against the hunter row automatically because it reads the
    QP's own `xi_kept`. The recovery algorithm itself is not touched.

ROW CONVENTION, identical to the frozen LiDAR rows
    h       = ||p - p_h|| - r_robot - r_hunter      (surface clearance, radius folded in once)
    grad_h  = (p - p_h) / ||p - p_h||               (unit, points away from the hunter)
    dh_dt   = -grad_h . v_hunter                    (explicit time derivative, as for a tracked obstacle)
"""
from __future__ import annotations

import numpy as np

from dr_control.estimated_cbf import EstimatedLidarBarrierSource


class HunterAugmentedSource(EstimatedLidarBarrierSource):
    IS_KNOWN_STATE = True

    def __init__(self, *, hunter_radius=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hunter_radius = float(hunter_radius)
        self._hunter_pos = None
        self._hunter_vel = None
        self.n_hunter_rows = 0

    def reset(self):
        super().reset()
        self._hunter_pos = None
        self._hunter_vel = None
        self.n_hunter_rows = 0

    def push_hunter(self, pos, vel):
        """Ground-truth channel, supplied by the harness from the SHARED PRE-STEP state."""
        self._hunter_pos = np.asarray(pos, dtype=float).reshape(2).copy()
        self._hunter_vel = np.asarray(vel, dtype=float).reshape(2).copy()

    def clear_hunter(self):
        """Withdraw the hunter row (Track A: no detection this step).

        With no row appended the protagonist's barrier set is exactly the frozen navigation one.
        """
        self._hunter_pos = None
        self._hunter_vel = None

    def hunter_row(self, p):
        """(h, grad_h (2,), dh_dt) for the hunter, or None if no hunter state was pushed."""
        if self._hunter_pos is None:
            return None
        p = np.asarray(p, dtype=float).reshape(2)
        d = p - self._hunter_pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-12:
            return -self.r_robot - self.hunter_radius, np.array([1.0, 0.0]), 0.0
        g = d / dist
        return (dist - self.r_robot - self.hunter_radius, g,
                -float(g @ self._hunter_vel))

    def samples(self, p):
        h, grads, dh_dt = super().samples(p)          # frozen LiDAR rows, untouched
        row = self.hunter_row(p)
        if row is None:
            return h, grads, dh_dt
        hh, g, dd = row
        self.n_hunter_rows += 1
        return (np.append(h, hh), np.column_stack([grads, g.reshape(2, 1)]),
                np.append(dh_dt, dd))
