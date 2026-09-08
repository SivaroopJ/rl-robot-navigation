"""Week5-Phase7 / Stage 3 / M1 intervention: prevent physically impossible closing rates.

NEW `[extension]`. Nothing frozen is modified: the DR-CBF controller, alpha, tau, eps, the
objective, sample selection and the u = 0 fallback are all untouched, and V0 remains the solver.
The intervention lives strictly UPSTREAM, in xi construction.

WHY  Stage 2 found that 37.8 % of infeasible steps have a minimal infeasible subset of
cardinality ONE, and that a single row can only be unsatisfiable inside the action box when the
ESTIMATED closing rate exceeds anything an obstacle can physically produce. Stage 2b confirmed
that causally: supplying a physical velocity by any of three routes removes all 884 singletons.
That class is mechanism M1. The residual multi-constraint class M2 is untouched here and is
Stage 4's problem.

THE CAP IS DERIVED, NOT TUNED.  A single row is unsatisfiable iff
    max_v * ||grad h||_1  <  tau - alpha*h - dh_dt
With the clamp dh_dt >= -v_cap, and UNDER THE OBSERVED CORPUS CONDITIONS h >= 0 and
||grad h||_1 = 1 (the axis-aligned worst case; off-axis gradients give ||.||_1 in (1, sqrt 2]
and are strictly easier),
    v_cap <= max_v - tau = 1.0 - 0.04 = 0.96   ==>   no cardinality-1 subset can exist.
The guarantee is NOT claimed beyond those conditions -- not for h < 0, and not for any other
max_v or tau. `assert_singleton_guarantee` checks the preconditions rather than assuming them.

INFORMATION LEDGER.  v_cap is derived from max_v and tau, both controller-internal constants.
It uses NO knowledge of how fast obstacles actually are. Stage 2b's A4 used the configured
obstacle speed (0.675 / 0.75), which is environment ground truth: legitimate for a diagnostic,
inadmissible as an intervention. Because 0.96 is LOOSER than 0.675/0.75 it clamps less, so
A4's -52.1 % is an UPPER BOUND on what T1 can achieve here, not a target.

TWO TREATMENTS
    T1  projection clamp   dh_dt := max(dh_dt, -v_cap)
    T2  velocity clamp     v_hat := v_hat * min(1, v_cap/||v_hat||), so dh_dt := s * dh_dt
For a CLOSING row (dh_dt < 0), T2 is at least as permissive as T1; on a RECEDING row (dh_dt >= 0)
T1 is a no-op while T2 still shrinks the magnitude. They are therefore not ordered in general,
which is why both are run.
"""
from __future__ import annotations

import numpy as np

from dr_control.estimated_cbf import EstimatedLidarBarrierSource

#: Derived from the frozen configuration: max_v - tau = 1.0 - 0.004/0.1.
V_CAP_DERIVED = 0.96


def derived_cap(max_v=1.0, wasserstein_r=0.004, epsilon=0.1):
    """The largest cap that eliminates the singleton class, under h >= 0 and ||grad h||_1 = 1."""
    return float(max_v - wasserstein_r / epsilon)


def assert_singleton_guarantee(v_cap, *, max_v=1.0, tau=0.04, h_min=0.0, grad_l1_min=1.0):
    """Check the guarantee's PRECONDITIONS instead of assuming them. Raises if they fail."""
    if h_min < 0:
        raise ValueError(f"guarantee requires h >= 0 on the sample set, got h_min={h_min}")
    if grad_l1_min < 1.0 - 1e-12:
        raise ValueError(f"guarantee requires ||grad h||_1 >= 1, got {grad_l1_min}")
    if v_cap > max_v * grad_l1_min - tau + 1e-12:
        raise ValueError(f"v_cap={v_cap} exceeds max_v*||grad||_1 - tau = "
                         f"{max_v * grad_l1_min - tau}; the singleton guarantee does not hold")
    return True


class ProjectionCappedSource(EstimatedLidarBarrierSource):
    """T1. Clamps the PROJECTED closing rate. Touches nothing but column 0 of xi."""

    def __init__(self, *, v_cap=V_CAP_DERIVED, **kwargs):
        super().__init__(**kwargs)
        self.v_cap = float(v_cap)
        self.n_clamped_rows = 0
        self.n_clamped_steps = 0

    def samples(self, p):
        h, grads, dh_dt = super().samples(p)          # frozen computation, untouched
        capped = np.maximum(dh_dt, -self.v_cap)
        n = int(np.sum(capped > dh_dt + 1e-15))
        self.n_clamped_rows += n
        self.n_clamped_steps += int(n > 0)
        return h, grads, capped


class VectorCappedSource(EstimatedLidarBarrierSource):
    """T2. Clamps the VELOCITY VECTOR magnitude, then re-projects.

    Since dh_dt = -grad_h . v and ||grad_h|| = 1, scaling v by s = min(1, v_cap/||v||) scales
    dh_dt by the same s. The owning track per row is recovered by re-deriving the nearest-point
    argmin -- the SAME recomputation Stage 0 validated with `geometry_recompute_mismatch = 0`
    over all 69 226 corpus steps -- so the frozen `samples()` is called unchanged and never
    duplicated. `test_vector_cap_is_inert_at_infinite_cap` pins the equivalence.
    """

    def __init__(self, *, v_cap=V_CAP_DERIVED, **kwargs):
        super().__init__(**kwargs)
        self.v_cap = float(v_cap)
        self.n_clamped_rows = 0
        self.n_clamped_steps = 0

    def _row_speeds(self, p):
        """Estimated speed of the track owning each row, in the frozen row order."""
        p = np.asarray(p, dtype=float).reshape(2)
        out = []
        for pts, (ids, _conf) in zip(self.buffer, self.track_buffer):
            d = np.linalg.norm(pts - p[None, :], axis=1)
            j = int(np.argmin(d))
            tid = int(ids[j])
            tr = self.tracker.track_by_id(tid) if tid >= 0 else None
            trusted = tr is not None and (tr.confirmed or not self.tracker.use_confirmation)
            out.append(float(np.linalg.norm(tr.velocity)) if trusted else 0.0)
        return np.asarray(out)

    def samples(self, p):
        h, grads, dh_dt = super().samples(p)          # frozen computation, untouched
        speed = self._row_speeds(p)
        s = np.where(speed > self.v_cap, self.v_cap / np.maximum(speed, 1e-12), 1.0)
        capped = s * dh_dt
        n = int(np.sum(s < 1.0 - 1e-15))
        self.n_clamped_rows += n
        self.n_clamped_steps += int(n > 0)
        return h, grads, capped


class OracleVelocitySource(EstimatedLidarBarrierSource):
    """D-oracle. DIAGNOSTIC ONLY -- uses ground-truth obstacle state and therefore BREAKS the
    information ledger by construction. Never a headline arm; every result carries the label.

    Mirrors Stage 2b's arm A2: the buffered surface point is attributed to the true obstacle
    whose surface it lay on when the scan was taken, and that obstacle's CURRENT true velocity
    is projected -- the same semantics as the frozen pipeline, which binds a track at push time
    and reads its current velocity.
    """

    IS_DIAGNOSTIC_ONLY = True

    def __init__(self, *, obs_radius=0.3, tol=0.15, **kwargs):
        super().__init__(**kwargs)
        self.obs_radius = float(obs_radius)
        self.tol = float(tol)
        self._truth_buffer = []          # true obstacle positions, aligned with self.buffer
        self._truth_now = None

    def reset(self):
        super().reset()
        self._truth_buffer = []
        self._truth_now = None

    def push_truth(self, obs_pos, obs_vel):
        """Ground-truth channel, supplied by the harness. Explicitly separate from push()."""
        self._truth_buffer.insert(0, np.asarray(obs_pos, dtype=float).copy())
        if len(self._truth_buffer) > self.k_scans:
            self._truth_buffer.pop()
        self._truth_now = np.asarray(obs_vel, dtype=float).copy()

    def samples(self, p):
        h, grads, _ = super().samples(p)
        p = np.asarray(p, dtype=float).reshape(2)
        g = np.asarray(grads, dtype=float)
        q = p[None, :] - (np.asarray(h)[:, None] + self.r_robot) * g.T
        dd = np.zeros(len(h))
        for j in range(len(h)):
            if j >= len(self._truth_buffer) or self._truth_now is None:
                continue
            hp = self._truth_buffer[j]
            if not len(hp):
                continue
            dist = np.linalg.norm(hp - q[j][None, :], axis=1)
            o = int(np.argmin(dist))
            if abs(float(dist[o]) - self.obs_radius) < self.tol and o < len(self._truth_now):
                dd[j] = -float(g[:, j] @ self._truth_now[o])
        return h, grads, dd
