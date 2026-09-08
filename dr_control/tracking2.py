"""Week5-Phase7 / Stage 5 / E1: geometry-derived reconstruction covariance. NEW `[extension]`.

Pre-registered in PHASE7_PLAN.md 11.9.8, with the n = 1 derivation corrected in 11.9.18 BEFORE
this file was written (3b3a629). NO GROUND TRUTH REACHES THIS MODULE: like the frozen tracker it
takes only (p_ego, ranges), and it adds no obstacle argument, no obstacle speed bound, no
identity and no per-object radius. The configured obstacle speed range (0.6-0.75) appears
NOWHERE here -- it is a Stage-2 METRIC threshold and was Stage-2b's privileged input, both
inadmissible as an intervention.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
Exactly ONE thing differs from `LidarVelocityTracker`: the measurement-noise covariance R.
Motion model (CV), process noise, gate, association, confirmation, coasting, segmentation,
static rejection, centre reconstruction and `r_nominal` are all the frozen ones, inherited
unchanged. This is NOT IMM, NOT constant-turn, NOT covariance-into-the-barrier and NOT an
uncertainty margin -- those are Stage 6 and remain gated (11.9.13-E/F).

`Track.__init__`'s initial position covariance keeps its frozen SIGMA_R_BASE^2 block. Replacing
it with R would be a SECOND change and 11.9.7 allows exactly one. Recorded as a known
asymmetry, not an oversight.

WHY -- THE FAILURE MECHANISM, TRACED THROUGH THE STAGES
-------------------------------------------------------
Stage 2 found infeasible steps dominated by super-physical estimated closing rates (risk ratios
15.1x / 29.8x); Stage 2b showed those estimates CAUSE the singleton class; Stage 3's T1 removed
the class by clamping dh_dt at the barrier -- the SYMPTOM -- and measured that the estimate
itself did not improve (P(e>0|TTC<1s) 0.485 -> 0.498 randomized, WORSE). E1 acts on the estimate.

Where the wrongness comes from: at the ranges where the barrier binds (h < 1 m) an obstacle is
1-4 LiDAR points. For a one-point return the fixed-radius reconstruction c = q + r*u_ray places
the centre ON the ray line, while the true centre sits off it by the ray's perpendicular offset
s. The frozen filter is told R = diag(s^2, s^2) -- ISOTROPIC -- with s inflated by a hand-set
POINT_COUNT_FACTOR = {1: 3.0, 2: 1.5}. So it understates the error where it is large
(tangential) and overstates it where it is small (radial). Frame to frame the reconstructed
centre slides tangentially along the surface and the filter, trusting that displacement, turns
it into velocity. That is a structurally plausible source of super-physical closing rates.

THE HONEST FAILURE MODE, PRE-REGISTERED (11.9.8)
------------------------------------------------
Inflating R makes the filter sluggish; a sluggish filter reports SMALLER velocities, which
mechanically lowers the super-physical rate while RAISING optimistic error -- it understates
genuine closing. The super-physical rate is therefore NOT admissible as E1's primary endpoint,
and admission A4 requires the optimistic tail AND the super-physical rate to move together.

ZERO NEW TUNED PARAMETERS
-------------------------
Only `r_nominal` (0.3), the existing SIGMA_R_BASE (0.02) and its retained range-growth factor
(1 + d/SIGMA_R_RANGE_REF) appear. POINT_COUNT_FACTOR is removed, as pre-registered. Nothing here
was selected from data, and 11.9.15 forbids selecting it later.
"""
from __future__ import annotations

import numpy as np

from dr_control.velocity_tracker import (GATE_CHI2, SIGMA_R_BASE, SIGMA_R_RANGE_REF,
                                         LidarVelocityTracker, Track, reconstruct_centre)
from scipy.optimize import linear_sum_assignment

#: E[e_t^2] = r^2/3 for a one-point return with ray offset s ~ U(-r, r). See 11.9.18.
MSE_T_COEF = 1.0 / 3.0
#: E[e_r^2] = r^2 (2 - pi/2 - 1/3). A MEAN SQUARE, not a variance: e_r has mean r(1 - pi/4),
#: a bias a Kalman R cannot represent, folded in conservatively as Var + bias^2. See 11.9.18.
MSE_R_COEF = 2.0 - np.pi / 2.0 - 1.0 / 3.0
#: Central-difference step for the n = 2 Jacobian. NUMERICAL, not a model parameter: it
#: differentiates the exact frozen two-root map, and any step in [1e-8, 1e-4] gives the same R
#: to 6 figures (pinned by test_two_point_jacobian_is_step_insensitive).
FD_STEP = 1e-6


def sigma_range(d):
    """Range-measurement std at distance d. The frozen growth law, RETAINED UNCHANGED, with the
    hand-set POINT_COUNT_FACTOR removed -- that is what E1 replaces with geometry."""
    return SIGMA_R_BASE * (1.0 + float(d) / SIGMA_R_RANGE_REF)


def _floor_eigenvalues(R, floor_var):
    """Symmetrise and floor. A reconstruction can never be more certain than the range
    measurement it is built from, so eigenvalues below that variance are raised to it. Derived,
    not tuned, and it keeps R SPD when a fit is ill-conditioned."""
    R = 0.5 * (np.asarray(R, float) + np.asarray(R, float).T)
    w, V = np.linalg.eigh(R)
    return V @ np.diag(np.maximum(w, float(floor_var))) @ V.T


def covariance_1pt(u_ray, r_nominal, sig_r):
    """Sampling covariance of a one-point fixed-radius reconstruction (11.9.18).

    e_tangential = -s and e_radial = r - sqrt(r^2 - s^2) EXACTLY, with s the ray's perpendicular
    offset from the centre. With s ~ U(-r, r) -- the only distribution available, since nothing
    observed says where on the disc the ray landed -- the mean squares are closed-form. Range
    noise adds to the RADIAL axis only, because it displaces the point along the ray.

    The large axis is TANGENTIAL. This is the direction admission check A2 tests.
    """
    u = np.asarray(u_ray, float).reshape(2)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    t = np.array([-u[1], u[0]])
    r2 = float(r_nominal) ** 2
    var_rad = MSE_R_COEF * r2 + float(sig_r) ** 2
    var_tan = MSE_T_COEF * r2
    return var_rad * np.outer(u, u) + var_tan * np.outer(t, t)


def covariance_2pt(points, dirs, ranges_seg, p_ego, r_nominal, sig):
    """First-order propagation of the two range measurements through the EXACT frozen two-root
    map c = m +- b*n_hat, b = sqrt(r^2 - a^2).

    R = J Sigma_rho J^T with Sigma_rho = diag(sigma(rho_1)^2, sigma(rho_2)^2) and J = dc/d(rho)
    by central differences ON THAT SAME MAP, so the covariance describes the reconstruction
    actually used rather than an idealisation of it. This reproduces the a -> r blow-up along
    n_hat that the frozen code merely flags.

    Returns (R, source). A degenerate chord (a >= r) is not differentiable there, so the
    conservative 1-point form is used instead.
    """
    pts = np.asarray(points, float).reshape(2, 2)
    dirs = np.asarray(dirs, float).reshape(2, 2)
    rho = np.asarray(ranges_seg, float).reshape(2)
    p_ego = np.asarray(p_ego, float).reshape(2)

    a = 0.5 * float(np.linalg.norm(pts[1] - pts[0]))
    if a >= float(r_nominal):
        j = int(np.argmin(rho))
        return covariance_1pt(dirs[j], r_nominal, sig[j]), "1pt_degenerate_chord"

    def centre_of(rho_vec):
        q = p_ego[None, :] + np.asarray(rho_vec, float)[:, None] * dirs
        c, _, _ = reconstruct_centre(q, dirs, p_ego, r_nominal)
        return np.asarray(c, float)

    J = np.zeros((2, 2))
    for k in range(2):
        step = FD_STEP * max(1.0, abs(float(rho[k])))
        rp, rm = rho.astype(float).copy(), rho.astype(float).copy()
        rp[k] += step
        rm[k] -= step
        J[:, k] = (centre_of(rp) - centre_of(rm)) / (2.0 * step)
    S = np.diag(np.asarray(sig, float) ** 2)
    return J @ S @ J.T, "2pt_jacobian"


def covariance_npt(points, dirs, centre, sig):
    """First-order propagation through the Gauss-Newton fixed-radius fit, n >= 3.

    Residual f_i(c) = ||c - q_i|| - r has dc-gradient n_i = (c - q_i)/||c - q_i|| and range
    sensitivity df_i/drho_i = -n_i . u_i, so

        R = (J^T J)^-1 J^T Sigma_f J (J^T J)^-1 ,  Sigma_f = diag(sigma_i^2 (n_i . u_i)^2)

    which is the correct first-order form rather than the cruder sigma^2 (J^T J)^-1.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    dirs = np.asarray(dirs, float).reshape(-1, 2)
    c = np.asarray(centre, float).reshape(2)
    d = c[None, :] - pts
    rho = np.linalg.norm(d, axis=1)
    ok = rho > 1e-9
    if int(ok.sum()) < 2:
        return covariance_1pt(dirs[0], 0.3, sig[0]), "1pt_degenerate_fit"
    J = d[ok] / rho[ok, None]                       # rows n_i
    s_f = (np.asarray(sig, float)[ok] * np.abs(np.sum(J * dirs[ok], axis=1))) ** 2
    JtJ = J.T @ J
    if abs(float(np.linalg.det(JtJ))) < 1e-12:      # collinear points: no transverse information
        Ji = np.linalg.pinv(JtJ)
    else:
        Ji = np.linalg.inv(JtJ)
    return Ji @ (J.T @ np.diag(s_f) @ J) @ Ji, "npt_gauss_newton"


def reconstruction_covariance(points, dirs, ranges_seg, p_ego, centre, r_nominal):
    """Geometry-derived R for one detection. Dispatches on point count, then floors."""
    pts = np.asarray(points, float).reshape(-1, 2)
    sig = np.array([sigma_range(r) for r in np.asarray(ranges_seg, float).reshape(-1)])
    n = len(pts)
    if n == 1:
        R, src = covariance_1pt(dirs[0], r_nominal, sig[0]), "1pt_sampling"
    elif n == 2:
        R, src = covariance_2pt(pts, dirs, ranges_seg, p_ego, r_nominal, sig)
    else:
        R, src = covariance_npt(pts, dirs, centre, sig)
    return _floor_eigenvalues(R, float(np.min(sig)) ** 2), src


class GeometricLidarVelocityTracker(LidarVelocityTracker):
    """E1. The frozen tracker with ONE substitution: where R comes from.

    `use_geometric_R=False` restores the frozen behaviour exactly, which is what admission check
    A1 (inertness) pins bit-identically over a real scan trace.

    WHY `update()` IS OVERRIDDEN. The frozen `Track.measurement_noise(n_points,
    range_to_sensor)` signature cannot carry the ray geometry R depends on. The algorithm below
    is the frozen one; the single substitution is marked E1. A1 is the guarantee against drift.
    """

    def __init__(self, *, use_geometric_R=True, **kwargs):
        super().__init__(**kwargs)
        self.use_geometric_R = bool(use_geometric_R)
        self.n_R_source = {}                       # diagnostics only; never fed back

    def detect(self, p_ego, ranges):
        """Frozen detection, plus a geometry-derived R attached to each detection."""
        p_ego = np.asarray(p_ego, float).reshape(2)
        ranges = np.asarray(ranges, float).reshape(-1)
        dets = super().detect(p_ego, ranges)
        for d in dets:
            seg = d["rays"]
            pts = p_ego[None, :] + ranges[seg, None] * self.dirs[seg]
            R, src = reconstruction_covariance(pts, self.dirs[seg], ranges[seg], p_ego,
                                               d["centre"], self.r_nominal)
            d["R"] = R
            d["R_source"] = src
            self.n_R_source[src] = self.n_R_source.get(src, 0) + 1
        return dets

    def _R_for(self, track, det):
        """E1: the one substitution. Geometry when enabled, the frozen rule otherwise."""
        if self.use_geometric_R and det.get("R") is not None:
            return det["R"]
        return track.measurement_noise(det["n_points"], det["range"])

    def update(self, p_ego, ranges):
        """Frozen algorithm. ONLY (p_ego, ranges) enter. R comes from `_R_for` -- see above."""
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
                    R = self._R_for(t, d)                              # E1 substitution
                    g = t.gate_distance(d["centre"], R)
                    if g <= GATE_CHI2:
                        cost[i, j] = g
            if self.association == "hungarian":
                rows, cols = linear_sum_assignment(cost)
                pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
            else:
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
                t.update(d["centre"], self._R_for(t, d),               # E1 substitution
                         n_points=d["n_points"], k_confirm=self.k_confirm,
                         n_window=self.n_confirm_window)
            else:
                t.mark_missed(self.k_confirm, self.n_confirm_window)

        for j, d in enumerate(dets):
            if j not in matched_dets:
                self.tracks.append(Track(d["centre"], self.dt, n_points=d["n_points"]))

        self.tracks = [t for t in self.tracks if t.misses <= self.n_coast]
        return self.tracks

    def reset(self):
        super().reset()
        self.n_R_source = {}
