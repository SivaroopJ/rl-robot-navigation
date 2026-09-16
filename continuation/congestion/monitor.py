"""Predictive feasibility monitor: grid estimate of M(s) - tau_eff. No QP solves.

WHAT THE QP REQUIRES (verified, collapsed regime eps * N_keep <= 1, |u_i| <= 1): the DRCCP block
reduces to min_i CBC_i(u) >= tau_eff = r_W max(1, alpha) / eps = 0.12 on the 5 kept rows, and the
CLF row has an unbounded slack. So the QP is feasible iff max_{box} min_i CBC_i >= tau_eff.

    M(s) = max_{u in grid} min_i CBC_i(s, u),   grid = 9 x 9 over the per-axis box (incl. corners, 0)

computed with the existing `random_recovery.margins` (the same function Random recovery scores
with). The grid is a subset of the box, so M_grid <= M_true: M_grid >= tau_eff CERTIFIES the DR
constraint is satisfiable; M_grid < tau_eff does NOT prove infeasibility. Solver-status failures
(optimal_inaccurate, DCPError) are outside this estimate.

ROWS
    t = 0     the xi the controller is about to solve with (identical rows, same sort)
    t > 0     `samples()` of a SHALLOW COPY of the agent's own source in which every buffered point
              bound to a confirmed track is shifted by v t (and the known hunter row, if any, is
              moved by v_h t), evaluated at p + t u. The copy keeps the source's metric counters
              from being touched. The row equations are therefore the source's own.
    keep      the controller's criticality sort, argsort(alpha h + dh_dt)[:n_keep]
              (ClfCbfDrccpController.generate_controller, reproduced as the one-line key and
              pinned by a test against the controller's recorded xi_kept).
"""
from __future__ import annotations

import copy

import numpy as np

from continuation.random_recovery import margins
from continuation.sensed_obstacles import confirmed_tracks
from dr_control.drccp_controller import build_xi


def control_grid(max_v, n=9):
    lin = np.linspace(-max_v, max_v, n)
    gx, gy = np.meshgrid(lin, lin)
    return np.column_stack([gx.ravel(), gy.ravel()])


def keep_rows(xi, alpha, n_keep):
    xi = np.atleast_2d(np.asarray(xi, float))
    order = np.argsort(xi[:, 1] * alpha + xi[:, 0])
    return xi[order][:n_keep]


def grid_margin(kept, alpha, grid):
    return float(np.max(margins(kept, grid, alpha)))


def predicted_source(src, t, strict_tracks=False):
    cp = copy.copy(src)
    tracks = ({tr.id: tr.velocity for tr in confirmed_tracks(src.tracker, strict=strict_tracks)}
              if t else {})
    buf = []
    for pts, (ids, _) in zip(src.buffer, src.track_buffer):
        q = np.array(pts, dtype=float, copy=True)
        if tracks:
            for i, tid in enumerate(ids):
                v = tracks.get(int(tid))
                if v is not None:
                    q[i] += v * t
        buf.append(q)
    cp.buffer = buf
    cp.track_buffer = list(src.track_buffer)
    if getattr(src, "_hunter_pos", None) is not None:
        cp._hunter_pos = src._hunter_pos + src._hunter_vel * t
    return cp


def predicted_xi(src, p, t, strict_tracks=False):
    h, g, dd = predicted_source(src, t, strict_tracks).samples(np.asarray(p, float).reshape(2))
    return build_xi(h, g, dd)


class FeasibilityMonitor:
    def __init__(self, *, alpha, tau, max_v, n_keep=5, grid_n=9, lookahead=(0.0, 0.25, 0.5),
                 strict_tracks=False, clearance_mode="sensed"):
        #: clearance_mode "rows" (congestion fix 1) reads the clearance from the controller's own
        #: barrier value h instead of a second geometric definition, so the wall inset is counted
        #: once rather than twice.
        self.strict_tracks = bool(strict_tracks)
        self.clearance_mode = clearance_mode
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.n_keep = int(n_keep)
        self.grid = control_grid(max_v, grid_n)
        self.lookahead = tuple(float(t) for t in lookahead)

    def margin_at(self, xi):
        return grid_margin(keep_rows(xi, self.alpha, self.n_keep), self.alpha, self.grid)

    def evaluate(self, src, p, u, *, xi_now=None, obstacles=None):
        p = np.asarray(p, float).reshape(2)
        u = np.asarray(u, float).reshape(2)
        Ms, states, h_min = [], [], []
        for t in self.lookahead:
            q = p + t * u
            xi = (xi_now if (t == 0.0 and xi_now is not None)
                  else predicted_xi(src, q, t, self.strict_tracks))
            Ms.append(self.margin_at(xi))
            h_min.append(float(np.min(np.atleast_2d(xi)[:, 1])))
            states.append(q)
        out = {"M": Ms, "M_look": float(min(Ms)), "margin": float(min(Ms)) - self.tau}
        if self.clearance_mode == "rows":
            out.update(clr=[float(c) for c in h_min], clr_look=float(min(h_min)))
        elif obstacles is not None:
            clr = obstacles.clearance(np.asarray(states), self.lookahead)
            out.update(clr=[float(c) for c in clr], clr_look=float(np.min(clr)))
        return out
