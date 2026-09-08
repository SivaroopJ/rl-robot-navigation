"""Baseline controllers for the Phase 1 reduction tests.

PROVENANCE
----------
`ClfCbfQpController` is ported from DR_Safe_Navigation:
    src/erl_clf_cbf_controller/src/erl_clf_cbf_controller/clf_cbf_controller.py
    class ClfCbfController
`ClfOnlyController` from .../clf_only_controller.py (class ClfQPController).

Not ported: `robust_cbf_socp`. The reference node dispatches to a class
`ClfCbf_Robust_SOCP_Controller` (clf_cbf_controller_node.py:205) that is never defined or
imported anywhere in DR_Safe_Navigation, so selecting it raises NameError.
[reference implementation issue]

WHY THE OBJECTIVE IS IMPORTED, NOT REWRITTEN
--------------------------------------------
Phase 1's central test is that the DRCCP reduces EXACTLY to this nominal CBF-QP when
r_W = 0, eps = 1, N = 1. The derivation shows the two FEASIBLE SETS coincide:

    constraints become   0 <= t - s_1,   s_1 >= 0,   s_1 >= t - u_bar.xi_1
    if u_bar.xi_1 >= 0:  t = s_1 = 0 is feasible
    if u_bar.xi_1 <  0:  s_1 > t contradicts s_1 <= t  ->  infeasible
    => feasible set is exactly { u : u_bar.xi_1 >= 0 } = { CBC >= 0 }

Equal feasible sets give equal minimizers only if the OBJECTIVES also match, so this module
imports `build_objective` from drccp_controller rather than restating it. If the objectives
were allowed to drift the test would silently stop meaning anything.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from dr_control.drccp_controller import (
    XI_DIM, K_V_REFERENCE, build_objective, clf_terms, nominal_action)


class ClfCbfQpController:
    """Nominal CLF-CBF-QP on a single (h, grad_h, dh_dt) sample.

    Constraint (reference clf_cbf_controller.py:92-99, adapted to the single integrator and
    to the clearance convention -- alpha multiplies h directly, the radius is not subtracted
    again):

        grad_h . u + alpha * h + dh_dt >= 0
    """

    name = "cbf_qp"

    def __init__(self, *, clf_rate=1.0, cbf_rate=0.4, max_v=1.0, k_v=K_V_REFERENCE, p0=3.0,
                 solver="SCS"):
        self.rateV = clf_rate
        self.rateh = cbf_rate
        self.max_v = max_v
        self.k_v = k_v
        self.p0 = p0
        self.solver = solver
        self.prev_u = np.zeros(2)
        self.solve_fail = True

    def reset(self):
        self.prev_u = np.zeros(2)
        self.solve_fail = True

    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        """`xi` may hold several samples; the most critical one is used, as in the reference."""
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        if xi.shape[1] != XI_DIM:
            raise ValueError(f"xi must have {XI_DIM} columns, got {xi.shape}")
        if u_nom is None:
            u_nom = nominal_action(p, gamma, self.max_v)

        order = np.argsort(xi[:, 1] * self.rateh + xi[:, 0])
        xi = xi[order]
        dh_dt, h, gx, gy = xi[0]
        h_crit = float(h)

        u = cp.Variable(2)
        delta = cp.Variable()
        V, dV = clf_terms(p, gamma, self.k_v)

        cons = [
            gx * u[0] + gy * u[1] + self.rateh * h + dh_dt >= 0,
            dV @ u + self.rateV * V <= delta,
            delta >= 0,
            cp.abs(u) <= self.max_v,
        ]
        obj, weights = build_objective(
            u, delta, u_prev=self.prev_u, u_nom=u_nom, h_crit=h_crit, p0=self.p0)

        # The nominal QP constrains exactly ONE sample -- the most critical. Reporting that
        # subset lets the diagnostics distinguish "the constraint was violated" (a real
        # failure) from "another, unconstrained obstacle was unsafe" (the expected weakness
        # of a single-constraint CBF, and the reason the paper uses N samples).
        info = {"h_crit": h_crit, "weights": weights, "V": V, "n_samples": 1,
                "xi_kept": xi[:1],
                "u_nom": np.asarray(u_nom, dtype=float).reshape(2)}
        prob = cp.Problem(obj, cons)
        t0 = time.perf_counter()
        try:
            prob.solve(solver=self.solver, verbose=False)
        except cp.error.DCPError as exc:
            info.update(status="DCPError", total_time=time.perf_counter() - t0,
                        solver_time=float("nan"), delta=float("nan"), error=str(exc))
            if record is not None:
                record.update(info)
            self.prev_u = np.zeros(2)
            self.solve_fail = True
            return np.zeros(2)
        total = time.perf_counter() - t0

        solver_time = getattr(prob.solver_stats, "solve_time", None)
        solver_time = float("nan") if solver_time is None else float(solver_time)
        info.update(status=prob.status, total_time=total, solver_time=solver_time,
                    canon_time=total - solver_time if solver_time == solver_time else float("nan"),
                    delta=float(delta.value) if delta.value is not None else float("nan"))
        if record is not None:
            record.update(info)

        if prob.status != "optimal" or u.value is None:
            self.solve_fail = True
            self.prev_u = np.zeros(2)
            return np.zeros(2)
        self.solve_fail = False
        # Same first-order-solver clip as the DRCCP path; the reference clips downstream in
        # post_process.ClfCbfPostprocess.send_cmd(clip_ctrl=True).
        u_raw = np.asarray(u.value, dtype=float).reshape(2)
        info["box_overshoot"] = float(np.max(np.abs(u_raw)) - self.max_v)
        self.prev_u = np.clip(u_raw, -self.max_v, self.max_v)
        return self.prev_u


class ClfOnlyController:
    """CLF only, no barrier constraint -- the sanity floor.

    Port of clf_only_controller.py (ClfQPController), adapted to the single integrator. With
    no CBF this is just the saturated goal-seeking action, so it is solved in closed form
    rather than through a QP; a test asserts it matches the QP form.
    """

    name = "clf_only"

    def __init__(self, *, max_v=1.0, **_ignored):
        self.max_v = max_v
        self.prev_u = np.zeros(2)
        self.solve_fail = False

    def reset(self):
        self.prev_u = np.zeros(2)

    def generate_controller(self, p, gamma, xi=None, *, u_nom=None, record=None):
        u = nominal_action(p, gamma, self.max_v) if u_nom is None else np.asarray(u_nom)
        u = np.clip(np.asarray(u, dtype=float).reshape(2), -self.max_v, self.max_v)
        if record is not None:
            record.update(status="optimal", total_time=0.0, solver_time=0.0,
                          delta=0.0, h_crit=float("nan"), n_samples=0,
                          weights={}, V=float("nan"), u_nom=u.copy())
        self.prev_u = u
        return u
