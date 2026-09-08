"""Week5-Phase7 / Stage 4: recovery of the residual M2 (multi-constraint) infeasibility.

NEW `[extension]`. V0 is FROZEN and is not modified. The ladder acts ONLY on steps the frozen
problem declares infeasible; on feasible steps it is inert and the frozen action is returned
untouched. alpha, tau, eps, v_cap, the objective, sample selection, the planner and the LiDAR
pipeline are all unchanged, and no estimator, IMM/CT, covariance or governor logic appears here.

THE LADDER, and what each rung gives up
    R0   min_i CBC_i >= tau      the frozen DR constraint          nothing given up
    R1   min_i CBC_i >= 0        tau := 0                          distributional robustness only
    R2   maximise min_i CBC_i,   then minimise the frozen           forward invariance; the
         objective among the maximisers                            achieved margin m is reported

A CRITICAL INTERPRETATION SAFEGUARD, per the Stage 4 approval:
    R1 "resolves" a step only in the sense that it makes the tau = 0 optimisation FEASIBLE. The
    resulting action is NOT thereby safe: it may still have m < 0 with respect to the original
    DR constraint once m is recomputed from the returned action. **No recovered action is ever
    described as safe.** Every record carries the achieved margin m and its guarantee tier.

THE WHOLE LADDER IS DETERMINED BY ONE NUMBER.  Let c_i(u) = grad_h_i . u + alpha*h_i + dh_dt_i
and m* = max_{|u|<=max_v} min_i c_i(u). Then
    m* >= tau   <=>  R0 feasible          0 <= m* < tau  <=>  R1 feasible, R0 not
    m* < 0      <=>  neither; R2 is the terminal rung
so the rung, the margin and the tier are consistent by construction rather than by convention.
The collapse identity that makes `min_i c_i >= tau` the exact DR constraint (eps*N < 1) is the
one pinned by tests/test_frozen_phase1_6.py and is not re-litigated here.

TIERS (plan section 0)
    T0  m >= tau   DR guarantee intact
    T1  0 <= m < tau   nominal CBF holds; only the Wasserstein margin is consumed
    T2  m < 0      forward invariance lost; the worst-case decay bound below is reported
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from dr_control.drccp_controller import build_objective, clf_terms

ETA = 1e-9          # declared lexicographic tolerance, not tuned
MODES = ("none", "R1", "R2", "ladder", "R15")


def cbc(kept, u, alpha):
    """min_i CBC_i for a returned action -- the achieved margin m."""
    u = np.asarray(u, dtype=float).reshape(2)
    return float(np.min(np.asarray(kept) @ np.array([1.0, alpha, u[0], u[1]])))


def tier_of(m, tau):
    """T0 / T1 / T2 from the achieved margin. Never a statement that an action is safe."""
    if m >= tau:
        return 0
    return 1 if m >= 0.0 else 2


def decay_floor(m, alpha):
    """For a T2 action, the asymptotic floor of the worst-case bound h(t) >= ... - |m|/alpha.

    With min_i CBC_i = m < 0 the barrier can decay as h_dot >= -alpha*h + m, whose solution is
    h(t) = (h0 + m/alpha) e^{-alpha t} - m/alpha, so h can approach m/alpha (negative). Returned
    as a NEGATIVE number for m < 0, and 0.0 otherwise. This is a bound, not a prediction.
    """
    return float(m / alpha) if m < 0 else 0.0


def _solve(kept, p, gamma, u_prev, u_nom, *, floor, alpha, max_v, k_v, rate_V, p0, solver):
    """The frozen objective and CLF, with the barrier rows required to clear `floor`.

    `floor = tau` reproduces R0's feasible set; `floor = 0` is R1. The objective builder is the
    FROZEN one, imported unchanged, so the recovered action minimises the same cost the
    controller was minimising.
    """
    kept = np.atleast_2d(np.asarray(kept, dtype=float))
    u = cp.Variable(2)
    delta = cp.Variable()
    V, dV = clf_terms(p, gamma, k_v)
    u_bar = cp.hstack([1.0, alpha, u[0], u[1]])
    cons = [kept @ u_bar >= floor, dV @ u + rate_V * V <= delta, delta >= 0,
            cp.abs(u) <= max_v]
    obj, _ = build_objective(u, delta, u_prev=u_prev, u_nom=u_nom,
                             h_crit=float(kept[0, 1]), p0=p0)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver, verbose=False)
    except (cp.error.SolverError, cp.error.DCPError):
        return None, "error"
    if prob.status != "optimal" or u.value is None:
        return None, prob.status
    return np.clip(np.asarray(u.value, dtype=float).reshape(2), -max_v, max_v), prob.status


def _max_min_margin(kept, *, alpha, max_v, solver):
    """m* = max_{|u|<=max_v} min_i CBC_i, as an LP. R2 stage A."""
    kept = np.atleast_2d(np.asarray(kept, dtype=float))
    u = cp.Variable(2)
    s = cp.Variable()
    u_bar = cp.hstack([1.0, alpha, u[0], u[1]])
    prob = cp.Problem(cp.Maximize(s), [kept @ u_bar >= s, cp.abs(u) <= max_v])
    try:
        prob.solve(solver=solver, verbose=False)
    except (cp.error.SolverError, cp.error.DCPError):
        return None, None
    if prob.status != "optimal" or u.value is None:
        return None, None
    return float(s.value), np.clip(np.asarray(u.value, float).reshape(2), -max_v, max_v)


def r15_admission(kept, *, alpha, dt, alpha_cap=None):
    """The three admission checks of plan section 3.3-D2. R1.5 runs ONLY if all three pass."""
    kept = np.atleast_2d(np.asarray(kept, dtype=float))
    reasons = []
    if float(kept[:, 1].min()) < 0:
        reasons.append("h_min < 0: raising alpha TIGHTENS such a row, monotonicity fails")
    cap = alpha_cap if alpha_cap is not None else 1.0 / dt
    if alpha >= cap:
        reasons.append(f"alpha {alpha} already at the discrete-time cap 1/dt = {cap}")
    # coupling: alpha' is substituted ONLY in the barrier rows, holding sample selection,
    # tau and the objective weights at their alpha-derived values. That is checkable here.
    if not np.all(np.isfinite(kept)):
        reasons.append("non-finite sample")
    return (not reasons), reasons


class RecoveryLadder:
    """Wraps a live frozen controller instance. The frozen module is never edited."""

    def __init__(self, ctrl, *, mode="ladder", alpha=None, tau=None, max_v=None,
                 k_v=None, rate_V=None, p0=None, solver=None, eta=ETA):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.ctrl = ctrl
        self.mode = mode
        self.alpha = ctrl.rateh if alpha is None else alpha
        self.tau = (ctrl.wasserstein_r / ctrl.epsilon) if tau is None else tau
        self.max_v = ctrl.max_v if max_v is None else max_v
        self.k_v = ctrl.k_v if k_v is None else k_v
        self.rate_V = ctrl.rateV if rate_V is None else rate_V
        self.p0 = ctrl.p0 if p0 is None else p0
        self.solver = ctrl.solver if solver is None else solver
        self.eta = float(eta)
        self.records = []
        self._orig = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._orig

    # ------------------------------------------------------------------ the wrapper
    def _alpha_relaxed(self, kept, *, cap=None, tol=1e-3):
        """R1.5 ABLATION. Smallest alpha' >= alpha making the barrier rows jointly satisfiable.

        alpha' is substituted ONLY in the barrier rows; sample selection, tau and the objective
        weights stay at their alpha-derived values, which is what makes this a relaxation of the
        original problem rather than a different problem (plan section 3.3-D2). Valid only while
        every kept h >= 0 -- checked by `r15_admission`, not assumed.
        """
        cap = cap if cap is not None else 1.0 / 0.1
        g, h, dd = kept[:, 2:4], kept[:, 1], kept[:, 0]

        def ok(a):
            b = self.tau - a * h - dd
            # max over the box of g_i . u, per row; jointly feasible iff the LP is
            u = cp.Variable(2)
            pr = cp.Problem(cp.Minimize(0), [g @ u >= b, cp.abs(u) <= self.max_v])
            try:
                pr.solve(solver=self.solver, verbose=False)
            except Exception:
                return False
            return pr.status == "optimal"

        if not ok(cap):
            return None
        lo, hi = self.alpha, cap
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u_prev = np.asarray(self.ctrl.prev_u, dtype=float).copy()
        u = self._orig(p, gamma, xi, u_nom=u_nom, record=rec)     # FROZEN call, verbatim
        status = str(rec.get("status"))
        kept = np.atleast_2d(np.asarray(rec.get("xi_kept", xi), dtype=float))
        un = np.asarray(rec.get("u_nom", [0.0, 0.0]), dtype=float).reshape(2)

        entry = {"status": status, "infeasible": "infeasible" in status,
                 "rung": "R0", "recovered": False,
                 "m": cbc(kept, u, self.alpha), "h_crit": float(kept[0, 1])}
        if not entry["infeasible"] or self.mode == "none":
            # INERT: the frozen action is returned untouched. On an infeasible step under
            # mode="none" that action is the frozen u = 0, and the rung is labelled as the
            # fallback rather than as R0 -- R0 means the DR problem was SOLVED.
            if entry["infeasible"]:
                entry["rung"] = "fallback_u0"
            entry["tier"] = tier_of(entry["m"], self.tau) if status == "optimal" else None
            entry["decay_floor"] = None
            self.records.append(entry)
            rec.update(entry)
            return u

        kw = dict(alpha=self.alpha, max_v=self.max_v, k_v=self.k_v, rate_V=self.rate_V,
                  p0=self.p0, solver=self.solver)
        u_out, rung = None, None

        if self.mode in ("R1", "ladder"):
            cand, st = _solve(kept, p, gamma, u_prev, un, floor=0.0, **kw)
            if cand is not None:
                u_out, rung = cand, "R1"

        if u_out is None and self.mode == "R15":
            adm, why = r15_admission(kept, alpha=self.alpha, dt=0.1)
            entry["r15_admission"] = adm
            entry["r15_reasons"] = why
            if adm:
                ap = self._alpha_relaxed(kept)
                if ap is not None:
                    entry["alpha_prime"] = ap
                    kept_r = kept.copy()
                    kept_r[:, 1] = kept[:, 1] * (ap / self.alpha)   # alpha' only in the rows
                    cand, st = _solve(kept_r, p, gamma, u_prev, un, floor=self.tau, **kw)
                    if cand is not None:
                        u_out, rung = cand, "R1.5"

        if u_out is None and self.mode in ("R2", "ladder"):
            m_star, u_lp = _max_min_margin(kept, alpha=self.alpha, max_v=self.max_v,
                                           solver=self.solver)
            if m_star is not None:
                cand, st = _solve(kept, p, gamma, u_prev, un, floor=m_star - self.eta, **kw)
                u_out, rung = (cand, "R2") if cand is not None else (u_lp, "R2_lp")
                entry["m_star"] = m_star

        if u_out is None:                       # R1 arm on a step R1 cannot fix
            entry.update(rung="fallback_u0", recovered=False, tier=None, decay_floor=None)
            self.records.append(entry)
            rec.update(entry)
            return u                            # the FROZEN u = 0, unchanged

        m = cbc(kept, u_out, self.alpha)
        entry.update(rung=rung, recovered=True, m=m, tier=tier_of(m, self.tau),
                     decay_floor=decay_floor(m, self.alpha),
                     u_dev=float(np.linalg.norm(u_out - un)))
        self.records.append(entry)
        rec.update(entry)
        self.ctrl.prev_u = np.asarray(u_out, dtype=float).copy()   # keep frozen state coherent
        return np.asarray(u_out, dtype=float)
