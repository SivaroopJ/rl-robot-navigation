"""Week5-Phase7 / Stage 1 / Direction 2: accelerated CLF-DR-CBF. `[extension]`.

FROZEN CODE IS NOT TOUCHED. `dr_control/drccp_controller.py` remains the reference and the
arbiter; this module adds faster paths beside it and is only ever accepted on evidence of
EQUIVALENCE (gate G1, criteria E1/E2). No runtime number from here is reportable until
feasibility status, action, objective, CBC and closed-loop behaviour have first agreed.

WHY ACCELERATION IS POSSIBLE AT ALL
    Measured on the frozen controller: 27.4-29.3 ms per step, of which 95.0-95.9 % is CVXPY
    canonicalization and 1.1-1.5 ms is the SCS solve. The frozen controller rebuilds
    `cp.Variable`, `cp.Problem` and N+3 constraint objects on EVERY call, exactly as the
    reference implementation does. Two independent fixes follow:

        V1/V2  keep the problem and canonicalize once, feeding data through cp.Parameter
        V3/V4  skip CVXPY entirely, using an exact algebraic reduction of the same problem

THE EXACT REDUCTION (V3/V4) -- A THEOREM, NOT A TOLERANCE
    The frozen DR constraint is

        r_W ||u_bar||_inf / eps <= t - (1/(N eps)) sum_i s_i ,  s_i >= 0 ,  s_i >= t - c_i

    with c_i = u_bar . xi_i. For fixed u the tightest choice is s_i = max(0, t - c_i), so
    feasibility in (t, s) is  max_t f(t) >= r_W ||u_bar||_inf / eps  with

        f(t) = t - (1/(N eps)) sum_i (t - c_i)_+

    f is concave piecewise linear with slope 1 - #{i : c_i < t}/(N eps). When eps*N < 1 that
    slope is +1 below min_i c_i and <= -1 above it, so max_t f = f(min_i c_i) = min_i c_i and

        DR constraint  <=>  min_i c_i >= r_W ||u_bar||_inf / eps  =: tau        (exact)

    ||u_bar||_inf = max(1, alpha, |u_x|, |u_y|), which equals 1 for EVERY admissible u when
    max_v <= max(1, alpha). Under the frozen configuration (alpha 0.4, max_v 1.0, eps 0.1,
    n_keep 5) both preconditions hold and tau = r_W/eps = 0.04 is a CONSTANT, so the whole
    DRCCP collapses to a 3-variable QP:

        min  p0||u - u_prev||^2 + p1||u - u_nom||^2 + p3 delta^2
        s.t. grad_h_i . u >= tau - alpha h_i - dh_dt_i        (N rows)
             grad_V . u - delta <= -rateV V ,  delta >= 0 ,  |u| <= max_v

    t and s_i are projected out; they never appear in the objective, so the minimiser in
    (u, delta) is unchanged. PRECONDITIONS ARE ASSERTED AT CONSTRUCTION and the class refuses
    to build if they fail -- the reduction is not valid outside them, and
    tests/test_frozen_phase1_6.py carries a negative control proving it fails for eps*N >= 1.

TWO BRANCHES THAT DELEGATE TO THE FROZEN CONTROLLER RATHER THAN REIMPLEMENT IT
    1. h_crit < 0. The reference computes p1 = 4*h_crit and p3 = 5*h_crit, which go negative
       and make the objective concave, so CVXPY raises DCPError -- deliberately reproduced in
       Phase 1 rather than repaired. An accelerated path must reproduce that behaviour, not
       improve on it, so this case is handed to the frozen controller verbatim. Measured
       frac_h_negative = 0 across Phases 4-6 and across all 69 226 Stage 0 corpus steps, so
       the branch is expected to be dead; it is tested anyway.
    2. Any variant-specific solver failure that the frozen path would not have had.
       Delegation keeps `prev_u` in sync in both directions.

WHAT IS *NOT* CLAIMED
    Bit-identical output (E0). SCS is first-order, OSQP is ADMM, CLARABEL is interior point;
    the frozen controller already records ~1e-6 box overshoot of its own. The pre-registered
    target is E1 (identical status, action within 1e-6) and E2 (identical episode outcomes on
    paired seeds), and the measured agreement is reported rather than assumed.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import osqp
import scipy.sparse as sp

from dr_control.drccp_controller import (K_V_REFERENCE, XI_DIM, ClfCbfDrccpController,
                                         build_objective, clf_terms, nominal_action)

VARIANTS = ("param_scs", "param_clarabel", "reduced_osqp", "reduced_osqp_ws")

#: OSQP status string -> the CVXPY vocabulary the frozen controller and Phase 6 record in.
#: Kept explicit: Stage 0 found `optimal_inaccurate` steps that must not be folded into either
#: neighbour, and Stage 1's L1 gate compares these labels directly.
OSQP_STATUS = {
    "solved": "optimal",
    "solved inaccurate": "optimal_inaccurate",
    "primal infeasible": "infeasible",
    "primal infeasible inaccurate": "infeasible_inaccurate",
    "dual infeasible": "unbounded",
    "dual infeasible inaccurate": "unbounded_inaccurate",
    "maximum iterations reached": "solver_error",
    "run time limit reached": "solver_error",
    "problem non convex": "solver_error",
    "interrupted": "solver_error",
}

#: Tight enough that ADMM is not the accuracy bottleneck against SCS's own ~1e-6.
#: `polishing` is OFF: measured accuracy against the high-accuracy arbiter is ~1e-14 without
#: it (L0/L2 of exp7_1_equivalence), so it buys nothing here, and osqp 1.1.3 prints
#: "Polishing not needed" to stdout even under verbose=False.
OSQP_SETTINGS = dict(verbose=False, eps_abs=1e-9, eps_rel=1e-9, eps_prim_inf=1e-9,
                     eps_dual_inf=1e-9, max_iter=20000, polishing=False)


class FastClfCbfDrccpController:
    """Accelerated DRCCP with the frozen controller's exact interface and semantics."""

    name = "drccp_fast"

    def __init__(self, *, variant="reduced_osqp", clf_rate=1.0, cbf_rate=0.4,
                 wasserstein_r=0.004, epsilon=0.1, max_v=1.0, k_v=K_V_REFERENCE, p0=3.0,
                 n_keep=5, solver="SCS"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.rateV = clf_rate
        self.rateh = cbf_rate
        self.wasserstein_r = wasserstein_r
        self.epsilon = epsilon
        self.max_v = max_v
        self.k_v = k_v
        self.p0 = p0
        self.n_keep = n_keep
        self.solver = solver

        if variant.startswith("reduced"):
            # The reduction is a theorem only inside these preconditions. Refuse otherwise.
            if not epsilon * n_keep < 1.0:
                raise ValueError(
                    f"reduced variant requires eps*n_keep < 1 (got {epsilon*n_keep}); the CVaR "
                    "does not collapse to the worst sample above 1 and the (t, s_i) form is "
                    "required -- use variant='param_scs'")
            if not max_v <= max(1.0, cbf_rate):
                raise ValueError(
                    f"reduced variant requires max_v <= max(1, alpha) (got max_v={max_v}, "
                    f"alpha={cbf_rate}); above it ||u_bar||_inf becomes u-dependent and tau is "
                    "no longer constant -- use variant='param_scs'")
        #: The constant Wasserstein margin. Valid only under the preconditions above.
        self.tau = wasserstein_r * max(1.0, cbf_rate) / epsilon

        #: Reference instance used ONLY for the delegated branches (h_crit < 0). prev_u is
        #: synchronised in both directions so delegation is transparent.
        self._ref = ClfCbfDrccpController(
            clf_rate=clf_rate, cbf_rate=cbf_rate, wasserstein_r=wasserstein_r,
            epsilon=epsilon, max_v=max_v, k_v=k_v, p0=p0, n_keep=n_keep, solver=solver)

        self.prev_u = np.zeros(2)
        self.solve_fail = True
        self._cvx_cache = {}        # N -> (problem, parameters, variables)
        self._osqp_cache = {}       # N -> OSQP model (pattern fixed by _structure)
        self._struct_cache = {}     # N -> CSC index structure
        self._ws = {}               # N -> (x, y) warm-start state
        self.n_delegated = 0        # metrics only

    def reset(self):
        self.prev_u = np.zeros(2)
        self.solve_fail = True
        self._ref.reset()
        self._ws = {}

    # ------------------------------------------------------------------ shared prologue
    def _prepare(self, p, gamma, xi, u_nom):
        """Sort, truncate and derive weights EXACTLY as the frozen controller does."""
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        if xi.shape[1] != XI_DIM:
            raise ValueError(f"xi must have {XI_DIM} columns, got {xi.shape}")
        if u_nom is None:
            u_nom = nominal_action(p, gamma, self.max_v)
        order = np.argsort(xi[:, 1] * self.rateh + xi[:, 0])
        kept = xi[order][: self.n_keep]
        return kept, float(kept[0, 1]), np.asarray(u_nom, dtype=float).reshape(2)

    def _delegate(self, p, gamma, xi, u_nom, info):
        """Hand the step to the frozen controller and mirror its state. Used for h_crit < 0."""
        self.n_delegated += 1
        self._ref.prev_u = np.asarray(self.prev_u, dtype=float).copy()
        rec = {}
        u = self._ref.generate_controller(p, gamma, xi, u_nom=u_nom, record=rec)
        info.update(rec)
        info["delegated"] = True
        self.prev_u = np.asarray(u, dtype=float).copy()
        self.solve_fail = self._ref.solve_fail
        return np.asarray(u, dtype=float)

    # ------------------------------------------------------------------ entry point
    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        t_call = time.perf_counter()
        kept, h_crit, u_nom = self._prepare(p, gamma, xi, u_nom)
        info = {"h_crit": h_crit, "n_samples": kept.shape[0], "xi_kept": kept,
                "u_nom": u_nom, "variant": self.variant, "delegated": False}

        if h_crit < 0.0:
            # Reference behaviour is a DCPError here and Phase 1 recorded it deliberately.
            out = self._delegate(p, gamma, xi, u_nom, info)
            info.setdefault("total_time", time.perf_counter() - t_call)
            if record is not None:
                record.update(info)
            return out

        V, dV = clf_terms(p, gamma, self.k_v)
        info["V"] = V
        p1, p3 = 4.0 * h_crit, 5.0 * h_crit
        info["weights"] = {"p0": self.p0, "p1": p1, "p3": p3}

        if self.variant.startswith("reduced"):
            u, ok = self._solve_reduced(kept, u_nom, dV, V, p1, p3, info)
        else:
            u, ok = self._solve_parameterized(kept, u_nom, dV, V, p1, p3, info)

        info["total_time"] = time.perf_counter() - t_call
        if record is not None:
            record.update(info)
        self.prev_u = u
        self.solve_fail = not ok
        return u

    # ------------------------------------------------------------------ V3 / V4
    def _structure(self, N):
        """CSC index structure of the reduced QP, built once per sample count.

        Constructed EXPLICITLY rather than by converting a dense array, because a gradient
        component that happens to be exactly zero would otherwise be dropped and the sparsity
        pattern would change between steps -- forcing a full OSQP re-setup and silently losing
        the speedup. The pattern here depends only on N.

            x = [u_x, u_y, delta]
            col 0: rows 0..N-1 (grad_h_x), row N (grad_V_x), row N+2 (box on u_x)
            col 1: rows 0..N-1 (grad_h_y), row N (grad_V_y), row N+3 (box on u_y)
            col 2: row N (-1, CLF slack), row N+1 (delta >= 0)
        """
        cached = self._struct_cache.get(N)
        if cached is not None:
            return cached
        rows = np.arange(N)
        A_ind = np.concatenate([rows, [N, N + 2], rows, [N, N + 3], [N, N + 1]]).astype(np.int32)
        A_ptr = np.array([0, N + 2, 2 * N + 4, 2 * N + 6], dtype=np.int32)
        P_ind = np.array([0, 1, 2], dtype=np.int32)
        P_ptr = np.array([0, 1, 2, 3], dtype=np.int32)
        shape = (N + 4, 3)
        cached = (A_ind, A_ptr, P_ind, P_ptr, shape)
        self._struct_cache[N] = cached
        return cached

    def _reduced_data(self, kept, u_nom, dV, V, p1, p3):
        """(P_data, q, A_data, l, u, const) of the reduced QP, in the cached CSC order."""
        N = kept.shape[0]
        G = kept[:, 2:4]
        b = self.tau - self.rateh * kept[:, 1] - kept[:, 0]

        w = self.p0 + p1
        P_data = np.array([2.0 * w, 2.0 * w, 2.0 * p3])
        lin = -2.0 * (self.p0 * self.prev_u + p1 * u_nom)
        q = np.array([lin[0], lin[1], 0.0])
        const = self.p0 * float(self.prev_u @ self.prev_u) + p1 * float(u_nom @ u_nom)

        A_data = np.concatenate([G[:, 0], [dV[0], 1.0], G[:, 1], [dV[1], 1.0], [-1.0, 1.0]])
        lo = np.concatenate([b, [-np.inf, 0.0, -self.max_v, -self.max_v]])
        hi = np.concatenate([np.full(N, np.inf), [-self.rateV * V, np.inf,
                                                  self.max_v, self.max_v]])
        return P_data, q, A_data, lo, hi, const

    def _solve_reduced(self, kept, u_nom, dV, V, p1, p3, info):
        N = kept.shape[0]
        A_ind, A_ptr, P_ind, P_ptr, shape = self._structure(N)
        P_data, q, A_data, lo, hi, const = self._reduced_data(kept, u_nom, dV, V, p1, p3)

        t0 = time.perf_counter()
        m = self._osqp_cache.get(N)
        if m is None:
            P = sp.csc_matrix((P_data, P_ind, P_ptr), shape=(3, 3))
            A = sp.csc_matrix((A_data, A_ind, A_ptr), shape=shape)
            m = osqp.OSQP()
            m.setup(P=P, q=q, A=A, l=lo, u=hi, **OSQP_SETTINGS)
            self._osqp_cache[N] = m
        else:
            # Pattern is fixed by _structure, so only the VALUES move: no re-setup, no
            # re-factorisation of the sparsity structure.
            m.update(Px=P_data, Ax=A_data, q=q, l=lo, u=hi)
        setup = time.perf_counter() - t0

        if self.variant == "reduced_osqp_ws" and N in self._ws:
            x0, y0 = self._ws[N]
            if x0.shape[0] == 3 and y0.shape[0] == N + 4:
                m.warm_start(x=x0, y=y0)

        t1 = time.perf_counter()
        res = m.solve()
        solver_time = time.perf_counter() - t1
        status = OSQP_STATUS.get(str(res.info.status), f"osqp:{res.info.status}")

        info.update(status=status, solver_time=solver_time, canon_time=setup,
                    setup_time=setup, osqp_status=str(res.info.status),
                    osqp_iter=int(getattr(res.info, "iter", -1)))

        if status != "optimal" or res.x is None or not np.all(np.isfinite(res.x)):
            info["delta"] = float("nan")
            info["objective"] = float("nan")
            self._ws.pop(N, None)
            return np.zeros(2), False       # frozen fallback semantics, unchanged

        if self.variant == "reduced_osqp_ws":
            self._ws[N] = (np.asarray(res.x).copy(), np.asarray(res.y).copy())

        u_raw = np.asarray(res.x[:2], dtype=float)
        u_clipped = np.clip(u_raw, -self.max_v, self.max_v)
        info["box_overshoot"] = float(np.max(np.abs(u_raw)) - self.max_v)
        info["delta"] = float(res.x[2])
        info["objective"] = float(res.info.obj_val) + const
        return u_clipped, True

    # ------------------------------------------------------------------ V1 / V2
    def _build_cvx(self, N):
        """One DPP-compliant problem per sample count, canonicalized once and reused.

        DPP NOTE, and this is the whole reason V1 is written this way. The frozen objective is
        `p1 * sum_squares(u - u_nom)` with BOTH p1 and u_nom data. A parameter multiplying a
        quadratic is not DPP, and neither is `sqrt_p1 * (u - u_nom_param)` -- that is a product
        of two parameter-dependent factors. Writing it as

            sum_squares(sqrt_p1 * u - sqrt_p1_u_nom)

        with `sqrt_p1` and `sqrt_p1_u_nom` as separate parameters is DPP (each product has one
        parameter-dependent factor) and is algebraically the SAME number. The sqrt/square round
        trip perturbs the objective at the 1e-16 level, far inside the E1 threshold.
        """
        u = cp.Variable(2)
        si = cp.Variable(N)
        t = cp.Variable()
        delta = cp.Variable()
        xi_p = cp.Parameter((N, XI_DIM))
        u_prev_p = cp.Parameter(2)
        sqrt_p1 = cp.Parameter(nonneg=True)
        sqrt_p1_unom = cp.Parameter(2)
        sqrt_p3 = cp.Parameter(nonneg=True)
        dV_p = cp.Parameter(2)
        rateV_V = cp.Parameter()

        u_bar = cp.hstack([1.0, self.rateh, u[0], u[1]])
        eps = self.epsilon
        cons = [self.wasserstein_r * cp.abs(u_bar) / eps
                <= (t - cp.sum(si) / (N * eps)) * np.ones(XI_DIM), si >= 0]
        cons += [si[i] >= t - u_bar @ xi_p[i] for i in range(N)]
        cons += [dV_p @ u + rateV_V <= delta, delta >= 0, cp.abs(u) <= self.max_v]

        obj = cp.Minimize(self.p0 * cp.sum_squares(u - u_prev_p)
                          + cp.sum_squares(sqrt_p1 * u - sqrt_p1_unom)
                          + cp.square(sqrt_p3 * delta))
        prob = cp.Problem(obj, cons)
        if not prob.is_dcp(dpp=True):
            raise RuntimeError("parameterized problem is not DPP; it would re-canonicalize "
                               "every step and deliver no speedup")
        pars = dict(xi=xi_p, u_prev=u_prev_p, sqrt_p1=sqrt_p1, sqrt_p1_unom=sqrt_p1_unom,
                    sqrt_p3=sqrt_p3, dV=dV_p, rateV_V=rateV_V)
        return prob, pars, {"u": u, "delta": delta}

    def _solve_parameterized(self, kept, u_nom, dV, V, p1, p3, info):
        N = kept.shape[0]
        if N not in self._cvx_cache:
            self._cvx_cache[N] = self._build_cvx(N)
        prob, pars, vars_ = self._cvx_cache[N]

        s1 = float(np.sqrt(max(p1, 0.0)))
        pars["xi"].value = kept
        pars["u_prev"].value = np.asarray(self.prev_u, dtype=float)
        pars["sqrt_p1"].value = s1
        pars["sqrt_p1_unom"].value = s1 * u_nom
        pars["sqrt_p3"].value = float(np.sqrt(max(p3, 0.0)))
        pars["dV"].value = np.asarray(dV, dtype=float)
        pars["rateV_V"].value = self.rateV * V

        solver = "SCS" if self.variant == "param_scs" else "CLARABEL"
        t0 = time.perf_counter()
        try:
            prob.solve(solver=solver, warm_start=False)
        except cp.error.SolverError as exc:
            info.update(status="SolverError", solver_time=float("nan"),
                        canon_time=float("nan"), delta=float("nan"), error=str(exc))
            return np.zeros(2), False
        total = time.perf_counter() - t0
        st = getattr(prob.solver_stats, "solve_time", None)
        st = float("nan") if st is None else float(st)
        info.update(status=prob.status, solver_time=st,
                    canon_time=total - st if st == st else float("nan"),
                    delta=float(vars_["delta"].value)
                    if vars_["delta"].value is not None else float("nan"),
                    objective=float(prob.value) if prob.value is not None else float("nan"))

        if prob.status != "optimal" or vars_["u"].value is None:
            return np.zeros(2), False
        u_raw = np.asarray(vars_["u"].value, dtype=float).reshape(2)
        u_clipped = np.clip(u_raw, -self.max_v, self.max_v)
        info["box_overshoot"] = float(np.max(np.abs(u_raw)) - self.max_v)
        return u_clipped, True


def reference_objective(u, delta, *, u_prev, u_nom, h_crit, p0=3.0):
    """The frozen objective evaluated numerically -- the L3 comparison quantity."""
    u = np.asarray(u, dtype=float).reshape(2)
    u_prev = np.asarray(u_prev, dtype=float).reshape(2)
    u_nom = np.asarray(u_nom, dtype=float).reshape(2)
    return (p0 * float((u - u_prev) @ (u - u_prev))
            + 4.0 * h_crit * float((u - u_nom) @ (u - u_nom))
            + 5.0 * h_crit * float(delta) ** 2)
