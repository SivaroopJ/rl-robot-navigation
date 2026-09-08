"""Single-integrator CLF-DR-CBF controller (DRCCP), Phase 1.

PROVENANCE
----------
Ported from DR_Safe_Navigation:
    src/erl_clf_cbf_controller/src/erl_clf_cbf_controller/clf_cbf_drccp_dynamic_controller.py
    class ClfCbfDrccp_dynamic_Controller  (author: kehan)
Reference parameter values from:
    src/erl_clf_cbf_controller/src/erl_clf_cbf_controller/controller_config.json  ("drccp")
        wheel_offset 0.08, cbf_rate 0.4, clf_rate 1.0,
        wasserstein_r 0.004, epsilon 0.1, noise_level 0.01

The DRCCP constraint block below is a faithful transcription of the reference's
`generate_controller` (reference lines 100-135). Nothing about the reformulation is
"improved" or redesigned here.

THE REFORMULATION (paper Prop. 6.5, as coded in the reference)
--------------------------------------------------------------
    r_W * |u_bar| / eps  <=  (t - (1/N) * sum(s_i) / eps) * 1     (componentwise)
    s_i >= t - u_bar . xi_i
    s_i >= 0

which is the CVaR-at-level-eps reformulation of

    inf_{P in B_r(P_N)}  CVaR_eps[ CBC ]  >=  0.

WHAT CHANGED FOR THIS ENVIRONMENT, AND WHY
------------------------------------------
[adaptation] Dynamics. The reference is a unicycle with a look-ahead ("off-wheel") point:

    g(x) = [[cos th, -l sin th],
            [sin th,  l cos th],
            [0,       1       ]]        x = [x, y, th],  u = [v, omega]

This environment is a holonomic single integrator, `p_dot = u`, i.e. f(x) = 0 and g(x) = I_2.
The reference packs the barrier constraint as an inner product `u_bar . xi_i`; with the
unicycle that vector is 5-dimensional (reference lines 96-98, 118-124):

    u_bar_uni = [ 1, alpha, (g(x)u)_x, (g(x)u)_y, (g(x)u)_th ]
    xi_uni_i  = [ dh_dt_i, d_i - r_robot, grad_h_x_i, grad_h_y_i, 0 ]

The trailing 0 is there because h is a spatial function in 2-D while the unicycle state is
3-D, so the theta-row of g(x)u contributes nothing. With g(x) = I_2 there is no theta-row at
all and the vector drops to 4 dimensions with no other change:

    u_bar = [ 1, alpha, u_x, u_y ]
    xi_i  = [ dh_dt_i, h_i, grad_h_x_i, grad_h_y_i ]
    u_bar . xi_i = dh_dt_i + alpha * h_i + grad_h_i . u = CBC_i

[adaptation] Barrier convention: `h` is the robot-body clearance (see dr_control/__init__.py).
The reference stores a raw sensor distance `d` and subtracts `robot_radius` inside the
controller; `d - r_robot` and our `h` are the same number, so slot 2 carries `h` directly and
the radius is never subtracted twice.

[adaptation] CLF. The reference's atan2 pose-tracking CLF exists to steer a unicycle's
heading. A holonomic agent has no heading, so V = 0.5 * k_v * ||p - gamma||^2.

[adaptation] Control bounds. The reference clamps |v| <= 1.2 and |omega| <= 1.0; here the
box is per-axis at MAX_SPEED, matching the environment's own action clip exactly.

[adaptation] Objective. The reference minimises

    p0*||u_prev - u||^2 + p1*(u0 - max_v)^2 + p2*u1^2 + p3*delta^2

The p1 term pulls the FORWARD speed toward max_v and the p2 term penalises the ANGULAR rate;
neither has a literal single-integrator analogue. The single-integrator translation keeps the
first, third and fourth terms and expresses "drive as fast as possible toward the reference"
as ||u - u_nom||^2 with u_nom the saturated goal-seeking velocity. The p2 angular-rate term is
dropped; smoothness is already carried by p0*||u - u_prev||^2. See `build_objective`.

[reference implementation correction] Portability, all mechanical:
  - `np.row_stack` (reference lines 78, 116) was removed in NumPy 2.0; this venv has NumPy
    2.4.6. Uses np.vstack / np.hstack.
  - The reference prints h_samples on every solve; removed.

NOT CORRECTED IN THIS ROUND, DELIBERATELY
-----------------------------------------
The reference recomputes p1 = 4*h and p3 = 5*h from the most critical sample every call
(reference lines 138-150), which goes negative once h < 0 and makes the objective concave, so
CVXPY raises DCPError. Phase 1 reproduces this verbatim and MEASURES the h < 0 rate rather
than silently flooring h. Any correction is a separately labelled change made after the
reference behaviour is on record.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

#: Slot ordering of the uncertainty vector xi. Asserted by tests/test_dr_control.py.
#: Mirrors the first four slots of the reference's 5-vector.
XI_SLOTS = ("dh_dt", "h", "grad_h_x", "grad_h_y")
XI_DIM = 4


def build_xi(h, grad_h, dh_dt):
    """Assemble the sample matrix `xi` with the ordering fixed by `XI_SLOTS`.

    Parameters
    h: array (N,)
        Robot-body clearance per sample. Already radius-corrected -- see the module docstring;
        `r_robot` is NOT subtracted again here or anywhere downstream.
    grad_h: array (2, N)
        Unit gradients, columns matching `h`.
    dh_dt: array (N,)
        Explicit time derivative of the barrier, -grad_h . v_obstacle (0 for static).

    Returns
    xi: array (N, 4)
        Row i is [dh_dt_i, h_i, grad_h_x_i, grad_h_y_i].
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    dh_dt = np.asarray(dh_dt, dtype=float).reshape(-1)
    grad_h = np.asarray(grad_h, dtype=float).reshape(2, -1)
    if not (len(h) == len(dh_dt) == grad_h.shape[1]):
        raise ValueError(
            f"inconsistent sample counts: h={len(h)}, dh_dt={len(dh_dt)}, "
            f"grad_h={grad_h.shape}")
    return np.column_stack([dh_dt, h, grad_h[0], grad_h[1]])


def cbc_value(xi_row, u, cbf_rate):
    """CBC = grad_h . u + alpha * h + dh_dt, evaluated numerically.

    The plain-NumPy statement of the constraint the optimizer enforces. Tests check that this
    agrees with `u_bar . xi` built inside the QP, which is what pins the slot ordering.
    """
    xi_row = np.asarray(xi_row, dtype=float)
    u = np.asarray(u, dtype=float)
    return float(u_bar_value(u, cbf_rate) @ xi_row)


def u_bar_value(u, cbf_rate):
    """The decision vector u_bar = [1, alpha, u_x, u_y], as plain NumPy."""
    u = np.asarray(u, dtype=float).reshape(2)
    return np.array([1.0, float(cbf_rate), u[0], u[1]])


def build_objective(u, delta, *, u_prev, u_nom, h_crit, p0=3.0, p2=0.3):
    """The objective shared by the DRCCP controller and the nominal CLF-CBF-QP.

    Shared on purpose: the Phase 1 exact-reduction test compares the two controllers'
    minimizers, and that comparison is only meaningful if their objectives are identical.

    Weights follow the reference verbatim (reference lines 138-150):
        p0 = 3            constant, smoothness
        p1 = 4 * h_crit   proximity gain-scheduling on the goal-seeking term
        p3 = 5 * h_crit   proximity gain-scheduling on the CLF slack
    so both scheduled weights vanish as the robot approaches contact, freeing the controller
    to slow down and abandon CLF descent in favour of the barrier constraint.

    NOTE, and this is the point of not flooring h: for h_crit < 0 both weights are negative,
    p1*square(.) becomes concave, and CVXPY raises DCPError. That is the reference's
    behaviour and Phase 1 records how often it happens instead of hiding it.

    `p2` is accepted and ignored -- kept in the signature to document that the reference's
    angular-rate penalty has no single-integrator analogue [adaptation].
    """
    p1 = 4.0 * h_crit
    p3 = 5.0 * h_crit
    return cp.Minimize(
        p0 * cp.sum_squares(u - u_prev)
        + p1 * cp.sum_squares(u - u_nom)
        + p3 * cp.square(delta)
    ), {"p0": p0, "p1": p1, "p3": p3}


#: Positional CLF gain. This is the reference's `linear_gain_sq` (0.05), the coefficient on
#: the positional term of its CLF (clf_cbf_drccp_dynamic_controller.py:41, 72-76). Keeping the
#: reference value matters numerically as well as for provenance: at k_v = 1.0 the CLF value
#: over a 10x10 arena reaches V ~ 37, and since the objective carries p3 * delta^2 with
#: delta >= rate_V * V, the slack term then dominates the QP by four orders of magnitude and
#: SCS returns solutions that violate the action box by ~1e-2.
K_V_REFERENCE = 0.05


def clf_terms(p, gamma, k_v):
    """V = 0.5 * k_v * ||p - gamma||^2 and its gradient k_v * (p - gamma).  [adaptation]

    The reference's CLF is a pose-tracking function whose positional term is
    0.5 * linear_gain_sq * ||p - gamma||^2 plus an atan2 heading term. A holonomic agent has
    no heading, so only the positional term survives, with k_v = linear_gain_sq.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    gamma = np.asarray(gamma, dtype=float).reshape(2)
    e = p - gamma
    return 0.5 * k_v * float(e @ e), k_v * e


def nominal_action(p, gamma, max_v):
    """Saturated goal-seeking velocity: the u the controller would pick with no obstacles."""
    p = np.asarray(p, dtype=float).reshape(2)
    gamma = np.asarray(gamma, dtype=float).reshape(2)
    d = gamma - p
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.zeros(2)
    return max_v * d / n


class ClfCbfDrccpController:
    """CLF-DR-CBF via the DRCCP reformulation, single integrator.

    Port of ClfCbfDrccp_dynamic_Controller. Stage 1 of the solver plan: SCS, problem rebuilt
    every call, exactly as the reference does. Parameterization is deliberately deferred so
    that solver, mathematics and environment are not all changed at once.
    """

    name = "drccp"

    def __init__(self, *, clf_rate=1.0, cbf_rate=0.4, wasserstein_r=0.004, epsilon=0.1,
                 max_v=1.0, k_v=K_V_REFERENCE, p0=3.0, n_keep=5, solver="SCS"):
        self.rateV = clf_rate          # reference: clf_rate 1.0
        self.rateh = cbf_rate          # reference: cbf_rate 0.4
        self.wasserstein_r = wasserstein_r
        self.epsilon = epsilon
        self.max_v = max_v             # [adaptation] env MAX_SPEED, not the reference's 1.2
        self.k_v = k_v
        self.p0 = p0
        #: Reference keeps the 5 most critical samples (clf_cbf_controller_node.py:345).
        self.n_keep = n_keep
        self.solver = solver
        self.prev_u = np.zeros(2)
        self.solve_fail = True

    def reset(self):
        self.prev_u = np.zeros(2)
        self.solve_fail = True

    # -- the DRCCP constraint block, transcribed from reference lines 100-135 -------------
    def _dro_constraints(self, u, xi, t, si):
        N = xi.shape[0]
        eps = self.epsilon
        # u_bar = [1, alpha, u_x, u_y]   (the reference's `stacked_vector`, minus the theta row)
        u_bar = cp.hstack([1.0, self.rateh, u[0], u[1]])
        ones = np.ones(XI_DIM)
        cons = [
            self.wasserstein_r * cp.abs(u_bar) / eps
            <= (t - cp.sum(si) / (N * eps)) * ones,
            si >= 0,
        ]
        cons += [si[i] >= t - u_bar @ xi[i] for i in range(N)]
        return cons, u_bar

    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        """Solve for u.

        Parameters
        p: array (2,)      robot position
        gamma: array (2,)  CLF reference (the goal itself in Phase 1 -- no governor)
        xi: array (N, 4)   samples, ordering per XI_SLOTS
        u_nom: array (2,) or None
        record: dict or None   filled with diagnostics if given

        Returns
        u: array (2,)      zero on solver failure, as in the reference
        """
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        if xi.shape[1] != XI_DIM:
            raise ValueError(f"xi must have {XI_DIM} columns, got {xi.shape}")
        if u_nom is None:
            u_nom = nominal_action(p, gamma, self.max_v)

        # The reference sorts samples by criticality and KEEPS ONLY THE MOST CRITICAL FIVE
        # (clf_cbf_controller_node.py:345, `argsort(...)[0:5]`), then schedules the objective
        # off the most critical one. Under our clearance convention the sort key is
        # alpha*h + dh_dt.
        #
        # The truncation is load-bearing, not cosmetic. CVaR at level eps over N samples
        # averages the worst eps*N of them; with the reference's N = 5 and eps = 0.1,
        # eps*N = 0.5 < 1, so CVaR collapses to the single WORST sample and the DR constraint
        # is effectively worst-case robust. Keeping more samples (eps*N > 1) averages the tail
        # instead and materially weakens the constraint -- measured here as min CBC going
        # negative and the collision rate rising above the nominal QP's.
        order = np.argsort(xi[:, 1] * self.rateh + xi[:, 0])
        xi_all = xi
        xi = xi[order][: self.n_keep]
        h_crit = float(xi[0, 1])

        N = xi.shape[0]
        u = cp.Variable(2)
        si = cp.Variable(N)
        t = cp.Variable()
        delta = cp.Variable()

        V, dV = clf_terms(p, gamma, self.k_v)
        cons, _ = self._dro_constraints(u, xi, t, si)
        cons += [
            dV @ u + self.rateV * V <= delta,
            delta >= 0,
            cp.abs(u) <= self.max_v,
        ]

        obj, weights = build_objective(
            u, delta, u_prev=self.prev_u, u_nom=u_nom, h_crit=h_crit, p0=self.p0)

        info = {"h_crit": h_crit, "weights": weights, "V": V, "n_samples": N,
                "xi_kept": xi,
                "u_nom": np.asarray(u_nom, dtype=float).reshape(2)}
        u_out, ok = self._solve(cp.Problem(obj, cons), u, delta, info)
        info["dcp_error"] = info.get("status") == "DCPError"
        if record is not None:
            record.update(info)
        self.prev_u = u_out
        self.solve_fail = not ok
        return u_out

    def _solve(self, prob, u, delta, info):
        """Solve, timing canonicalization separately from the solver itself.

        DCPError is caught rather than allowed to propagate: for h_crit < 0 the reference's
        objective is genuinely non-DCP (see build_objective), and Phase 1 needs to COUNT
        those events, not crash on the first one.
        """
        t0 = time.perf_counter()
        try:
            prob.solve(solver=self.solver, verbose=False)
        except cp.error.DCPError as exc:
            info.update(status="DCPError", total_time=time.perf_counter() - t0,
                        solver_time=float("nan"), canon_time=float("nan"),
                        delta=float("nan"), error=str(exc))
            return np.zeros(2), False
        except cp.error.SolverError as exc:
            info.update(status="SolverError", total_time=time.perf_counter() - t0,
                        solver_time=float("nan"), canon_time=float("nan"),
                        delta=float("nan"), error=str(exc))
            return np.zeros(2), False
        total = time.perf_counter() - t0

        solver_time = getattr(prob.solver_stats, "solve_time", None)
        solver_time = float("nan") if solver_time is None else float(solver_time)
        info.update(status=prob.status, total_time=total, solver_time=solver_time,
                    canon_time=total - solver_time if solver_time == solver_time else float("nan"),
                    delta=float(delta.value) if delta.value is not None else float("nan"))

        if prob.status != "optimal" or u.value is None:
            # Reference behaviour: on a non-optimal solve, command zero and reset u_prev.
            return np.zeros(2), False

        # SCS is a first-order solver and returns box-feasible points only to its tolerance
        # (~1e-6 overshoot observed). The reference clips the command downstream in
        # post_process.ClfCbfPostprocess.send_cmd(clip_ctrl=True); we do the same here so the
        # returned action is always admissible. Recorded so the overshoot stays visible.
        u_raw = np.asarray(u.value, dtype=float).reshape(2)
        u_clipped = np.clip(u_raw, -self.max_v, self.max_v)
        info["box_overshoot"] = float(np.max(np.abs(u_raw)) - self.max_v)
        return u_clipped, True
