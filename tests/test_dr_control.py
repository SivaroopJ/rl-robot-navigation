"""Phase 1 gate: controller mathematics, no sensor.

Hard tests (must pass):
    1. u_bar . xi equals the hand-computed CBC
    2. xi slot ordering is exactly [dh_dt, h, grad_h_x, grad_h_y]
    3. exact reduction: r_W = 0, eps = 1, N = 1  ==>  DRCCP == nominal CBF-QP

Diagnostic checks (recorded, not gating) live in experiments/exp0_analytic.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.baselines import ClfCbfQpController, ClfOnlyController
from dr_control.cbf_sources import (
    AnalyticBarrierSource, circle_barrier, rect_barrier, wall_barriers)
from dr_control.drccp_controller import (
    XI_DIM, XI_SLOTS, ClfCbfDrccpController, build_xi, cbc_value, clf_terms,
    nominal_action, u_bar_value)

ALPHA = 0.4          # reference cbf_rate
MAX_V = 1.0          # env MAX_SPEED
R_ROBOT = 0.3        # env AGENT_RADIUS
R_OBS = 0.3          # env OBSTACLE_RADIUS


# ---------------------------------------------------------------- hard test 2: ordering
def test_xi_slot_ordering_is_fixed():
    assert XI_SLOTS == ("dh_dt", "h", "grad_h_x", "grad_h_y")
    assert XI_DIM == 4

    xi = build_xi(h=[1.5], grad_h=np.array([[0.6], [0.8]]), dh_dt=[-0.25])
    assert xi.shape == (1, 4)
    np.testing.assert_allclose(xi[0], [-0.25, 1.5, 0.6, 0.8])


def test_u_bar_ordering_matches_xi():
    u = np.array([0.3, -0.7])
    np.testing.assert_allclose(u_bar_value(u, ALPHA), [1.0, ALPHA, 0.3, -0.7])


# ------------------------------------------------------- hard test 1: CBC algebra
@pytest.mark.parametrize("h, grad, dh_dt, u", [
    (1.5, (0.6, 0.8), -0.25, (0.3, -0.7)),
    (0.0, (1.0, 0.0), 0.0, (-1.0, 0.0)),
    (-0.2, (0.0, -1.0), 0.5, (0.25, 0.25)),
    (3.0, (-0.6, 0.8), 0.1, (1.0, 1.0)),
])
def test_cbc_equals_hand_computed(h, grad, dh_dt, u):
    grad = np.asarray(grad, dtype=float)
    u = np.asarray(u, dtype=float)
    expected = float(grad @ u) + ALPHA * h + dh_dt          # CBC = grad_h.u + alpha*h + dh_dt
    xi = build_xi([h], grad.reshape(2, 1), [dh_dt])
    assert cbc_value(xi[0], u, ALPHA) == pytest.approx(expected)


def test_cbc_matches_inner_product_form():
    """The QP enforces u_bar . xi; the plain-NumPy CBC must be the same number."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        h = float(rng.uniform(-1, 5))
        theta = float(rng.uniform(0, 2 * np.pi))
        grad = np.array([np.cos(theta), np.sin(theta)])
        dh_dt = float(rng.uniform(-1, 1))
        u = rng.uniform(-MAX_V, MAX_V, 2)
        xi = build_xi([h], grad.reshape(2, 1), [dh_dt])
        assert cbc_value(xi[0], u, ALPHA) == pytest.approx(float(u_bar_value(u, ALPHA) @ xi[0]))


def test_build_xi_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_xi(h=[1.0, 2.0], grad_h=np.ones((2, 1)), dh_dt=[0.0, 0.0])


# ------------------------------------------------------------ barrier source geometry
def test_circle_barrier_is_robot_body_clearance():
    """h = 0 exactly at the environment's collision threshold, r_robot + r_obs."""
    h, grad, dh = circle_barrier([0.6, 0.0], [0.0, 0.0], R_OBS, R_ROBOT)
    assert h == pytest.approx(0.0)
    np.testing.assert_allclose(grad, [1.0, 0.0])
    assert dh == 0.0

    h, _, _ = circle_barrier([2.0, 0.0], [0.0, 0.0], R_OBS, R_ROBOT)
    assert h == pytest.approx(2.0 - 0.6)


def test_circle_barrier_dh_dt_sign():
    """An obstacle closing on the robot must give dh_dt < 0."""
    # Robot at origin, obstacle at +x moving in -x (toward the robot).
    _, grad, dh = circle_barrier([0.0, 0.0], [2.0, 0.0], R_OBS, R_ROBOT, v_obs=(-1.0, 0.0))
    np.testing.assert_allclose(grad, [-1.0, 0.0])
    assert dh == pytest.approx(-1.0)
    # Receding.
    _, _, dh = circle_barrier([0.0, 0.0], [2.0, 0.0], R_OBS, R_ROBOT, v_obs=(1.0, 0.0))
    assert dh == pytest.approx(1.0)


def test_rect_barrier_matches_env_collision_geometry():
    """Reproduces RobotNavEnv._check_collision_type: clip to the rect, then distance."""
    rect = (2.0, 2.0, 0.5, 1.5)      # a real static obstacle from config.json
    rng = np.random.default_rng(1)
    for _ in range(500):
        p = rng.uniform(0, 10, 2)
        cx, cy, hw, hh = rect
        closest = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
        expected = float(np.linalg.norm(p - closest)) - R_ROBOT
        h, _, _ = rect_barrier(p, rect, R_ROBOT)
        if np.linalg.norm(p - closest) > 1e-9:      # outside: exact
            assert h == pytest.approx(expected, abs=1e-9)


def test_wall_barrier_matches_env_clamp():
    """Env collides when the centre leaves [r_robot, W - r_robot]; h = 0 there."""
    for h, _, _ in wall_barriers([R_ROBOT, 5.0], 10.0, R_ROBOT):
        pass
    hs = [h for h, _, _ in wall_barriers([R_ROBOT, 5.0], 10.0, R_ROBOT)]
    assert min(hs) == pytest.approx(0.0)
    hs = [h for h, _, _ in wall_barriers([5.0, 5.0], 10.0, R_ROBOT)]
    assert min(hs) == pytest.approx(4.7)


def test_analytic_source_shapes():
    src = AnalyticBarrierSource(world_size=10.0, r_robot=R_ROBOT,
                                static_rects=[(2.0, 2.0, 0.5, 1.5)])
    h, g, dd = src.samples([5.0, 5.0], circles=[[6.0, 5.0]], circle_radius=R_OBS)
    assert h.shape == (6,) and g.shape == (2, 6) and dd.shape == (6,)
    np.testing.assert_allclose(np.linalg.norm(g, axis=0), 1.0, atol=1e-9)


# ------------------------------------------------- hard test 3: the exact reduction
def reduction_debug_record(p, gamma, xi_row, *, alpha=ALPHA):
    """Compact numerical record for one fixed state, for debugging a reduction failure.

    Contains everything needed to localise a failure to algebra / objective / constraints /
    solver without re-deriving anything by hand.
    """
    from dr_control.drccp_controller import build_objective
    import cvxpy as cp

    dr = ClfCbfDrccpController(cbf_rate=alpha, wasserstein_r=0.0, epsilon=1.0, max_v=MAX_V)
    qp = ClfCbfQpController(cbf_rate=alpha, max_v=MAX_V)
    xi = np.atleast_2d(xi_row)

    dr_info, qp_info = {}, {}
    u_dr = dr.generate_controller(p, gamma, xi, record=dr_info)
    u_qp = qp.generate_controller(p, gamma, xi, record=qp_info)

    u_nom = nominal_action(p, gamma, MAX_V)
    V, dV = clf_terms(p, gamma, dr.k_v)
    h_crit = float(xi[0, 1])

    return {
        "state": {"p": np.asarray(p).tolist(), "gamma": np.asarray(gamma).tolist()},
        "xi": xi.tolist(),
        "xi_slots": list(XI_SLOTS),
        "u_bar(u_dr)": u_bar_value(u_dr, alpha).tolist(),
        "u_bar(u_qp)": u_bar_value(u_qp, alpha).tolist(),
        "u_nom": u_nom.tolist(),
        "objective_coeffs": {"p0": dr.p0, "p1": 4.0 * h_crit, "p3": 5.0 * h_crit},
        "clf": {"V": V, "grad_V": dV.tolist(), "clf_rate": dr.rateV},
        "drccp_params": {"r_W": dr.wasserstein_r, "epsilon": dr.epsilon, "alpha": alpha,
                         "N": int(xi.shape[0])},
        "nominal_cbf_constraint_value": cbc_value(xi[0], u_qp, alpha),
        "drccp_cbc_at_solution": [cbc_value(row, u_dr, alpha) for row in xi],
        "u_drccp": np.asarray(u_dr).tolist(),
        "u_nominal_qp": np.asarray(u_qp).tolist(),
        "abs_diff": float(np.linalg.norm(np.asarray(u_dr) - np.asarray(u_qp))),
        "status_drccp": dr_info.get("status"),
        "status_nominal_qp": qp_info.get("status"),
        "delta_drccp": dr_info.get("delta"),
        "delta_nominal_qp": qp_info.get("delta"),
    }


REDUCTION_STATES = [
    # (p, gamma, xi = [dh_dt, h, gx, gy])
    ([1.0, 1.0], [8.0, 8.0], [0.0, 2.0, -0.7071, -0.7071]),   # far, inactive barrier
    ([4.0, 5.0], [9.0, 5.0], [0.0, 0.35, -1.0, 0.0]),         # close, barrier binding
    ([4.0, 5.0], [9.0, 5.0], [-0.5, 0.35, -1.0, 0.0]),        # closing obstacle
    ([2.0, 2.0], [2.5, 9.0], [0.0, 1.2, 0.0, -1.0]),          # gradient orthogonal to goal
    ([5.0, 5.0], [5.0, 0.5], [0.3, 0.05, 0.0, 1.0]),          # near contact, receding
]


@pytest.mark.parametrize("p, gamma, xi_row", REDUCTION_STATES)
def test_exact_reduction_drccp_equals_nominal_cbf_qp(p, gamma, xi_row):
    """r_W = 0, eps = 1, N = 1  =>  identical feasible set, identical objective, same u.

    Derivation (see dr_control/baselines.py): the DRCCP constraints collapse to
    { u : u_bar . xi_1 >= 0 }, which is exactly the nominal CBF constraint. Objectives are
    shared via build_objective, so the minimizers must agree.
    """
    rec = reduction_debug_record(p, gamma, xi_row)
    assert rec["status_drccp"] == "optimal", rec
    assert rec["status_nominal_qp"] == "optimal", rec
    # SCS is a first-order solver; 1e-3 is its realistic agreement level on these QPs.
    assert rec["abs_diff"] < 1e-3, rec


def test_reduction_holds_over_random_states():
    rng = np.random.default_rng(7)
    worst, worst_rec = 0.0, None
    for _ in range(40):
        p = rng.uniform(1.0, 9.0, 2)
        gamma = rng.uniform(1.0, 9.0, 2)
        theta = rng.uniform(0, 2 * np.pi)
        xi_row = [rng.uniform(-0.5, 0.5), rng.uniform(0.05, 3.0),
                  np.cos(theta), np.sin(theta)]
        rec = reduction_debug_record(p, gamma, xi_row)
        if rec["status_drccp"] != "optimal" or rec["status_nominal_qp"] != "optimal":
            continue
        if rec["abs_diff"] > worst:
            worst, worst_rec = rec["abs_diff"], rec
    assert worst < 1e-3, worst_rec


# ------------------------------------------------------------------- sundry invariants
def test_drccp_respects_the_action_box():
    dr = ClfCbfDrccpController(max_v=MAX_V)
    xi = build_xi([5.0], np.array([[1.0], [0.0]]), [0.0])
    u = dr.generate_controller([1.0, 1.0], [9.0, 9.0], xi)
    assert np.all(np.abs(u) <= MAX_V + 1e-6)


def test_drccp_records_dcp_error_for_negative_h_without_flooring():
    """h < 0 makes the reference's objective concave. Phase 1 records it; it does not floor h.

    This pins the B.1 behaviour: the failure must be visible, not silently repaired.
    """
    dr = ClfCbfDrccpController(max_v=MAX_V)
    xi = build_xi([-0.4], np.array([[1.0], [0.0]]), [0.0])
    info = {}
    u = dr.generate_controller([1.0, 1.0], [9.0, 9.0], xi, record=info)
    assert info["weights"]["p1"] < 0 and info["weights"]["p3"] < 0
    assert info["status"] == "DCPError"
    assert info["dcp_error"] is True
    np.testing.assert_allclose(u, [0.0, 0.0])


def test_clf_only_is_the_saturated_goal_seeking_action():
    ctrl = ClfOnlyController(max_v=MAX_V)
    u = ctrl.generate_controller([0.0, 0.0], [3.0, 4.0])
    np.testing.assert_allclose(u, [0.6, 0.8], atol=1e-9)


def test_clf_terms_and_nominal_action():
    V, dV = clf_terms([1.0, 2.0], [4.0, 6.0], k_v=2.0)
    assert V == pytest.approx(0.5 * 2.0 * 25.0)
    np.testing.assert_allclose(dV, [2.0 * -3.0, 2.0 * -4.0])
    np.testing.assert_allclose(nominal_action([0.0, 0.0], [0.0, 0.0], MAX_V), [0.0, 0.0])
