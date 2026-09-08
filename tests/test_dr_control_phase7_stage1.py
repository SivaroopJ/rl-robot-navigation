"""Week5-Phase7 / Stage 1 gate: the accelerated controller.

Hard tests:
    1. the exact reduction is a THEOREM, and its preconditions are enforced, not assumed
    2. V1's parameterized problem is DPP and does NOT re-canonicalize
    3. h_crit < 0 delegates to the frozen controller, preserving the DCPError behaviour
    4. the fallback semantics are unchanged: non-optimal => u = 0
    5. the accelerated variants are CLOSER to the true optimum than the frozen reference is
    6. the frozen controller is untouched (covered additionally by test_frozen_phase1_6)
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.fast_drccp import (OSQP_STATUS, VARIANTS, FastClfCbfDrccpController,
                                   reference_objective)

ALPHA, TAU, MAX_V = 0.4, 0.04, 1.0


def random_xi(rng, n=5, *, feasible=True):
    ang = rng.uniform(-0.7, 0.7, n) if feasible else rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0.2, 1.5, n) if feasible else rng.uniform(0.0, 0.6, n)
    dd = rng.uniform(-0.3, 0.0, n) if feasible else rng.uniform(-0.8, -0.3, n)
    return build_xi(h, np.stack([np.cos(ang), np.sin(ang)]), dd)


# ------------------------------------------------------- 1: the reduction and its preconditions
def test_reduced_variant_refuses_when_eps_times_n_keep_is_at_least_one():
    """Above eps*N = 1 the CVaR no longer collapses to the worst sample; the class must
    refuse rather than silently apply an invalid reduction."""
    with pytest.raises(ValueError, match="eps.*n_keep"):
        FastClfCbfDrccpController(variant="reduced_osqp", epsilon=0.5, n_keep=5)
    FastClfCbfDrccpController(variant="param_scs", epsilon=0.5, n_keep=5)   # general form is fine


def test_reduced_variant_refuses_when_max_v_exceeds_max_one_alpha():
    """Above it, ||u_bar||_inf becomes u-dependent and tau is no longer constant."""
    with pytest.raises(ValueError, match="max_v"):
        FastClfCbfDrccpController(variant="reduced_osqp", max_v=1.5, cbf_rate=0.4)
    FastClfCbfDrccpController(variant="reduced_osqp", max_v=1.5, cbf_rate=2.0)  # alpha >= max_v


def test_tau_matches_the_frozen_configuration():
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    assert c.tau == pytest.approx(0.004 / 0.1)


def test_reduced_feasible_set_equals_the_frozen_one():
    """The reduction claim, tested as a set equality on the FROZEN controller's own answers:
    whenever the frozen solver says infeasible, no admissible u meets min_i CBC_i >= tau, and
    whenever it solves, the returned u does."""
    rng = np.random.default_rng(0)
    ref = ClfCbfDrccpController()
    n_inf = n_ok = 0
    for k in range(60):
        xi = random_xi(rng, feasible=(k % 2 == 0))
        p, gamma = rng.uniform(1, 9, 2), rng.uniform(1, 9, 2)
        ref.reset()
        info = {}
        u = ref.generate_controller(p, gamma, xi, record=info)
        kept = np.asarray(info["xi_kept"])
        if info["status"] == "optimal":
            n_ok += 1
            ub = np.array([1.0, ALPHA, u[0], u[1]])
            assert float(np.min(kept @ ub)) >= TAU - 5e-4
        elif "infeasible" in str(info["status"]):
            n_inf += 1
            grid = np.linspace(-MAX_V, MAX_V, 41)
            best = max(float(np.min(kept @ np.array([1.0, ALPHA, ux, uy])))
                       for ux in grid for uy in grid)
            assert best < TAU + 1e-3, f"reduced-feasible point exists but frozen said infeasible"
    assert n_ok >= 10 and n_inf >= 5, (n_ok, n_inf)


# ------------------------------------------------------- 2: V1 is DPP and reuses the problem
def test_parameterized_problem_is_dpp():
    c = FastClfCbfDrccpController(variant="param_scs")
    prob, _, _ = c._build_cvx(5)
    assert prob.is_dcp(dpp=True)


def test_parameterized_problem_is_built_once_per_sample_count():
    rng = np.random.default_rng(1)
    c = FastClfCbfDrccpController(variant="param_scs")
    for _ in range(10):
        c.generate_controller(np.array([4.0, 4.0]), np.array([8.0, 8.0]), random_xi(rng))
    assert set(c._cvx_cache) == {5}, "a new problem was built per step"


def test_osqp_model_is_built_once_per_sample_count():
    rng = np.random.default_rng(2)
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    for n in (3, 5, 5, 5, 3):
        c.generate_controller(np.array([4.0, 4.0]), np.array([8.0, 8.0]), random_xi(rng, n))
    assert set(c._osqp_cache) == {3, 5}


def test_csc_pattern_is_stable_when_a_gradient_component_is_exactly_zero():
    """An exact zero in grad_h must NOT be dropped from the sparse pattern; if it were, OSQP
    would need a full re-setup every time and the speedup would silently vanish."""
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    xi = build_xi([0.5] * 5, np.array([[1.0, 0.0, 1.0, 0.0, 1.0],
                                       [0.0, 1.0, 0.0, 1.0, 0.0]]), [-0.1] * 5)
    for _ in range(5):
        c.generate_controller(np.array([4.0, 4.0]), np.array([8.0, 8.0]), xi)
    A_ind, A_ptr, _, _, shape = c._structure(5)
    # N + 4 rows, 3 cols; nnz = (N grad_h + grad_V + box) x 2 cols + (CLF slack, delta >= 0)
    assert shape == (9, 3)
    assert len(A_ind) == 2 * 5 + 6 == 16 and A_ptr[-1] == 16
    assert set(c._osqp_cache) == {5}, "the pattern changed and forced a re-setup"


# ------------------------------------------------------- 3: the delegated branch
def test_negative_h_crit_delegates_and_reproduces_the_dcp_error():
    """The reference's p1 = 4h, p3 = 5h go negative for h < 0 and CVXPY raises DCPError.
    Phase 1 recorded that deliberately, so an accelerated path must reproduce it, not fix it."""
    xi = build_xi([-0.05, 0.4, 0.5, 0.6, 0.7],
                  np.array([[1.0, 0.9, 0.8, 0.7, 0.6], [0.0, 0.44, 0.6, 0.71, 0.8]]),
                  [0.0] * 5)
    p, gamma = np.array([4.0, 4.0]), np.array([7.0, 6.0])
    ref_info, var_info = {}, {}
    u_ref = ClfCbfDrccpController().generate_controller(p, gamma, xi, record=ref_info)
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    u_var = c.generate_controller(p, gamma, xi, record=var_info)
    assert ref_info["status"] == "DCPError"
    assert var_info["status"] == "DCPError" and var_info["delegated"]
    assert c.n_delegated == 1
    np.testing.assert_array_equal(u_var, u_ref)


def test_delegation_keeps_prev_u_synchronised():
    rng = np.random.default_rng(3)
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    xi_ok = random_xi(rng)
    u = c.generate_controller(np.array([4.0, 4.0]), np.array([8.0, 8.0]), xi_ok)
    np.testing.assert_array_equal(c.prev_u, u)
    xi_bad = build_xi([-0.1] + [0.5] * 4, np.array([[1.0] * 5, [0.0] * 5]), [0.0] * 5)
    c.generate_controller(np.array([4.0, 4.0]), np.array([8.0, 8.0]), xi_bad)
    np.testing.assert_array_equal(c.prev_u, np.zeros(2))     # DCPError -> u = 0


# ------------------------------------------------------- 4: fallback semantics unchanged
@pytest.mark.parametrize("variant", VARIANTS)
def test_infeasible_step_still_commands_zero(variant):
    """Direction 1 changes this. Direction 2 must NOT."""
    xi = build_xi([0.0], np.array([[1.0], [0.0]]), [-50.0])
    c = FastClfCbfDrccpController(variant=variant)
    info = {}
    u = c.generate_controller(np.zeros(2), np.array([5.0, 0.0]), xi, record=info)
    assert info["status"] != "optimal"
    np.testing.assert_array_equal(u, np.zeros(2))
    assert c.solve_fail


@pytest.mark.parametrize("variant", VARIANTS)
def test_actions_stay_inside_the_action_box(variant):
    rng = np.random.default_rng(4)
    c = FastClfCbfDrccpController(variant=variant)
    for _ in range(30):
        u = c.generate_controller(rng.uniform(1, 9, 2), rng.uniform(1, 9, 2), random_xi(rng))
        assert np.max(np.abs(u)) <= c.max_v + 1e-12


def test_osqp_status_map_is_total_over_the_documented_statuses():
    for s in ("solved", "solved inaccurate", "primal infeasible",
              "primal infeasible inaccurate", "dual infeasible",
              "maximum iterations reached"):
        assert s in OSQP_STATUS


# ------------------------------------------------------- 5: accuracy against a neutral arbiter
def test_accelerated_variants_are_nearer_the_true_optimum_than_the_frozen_reference():
    """The finding that forced the E1' amendment.

    The pre-registered E1 threshold (||u - u_frozen|| <= 1e-6) implicitly treated the frozen
    output as ground truth. It is not: SCS is first-order. Measured against the FROZEN
    formulation solved by CLARABEL at tol 1e-14 -- a reference belonging to neither
    implementation -- the frozen controller's own error reaches ~1e-4 while the reduced QP's
    is ~1e-9. A threshold tighter than the reference's own noise cannot be met by any correct
    implementation, so the criterion moved to the arbiter and BOTH numbers are reported.
    """
    from experiments.exp7_1_equivalence import arbiter_solve
    rng = np.random.default_rng(5)
    d_ref, d_var = [], []
    for _ in range(25):
        xi = random_xi(rng)
        p, gamma = rng.uniform(1, 9, 2), rng.uniform(1, 9, 2)
        u_star, _, st = arbiter_solve(p, gamma, xi, np.zeros(2))
        if u_star is None or st != "optimal":
            continue
        u_ref = ClfCbfDrccpController().generate_controller(p, gamma, xi)
        u_var = FastClfCbfDrccpController(variant="reduced_osqp").generate_controller(
            p, gamma, xi)
        d_ref.append(np.max(np.abs(u_ref - u_star)))
        d_var.append(np.max(np.abs(u_var - u_star)))
    assert len(d_ref) >= 15
    assert max(d_var) < max(d_ref), (max(d_var), max(d_ref))
    assert max(d_var) < 1e-7, max(d_var)


def test_reference_objective_matches_the_frozen_weights():
    u, delta, u_prev, u_nom, h = np.array([0.3, -0.2]), 0.7, np.zeros(2), np.array([1.0, 0.0]), 0.5
    expected = (3.0 * float(u @ u) + 4.0 * h * float((u - u_nom) @ (u - u_nom))
                + 5.0 * h * delta ** 2)
    assert reference_objective(u, delta, u_prev=u_prev, u_nom=u_nom,
                               h_crit=h) == pytest.approx(expected)


# ------------------------------------------------------- 6: interface parity
@pytest.mark.parametrize("variant", VARIANTS)
def test_interface_matches_the_frozen_controller(variant):
    c = FastClfCbfDrccpController(variant=variant)
    for attr in ("generate_controller", "reset", "prev_u", "solve_fail",
                 "rateV", "rateh", "wasserstein_r", "epsilon", "max_v", "k_v", "n_keep"):
        assert hasattr(c, attr), attr
    assert (c.rateh, c.wasserstein_r, c.epsilon, c.max_v) == (0.4, 0.004, 0.1, 1.0)
    c.reset()
    np.testing.assert_array_equal(c.prev_u, np.zeros(2))


@pytest.mark.parametrize("variant", VARIANTS)
def test_records_the_same_diagnostic_fields(variant):
    """EpisodeDiagnostics.record reads these; a missing key would silently become NaN."""
    rng = np.random.default_rng(6)
    info = {}
    FastClfCbfDrccpController(variant=variant).generate_controller(
        np.array([4.0, 4.0]), np.array([8.0, 8.0]), random_xi(rng), record=info)
    for key in ("status", "h_crit", "weights", "xi_kept", "u_nom", "delta",
                "total_time", "solver_time", "canon_time"):
        assert key in info, key
