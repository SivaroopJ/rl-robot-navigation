"""Week5-Phase7 / Stage 0 / regression gate: the Phase 1-6 system must not move.

Phase 7 adds new modules BESIDE the frozen system and is never allowed to edit it. This test
is what makes that a guarantee rather than an intention: every frozen file is hashed against
MD_files/week5/phase7/FROZEN_MANIFEST.sha256 on every test run.

Covered: robot_env/, evaluation/, dr_control/, optional_navigation/, experiments/exp0..6,
tests/test_dr_control*, tests/test_week4_env.py, results/phase1..6 JSON, config.json.
Not covered, deliberately: requirements.txt, *.log, Phase 7's own files (see the generator's
docstring for the reasoning).

If this test fails, DO NOT regenerate the manifest to make it pass. Find out what moved.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.exp7_0_frozen_manifest import (MANIFEST, compute, read_manifest, verify)


def test_manifest_exists_and_is_populated():
    assert MANIFEST.exists(), f"missing {MANIFEST}; run exp7_0_frozen_manifest --write"
    entries = read_manifest()
    assert len(entries) >= 50, f"manifest looks truncated: {len(entries)} entries"


def test_no_frozen_file_has_been_modified():
    ok, modified, missing, untracked = verify()
    assert not modified, f"FROZEN FILES MODIFIED: {modified}"
    assert not missing, f"FROZEN FILES MISSING: {missing}"
    assert ok


def test_no_new_file_has_appeared_in_the_frozen_area():
    """A new dr_control/ or results/phase*/ file is drift too -- Phase 7 has its own homes."""
    _, _, _, untracked = verify()
    assert not untracked, (
        "new files in the frozen area; Phase 7 output belongs in results/week5_phase7/ "
        f"and MD_files/week5/phase7/: {untracked}")


def test_manifest_covers_the_load_bearing_modules():
    """A manifest that silently stopped matching anything would pass the tests above."""
    entries = read_manifest()
    for required in ("dr_control/drccp_controller.py", "dr_control/policy.py",
                     "dr_control/velocity_tracker.py", "dr_control/estimated_cbf.py",
                     "dr_control/lidar_cbf.py", "optional_navigation/planner.py",
                     "robot_env/robot_nav_env.py", "robot_env/lidar_core.py",
                     "evaluation/shortest_path.py", "evaluation/evaluate_week4.py",
                     "config.json"):
        assert required in entries, f"{required} is not in the frozen manifest"


def test_manifest_digests_are_reproducible():
    """Hashing twice must agree -- guards against a non-deterministic file list."""
    assert compute() == compute()


# --------------------------------------------------------------- the frozen constants
def test_frozen_controller_parameters_are_unchanged():
    """Belt and braces: the numbers Phases 1-6 were run at, asserted directly."""
    from dr_control.drccp_controller import (K_V_REFERENCE, XI_DIM, XI_SLOTS,
                                             ClfCbfDrccpController)
    c = ClfCbfDrccpController()
    assert (c.rateV, c.rateh, c.wasserstein_r, c.epsilon) == (1.0, 0.4, 0.004, 0.1)
    assert (c.max_v, c.k_v, c.p0, c.n_keep, c.solver) == (1.0, 0.05, 3.0, 5, "SCS")
    assert K_V_REFERENCE == 0.05
    assert XI_SLOTS == ("dh_dt", "h", "grad_h_x", "grad_h_y") and XI_DIM == 4


def test_frozen_fallback_is_still_u_equals_zero():
    """Direction 1 replaces this ONLY in new code. The frozen behaviour is the control arm."""
    from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
    # a barrier that no admissible u can satisfy: gradient +x, huge required rate
    xi = build_xi(h=[0.0], grad_h=np.array([[1.0], [0.0]]), dh_dt=[-50.0])
    ctrl = ClfCbfDrccpController(max_v=1.0)
    info = {}
    u = ctrl.generate_controller(np.zeros(2), np.array([5.0, 0.0]), xi, record=info)
    assert info["status"] != "optimal"
    np.testing.assert_array_equal(u, np.zeros(2))


# --------------------------------------------------------------- the tau identity
@pytest.mark.parametrize("solver,tol", [("SCS", 1e-4), ("CLARABEL", 1e-6)])
def test_cvar_collapses_to_the_worst_sample_when_eps_N_lt_1(solver, tol):
    """The identity Directions 1 and 2 both rest on:

        eps*N < 1   =>   DR constraint  <=>  min_i CBC_i >= r_W*||u_bar||_inf/eps

    Checked in BOTH directions against the CVXPY feasible set.

    TOLERANCE IS PER SOLVER, ON PURPOSE. Measured worst deviation of the optimum from the
    reduced constraint is 1.2e-6 with SCS (first-order) and 1.9e-8 with CLARABEL (interior
    point), and at the optimum t* == tau to four decimals with sum_i s_i == 0 -- exactly the
    structure the derivation predicts. Splitting the tolerance by solver is what distinguishes
    "the identity is wrong" from "SCS is a first-order solver"; a single loose tolerance would
    hide the difference.
    """
    import cvxpy as cp
    rng = np.random.default_rng(0)
    alpha, r_W, eps, max_v, N = 0.4, 0.004, 0.1, 1.0, 5
    assert eps * N < 1
    tau = r_W / eps

    n_feasible = n_infeasible = 0
    worst = 0.0
    for trial in range(40):
        # alternate an EASY family (gradients in a cone -> usually feasible) with an
        # ADVERSARIAL one (gradients anywhere -> usually infeasible), so both branches run
        if trial % 2 == 0:
            ang = rng.uniform(-0.6, 0.6, N)
            xi = np.column_stack([rng.uniform(-0.3, 0.0, N), rng.uniform(0.2, 1.5, N),
                                  np.cos(ang), np.sin(ang)])
        else:
            ang = rng.uniform(0, 2 * np.pi, N)
            xi = np.column_stack([rng.uniform(-0.8, 0.0, N), rng.uniform(0.0, 1.5, N),
                                  np.cos(ang), np.sin(ang)])

        u = cp.Variable(2)
        si = cp.Variable(N)
        t = cp.Variable()
        u_bar = cp.hstack([1.0, alpha, u[0], u[1]])
        cons = [r_W * cp.abs(u_bar) / eps <= (t - cp.sum(si) / (N * eps)) * np.ones(4),
                si >= 0, cp.abs(u) <= max_v]
        cons += [si[i] >= t - u_bar @ xi[i] for i in range(N)]
        c = rng.normal(size=2)                       # random objective -> different faces
        prob = cp.Problem(cp.Maximize(c @ u), cons)
        prob.solve(solver=solver)

        if prob.status == "optimal":
            n_feasible += 1
            uv = np.asarray(u.value).reshape(2)
            m = float(np.min(xi @ np.array([1.0, alpha, uv[0], uv[1]])))
            worst = min(worst, m - tau)
            assert m >= tau - tol, f"DRCCP optimum violates the reduced constraint by {tau-m:g}"
        elif "infeasible" in str(prob.status):
            n_infeasible += 1
            # converse: no admissible u may satisfy the reduced form either
            for _ in range(400):
                uu = rng.uniform(-max_v, max_v, 2)
                if float(np.min(xi @ np.array([1.0, alpha, uu[0], uu[1]]))) >= tau + tol:
                    raise AssertionError("reduced-feasible point exists but DRCCP is infeasible")

    # the test must not silently become vacuous
    assert n_feasible >= 5 and n_infeasible >= 5, (n_feasible, n_infeasible)
    assert worst >= -tol


def test_cvar_does_not_collapse_when_eps_N_ge_1():
    """NEGATIVE CONTROL. With eps*N >= 1 the CVaR averages the tail instead of taking the
    minimum, so the reduced form must be STRICTLY LOOSER -- there must exist a u that the
    DRCCP admits and `min_i CBC_i >= tau` rejects. If this ever stops holding, the reduction
    is being applied outside its precondition."""
    import cvxpy as cp
    rng = np.random.default_rng(1)
    alpha, r_W, eps, max_v, N = 0.4, 0.004, 0.5, 1.0, 5      # eps*N = 2.5 >= 1
    tau = r_W / eps
    found_looser = False
    for _ in range(30):
        ang = rng.uniform(0, 2 * np.pi, N)
        xi = np.column_stack([rng.uniform(-0.8, 0.0, N), rng.uniform(0.0, 1.0, N),
                              np.cos(ang), np.sin(ang)])
        u = cp.Variable(2)
        si = cp.Variable(N)
        t = cp.Variable()
        u_bar = cp.hstack([1.0, alpha, u[0], u[1]])
        cons = [r_W * cp.abs(u_bar) / eps <= (t - cp.sum(si) / (N * eps)) * np.ones(4),
                si >= 0, cp.abs(u) <= max_v]
        cons += [si[i] >= t - u_bar @ xi[i] for i in range(N)]
        prob = cp.Problem(cp.Maximize(rng.normal(size=2) @ u), cons)
        prob.solve(solver="CLARABEL")
        if prob.status == "optimal":
            uv = np.asarray(u.value).reshape(2)
            if float(np.min(xi @ np.array([1.0, alpha, uv[0], uv[1]]))) < tau - 1e-6:
                found_looser = True
                break
    assert found_looser, "eps*N >= 1 behaved like the collapsed form; check the precondition"


def test_tau_is_constant_under_the_frozen_configuration():
    """||u_bar||_inf = max(1, alpha, |u_x|, |u_y|) = 1 for every admissible u."""
    rng = np.random.default_rng(0)
    alpha, max_v = 0.4, 1.0
    for _ in range(1000):
        u = rng.uniform(-max_v, max_v, 2)
        assert max(1.0, alpha, abs(u[0]), abs(u[1])) == 1.0


def test_the_reduction_precondition_is_stated_and_checkable():
    """eps*n_keep < 1 and max_v <= max(1, alpha): Stage 1's fast path may assume no more."""
    from dr_control.drccp_controller import ClfCbfDrccpController
    c = ClfCbfDrccpController()
    assert c.epsilon * c.n_keep < 1.0
    assert c.max_v <= max(1.0, c.rateh)
