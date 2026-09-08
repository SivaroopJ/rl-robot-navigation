"""Week5-Phase7 / Stage 2b gate: the diagnostic machinery, not a controller.

Stage 2b substitutes one channel of the recorded problem and re-tests feasibility. Two pieces
of machinery carry that argument and are pinned here:

    1. the fast exact 2-D feasibility test agrees with the LP definition
    2. the arm constructions change ONLY dh_dt, leaving h, grad_h and the row set untouched

No controller is built or modified by Stage 2b; V0 stays frozen and is covered by
tests/test_frozen_phase1_6.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.exp7_2b_velocity_counterfactual import (ALPHA, MAX_V, OBS_R, R_ROBOT, TAU,
                                                         build_arms, feas, feas_lp,
                                                         owner_dh_dt)


def random_xi(rng, n=None):
    n = int(rng.integers(1, 6)) if n is None else n
    ang = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([rng.uniform(-3.0, 0.2, n), rng.uniform(-0.05, 1.6, n),
                            np.cos(ang), np.sin(ang)])


def test_fast_feasibility_agrees_with_the_lp_definition():
    """The LP is the definition; the vertex-enumeration test exists only for speed.

    Both branches must be exercised, or the test could pass by always answering one way.
    """
    rng = np.random.default_rng(0)
    n_feasible = n_infeasible = 0
    for _ in range(1500):
        xi = random_xi(rng)
        a, b = feas(xi), feas_lp(xi)
        assert a == b, xi
        n_feasible += a
        n_infeasible += not a
    assert n_feasible > 100 and n_infeasible > 100, (n_feasible, n_infeasible)


def test_fast_feasibility_on_hand_built_cases():
    # a single row the box can satisfy comfortably
    assert feas(np.array([[0.0, 1.0, 1.0, 0.0]]))
    # a single row demanding more closing-rate compensation than the box can deliver
    assert not feas(np.array([[-3.0, 0.0, 1.0, 0.0]]))
    # two rows demanding opposite directions
    assert not feas(np.array([[-0.9, 0.0, 1.0, 0.0], [-0.9, 0.0, -1.0, 0.0]]))


# ------------------------------------------------------------------ arm construction
def test_arms_change_only_dh_dt():
    """h, grad_h and the row set are the SAME object in every arm -- the whole point of the
    counterfactual is that only the velocity channel moves."""
    rng = np.random.default_rng(1)
    xi = random_xi(rng, 5)
    ages = np.arange(5)
    p = np.array([5.0, 5.0])
    hist = {a: np.array([[6.0, 5.0], [2.0, 2.0]]) for a in range(5)}
    vel = np.array([[0.5, 0.0], [0.0, 0.0]])
    arms, _ = build_arms(xi, ages, p, hist, vel, 0.675, 0.15)
    for name, x in arms.items():
        assert x.shape == xi.shape, name
        np.testing.assert_array_equal(x[:, 1:], xi[:, 1:])       # h and grad_h untouched
    np.testing.assert_array_equal(arms["A1_frozen_estimate"], xi)
    np.testing.assert_array_equal(arms["A3_zero"][:, 0], np.zeros(5))


def test_cap_arm_never_loosens_below_the_physical_bound():
    """A4 clamps the PROJECTION at -v_phys. It may only raise dh_dt, never lower it."""
    rng = np.random.default_rng(2)
    xi = random_xi(rng, 5)
    arms, _ = build_arms(xi, np.arange(5), np.zeros(2), {}, None, 0.675, 0.15)
    a4 = arms["A4_capped_estimate"][:, 0]
    assert np.all(a4 >= xi[:, 0] - 1e-15)
    assert np.all(a4 >= -0.675 - 1e-12)


def test_cap_is_a_bound_on_clamping_the_velocity_vector():
    """Arm 4 clamps the PROJECTION and is the MORE CONSERVATIVE of the two clamps.

    With s = min(1, v_phys/||v||): the vector clamp gives s*dh_dt, the projection clamp gives
    max(dh_dt, -v_phys), and ||grad_h|| = 1 implies |dh_dt| <= ||v||, hence
    s*dh_dt >= max(dh_dt, -v_phys) FOR CLOSING ROWS (dh_dt < 0). Closing rows are the only
    ones that can tighten a constraint, so whatever infeasibility arm 4 removes is a LOWER
    BOUND on what clamping the velocity vector would remove. For a RECEDING row (dh_dt >= 0)
    the inequality does not hold and does not need to: the projection clamp is a no-op there,
    and a receding row only loosens its constraint. Both halves are asserted below.
    """
    v_phys = 0.675
    g = np.array([1.0, 0.0])

    def proj_clamp(v):
        return max(-float(g @ v), -v_phys)

    def vec_clamp(v):
        return -float(g @ (v * min(1.0, v_phys / np.linalg.norm(v))))

    # parallel and CLOSING (dh_dt < 0): the two coincide
    v = np.array([2.0, 0.0])
    assert -float(g @ v) < 0
    assert proj_clamp(v) == pytest.approx(vec_clamp(v))

    # the inequality on CLOSING rows; no-op on receding rows
    rng = np.random.default_rng(7)
    n_closing = n_receding = 0
    for _ in range(2000):
        th = rng.uniform(0, 2 * np.pi)
        g = np.array([np.cos(th), np.sin(th)])
        v = rng.normal(scale=1.5, size=2)
        dh = -float(g @ v)
        if dh < 0:
            n_closing += 1
            assert vec_clamp(v) >= proj_clamp(v) - 1e-12, (g, v, dh)
        else:
            n_receding += 1
            assert proj_clamp(v) == pytest.approx(dh)
    assert n_closing > 200 and n_receding > 200, (n_closing, n_receding)


# ------------------------------------------------------------------ ground-truth attribution
def test_owner_attribution_picks_the_obstacle_whose_surface_the_point_lies_on():
    """A buffered point on an obstacle surface gets that obstacle's velocity; a point on no
    surface is static structure and gets zero."""
    p = np.array([0.0, 0.0])
    # a surface point 1.0 m away along +x, on an obstacle centred at 1.3 (radius 0.3)
    h = 1.0 - R_ROBOT
    xi = np.array([[0.0, h, -1.0, 0.0]])          # grad points from obstacle back to the robot
    hist = {0: np.array([[1.3, 0.0]])}
    vel = np.array([[-0.5, 0.0]])                 # closing on the robot
    dh, n_dyn = owner_dh_dt(xi, np.array([0]), p, hist, vel, 0.15)
    assert n_dyn == 1
    assert dh[0] == pytest.approx(-float(xi[0, 2:4] @ vel[0]))
    # move the obstacle far away: the point now lies on no surface -> static, v = 0
    dh2, n_dyn2 = owner_dh_dt(xi, np.array([0]), p, {0: np.array([[9.0, 9.0]])}, vel, 0.15)
    assert n_dyn2 == 0 and dh2[0] == 0.0


def test_buffered_point_reconstruction_is_exact():
    """q = p - (h + r_robot) * grad_h inverts the frozen sample construction."""
    p = np.array([3.0, 4.0])
    q_true = np.array([3.0, 5.2])
    dist = float(np.linalg.norm(q_true - p))
    g = (p - q_true) / dist
    xi = np.array([[0.0, dist - R_ROBOT, g[0], g[1]]])
    q = p[None, :] - (xi[:, 1:2] + R_ROBOT) * xi[:, 2:4]
    np.testing.assert_allclose(q[0], q_true, atol=1e-12)


def test_constants_match_the_frozen_configuration():
    assert (ALPHA, TAU, MAX_V, R_ROBOT, OBS_R) == (0.4, 0.04, 1.0, 0.3, 0.3)
