"""Week5-Phase7 / Stage 4 gate: the M2 recovery ladder.

    1. INERTNESS -- on feasible steps every arm is bit-identical to B4
    2. rung ordering -- the R0/R1/R2 feasible sets are nested, and the rung follows m*
    3. margin/tier recomputation is done from the RETURNED action, not from the solver
    4. genuinely infeasible M2 states are handled: R2 always returns an action
    5. the frozen controller and the u = 0 fallback are untouched outside treatment
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.recovery import (ETA, MODES, RecoveryLadder, cbc, decay_floor, r15_admission,
                                 tier_of)

ALPHA, TAU, MAX_V = 0.4, 0.04, 1.0
P, G = np.array([4.0, 4.0]), np.array([8.0, 7.0])


def feasible_xi():
    ang = np.linspace(-0.5, 0.5, 5)
    return build_xi([0.8] * 5, np.stack([np.cos(ang), np.sin(ang)]), [-0.1] * 5)


def infeasible_m2_xi():
    """A genuine MULTI-constraint conflict: opposed directions, each satisfiable alone."""
    return build_xi([0.05, 0.05], np.array([[1.0, -1.0], [0.0, 0.0]]), [-0.6, -0.6])


def run(xi, mode, u_prev=None):
    c = ClfCbfDrccpController()
    if u_prev is not None:
        c.prev_u = np.asarray(u_prev, float).copy()
    lad = RecoveryLadder(c, mode=mode)
    info = {}
    u = c.generate_controller(P, G, xi, record=info)
    lad.detach()
    return np.asarray(u, float), info, lad.records[-1]


# ------------------------------------------------------------------ 1: inertness
@pytest.mark.parametrize("mode", MODES)
def test_inert_on_feasible_steps(mode):
    """Bit-identical to the untouched frozen controller whenever the step is feasible."""
    xi = feasible_xi()
    base = ClfCbfDrccpController().generate_controller(P, G, xi)
    u, info, r = run(xi, mode)
    assert info["status"] == "optimal"
    np.testing.assert_array_equal(u, base)
    assert r["rung"] == "R0" and not r["recovered"]


def test_inert_over_a_sequence_of_feasible_steps():
    rng = np.random.default_rng(0)
    c0, c1 = ClfCbfDrccpController(), ClfCbfDrccpController()
    lad = RecoveryLadder(c1, mode="ladder")
    for _ in range(20):
        ang = rng.uniform(-0.6, 0.6, 5)
        xi = build_xi(rng.uniform(0.4, 1.5, 5), np.stack([np.cos(ang), np.sin(ang)]),
                      rng.uniform(-0.2, 0.0, 5))
        np.testing.assert_array_equal(c1.generate_controller(P, G, xi),
                                      c0.generate_controller(P, G, xi))
    lad.detach()


# ------------------------------------------------------------------ 2: rung ordering
def test_r0_r1_r2_feasible_sets_are_nested():
    """{m >= tau} subset {m >= 0} subset {any action}. Checked on a grid, no solver."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        ang = rng.uniform(0, 2 * np.pi, 3)
        xi = build_xi(rng.uniform(0.0, 1.0, 3), np.stack([np.cos(ang), np.sin(ang)]),
                      rng.uniform(-1.0, 0.0, 3))
        gr = np.linspace(-MAX_V, MAX_V, 21)
        r0 = r1 = 0
        for ux in gr:
            for uy in gr:
                m = cbc(xi, [ux, uy], ALPHA)
                r0 += m >= TAU
                r1 += m >= 0.0
        assert r1 >= r0                       # R1's set contains R0's


def test_rung_follows_the_max_min_margin():
    """R2 is reached only when no action attains m >= 0, i.e. when m* < 0."""
    u, info, r = run(infeasible_m2_xi(), "ladder")
    assert r["infeasible"] and r["recovered"]
    if r["rung"] == "R1":
        assert r["m"] >= -1e-6
    else:
        assert r["rung"].startswith("R2") and r["m"] < TAU


def test_r1_arm_falls_back_to_u_zero_when_r1_cannot_help():
    """The R1-only arm must NOT invent an action; it returns the frozen u = 0."""
    xi = build_xi([0.0], np.array([[1.0], [0.0]]), [-50.0])      # unsatisfiable at any floor
    u, info, r = run(xi, "R1")
    assert r["infeasible"] and not r["recovered"] and r["rung"] == "fallback_u0"
    np.testing.assert_array_equal(u, np.zeros(2))


# ------------------------------------------------------------------ 3: margin and tier
def test_margin_is_recomputed_from_the_returned_action():
    xi = infeasible_m2_xi()
    u, info, r = run(xi, "ladder")
    assert r["m"] == pytest.approx(cbc(xi, u, ALPHA), abs=1e-9)


def test_tier_boundaries():
    assert tier_of(TAU, TAU) == 0 and tier_of(TAU + 1, TAU) == 0
    assert tier_of(0.0, TAU) == 1 and tier_of(TAU - 1e-12, TAU) == 1
    assert tier_of(-1e-12, TAU) == 2 and tier_of(-1.0, TAU) == 2


def test_decay_floor_matches_the_closed_form():
    """h_dot >= -alpha h + m has floor m/alpha for m < 0, and none for m >= 0."""
    assert decay_floor(-0.2, 0.4) == pytest.approx(-0.5)
    assert decay_floor(0.0, 0.4) == 0.0 and decay_floor(0.1, 0.4) == 0.0
    h, m, a, dt = 0.5, -0.2, 0.4, 0.001
    for _ in range(200000):
        h += (-a * h + m) * dt
    assert h == pytest.approx(decay_floor(m, a), abs=1e-3)


def test_no_record_asserts_safety():
    """The records carry m and tier -- never a boolean called 'safe'."""
    _, _, r = run(infeasible_m2_xi(), "ladder")
    assert "safe" not in {k.lower() for k in r}
    assert "m" in r and "tier" in r


# ------------------------------------------------------------------ 4: genuine M2 states
def test_r2_always_returns_an_action_on_adversarial_conflicts():
    rng = np.random.default_rng(2)
    n_inf = 0
    for _ in range(60):
        ang = rng.uniform(0, 2 * np.pi, 4)
        xi = build_xi(rng.uniform(0.0, 0.3, 4), np.stack([np.cos(ang), np.sin(ang)]),
                      rng.uniform(-1.2, -0.4, 4))
        u, info, r = run(xi, "R2")
        assert np.all(np.isfinite(u)) and np.max(np.abs(u)) <= MAX_V + 1e-9
        if r["infeasible"]:
            n_inf += 1
            assert r["recovered"] and r["rung"].startswith("R2")
            assert r["m"] == pytest.approx(cbc(xi, u, ALPHA), abs=1e-6)
    assert n_inf >= 10


def test_recovered_action_respects_the_action_box():
    u, _, _ = run(infeasible_m2_xi(), "ladder")
    assert np.max(np.abs(u)) <= MAX_V + 1e-9


# ------------------------------------------------------------------ 5: frozen untouched
def test_frozen_controller_is_restored_on_detach():
    c = ClfCbfDrccpController()
    orig = c.generate_controller
    lad = RecoveryLadder(c, mode="ladder")
    assert c.generate_controller is not orig
    lad.detach()
    assert c.generate_controller == orig


def test_frozen_defaults_untouched():
    c = ClfCbfDrccpController()
    RecoveryLadder(c, mode="ladder").detach()
    assert (c.rateh, c.wasserstein_r, c.epsilon, c.max_v) == (0.4, 0.004, 0.1, 1.0)
    assert ClfCbfDrccpController().rateh == 0.4


def test_r15_admission_checks_are_enforced():
    ok, why = r15_admission(np.array([[0.0, 0.5, 1.0, 0.0]]), alpha=0.4, dt=0.1)
    assert ok and not why
    ok, why = r15_admission(np.array([[0.0, -0.1, 1.0, 0.0]]), alpha=0.4, dt=0.1)
    assert not ok and "h_min < 0" in why[0]
    ok, why = r15_admission(np.array([[0.0, 0.5, 1.0, 0.0]]), alpha=12.0, dt=0.1)
    assert not ok


def test_eta_is_declared_not_tuned():
    assert ETA == 1e-9
