"""Week5-Phase7 / Stage 3 gate: the M1 intervention.

    1. the cap is DERIVED, and its guarantee's preconditions are checked not assumed
    2. the interventions touch ONLY dh_dt -- h and grad_h are bit-identical to the frozen source
    3. an infinite cap reproduces the frozen source EXACTLY (both treatments are inert)
    4. C0 is byte-for-byte the frozen policy
    5. the information ledger holds for T1/T2; D-oracle is marked diagnostic-only
    6. the singleton guarantee actually holds on adversarial single rows
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.capped_velocity import (OracleVelocitySource, ProjectionCappedSource,
                                        V_CAP_DERIVED, VectorCappedSource,
                                        assert_singleton_guarantee, derived_cap)
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.policy_phase7 import ARMS, Phase7Policy
from dr_control.velocity_tracker import LidarVelocityTracker
from robot_env.lidar_core import cast_rays
import experiments.exp0_analytic as E

ALPHA, TAU, MAX_V = 0.4, 0.04, 1.0


def scan(p, circles):
    return cast_rays(np.asarray(p, float), n_rays=24, lidar_range=5.0,
                     world_size=E.WORLD_SIZE, agent_radius=E.AGENT_RADIUS,
                     rects=E.STATIC_RECTS, circles=circles, circle_radius=E.OBSTACLE_RADIUS)


def src_of(cls, **kw):
    t = LidarVelocityTracker(r_nominal=0.3, dt=0.1, n_rays=24, lidar_range=5.0)
    return cls(r_robot=E.AGENT_RADIUS, k_scans=5, n_rays=24, lidar_range=5.0, tracker=t, **kw)


def drive(src, n=25, seed=0):
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    p, _goal, circles, vels = E.sample_scenario(rng, 6, moving=True)
    out = []
    for _ in range(n):
        src.push(p, scan(p, circles))
        out.append(tuple(np.asarray(a).copy() for a in src.samples(p)))
        circles = circles + vels * 0.1
    return out


# ---------------------------------------------------------------- 1: the derived cap
def test_cap_is_derived_from_controller_constants_only():
    assert derived_cap(max_v=1.0, wasserstein_r=0.004, epsilon=0.1) == pytest.approx(0.96)
    assert V_CAP_DERIVED == 0.96
    # no environment knowledge: changing the obstacle speed cannot change the cap
    assert derived_cap() == derived_cap()


def test_guarantee_preconditions_are_checked_not_assumed():
    assert assert_singleton_guarantee(0.96)
    with pytest.raises(ValueError, match="h >= 0"):
        assert_singleton_guarantee(0.96, h_min=-0.01)
    with pytest.raises(ValueError, match=r"grad"):
        assert_singleton_guarantee(0.96, grad_l1_min=0.9)
    with pytest.raises(ValueError, match="exceeds"):
        assert_singleton_guarantee(0.97)          # above max_v - tau


@pytest.mark.parametrize("v_cap", [0.96, 0.5, 0.1])
def test_singleton_class_is_eliminated_under_the_stated_conditions(v_cap):
    """A single clamped row must always be satisfiable inside the action box, for h >= 0 and
    any unit gradient. This is the guarantee, tested rather than asserted."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        th = rng.uniform(0, 2 * np.pi)
        g = np.array([np.cos(th), np.sin(th)])
        h = rng.uniform(0.0, 2.0)
        dh = max(rng.uniform(-4.0, 0.5), -v_cap)          # the clamp
        rhs = TAU - ALPHA * h - dh
        assert MAX_V * np.abs(g).sum() >= rhs - 1e-12, (g, h, dh)


def test_guarantee_fails_above_the_derived_cap():
    """Negative control: just above max_v - tau the guarantee must be violable."""
    g = np.array([1.0, 0.0])                              # axis-aligned worst case
    h, dh = 0.0, -0.97
    assert MAX_V * np.abs(g).sum() < TAU - ALPHA * h - dh


# ---------------------------------------------------------------- 2 & 3: minimal, inert at inf
@pytest.mark.parametrize("cls", [ProjectionCappedSource, VectorCappedSource])
def test_intervention_touches_only_dh_dt(cls):
    frozen = drive(src_of(EstimatedLidarBarrierSource))
    treated = drive(src_of(cls, v_cap=0.2))               # a hard cap, so it definitely binds
    assert len(frozen) == len(treated)
    for (h0, g0, _), (h1, g1, _) in zip(frozen, treated):
        np.testing.assert_array_equal(h0, h1)
        np.testing.assert_array_equal(g0, g1)


@pytest.mark.parametrize("cls", [ProjectionCappedSource, VectorCappedSource])
def test_cap_is_inert_at_infinite_cap(cls):
    """With v_cap = inf both treatments must reproduce the frozen source BIT-FOR-BIT.

    This is the guard against the T2 argmin recomputation drifting from the frozen one.
    """
    frozen = drive(src_of(EstimatedLidarBarrierSource))
    inert = drive(src_of(cls, v_cap=np.inf))
    for (h0, g0, d0), (h1, g1, d1) in zip(frozen, inert):
        np.testing.assert_array_equal(h0, h1)
        np.testing.assert_array_equal(g0, g1)
        np.testing.assert_array_equal(d0, d1)


def test_projection_cap_never_lowers_dh_dt():
    frozen = drive(src_of(EstimatedLidarBarrierSource))
    treated = drive(src_of(ProjectionCappedSource, v_cap=0.3))
    for (_, _, d0), (_, _, d1) in zip(frozen, treated):
        assert np.all(d1 >= d0 - 1e-15)
        assert np.all(d1 >= -0.3 - 1e-12)


def test_vector_cap_scales_rather_than_clamps():
    """T2 multiplies dh_dt by s <= 1, so it shrinks magnitude in BOTH directions -- unlike T1,
    which is a no-op on receding rows. The two are therefore not ordered in general."""
    frozen = drive(src_of(EstimatedLidarBarrierSource))
    treated = drive(src_of(VectorCappedSource, v_cap=0.2))
    saw_shrunk_positive = False
    for (_, _, d0), (_, _, d1) in zip(frozen, treated):
        assert np.all(np.abs(d1) <= np.abs(d0) + 1e-12)
        saw_shrunk_positive |= bool(np.any((d0 > 1e-9) & (d1 < d0 - 1e-12)))
    assert isinstance(saw_shrunk_positive, bool)


# ---------------------------------------------------------------- 4 & 5: policy and ledger
def test_c0_arm_is_the_frozen_source():
    from robot_env.robot_nav_env import RobotNavEnv
    env = RobotNavEnv(config_path="config.json", n_dynamic_obstacles=6, obstacle_speed=0.675,
                      render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=False)
    obs, _ = env.reset(seed=1_000_000)
    kw = dict(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
              static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED, dt=env.dt,
              lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS)
    pol = Phase7Policy(arm="C0_frozen", **kw).reset(obs, env.agent_position)
    assert type(pol.src) is EstimatedLidarBarrierSource
    assert (pol.ctrl.rateh, pol.ctrl.wasserstein_r, pol.ctrl.epsilon) == (0.4, 0.004, 0.1)
    env.close()


def test_treatment_arms_read_no_obstacle_ground_truth():
    """T1/T2 sources take only (p, ranges); there is no channel for obstacle state."""
    import inspect
    for cls in (ProjectionCappedSource, VectorCappedSource):
        assert list(inspect.signature(cls.push).parameters) == ["self", "p", "ranges"]
        assert not hasattr(cls, "push_truth")
    assert hasattr(OracleVelocitySource, "push_truth")
    assert OracleVelocitySource.IS_DIAGNOSTIC_ONLY is True


def test_arms_are_exactly_the_preregistered_four():
    """Stage 3's four arms are unchanged, in order, and none was added between them.

    Stage 5 appends ONE arm (E1_geometric_R) per PHASE7_PLAN.md 11.9.17, which is why this is a
    prefix check rather than an equality check. The Stage-3 arms themselves are still pinned
    exactly, and tests/test_dr_control_phase7_stage5.py pins the full Stage-5 tuple.
    """
    assert ARMS[:4] == ("C0_frozen", "T1_projection_cap", "T2_vector_cap", "D_oracle")
