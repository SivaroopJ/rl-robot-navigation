"""Phase 3 gate: dynamic obstacles with an oracle velocity channel.

Hard tests:
    1. per-ray ownership is EXACT (reproduces the combined scan)
    2. dh_dt has the right sign and magnitude for a closing / receding obstacle
    3. the three dh_dt arms differ ONLY in dh_dt (h and grad_h identical)
    4. static-only worlds still give dh_dt == 0 under the oracle arm
    5. the oracle channel is opt-in: no path reaches ground-truth velocity without it
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.dynamic_cbf import (
    WALL, DynamicLidarBarrierSource, associate_nearest, ray_owners)
import experiments.exp0_analytic as E
from robot_env.lidar_core import cast_rays

W, RA, RO, LR, NR = 10.0, 0.3, 0.3, 5.0, 24
RECTS = E.STATIC_RECTS


def scan(p, circles):
    return cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                     rects=RECTS, circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=RO)


# ------------------------------------------------- test 1: ownership is exact
def test_ray_ownership_reproduces_the_combined_scan():
    """The per-object arg-min must equal the real scan, ray by ray.

    This is what makes association ground truth rather than a guess: if the reconstructed
    minimum matched only approximately, ownership would be an inference.
    """
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = rng.uniform(0.6, W - 0.6, 2)
        circles = rng.uniform(1.0, 9.0, (6, 2))
        owners, best = ray_owners(p, rects=RECTS, circles=circles, circle_radius=RO,
                                  n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA)
        np.testing.assert_allclose(best, scan(p, circles), atol=1e-5)
        assert owners.shape == (NR,)


def test_ray_ownership_identifies_the_right_object():
    """A lone circle directly along +x must own that ray; the opposite ray must be a wall."""
    circles = np.array([[8.0, 5.0]])
    owners, _ = ray_owners([5.0, 5.0], rects=(), circles=circles, circle_radius=RO,
                           n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA)
    assert owners[0] == 1000            # +x ray -> circle 0
    assert owners[12] == WALL           # -x ray -> wall


def test_ownership_distinguishes_rect_from_circle():
    circles = np.array([[5.0, 8.0]])
    owners, _ = ray_owners([5.0, 5.0], rects=[(8.0, 5.0, 0.5, 0.5)], circles=circles,
                           circle_radius=RO, n_rays=NR, lidar_range=LR,
                           world_size=W, agent_radius=RA)
    assert owners[0] == 0               # +x -> rect 0
    assert owners[6] == 1000            # +y -> circle 0


# --------------------------------------- test 2: dh_dt sign and magnitude, oracle arm
@pytest.mark.parametrize("v, expect", [
    ((-1.0, 0.0), -1.0),    # obstacle at +x moving -x: closing  -> dh_dt < 0
    ((+1.0, 0.0), +1.0),    # receding                          -> dh_dt > 0
    ((0.0, 1.0), 0.0),      # tangential                        -> dh_dt = 0
])
def test_oracle_dh_dt_sign_and_magnitude(v, expect):
    p = np.array([5.0, 5.0])
    circles = np.array([[8.0, 5.0]])
    src = DynamicLidarBarrierSource(r_robot=RA, k_scans=1, dh_dt_mode="oracle")
    owners, _ = ray_owners(p, rects=(), circles=circles, circle_radius=RO,
                           n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA)
    src.push(p, cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                          rects=(), circles=circles, circle_radius=RO), owners)
    h, g, dd = src.samples(p, np.array([v]))
    np.testing.assert_allclose(g[:, 0], [-1.0, 0.0], atol=1e-6)
    assert dd[0] == pytest.approx(expect, abs=1e-5)


def test_oracle_dh_dt_equals_minus_grad_dot_v():
    """The defining identity, on random geometry."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        p = np.array([5.0, 5.0])
        c = p + rng.uniform(1.0, 3.0) * np.array([np.cos(t := rng.uniform(0, 2 * np.pi)),
                                                  np.sin(t)])
        v = rng.uniform(-0.75, 0.75, 2)
        circles = c.reshape(1, 2)
        owners, _ = ray_owners(p, rects=(), circles=circles, circle_radius=RO, n_rays=NR,
                               lidar_range=LR, world_size=W, agent_radius=RA)
        src = DynamicLidarBarrierSource(r_robot=RA, k_scans=1, dh_dt_mode="oracle")
        src.push(p, cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                              rects=(), circles=circles, circle_radius=RO), owners)
        h, g, dd = src.samples(p, v.reshape(1, 2))
        if src.critical_owner(p) >= 1000:            # only when the circle is the critical one
            assert dd[0] == pytest.approx(-float(g[:, 0] @ v), abs=1e-6)


# ------------------------------- test 3: the arms differ ONLY in dh_dt
def test_arms_share_h_and_grad_and_differ_only_in_dh_dt():
    p = np.array([4.0, 5.0])
    circles = np.array([[6.5, 5.0], [4.0, 7.0]])
    vels = np.array([[-0.7, 0.0], [0.0, -0.7]])
    owners, _ = ray_owners(p, rects=RECTS, circles=circles, circle_radius=RO, n_rays=NR,
                           lidar_range=LR, world_size=W, agent_radius=RA)
    out = {}
    for mode in ("oracle", "zero", "conservative"):
        src = DynamicLidarBarrierSource(r_robot=RA, k_scans=3, dh_dt_mode=mode)
        for _ in range(3):
            src.push(p, scan(p, circles), owners)
        out[mode] = src.samples(p, vels)

    for mode in ("zero", "conservative"):
        np.testing.assert_array_equal(out[mode][0], out["oracle"][0])   # h identical
        np.testing.assert_array_equal(out[mode][1], out["oracle"][1])   # grad identical
    np.testing.assert_array_equal(out["zero"][2], np.zeros(3))
    np.testing.assert_allclose(out["conservative"][2], np.full(3, -0.75))


# ------------------------------- test 4/5: oracle is opt-in and static-safe
def test_oracle_arm_gives_zero_dh_dt_in_a_static_world():
    p = np.array([4.0, 4.0])
    owners, _ = ray_owners(p, rects=RECTS, circles=np.zeros((0, 2)), circle_radius=RO,
                           n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA)
    src = DynamicLidarBarrierSource(r_robot=RA, k_scans=4, dh_dt_mode="oracle")
    for _ in range(4):
        src.push(p, scan(p, np.zeros((0, 2))), owners)
    _, _, dd = src.samples(p, None)
    np.testing.assert_array_equal(dd, np.zeros(4))


def test_velocities_are_ignored_unless_the_oracle_arm_is_selected():
    """No silent fallback: 'zero' and 'conservative' must not read the velocity argument."""
    p = np.array([5.0, 5.0])
    circles = np.array([[7.0, 5.0]])
    owners, _ = ray_owners(p, rects=(), circles=circles, circle_radius=RO, n_rays=NR,
                           lidar_range=LR, world_size=W, agent_radius=RA)
    for mode, expect in (("zero", 0.0), ("conservative", -0.75)):
        src = DynamicLidarBarrierSource(r_robot=RA, k_scans=1, dh_dt_mode=mode)
        src.push(p, cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                              rects=(), circles=circles, circle_radius=RO), owners)
        _, _, dd = src.samples(p, np.array([[-99.0, 0.0]]))   # absurd velocity
        assert dd[0] == pytest.approx(expect)


def test_nearest_neighbour_association_is_only_a_measurement_helper():
    """associate_nearest exists to QUANTIFY association error, not to drive control."""
    circles = np.array([[8.0, 5.0]])
    assert associate_nearest([7.7, 5.0], circles, RO) == 1000     # on the circle surface
    assert associate_nearest([1.0, 1.0], circles, RO) == WALL     # far from any circle
