"""Phase 2 gate: LiDAR -> CBF conversion, static obstacles.

Separate file from tests/test_dr_control.py so the Phase 1 gate stays reproducible and
independently runnable.

Hard tests:
    1. what cast_rays measures, per surface family (validated BEFORE the radius constant)
    2. the uniform r_robot subtraction is exact for rect/circle, conservative for walls
    3. LiDAR-derived (h, grad_h) matches analytic geometry within the discretisation bound
    4. the paired ground-truth obstacle block obs[28:52] is never read
    5. dh_dt is exactly 0 in a static world
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.cbf_sources import AnalyticBarrierSource, circle_barrier, rect_barrier
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.lidar_cbf import (
    LidarBarrierSource, goal_from_observation, ranges_from_observation, surface_points)
from robot_env.lidar_core import cast_rays, ray_angles

W, RA, RO, LR, NR = 10.0, 0.3, 0.3, 5.0, 24


def scan(p, rects=(), circles=()):
    return cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                     rects=rects, circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=RO)


# ------------------------------------------- test 1: what does cast_rays measure?
def test_wall_reading_is_centre_to_inset_boundary():
    """The caster pre-insets the boundary by agent_radius, so the radius is already in."""
    r = scan([5.0, 5.0])
    assert r[0] == pytest.approx(W - RA - 5.0)        # +x ray -> 4.7, not 5.0


def test_rect_and_circle_readings_are_centre_to_obstacle_surface():
    r = scan([5.0, 5.0], circles=[[8.0, 5.0]])
    assert r[0] == pytest.approx(3.0 - RO)            # 2.7: centre->centre minus obs radius
    r = scan([5.0, 5.0], rects=[(8.0, 5.0, 0.5, 0.5)])
    assert r[0] == pytest.approx(2.5)                 # centre -> near face


# ------------------------------- test 2: the uniform subtraction, exact vs conservative
def test_uniform_radius_subtraction_is_exact_for_rect_and_circle():
    """At the env's own collision threshold, h = range - r_robot must be exactly 0."""
    # cast_rays computes in float32 on purpose (see robot_env/lidar_core.py docstring), so
    # exact equality is not available; 1e-5 is far below any physically meaningful margin.
    r = scan([5.0, 5.0], circles=[[5.0 + RA + RO, 5.0]])
    assert r[0] - RA == pytest.approx(0.0, abs=1e-5)

    r = scan([5.0, 5.0], rects=[(5.0 + RA + 0.5, 5.0, 0.5, 0.5)])
    assert r[0] - RA == pytest.approx(0.0, abs=1e-5)


def test_uniform_radius_subtraction_is_conservative_for_walls_by_exactly_r_robot():
    """True wall clearance is the reading itself; we use reading - r_robot, so we under-report
    by exactly r_robot. Conservative is safe; the magnitude is pinned here so it is known."""
    p = [9.0, 5.0]
    reading = scan(p)[0]
    true_clearance = (W - RA) - p[0]                  # centre to inset boundary
    assert reading == pytest.approx(true_clearance)
    assert true_clearance - (reading - RA) == pytest.approx(RA)


def test_wall_ray_is_degenerate_exactly_at_the_clamp_boundary():
    """Documented sensor limitation, deliberately NOT repaired (robot_env is untouched).

    cast_rays requires t > 0, so at x = W - RA (which the env's position clamp makes exactly
    reachable) the wall-facing ray reports lidar_range instead of 0.
    """
    assert scan([W - RA - 1e-4, 5.0])[0] == pytest.approx(1e-4, abs=1e-5)
    assert scan([W - RA, 5.0])[0] == pytest.approx(LR)      # blind at contact


# -------------------------------- test 3: LiDAR-derived barrier vs analytic ground truth
def test_surface_points_drop_misses():
    """A ray returning exactly lidar_range hit nothing; it must not become a phantom surface."""
    ranges = np.full(NR, LR)
    ranges[0] = 2.0
    pts, hit = surface_points([5.0, 5.0], ranges)
    assert len(pts) == 1 and hit.sum() == 1
    np.testing.assert_allclose(pts[0], [7.0, 5.0])


def expected_lidar_clearance(p, rects, circles=()):
    """What the LiDAR convention SHOULD report, given exact geometry.

    The two surface families bias in OPPOSITE directions and must not be pooled:
      - walls:      the reading already excludes r_robot, so subtracting it again
                    under-reports the true clearance by exactly r_robot (conservative);
      - rect/circle: subtraction is exact, and the only error left is discretisation,
                    which can only OVER-report (the rays sample the surface at intervals).
    """
    from dr_control.cbf_sources import wall_barriers
    vals = [h - RA for h, _, _ in wall_barriers(p, W, RA)]          # wall convention
    vals += [rect_barrier(p, r, RA)[0] for r in rects]              # exact
    vals += [circle_barrier(p, c, RO, RA)[0] for c in np.asarray(circles, float).reshape(-1, 2)]
    return min(vals)


def test_lidar_h_matches_analytic_within_discretisation_bound():
    """Against the convention-corrected reference, LiDAR may only OVER-report clearance.

    Discretisation samples a convex surface at ray intervals, so the nearest sampled point is
    never closer than the true nearest point: h_lidar >= h_expected. A negative residual would
    mean the sensor invented a closer obstacle than exists, which is the dangerous direction.
    """
    rects = [(2.0, 2.0, 0.5, 1.5), (7.0, 3.0, 1.0, 0.5)]
    analytic = AnalyticBarrierSource(world_size=W, r_robot=RA, static_rects=rects)
    src = LidarBarrierSource(r_robot=RA, k_scans=1)
    rng = np.random.default_rng(0)

    errs = []
    for _ in range(400):
        p = rng.uniform(0.6, W - 0.6, 2)
        if analytic.true_clearance(p) <= 0.05:
            continue
        src.reset()
        src.push(p, scan(p, rects=rects))
        errs.append(src.min_clearance_estimate(p) - expected_lidar_clearance(p, rects))
    errs = np.array(errs)
    assert errs.min() > -1e-4, f"LiDAR reported a CLOSER obstacle than exists: {errs.min()}"
    assert np.median(errs) < 0.25, f"discretisation bias too large: {np.median(errs)}"


def test_wall_convention_under_reports_by_exactly_r_robot():
    """Pinned separately, because it is the opposite bias from the rect/circle case."""
    analytic = AnalyticBarrierSource(world_size=W, r_robot=RA, static_rects=[])
    src = LidarBarrierSource(r_robot=RA, k_scans=1)
    for p in ([5.0, 1.0], [1.2, 5.0], [8.5, 5.0], [5.0, 8.8]):
        src.reset()
        src.push(p, scan(p))
        assert analytic.true_clearance(p) - src.min_clearance_estimate(p) == pytest.approx(
            RA, abs=1e-4)


def test_lidar_gradient_points_away_from_the_obstacle():
    """grad_h must point from the surface toward the robot, and be a unit vector."""
    circles = [[8.0, 5.0]]
    src = LidarBarrierSource(r_robot=RA, k_scans=1)
    p = np.array([5.0, 5.0])
    src.push(p, scan(p, circles=circles))
    h, g, dd = src.samples(p)
    assert g.shape == (2, 1)
    np.testing.assert_allclose(np.linalg.norm(g[:, 0]), 1.0, atol=1e-9)
    np.testing.assert_allclose(g[:, 0], [-1.0, 0.0], atol=1e-9)   # obstacle at +x
    assert dd[0] == 0.0


def test_buffer_keeps_k_scans_and_yields_one_sample_each():
    src = LidarBarrierSource(r_robot=RA, k_scans=5)
    for i in range(8):
        p = np.array([5.0 + 0.01 * i, 5.0])
        src.push(p, scan(p))
    assert src.ready() and len(src.buffer) == 5
    h, g, dd = src.samples([5.08, 5.0])
    assert h.shape == (5,) and g.shape == (2, 5) and dd.shape == (5,)


# ---------------------------------------- test 5: static world has exactly zero dh_dt
def test_dh_dt_is_identically_zero_in_a_static_world():
    src = LidarBarrierSource(r_robot=RA, k_scans=5)
    rng = np.random.default_rng(3)
    for _ in range(5):
        p = rng.uniform(1, 9, 2)
        src.push(p, scan(p, rects=[(5.0, 5.0, 0.5, 0.5)]))
    _, _, dd = src.samples([3.0, 3.0])
    np.testing.assert_array_equal(dd, np.zeros(5))


# ------------------- test 4: the ground-truth obstacle block must never influence control
def _action_from_observation(obs, p, src, ctrl):
    """The full Phase 2 control path, reading ONLY obs[0:2] and obs[4:28]."""
    ranges = ranges_from_observation(obs)
    goal = goal_from_observation(obs, p)
    src.push(p, ranges)
    h, g, dd = src.samples(p)
    return ctrl.generate_controller(p, goal, build_xi(h, g, dd))


def test_paired_obstacle_block_is_never_read():
    """Perturbing obs[28:52] -- exact obstacle positions and velocities -- must not move u.

    This is the enforcement of the LiDAR-only information ledger. If a future change starts
    reading ground truth, this test fails.
    """
    rng = np.random.default_rng(11)
    p = np.array([4.0, 4.0])
    ranges = scan(p, rects=[(6.0, 4.0, 0.5, 0.5)])

    obs = np.zeros(52)
    obs[0:2] = np.array([3.0, 2.0]) / W          # goal offset
    obs[2:4] = [0.1, 0.0]
    obs[4:28] = ranges / LR

    def act(o):
        s = LidarBarrierSource(r_robot=RA, k_scans=5)
        c = ClfCbfDrccpController(max_v=1.0)
        for _ in range(5):
            u = _action_from_observation(o, p, s, c)
        return u

    base = act(obs.copy())
    for _ in range(5):
        tampered = obs.copy()
        tampered[28:52] = rng.uniform(-1, 1, 24)   # nonsense ground truth
        np.testing.assert_array_equal(act(tampered), base)


def test_observation_helpers_read_only_their_own_slices():
    obs = np.arange(52, dtype=float) / 52.0
    np.testing.assert_allclose(ranges_from_observation(obs), obs[4:28] * LR)
    p = np.array([1.0, 2.0])
    np.testing.assert_allclose(goal_from_observation(obs, p), p + obs[0:2] * W)


# ------------------------------------------- 24-ray detection limit (derivation + check)
def test_guaranteed_detection_range_matches_the_angular_derivation():
    """A disc of radius R at range d subtends asin(R/d); worst-case ray misalignment is
    half the ray spacing. Guaranteed detection therefore needs asin(R/d) >= pi/n_rays,
    i.e. d <= R / sin(pi/n_rays) ~= 2.30 m for R = 0.3, n_rays = 24.

    This is a bound on GUARANTEED detection, not a claim that farther discs are usually
    missed; the empirical miss curve is measured in experiments/exp1_lidar_static.py.
    """
    d_guaranteed = RO / np.sin(np.pi / NR)
    assert d_guaranteed == pytest.approx(2.2985, abs=1e-3)

    # Worst-case alignment: put the disc centre exactly between two rays.
    ang = ray_angles(NR)
    bearing = 0.5 * (ang[0] + ang[1])
    for d, expect_hit in ((d_guaranteed - 0.15, True), (d_guaranteed + 1.0, False)):
        centre = np.array([5.0, 5.0]) + d * np.array([np.cos(bearing), np.sin(bearing)])
        r = scan([5.0, 5.0], circles=[centre])
        detected = bool(np.any(r < LR - 1e-6) and np.min(r) < d)
        assert detected == expect_hit, f"d={d}"
