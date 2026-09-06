"""Phase 4 gate: LiDAR-only velocity estimation.

Hard tests:
    1. sign convention of e = dh_dt_est - dh_dt_true, all four hand-constructed cases
    2. two-point centre reconstruction picks the physically correct (far) root
    3. segmentation / static rejection behave as designed, incl. the disc-vs-line margin
    4. LEAKAGE: signature, imports, runtime tripwire, and ground-truth perturbation
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from dr_control.drccp_controller import build_xi
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import (
    COMPACT_MARGIN, LINE_TOL, LidarVelocityTracker, line_residual, reconstruct_centre,
    segment_is_compact, segment_scan)
from robot_env.lidar_core import cast_rays, ray_angles
import experiments.exp0_analytic as E

W, RA, RO, LR, NR = 10.0, 0.3, 0.3, 5.0, 24


def scan(p, circles=(), rects=()):
    return cast_rays(p, n_rays=NR, lidar_range=LR, world_size=W, agent_radius=RA,
                     rects=rects, circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=RO)


# ============================================================ 1. SIGN CONVENTION
# Robot at origin, obstacle at +x  =>  grad_h = (-1, 0)  (points obstacle -> robot).
# dh_dt = -grad_h . v.  e = dh_dt_est - dh_dt_true.
# e > 0  <=>  the controller believes the obstacle is closing LESS / receding MORE than
# reality  <=>  optimistic  <=>  DANGEROUS. Verified in all four branches below.
GRAD_H = np.array([-1.0, 0.0])


def dh_dt(v):
    return -float(GRAD_H @ np.asarray(v, dtype=float))


@pytest.mark.parametrize("name, v_true, v_est, sign, dangerous", [
    ("approaching, under-estimated speed", (-0.75, 0.0), (-0.25, 0.0), +1, True),
    ("approaching, over-estimated speed", (-0.25, 0.0), (-0.75, 0.0), -1, False),
    ("receding, under-estimated speed", (+0.75, 0.0), (+0.25, 0.0), -1, False),
    ("receding, over-estimated speed", (+0.25, 0.0), (+0.75, 0.0), +1, True),
])
def test_projected_error_sign_convention(name, v_true, v_est, sign, dangerous):
    e = dh_dt(v_est) - dh_dt(v_true)
    assert np.sign(e) == sign, f"{name}: e = {e}"
    # "Dangerous" means the controller thinks the barrier decays more slowly than it really
    # does, i.e. it under-estimates the rate at which clearance is being lost.
    assert (dh_dt(v_est) > dh_dt(v_true)) == dangerous, name
    assert (e > 0) == dangerous, name


def test_tangential_velocity_error_is_not_penalised():
    """Only the gradient-projected component enters the constraint; ||v_hat - v|| would
    wrongly flag a purely tangential error as an error."""
    v_true, v_est = np.array([-0.5, 0.0]), np.array([-0.5, 0.6])
    assert np.linalg.norm(v_est - v_true) == pytest.approx(0.6)
    assert dh_dt(v_est) - dh_dt(v_true) == pytest.approx(0.0, abs=1e-12)


# ============================================ 2. TWO-POINT CENTRE: THE FAR ROOT
def test_two_point_centre_is_the_far_root():
    """A ray stops on the NEAR surface of an opaque body, so the centre lies beyond it.

    Constructed exactly: a disc at (2, 0) seen from the origin, two symmetric surface points.
    The near root would place the body between the sensor and its own surface.
    """
    p_ego = np.zeros(2)
    centre_true = np.array([2.0, 0.0])
    phi = np.deg2rad(25.0)
    q1 = centre_true + RO * np.array([-np.cos(phi), np.sin(phi)])
    q2 = centre_true + RO * np.array([-np.cos(phi), -np.sin(phi)])
    dirs = np.array([(q1 - p_ego) / np.linalg.norm(q1 - p_ego),
                     (q2 - p_ego) / np.linalg.norm(q2 - p_ego)])

    c, n, degen = reconstruct_centre([q1, q2], dirs, p_ego, RO)
    assert n == 2 and not degen
    np.testing.assert_allclose(c, centre_true, atol=1e-9)
    # and it is strictly the farther of the two candidate roots
    m = 0.5 * (q1 + q2)
    assert np.linalg.norm(c - p_ego) > np.linalg.norm(m - p_ego)


def test_two_point_centre_is_deterministic_under_point_order():
    p_ego = np.zeros(2)
    centre_true = np.array([0.0, 1.6])
    phi = np.deg2rad(30.0)
    q1 = centre_true + RO * np.array([np.sin(phi), -np.cos(phi)])
    q2 = centre_true + RO * np.array([-np.sin(phi), -np.cos(phi)])
    d = lambda q: (q - p_ego) / np.linalg.norm(q - p_ego)
    a, _, _ = reconstruct_centre([q1, q2], [d(q1), d(q2)], p_ego, RO)
    b, _, _ = reconstruct_centre([q2, q1], [d(q2), d(q1)], p_ego, RO)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a, centre_true, atol=1e-9)


def test_two_point_centre_degenerate_when_chord_exceeds_diameter():
    p_ego = np.zeros(2)
    q1, q2 = np.array([2.0, -0.5]), np.array([2.0, 0.5])       # 1.0 m apart > 2*0.3
    d = lambda q: (q - p_ego) / np.linalg.norm(q - p_ego)
    c, n, degen = reconstruct_centre([q1, q2], [d(q1), d(q2)], p_ego, RO)
    assert degen and n == 2
    np.testing.assert_allclose(c, 0.5 * (q1 + q2))             # clamped to the midpoint


def test_multi_point_fit_recovers_the_centre():
    p_ego = np.zeros(2)
    centre_true = np.array([1.5, 0.4])
    ang = np.linspace(np.pi - 0.5, np.pi + 0.5, 4)             # arc facing the sensor
    pts = centre_true + RO * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    dirs = np.stack([(q - p_ego) / np.linalg.norm(q - p_ego) for q in pts])
    c, n, _ = reconstruct_centre(pts, dirs, p_ego, RO)
    assert n == 4
    np.testing.assert_allclose(c, centre_true, atol=1e-6)


def test_one_point_fallback_is_exact_only_head_on():
    p_ego = np.zeros(2)
    centre_true = np.array([2.0, 0.0])
    q = centre_true - np.array([RO, 0.0])                      # head-on hit
    c, n, _ = reconstruct_centre([q], [np.array([1.0, 0.0])], p_ego, RO)
    assert n == 1
    np.testing.assert_allclose(c, centre_true, atol=1e-12)


# ============================================ 3. SEGMENTATION / STATIC REJECTION
def test_segmentation_splits_two_separated_objects():
    p = np.array([5.0, 5.0])
    segs = segment_scan(scan(p, circles=[[7.0, 5.0], [5.0, 7.0]]), n_rays=NR, lidar_range=LR)
    assert len(segs) >= 2


def test_a_disc_is_compact_and_a_multi_ray_wall_is_not():
    """Compactness separates a disc from an EXTENDED wall segment.

    It cannot separate a disc from a ONE-RAY wall fragment, and nothing in a single frame
    can: one range return carries no extent information. Distant walls do fragment this way
    once oblique rays fall out of range. Such fragments therefore reach the tracker as
    candidates -- which is benign for the reason pinned by the next test, and is measured as
    the static false-positive rate in exp4.
    """
    p = np.array([5.0, 5.0])
    ranges = scan(p, circles=[[6.2, 5.0]])
    dirs = np.stack([np.cos(ray_angles(NR)), np.sin(ray_angles(NR))], axis=1)
    checked_disc = checked_wall = False
    for seg in segment_scan(ranges, n_rays=NR, lidar_range=LR):
        pts = p[None, :] + ranges[seg, None] * dirs[seg]
        compact, extent = segment_is_compact(pts, RO)
        if np.min(ranges[seg]) < 1.5:                          # the disc
            assert compact and extent <= 2 * RO + COMPACT_MARGIN
            checked_disc = True
        elif len(seg) >= 3:                                    # an extended wall run
            assert not compact
            checked_wall = True
    assert checked_disc and checked_wall


def test_static_structure_yields_near_zero_estimated_velocity():
    """The property that makes static false-positives benign, and it must hold.

    A wall fragment misclassified as a dynamic candidate is harmless PROVIDED the filter
    estimates v ~ 0 for it -- static structure does not move in the world frame, and the ego
    pose is known, so its reconstructed centre should be stationary. If this failed, phantom
    dh_dt would be injected on walls.
    """
    tr = LidarVelocityTracker(r_nominal=RO)
    p = np.array([5.0, 5.0])
    for _ in range(20):
        tr.update(p, scan(p, rects=E.STATIC_RECTS))            # walls + rects only, all static
    speeds = [float(np.linalg.norm(t.velocity)) for t in tr.tracks if t.confirmed]
    assert all(s < 0.12 for s in speeds), speeds


def test_line_tolerance_leaves_a_real_disc_above_it():
    """The margin that keeps a genuine obstacle from being frozen at v = 0.

    A disc of radius 0.3 observed over two 15-degree ray steps departs from its chord by
    r*(1 - cos 15deg) = 0.0102 m. LINE_TOL must sit clearly below that or the static test
    would misclassify real obstacles in the 0.8-1.0 m band.
    """
    sagitta_two_steps = RO * (1 - np.cos(np.deg2rad(15.0)))
    assert sagitta_two_steps == pytest.approx(0.01022, abs=1e-4)
    assert LINE_TOL < 0.5 * sagitta_two_steps

    ang = np.deg2rad([-15.0, 0.0, 15.0])
    pts = np.array([2.0, 0.0]) + RO * np.stack([np.cos(np.pi + ang), np.sin(np.pi + ang)], 1)
    assert line_residual(pts) > LINE_TOL                       # a disc is NOT flat

    wall = np.stack([np.full(5, 3.0), np.linspace(-1, 1, 5)], axis=1)
    assert line_residual(wall) < LINE_TOL                      # a wall is


def test_static_rejection_keeps_a_moving_disc_and_drops_walls():
    p = np.array([5.0, 5.0])
    tr = LidarVelocityTracker(r_nominal=RO)
    dets = tr.detect(p, scan(p, circles=[[6.2, 5.0]], rects=E.STATIC_RECTS))
    assert len(dets) >= 1
    assert any(np.linalg.norm(d["centre"] - np.array([6.2, 5.0])) < 0.2 for d in dets)


# ------------------------------------------------- tracker recovers a known velocity
def test_tracker_recovers_a_constant_velocity():
    """Stationary robot, one disc crossing at a known constant velocity."""
    tr = LidarVelocityTracker(r_nominal=RO)
    p = np.array([5.0, 5.0])
    c = np.array([5.0, 7.5])
    v = np.array([0.0, -0.7])
    for _ in range(25):
        tr.update(p, scan(p, circles=[c]))
        c = c + v * 0.1
    v_hat, trusted, tid = tr.velocity_at(c - np.array([0.0, -RO]))
    assert trusted and tid >= 0
    assert np.linalg.norm(v_hat - v) < 0.15, v_hat


def test_unconfirmed_tracks_report_zero_velocity():
    tr = LidarVelocityTracker(r_nominal=RO)
    p = np.array([5.0, 5.0])
    c = np.array([5.0, 7.0])
    tr.update(p, scan(p, circles=[c]))                          # single frame -> unconfirmed
    v_hat, trusted, _ = tr.velocity_at(c - np.array([0.0, RO]))
    assert not trusted
    np.testing.assert_array_equal(v_hat, np.zeros(2))


# ============================================================ 4. LEAKAGE SAFEGUARDS
def test_leak_signature_no_obstacle_arguments():
    """Layer 1: there is no parameter through which ground truth could enter."""
    for fn in (LidarVelocityTracker.update, EstimatedLidarBarrierSource.push,
               EstimatedLidarBarrierSource.samples):
        params = set(inspect.signature(fn).parameters) - {"self"}
        assert not (params & {"circles", "obstacle_velocities", "vels", "owners",
                              "obstacles", "env", "velocity"}), (fn, params)
    assert set(inspect.signature(LidarVelocityTracker.update).parameters) == {
        "self", "p_ego", "ranges"}
    assert set(inspect.signature(EstimatedLidarBarrierSource.push).parameters) == {
        "self", "p", "ranges"}


def test_leak_imports_do_not_reach_ground_truth_modules():
    """Layer 2: the Phase 4 modules must not import ground-truth geometry helpers."""
    forbidden_mod = {"dr_control.dynamic_cbf", "dr_control.cbf_sources"}
    forbidden_name = {"ray_owners", "associate_nearest", "AnalyticBarrierSource",
                      "circle_barrier", "rect_barrier", "wall_barriers"}
    for rel in ("dr_control/velocity_tracker.py", "dr_control/estimated_cbf.py"):
        tree = ast.parse(Path(rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_mod, (rel, node.module)
                for a in node.names:
                    assert a.name not in forbidden_name, (rel, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name not in forbidden_mod, (rel, a.name)


def _build_scenario(seed=0):
    """Scenario SETUP, kept separate from the control loop.

    Placing obstacles in free space legitimately uses ground-truth geometry -- it is the
    simulator building the world. The runtime tripwire must therefore be armed AFTER this,
    so that it covers the control path only and does not fire on world construction.
    """
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    return E.sample_scenario(rng, 6, moving=True)


def _run_estimated_episode(n_steps=40, corrupt_velocities=False, seed=0, scenario=None):
    """A self-contained `estimated` rollout, returning actions and estimator outputs.

    Ground truth generates the SCAN (it must -- that is the sensor), and nothing else.
    """
    from dr_control.drccp_controller import ClfCbfDrccpController

    p, goal, circles, vels = scenario if scenario is not None else _build_scenario(seed)
    p, circles, vels = p.copy(), circles.copy(), vels.copy()
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=0.4)
    src = EstimatedLidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5)

    actions, dh_dts = [], []
    for step in range(n_steps):
        ranges = scan(p, circles=circles, rects=E.STATIC_RECTS)
        if corrupt_velocities:
            # Corrupt ground truth AFTER the scan is generated. The scan already exists, so
            # a correct implementation cannot be affected by this at all.
            vels = vels + 17.0
        src.push(p, ranges)
        h, g, dd = src.samples(p)
        u = ctrl.generate_controller(p, goal, build_xi(h, g, dd))
        actions.append(np.asarray(u).copy())
        if len(dd) == src.k_scans:      # the buffer fills over the first k_scans steps;
            dh_dts.append(np.asarray(dd).copy())   # only full-width rows are stackable
        p = np.clip(p + np.asarray(u) * E.DT, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)
        circles = circles + (vels if not corrupt_velocities else vels - 17.0) * E.DT
    return np.array(actions), np.array(dh_dts)


def test_leak_runtime_tripwire_ground_truth_geometry_is_never_touched(monkeypatch):
    """Layer 3: make ground-truth helpers explode, then run a full estimated rollout."""
    import dr_control.dynamic_cbf as dyn
    from dr_control.cbf_sources import AnalyticBarrierSource

    def boom(*a, **k):
        raise AssertionError("control path touched ground-truth geometry")

    scenario = _build_scenario()          # world construction happens BEFORE the tripwire
    monkeypatch.setattr(dyn, "ray_owners", boom)
    monkeypatch.setattr(dyn, "associate_nearest", boom)
    monkeypatch.setattr(AnalyticBarrierSource, "samples", boom)
    actions, _ = _run_estimated_episode(n_steps=30, scenario=scenario)
    assert len(actions) == 30


def test_leak_corrupting_ground_truth_velocity_changes_nothing_bit_for_bit():
    """Layer 4, strengthened: BOTH actions and estimator outputs must be bit-identical.

    The estimator is fully deterministic (no RNG anywhere in the tracker, and Hungarian
    assignment is deterministic), so exact equality is the correct requirement, not closeness.
    """
    a0, d0 = _run_estimated_episode(n_steps=40, corrupt_velocities=False)
    a1, d1 = _run_estimated_episode(n_steps=40, corrupt_velocities=True)
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_array_equal(d0, d1)


def test_leak_estimator_is_deterministic_across_repeated_runs():
    """Prerequisite for the bit-identical claim above."""
    a0, d0 = _run_estimated_episode(n_steps=25)
    a1, d1 = _run_estimated_episode(n_steps=25)
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_array_equal(d0, d1)


def test_leak_tracker_inputs_are_exactly_pose_and_ranges(monkeypatch):
    """Layer 5: audit what the estimator is actually handed, per step."""
    seen = []
    orig = LidarVelocityTracker.update

    def spy(self, p_ego, ranges):
        seen.append((np.asarray(p_ego).shape, np.asarray(ranges).shape))
        return orig(self, p_ego, ranges)

    monkeypatch.setattr(LidarVelocityTracker, "update", spy)
    _run_estimated_episode(n_steps=15)
    assert seen and all(s == ((2,), (NR,)) for s in seen)
