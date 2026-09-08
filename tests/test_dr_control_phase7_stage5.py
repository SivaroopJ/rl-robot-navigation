"""Week5-Phase7 / Stage 5 / E1 admission checks A1-A3 and A5 (PHASE7_PLAN.md 11.9.13).

A1 inertness   with use_geometric_R=False the subclass is BIT-IDENTICAL to the frozen tracker.
A2 correctness anisotropy in the derived direction, the isotropic limit, SPD, closed forms.
A3 leakage     signature, AST import scan, runtime tripwire, action invariance under corruption
               of the ground-truth observation slice obs[28:52].
A5 regression  covered by tests/test_frozen_phase1_6.py and the frozen sha256 manifest.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from dr_control.capped_velocity import V_CAP_DERIVED, ProjectionCappedSource
from dr_control.policy_phase7 import ARMS, Phase7Policy
from dr_control.tracking2 import (FD_STEP, MSE_R_COEF, MSE_T_COEF, GeometricLidarVelocityTracker,
                                  covariance_1pt, covariance_2pt, covariance_npt,
                                  reconstruction_covariance, sigma_range)
from dr_control.velocity_tracker import (POINT_COUNT_FACTOR, SIGMA_R_BASE, LidarVelocityTracker,
                                         reconstruct_centre)
from robot_env.lidar_core import ray_angles
from experiments.exp6_ppo_comparison import make_env

R_NOM = 0.3


def _trace(n_steps=60, seed=7):
    """A real scan trace from the environment: (p_ego, ranges) pairs, nothing else."""
    env = make_env(True)
    obs, _ = env.reset(seed=seed)
    out = []
    for _ in range(n_steps):
        out.append((env.agent_position.copy(), obs[4:28].copy() * env.LIDAR_RANGE))
        obs, _, term, trunc, _ = env.step(np.array([0.4, 0.3], dtype=np.float32))
        if term or trunc:
            obs, _ = env.reset(seed=seed + 1)
    return out


def _state(tracker):
    """Track.id comes from a process-wide counter shared by every tracker instance, so two
    trackers run side by side get different absolute ids for the same track. Identity is
    therefore compared by POSITION IN THE TRACK LIST, which the frozen algorithm determines,
    and the id itself is excluded."""
    return [(t.x.copy(), t.P.copy(), t.age, t.hits, t.misses, t.confirmed)
            for t in tracker.tracks]


# --------------------------------------------------------------- A1  inertness
def test_a1_inertness_bit_identical_to_frozen_tracker():
    """The ONLY difference is R. With it disabled the subclass must reproduce the frozen
    tracker exactly -- this is what guarantees the copied update() has not drifted."""
    froz = LidarVelocityTracker(r_nominal=R_NOM)
    geo = GeometricLidarVelocityTracker(r_nominal=R_NOM, use_geometric_R=False)
    for p, ranges in _trace():
        froz.update(p, ranges)
        geo.update(p, ranges)
        sf, sg = _state(froz), _state(geo)
        assert len(sf) == len(sg)
        for a, b in zip(sf, sg):
            assert a[2:] == b[2:]                      # age, hits, misses, confirmed
            assert np.array_equal(a[0], b[0])          # bit-identical state
            assert np.array_equal(a[1], b[1])          # bit-identical covariance


def test_a1_geometric_R_actually_changes_something():
    """Negative control for A1: with E1 ENABLED the trajectory of filter states must differ,
    otherwise the inertness test above would be vacuous."""
    froz = LidarVelocityTracker(r_nominal=R_NOM)
    geo = GeometricLidarVelocityTracker(r_nominal=R_NOM, use_geometric_R=True)
    differed = False
    for p, ranges in _trace():
        froz.update(p, ranges)
        geo.update(p, ranges)
        sf, sg = _state(froz), _state(geo)
        if len(sf) != len(sg) or any(not np.array_equal(a[0], b[0]) for a, b in zip(sf, sg)):
            differed = True
    assert differed


# --------------------------------------------------------------- A2  correctness
def test_a2_one_point_covariance_is_anisotropic_tangentially():
    """The large axis is TANGENTIAL to the ray -- the direction 11.9.8 predicted."""
    u = np.array([1.0, 0.0])
    R = covariance_1pt(u, R_NOM, sigma_range(1.0))
    var_rad = float(u @ R @ u)
    t = np.array([0.0, 1.0])
    var_tan = float(t @ R @ t)
    assert var_tan > var_rad
    assert var_tan == pytest.approx(MSE_T_COEF * R_NOM ** 2)
    assert var_rad == pytest.approx(MSE_R_COEF * R_NOM ** 2 + sigma_range(1.0) ** 2)


def test_a2_closed_forms_match_the_exact_geometry():
    """Hand-derived numeric case: e_t = -s, e_r = r - sqrt(r^2 - s^2), s ~ U(-r, r)."""
    rng = np.random.default_rng(0)
    s = rng.uniform(-R_NOM, R_NOM, 400_000)
    mse_t = float((s ** 2).mean())
    mse_r = float(((R_NOM - np.sqrt(np.maximum(R_NOM ** 2 - s ** 2, 0.0))) ** 2).mean())
    assert mse_t == pytest.approx(MSE_T_COEF * R_NOM ** 2, rel=2e-2)
    assert mse_r == pytest.approx(MSE_R_COEF * R_NOM ** 2, rel=2e-2)


def test_a2_one_point_is_more_conservative_tangentially_than_the_frozen_rule():
    """The corrected derivation (11.9.18) must NOT make a one-point return more trusted than
    the frozen isotropic rule -- the error the pre-registration caught."""
    for d in (0.6, 1.0, 2.0, 3.0):
        frozen_sd = SIGMA_R_BASE * POINT_COUNT_FACTOR[1] * (1.0 + d / 3.0)
        R = covariance_1pt(np.array([1.0, 0.0]), R_NOM, sigma_range(d))
        assert float(np.sqrt(np.linalg.eigvalsh(R).max())) > frozen_sd


def test_a2_rotation_equivariance():
    """R must rotate with the ray: geometry, not a fixed frame."""
    th = 0.7
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = covariance_1pt(np.array([1.0, 0.0]), R_NOM, sigma_range(1.5))
    B = covariance_1pt(Rot @ np.array([1.0, 0.0]), R_NOM, sigma_range(1.5))
    assert np.allclose(B, Rot @ A @ Rot.T)


def test_a2_all_branches_are_symmetric_positive_definite():
    ang = ray_angles(24)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    p = np.zeros(2)
    for seg in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 3]):
        rr = np.full(len(seg), 1.2)
        pts = p[None, :] + rr[:, None] * dirs[seg]
        c, _, _ = reconstruct_centre(pts, dirs[seg], p, R_NOM)
        R, src = reconstruction_covariance(pts, dirs[seg], rr, p, c, R_NOM)
        assert np.allclose(R, R.T)
        assert float(np.linalg.eigvalsh(R).min()) > 0.0
        assert src


def test_a2_eigenvalue_floor_is_the_range_variance():
    """A reconstruction can never be more certain than the measurement it is built from."""
    ang = ray_angles(24)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    p = np.zeros(2)
    seg = list(range(6))
    rr = np.full(len(seg), 0.7)
    pts = p[None, :] + rr[:, None] * dirs[seg]
    c, _, _ = reconstruct_centre(pts, dirs[seg], p, R_NOM)
    R, _ = reconstruction_covariance(pts, dirs[seg], rr, p, c, R_NOM)
    assert float(np.linalg.eigvalsh(R).min()) >= sigma_range(0.7) ** 2 - 1e-15


def test_a2_many_well_spread_points_approach_isotropy():
    """Isotropic limit: with points spread around the disc, R loses its directionality."""
    p = np.zeros(2)
    c_true = np.array([1.5, 0.0])
    phis = np.linspace(np.pi * 0.7, np.pi * 1.3, 9)
    pts = c_true[None, :] + R_NOM * np.stack([np.cos(phis), np.sin(phis)], axis=1)
    dirs = pts - p[None, :]
    rr = np.linalg.norm(dirs, axis=1)
    dirs = dirs / rr[:, None]
    R = covariance_npt(pts, dirs, c_true, np.array([sigma_range(r) for r in rr]))[0]
    w = np.linalg.eigvalsh(R)
    assert w.max() / w.min() < 4.0
    R1 = covariance_1pt(dirs[0], R_NOM, sigma_range(rr[0]))
    w1 = np.linalg.eigvalsh(R1)
    assert w.max() < w1.max()                # many points are more certain than one


def _mc_covariance(pts, dirs, rr, p, sig, n=20000, seed=0):
    """Empirical covariance of the EXACT frozen reconstruction under range noise."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2))
    for i in range(n):
        rho = rr + rng.normal(0.0, sig)
        q = p[None, :] + rho[:, None] * dirs
        out[i] = reconstruct_centre(q, dirs, p, R_NOM)[0]
    return np.cov(out.T)


def test_a2_two_point_covariance_matches_monte_carlo_of_the_exact_map():
    """First-order propagation must reproduce the covariance the reconstruction actually has.
    This validates covariance_2pt end-to-end against its own definition."""
    p = np.zeros(2)
    c = np.array([1.6, 0.0])
    pts = np.array([c + R_NOM * np.array([np.cos(t), np.sin(t)]) for t in (2.4, 3.9)])
    rr = np.linalg.norm(pts, axis=1)
    dirs = pts / rr[:, None]
    sig = np.array([sigma_range(x) for x in rr])
    R = covariance_2pt(pts, dirs, rr, p, R_NOM, sig)[0]
    M = _mc_covariance(pts, dirs, rr, p, sig)
    assert np.allclose(np.linalg.eigvalsh(R), np.linalg.eigvalsh(M), rtol=0.15, atol=1e-6)


def test_a2_npt_covariance_matches_monte_carlo_of_the_exact_map():
    p = np.zeros(2)
    c = np.array([1.2, 0.0])
    phis = np.linspace(2.5, 3.8, 4)
    pts = np.array([c + R_NOM * np.array([np.cos(t), np.sin(t)]) for t in phis])
    rr = np.linalg.norm(pts, axis=1)
    dirs = pts / rr[:, None]
    sig = np.array([sigma_range(x) for x in rr])
    R = covariance_npt(pts, dirs, c, sig)[0]
    M = _mc_covariance(pts, dirs, rr, p, sig)
    assert np.allclose(np.linalg.eigvalsh(R), np.linalg.eigvalsh(M), rtol=0.25, atol=1e-6)


def test_a2_closely_spaced_points_are_less_certain_than_well_spread_ones():
    """The dominant conditioning effect for n = 2: a short chord determines the centre badly.

    NOTE (11.9.18 addendum): the -a/b amplification as a -> r is present in the Jacobian but is
    NOT generally dominant here, because it enters multiplied by da/drho, which vanishes when
    the chord is perpendicular to the rays -- the usual symmetric LiDAR view of a disc. The
    verified, dominant effect is chord length, and that is what this test pins.
    """
    p = np.zeros(2)
    c = np.array([2.0, 0.0])
    sds = []
    for spread in (1.2, 0.6, 0.25):
        pts = np.array([c + R_NOM * np.array([np.cos(t), np.sin(t)])
                        for t in (np.pi - spread / 2, np.pi + spread / 2)])
        rr = np.linalg.norm(pts, axis=1)
        dirs = pts / rr[:, None]
        R = covariance_2pt(pts, dirs, rr, p, R_NOM,
                           np.array([sigma_range(x) for x in rr]))[0]
        sds.append(float(np.sqrt(np.linalg.eigvalsh(R).max())))
    assert sds[0] < sds[1] < sds[2]


def test_a2_degenerate_chord_falls_back_to_the_conservative_one_point_form():
    p = np.zeros(2)
    pts = np.array([[2.0, 0.4], [2.0, -0.4]])          # a = 0.4 >= r = 0.3
    dirs = pts / np.linalg.norm(pts, axis=1)[:, None]
    rr = np.linalg.norm(pts, axis=1)
    _, src = covariance_2pt(pts, dirs, rr, p, R_NOM, np.array([sigma_range(r) for r in rr]))
    assert src == "1pt_degenerate_chord"


def test_a2_two_point_jacobian_is_step_insensitive():
    """FD_STEP is numerical, not a model parameter."""
    import dr_control.tracking2 as m
    p = np.zeros(2)
    pts = np.array([[1.5, 0.12], [1.5, -0.12]])
    dirs = pts / np.linalg.norm(pts, axis=1)[:, None]
    rr = np.linalg.norm(pts, axis=1)
    sig = np.array([sigma_range(r) for r in rr])
    base = covariance_2pt(pts, dirs, rr, p, R_NOM, sig)[0]
    for step in (1e-8, 1e-5, 1e-4):
        m.FD_STEP = step
        try:
            assert np.allclose(covariance_2pt(pts, dirs, rr, p, R_NOM, sig)[0], base, rtol=1e-5)
        finally:
            m.FD_STEP = FD_STEP


def _executable_source(path):
    """Module source with every docstring stripped -- prose may name what code may not use."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_a2_point_count_factor_dependence_is_removed():
    """E1 must not CONSULT POINT_COUNT_FACTOR, as pre-registered. The docstring names it only
    to say what was replaced, so the check is on executable code, not on the file text."""
    assert "POINT_COUNT_FACTOR" not in _executable_source("dr_control/tracking2.py")
    tree = ast.parse(Path("dr_control/tracking2.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert "POINT_COUNT_FACTOR" not in imported


# --------------------------------------------------------------- A3  leakage
def test_a3_signature_has_no_obstacle_channel():
    for fn in (GeometricLidarVelocityTracker.update, GeometricLidarVelocityTracker.detect):
        params = set(inspect.signature(fn).parameters) - {"self"}
        assert params == {"p_ego", "ranges"}


def test_a3_no_privileged_imports():
    tree = ast.parse(Path("dr_control/tracking2.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    banned = {"robot_env.robot_nav_env", "robot_env.dynamic_obstacles",
              "dr_control.dynamic_cbf", "dr_control.capped_velocity"}
    assert not (mods & banned), mods & banned


def test_a3_no_obstacle_speed_constant_appears():
    """The configured obstacle speeds are a Stage-2 METRIC threshold and Stage-2b's privileged
    input. Neither may appear in an intervention."""
    src = _executable_source("dr_control/tracking2.py")
    for lit in ("0.675", "0.75", "obstacle_speed", "obstacle_velocit"):
        assert lit not in src, lit


def test_a3_runtime_tripwire_rejects_extra_arguments():
    tr = GeometricLidarVelocityTracker(r_nominal=R_NOM)
    with pytest.raises(TypeError):
        tr.update(np.zeros(2), np.full(24, 5.0), np.zeros((6, 2)))


def test_a3_action_is_invariant_to_ground_truth_observation_slice():
    """obs[28:52] carries exact obstacle positions and velocities. E1 must never read it."""
    env = make_env(True)
    obs, _ = env.reset(seed=11)
    acts = []
    for corrupt in (False, True):
        pol = Phase7Policy(arm="E1_geometric_R", v_cap=V_CAP_DERIVED,
                           world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                           static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                           dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS)
        o = obs.copy()
        if corrupt:
            o[28:52] = -7.5
        pol.reset(o, env.agent_position)
        a, _ = pol.predict(o, env.agent_position)
        acts.append(np.asarray(a, float))
    assert np.array_equal(acts[0], acts[1])


# --------------------------------------------------------------- wiring
def test_e1_arm_is_t1_in_every_respect_but_the_tracker():
    env = make_env(True)
    obs, _ = env.reset(seed=3)
    built = {}
    for arm in ("T1_projection_cap", "E1_geometric_R"):
        pol = Phase7Policy(arm=arm, v_cap=V_CAP_DERIVED, world_size=env.WORLD_SIZE,
                           agent_radius=env.AGENT_RADIUS, static_obstacles=env.static_obstacles,
                           max_speed=env.MAX_SPEED, dt=env.dt, lidar_range=env.LIDAR_RANGE,
                           n_rays=env.N_LIDAR_RAYS)
        pol.reset(obs, env.agent_position)
        built[arm] = pol.src
    for src in built.values():
        assert isinstance(src, ProjectionCappedSource)
        assert src.v_cap == V_CAP_DERIVED
    assert type(built["T1_projection_cap"].tracker) is LidarVelocityTracker
    assert type(built["E1_geometric_R"].tracker) is GeometricLidarVelocityTracker


def test_existing_arms_keep_the_frozen_tracker():
    assert ARMS == ("C0_frozen", "T1_projection_cap", "T2_vector_cap", "D_oracle",
                    "E1_geometric_R")
    env = make_env(True)
    obs, _ = env.reset(seed=5)
    for arm in ("T1_projection_cap", "T2_vector_cap", "D_oracle"):
        pol = Phase7Policy(arm=arm, world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                           static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                           dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS)
        pol.reset(obs, env.agent_position)
        assert type(pol.src.tracker) is LidarVelocityTracker
