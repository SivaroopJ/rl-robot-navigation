"""Hunter variants B1 / B1.5 / P2 / P2.5: focused tests (plan section 14, Tests A-E)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from continuation.pursuit.config import PURSUIT_BLOCKS, PursuitConfig
from continuation.pursuit.harness import build_protagonist, make_pursuit_env, run_episode
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.hunter_policies import (MAX_PREDICTION_HORIZON, VARIANT_BLOCKS,
                                                  VARIANTS, HunterVariant, ProtagFilteredSource,
                                                  direct_target, goal_direction,
                                                  predict_position)
from continuation.pursuit.source import HunterAugmentedSource
from continuation.seeds import BLOCKS as NAV_BLOCKS
from dr_control.drccp_controller import clf_terms
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import LidarVelocityTracker
from experiments.week6_continuation.common import load_random, load_tuned

REPO = Path(__file__).resolve().parents[1]
SEED = 11_000_003
TUNED, _ = load_tuned()
SPEC, _ = load_random()
PRED = dict(dt=0.1, v_max=1.0, world_size=10.0, body_radius=0.3)


@pytest.fixture(scope="module")
def cfg():
    return PursuitConfig()


@pytest.fixture()
def staged(cfg):
    """Protag at (5,5), hunter 1 m to its +x, dynamic obstacles parked far away: in the hunter's
    scan Protag's surface (0.7 m) is the nearest return, the rectangle at (7,3) is next (1.5 m)."""
    e = make_pursuit_env(cfg)
    e.reset(seed=SEED)
    e.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    e.hunter_position = np.array([6.0, 5.0])
    e.obstacle_positions = np.tile(np.array([[9.2, 0.8]], dtype=np.float32),
                                   (e.n_dynamic_obstacles, 1))
    yield e
    e.close()


def _hunter(filter_radius=None):
    return HunterController(params=TUNED, max_speed=1.0, filter_radius=filter_radius)


def _is_protag_row(row, p_h, p_e, r=0.3):
    g = (p_h - p_e) / np.linalg.norm(p_h - p_e)
    return (abs(float(row[1]) - (np.linalg.norm(p_h - p_e) - r - r)) < 1e-4
            and np.allclose(row[2:4], g, atol=1e-4))


# =============================================================== Test A: direct pursuit
def test_direct_target_is_protags_current_position():
    p = np.array([3.2, 4.1])
    t = direct_target(p)
    assert np.array_equal(t, p) and t is not p


def test_direct_clf_and_action_reduce_hunter_protag_distance(staged):
    e = staged
    h = _hunter()
    for _ in range(5):                                   # fill the scan buffer as an episode does
        u, info = h.act(e.hunter_position, e.hunter_lidar(), e.agent_position)
    V, dV = clf_terms(e.hunter_position, e.agent_position, TUNED.k_v)
    assert V > 0 and float(dV @ info["u_nom"]) < 0       # nominal action descends V
    d0 = np.linalg.norm(e.hunter_position - e.agent_position)
    assert np.linalg.norm(e.hunter_position + 0.1 * u - e.agent_position) < d0


def test_hunter_does_not_treat_protag_as_a_barrier_row_in_either_lidar_mode(staged):
    e = staged
    p_h, p_e = e.hunter_position, e.agent_position.astype(float)
    # sensitivity check: an UNFILTERED hunter fed the Protag-rendering scan does get the row
    raw = _hunter()
    _, info = raw.act(p_h, e.hunter_lidar(include_protagonist=True), p_e)
    assert any(_is_protag_row(r, p_h, p_e) for r in np.atleast_2d(info["xi_kept"]))
    for mode in ("remove", "filter"):
        v = HunterVariant("t", lidar_mode=mode)
        h = _hunter(v.filter_radius(0.3) if mode == "filter" else None)
        _, info = h.act(p_h, e.hunter_lidar(include_protagonist=v.protag_in_hunter_lidar), p_e,
                        filter_reference=p_e if mode == "filter" else None)
        assert not any(_is_protag_row(r, p_h, p_e) for r in np.atleast_2d(info["xi_kept"])), mode


def test_zero_relative_distance_is_handled(staged):
    e = staged
    h = _hunter()
    u, info = h.act(e.hunter_position, e.hunter_lidar(), e.hunter_position.copy())
    assert np.array_equal(info["u_nom"], np.zeros(2)) and np.all(np.isfinite(u))


# =============================================================== Test B: predictive pursuit
def test_prediction_matches_the_closed_form_when_no_speed_clip_binds():
    p, v, g = np.array([2.0, 2.0]), np.array([0.3, 0.1]), np.array([8.0, 2.0])
    q, info = predict_position(p, v, g, horizon=0.5, goal_accel=1.0, **PRED)
    want = p + v * 0.5 + 0.5 * 1.0 * np.array([1.0, 0.0]) * 0.25
    assert np.allclose(q, want, atol=1e-12) and not info["clipped"]


def test_prediction_changes_with_velocity_and_with_goal_direction():
    p, g = np.array([5.0, 5.0]), np.array([8.0, 5.0])
    base, _ = predict_position(p, np.array([0.2, 0.0]), g, horizon=0.5, goal_accel=1.0, **PRED)
    faster, _ = predict_position(p, np.array([0.6, 0.0]), g, horizon=0.5, goal_accel=1.0, **PRED)
    other_goal, _ = predict_position(p, np.array([0.2, 0.0]), np.array([5.0, 8.0]),
                                     horizon=0.5, goal_accel=1.0, **PRED)
    assert not np.allclose(base, faster) and not np.allclose(base, other_goal)
    assert other_goal[1] > base[1]                       # pulled toward the +y goal


def test_zero_velocity_and_protag_at_goal_are_safe():
    p = np.array([4.0, 4.0])
    q, info = predict_position(p, np.zeros(2), p.copy(), horizon=0.5, goal_accel=1.0, **PRED)
    assert np.array_equal(q, p) and np.array_equal(info["d_g"], np.zeros(2))
    assert np.array_equal(goal_direction(p, p + 1e-12), np.zeros(2))
    q, _ = predict_position(p, np.zeros(2), np.array([4.0, 9.0]), horizon=0.5, goal_accel=1.0,
                            **PRED)
    assert np.allclose(q, p + np.array([0.0, 0.125]))    # only the goal-directed term


def test_prediction_horizon_is_bounded_and_respected():
    p, v, g = np.array([5.0, 5.0]), np.array([1.0, 0.0]), np.array([9.0, 5.0])
    q0, _ = predict_position(p, v, g, horizon=0.0, goal_accel=1.0, **PRED)
    assert np.array_equal(q0, p)
    with pytest.raises(ValueError):
        predict_position(p, v, g, horizon=MAX_PREDICTION_HORIZON + 0.1, goal_accel=1.0, **PRED)
    with pytest.raises(ValueError):
        HunterVariant("bad", policy="predictive", prediction_horizon=5.0)
    # already at Protag's per-axis speed bound: acceleration cannot push it further
    q, info = predict_position(p, v, g, horizon=0.5, goal_accel=1.0, **PRED)
    assert info["clipped"] and q[0] - p[0] <= 1.0 * 0.5 + 1e-12


def test_prediction_stays_finite_and_inside_the_arena():
    q, _ = predict_position(np.array([9.6, 5.0]), np.array([1.0, 1.0]), np.array([9.7, 9.7]),
                            horizon=2.0, goal_accel=5.0, **PRED)
    assert np.all(np.isfinite(q)) and np.all(q <= 10.0 - 0.3) and np.all(q >= 0.3)
    q, info = predict_position(np.array([5.0, 5.0]), np.array([np.nan, 0.0]),
                               np.array([8.0, 8.0]), horizon=0.5, goal_accel=1.0, **PRED)
    assert info["fallback"] and np.array_equal(q, [5.0, 5.0])


def test_prediction_is_recalculated_every_step_from_the_pre_step_state(cfg):
    rec = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, variant=VARIANTS["P2"], trace=True)
    tr = rec["trace"]
    assert len(tr) == rec["steps"] > 5
    e = make_pursuit_env(cfg)
    for s in tr:
        want, _ = predict_position(s["p_prot"], s["v_prot"], s["goal"], horizon=0.5,
                                   goal_accel=1.0, dt=e.dt, v_max=e.MAX_SPEED,
                                   world_size=e.WORLD_SIZE, body_radius=e.AGENT_RADIUS)
        assert np.allclose(s["p_target"], want, atol=1e-12)
    e.close()
    assert len({tuple(np.round(s["p_target"], 9)) for s in tr}) > len(tr) // 2
    assert rec["prediction_horizon_steps"] == 5 and rec["prediction_n_evaluable"] > 0


# =============================================================== Test C: LiDAR removal
def test_protag_is_absent_from_the_default_hunter_scan_and_present_when_rendered(staged):
    e = staged
    removed, rendered = e.hunter_lidar(), e.hunter_lidar(include_protagonist=True)
    ray_minus_x = e.N_LIDAR_RAYS // 2                    # bearing pi: straight at Protag
    assert rendered[ray_minus_x] == pytest.approx(1.0 - 0.3, abs=1e-5)
    assert removed[ray_minus_x] > 1.0                    # passes through Protag
    hit = rendered < removed
    assert hit.any() and np.array_equal(rendered[~hit], removed[~hit])   # env returns unchanged


def test_environmental_obstacles_remain_detectable_in_both_scans(staged):
    e = staged
    e.obstacle_positions[0] = np.array([6.0, 6.0], dtype=np.float32)    # 1 m above the hunter
    up = e.N_LIDAR_RAYS // 4
    for inc in (False, True):
        assert e.hunter_lidar(include_protagonist=inc)[up] == pytest.approx(0.7, abs=1e-5)


def test_rendering_protag_in_the_hunter_scan_changes_neither_the_world_nor_protags_scan(staged):
    e = staged
    before = (e.agent_position.copy(), e.obstacle_positions.copy(), e.hunter_position.copy())
    scan_p = e._cast_lidar_rays().copy()
    e.hunter_lidar(include_protagonist=True)
    assert np.array_equal(before[0], e.agent_position)
    assert np.array_equal(before[1], e.obstacle_positions)
    assert np.array_equal(before[2], e.hunter_position)
    assert np.array_equal(scan_p, e._cast_lidar_rays())


# =============================================================== Test D: CBF filtering
def _sources(filter_radius=0.35):
    kw = dict(r_robot=0.3, k_scans=5, n_rays=24, lidar_range=5.0)
    return (EstimatedLidarBarrierSource(tracker=LidarVelocityTracker(), **kw),
            ProtagFilteredSource(filter_radius=filter_radius, tracker=LidarVelocityTracker(),
                                 **kw))


def test_filter_drops_only_candidates_within_the_radius(staged):
    e = staged
    ranges = e.hunter_lidar(include_protagonist=True)
    plain, filt = _sources()
    plain.push(e.hunter_position, ranges)
    filt.push(e.hunter_position, ranges, exclude_center=e.agent_position)
    p_e = e.agent_position.astype(float)
    d_plain = np.linalg.norm(plain.buffer[0] - p_e, axis=1)
    kept_plain = plain.buffer[0][d_plain > filt.filter_radius]
    assert (d_plain <= filt.filter_radius).sum() > 0
    assert np.array_equal(filt.buffer[0], kept_plain)    # survivors are bit-identical
    st = filt.filter_stats()
    assert st["filtered_protag"] == (d_plain <= filt.filter_radius).sum()
    assert st["filtered_env_false_positive"] == 0 and st["leaked_protag"] == 0


def test_raw_lidar_still_reaches_the_hunter_and_its_tracker(staged, monkeypatch):
    e = staged
    ranges = e.hunter_lidar(include_protagonist=True)
    copy = ranges.copy()
    h = _hunter(0.35)
    seen = []
    orig = h.src.tracker.update
    monkeypatch.setattr(h.src.tracker, "update", lambda p, r: (seen.append(np.array(r)),
                                                               orig(p, r))[1])
    h.act(e.hunter_position, ranges, e.agent_position, filter_reference=e.agent_position)
    assert np.array_equal(ranges, copy)                  # observation not mutated
    assert np.array_equal(seen[0], copy)                 # tracker saw the unfiltered scan


def test_environmental_point_near_protag_is_filtered_and_counted_as_false_positive(staged):
    e = staged
    # an obstacle surface 2 cm from Protag's surface, between hunter and Protag's flank
    e.obstacle_positions[0] = np.array([5.0, 5.62], dtype=np.float32)
    ranges = e.hunter_lidar(include_protagonist=True)
    _, filt = _sources(filter_radius=0.35)
    filt.push(e.hunter_position, ranges, exclude_center=e.agent_position)
    _, wide = _sources(filter_radius=2.0)
    wide.push(e.hunter_position, ranges, exclude_center=e.agent_position)
    assert wide.filter_stats()["filtered_env_false_positive"] > 0     # the limitation is logged
    assert filt.filter_stats()["filtered_env_false_positive"] <= \
        wide.filter_stats()["filtered_env_false_positive"]


def test_filtering_hunter_requires_a_reference_and_the_other_refuses_one(staged):
    e = staged
    with pytest.raises(ValueError):
        _hunter(0.35).act(e.hunter_position, e.hunter_lidar(), e.agent_position)
    with pytest.raises(ValueError):
        _hunter().act(e.hunter_position, e.hunter_lidar(), e.agent_position,
                      filter_reference=e.agent_position)


def test_filter_reference_is_current_position_not_the_prediction(cfg, monkeypatch):
    """P2.5: the pursuit target is p_hat, but the filter centre must be the pre-step p_e."""
    calls = []
    orig = ProtagFilteredSource.push

    def spy(self, p, ranges, *, exclude_center=None):
        calls.append(np.asarray(exclude_center, float).copy())
        return orig(self, p, ranges, exclude_center=exclude_center)

    monkeypatch.setattr(ProtagFilteredSource, "push", spy)
    rec = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, variant=VARIANTS["P2.5"], trace=True)
    assert len(calls) == rec["steps"]
    for c, s in zip(calls, rec["trace"]):
        assert np.array_equal(c, s["p_prot"])
    assert any(not np.allclose(s["p_target"], s["p_prot"]) for s in rec["trace"])


def test_hunter_collision_and_protag_cbf_are_untouched_by_the_lidar_mode(cfg, staged):
    e = staged
    obs, _ = e.reset(seed=SEED)
    pol, wrap = build_protagonist(e, obs, cfg, TUNED, SPEC, SEED)
    assert type(pol.src) is HunterAugmentedSource       # Protag's own barrier source
    wrap.detach()
    e.hunter_position = e.obstacle_positions[0].astype(float) + np.array([0.5, 0.0])
    e.hunter_disabled = False
    assert e._hunter_collision_type() == "dynamic"      # env geometry; no variant input exists


def test_filter_stats_in_an_episode_show_no_leaks(cfg):
    rec = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, variant=VARIANTS["B1.5"])
    st = rec["filter_stats"]
    assert st["leaked_protag"] == 0 and st["points_total"] > 0
    assert rec["filter_radius"] == pytest.approx(0.35)


# =============================================================== Test E: integration
def test_b1_reproduces_the_audited_hunter_exactly(cfg):
    old = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC)
    new = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, variant=VARIANTS["B1"])
    skip = ("_ms_mean", "_ms_p95", "episode_time_s")
    for k, v in old.items():
        if not k.endswith(skip):
            assert new[k] == v or (v != v and new[k] != new[k]), k


@pytest.mark.parametrize("name", list(VARIANTS))
def test_all_four_variants_run_on_m0_with_consistent_outcomes(cfg, name):
    e = make_pursuit_env(cfg)
    assert e.n_dynamic_obstacles == 6 and len(e.static_obstacles) == 5
    assert type(e.motion_model).__name__ == "SmoothStochasticMotion"
    rec = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, env=e, variant=VARIANTS[name],
                      trace=True)
    e.close()
    tr = rec["trace"]
    assert rec["hunter_variant"]["name"] == name
    assert rec["outcome"] in ("protagonist_collision", "capture", "goal", "timeout")
    assert rec["path_length"] > 0.5                                  # Protag navigates
    hp = np.asarray([s["p_hunter"] for s in tr])
    assert np.linalg.norm(hp[-1] - hp[0]) > 0.3 or rec["hunter_collision"]   # hunter moves
    nom = np.asarray([s["u_hunter_nominal"] for s in tr])
    dirs = np.asarray([np.subtract(s["p_target"], s["p_hunter"]) for s in tr])
    ok = np.linalg.norm(dirs, axis=1) > 1e-6
    assert np.all(np.einsum("ij,ij->i", nom[ok], dirs[ok]) > 0)     # pursues its target
    assert np.all(np.abs(np.asarray([s["u_hunter"] for s in tr])) <= 1.0 + 1e-6)
    assert (rec["outcome"] == "capture") == bool(rec["captured"]) or rec["protagonist_collision"]
    if rec["outcome"] == "capture":
        assert rec["capture_distance_at_capture"] <= 0.65 + 1e-9
    assert rec["hunter_n_no_barrier_rows"] == 0


def test_variants_share_layout_for_a_seed(cfg):
    e = make_pursuit_env(cfg)
    lay = set()
    for name in VARIANTS:
        e.reset(seed=SEED)
        lay.add((tuple(e.agent_position), tuple(e.target_position), tuple(e.hunter_position)))
    e.close()
    assert len(lay) == 1


def test_variant_seed_blocks_are_fresh_and_disjoint():
    used = [(b, b + 100_000) for b, _ in NAV_BLOCKS.values()]
    used += [(b, b + s) for b, s in PURSUIT_BLOCKS.values()]
    used += [(12_000_000, 13_999_999)]                     # Week 5.5 and Tracks A / A2 / B
    for name, (base, size) in VARIANT_BLOCKS.items():
        for lo, hi in used:
            assert base + size <= lo or base >= hi, name


def test_hunter_policy_module_adds_no_planner_or_learning():
    src = (REPO / "continuation/pursuit/hunter_policies.py").read_text()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert not any(k in m.lower() for m in mods for k in
                   ("planner", "astar", "a_star", "torch", "stable_baselines", "ladder"))
