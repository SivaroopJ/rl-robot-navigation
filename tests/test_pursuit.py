"""Pursuit-evasion: validation tests (plan section 11)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from continuation.params import DRCBFParams
from continuation.pursuit.config import PursuitConfig, PURSUIT_BLOCKS, pursuit_seeds
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.harness import build_protagonist, make_pursuit_env, run_episode
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.source import HunterAugmentedSource
from continuation.seeds import BLOCKS as NAV_BLOCKS
from experiments.exp6_ppo_comparison import make_env as make_nav_env
from experiments.week6_continuation.common import load_random, load_tuned

REPO = Path(__file__).resolve().parents[1]
SEED = 11_000_003
TUNED, _TF = load_tuned()
SPEC, _RF = load_random()


@pytest.fixture(scope="module")
def cfg():
    return PursuitConfig()


@pytest.fixture(scope="module")
def env(cfg):
    e = make_pursuit_env(cfg)
    yield e
    e.close()


# ----------------------------------------------------------------- capture / contact geometry
def test_capture_threshold_is_the_declared_value(cfg):
    assert cfg.capture_distance(0.3) == pytest.approx(0.65)
    assert cfg.contact_distance(0.3) == pytest.approx(0.60)
    assert cfg.capture_distance(0.3) > cfg.contact_distance(0.3)      # 5 cm margin before contact


def test_capture_fires_exactly_at_the_threshold(env, cfg):
    env.reset(seed=SEED)
    env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    # 0.65 exactly is a float-boundary case (5.0 + 0.65 lands at 0.6500000000000004 in float64),
    # so the threshold constant is asserted separately and the rule is probed either side of it.
    for d, want in ((0.60, True), (0.649, True), (0.651, False), (0.70, False)):
        env.hunter_position = np.array([5.0 + d, 5.0])
        env.hunter_disabled = False
        _, _, _, _, info = env.step_pursuit(np.zeros(2), np.zeros(2))
        assert info["captured"] is want, (d, info["hunter_distance"])


def test_capture_and_contact_are_recorded_separately(env):
    env.reset(seed=SEED)
    env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    env.hunter_position = np.array([5.62, 5.0])          # inside capture, outside contact
    env.hunter_disabled = False
    _, _, _, _, info = env.step_pursuit(np.zeros(2), np.zeros(2))
    assert info["captured"] and not info["agent_contact"]
    assert info["protagonist_collision_type"] is None    # capture is NOT a collision


def test_disabled_hunter_cannot_capture(env):
    env.reset(seed=SEED)
    env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    env.hunter_position = np.array([5.3, 5.0])
    env.hunter_disabled = True
    _, _, _, _, info = env.step_pursuit(np.zeros(2), np.zeros(2))
    assert not info["captured"] and info["hunter_disabled"]


# ----------------------------------------------------------------- collisions
def test_hunter_wall_collision_disables_it_and_episode_continues(env):
    env.reset(seed=SEED)
    env.hunter_position = np.array([0.35, 5.0])
    env.hunter_disabled = False
    _, _, term, _, info = env.step_pursuit(np.zeros(2), np.array([-1.0, 0.0]))
    assert info["hunter_collision_type"] == "wall" and info["hunter_disabled"]
    assert not term                                        # protagonist outcome still measurable


def test_hunter_static_collision_is_detected(env):
    env.reset(seed=SEED)
    cx, cy, hw, hh = env.static_obstacles[0]
    env.hunter_position = np.array([cx + hw + 0.31, cy])
    # Park every dynamic obstacle in the corner farthest from the hunter so the rectangle is the
    # only thing it can hit (emptying the array would break the env's own proximity reward).
    far = np.array([9.6, 9.6]) if np.linalg.norm(env.hunter_position - np.array([9.6, 9.6])) > 4 \
        else np.array([0.4, 0.4])
    env.obstacle_positions = np.tile(far, (len(env.obstacle_positions), 1)).astype(np.float32)
    env.hunter_disabled = False
    _, _, _, _, info = env.step_pursuit(np.zeros(2), np.array([-1.0, 0.0]))
    assert info["hunter_collision_type"] == "static"


def test_frozen_protagonist_collision_semantics_are_unchanged(env):
    """A protagonist obstacle collision still terminates and still reports its type."""
    env.reset(seed=SEED)
    env.agent_position = np.array([0.35, 5.0], dtype=np.float32)
    env.hunter_position = np.array([9.0, 9.0])
    _, _, term, _, info = env.step_pursuit(np.array([-1.0, 0.0]), np.zeros(2))
    assert info["protagonist_collision_type"] == "wall" and term
    assert info["outcome"] == "protagonist_collision"


# ----------------------------------------------------------------- simultaneity
def test_hunter_cannot_react_to_the_protagonists_new_position(env):
    """Same pre-step state + same actions => same hunter result, regardless of protagonist motion.

    If the hunter were stepped AFTER the protagonist and re-planned, moving the protagonist would
    change the hunter's post-step position. It must not.
    """
    out = []
    for a_prot in (np.array([1.0, 0.0]), np.array([-1.0, 0.0])):
        env.reset(seed=SEED)
        env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
        env.hunter_position = np.array([3.0, 5.0])
        _, _, _, _, info = env.step_pursuit(a_prot, np.array([1.0, 0.0]))
        out.append(info["hunter_position"].copy())
    assert np.array_equal(out[0], out[1])


def test_both_controllers_read_one_shared_pre_step_state(env):
    env.reset(seed=SEED)
    pre = env.pre_step_state()
    assert np.array_equal(pre["p_prot"], env.agent_position)
    assert np.array_equal(pre["p_hunter"], env.hunter_position)
    env.step_pursuit(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert not np.array_equal(pre["p_prot"], env.agent_position)      # pre-state is a snapshot
    assert not np.array_equal(pre["p_hunter"], env.hunter_position)


# ----------------------------------------------------------------- episode layout / seeds
def test_protagonist_layout_matches_the_navigation_env_for_the_same_seed(cfg):
    """The hunter is drawn AFTER start, goal and obstacles, so the navigation layout is intact."""
    pe = make_pursuit_env(cfg)
    ne = make_nav_env(True)
    for s in (11_000_000, 11_000_001):
        o1, _ = pe.reset(seed=s)
        o2, _ = ne.reset(seed=s)
        assert np.array_equal(pe.agent_position, ne.agent_position)
        assert np.array_equal(pe.target_position, ne.target_position)
        assert np.array_equal(pe.obstacle_positions, ne.obstacle_positions)
        assert np.array_equal(o1, o2)
    pe.close(); ne.close()


def test_hunter_spawns_far_enough_and_in_free_space(cfg):
    e = make_pursuit_env(cfg)
    for s in pursuit_seeds("P1_sanity", 10):
        e.reset(seed=s)
        d = np.linalg.norm(e.hunter_position - e.agent_position)
        assert d >= cfg.hunter_start_min_dist - 1e-6
        assert e._hunter_collision_type() is None
    e.close()


def test_pursuit_seed_blocks_are_disjoint_from_every_navigation_block():
    nav = [(b, b + 100_000) for b, _ in NAV_BLOCKS.values()]
    for name, (base, size) in PURSUIT_BLOCKS.items():
        assert base >= 11_000_000, name
        for lo, hi in nav:
            assert base + size <= lo or base >= hi, name


# ----------------------------------------------------------------- controllers
def test_protagonist_is_the_frozen_config_and_uses_no_ladder(cfg, env):
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist(env, obs, cfg, TUNED, SPEC, SEED)
    assert TUNED == DRCBFParams(alpha=0.8, clf_rate=1.0, wasserstein_r=0.012, epsilon=0.1,
                                k_v=0.10, k_scans=5)
    assert SPEC.K == 16 and SPEC.H == 1 and SPEC.speeds == (0.8, 1.0) and SPEC.accept_m == 0.0
    assert type(wrap).__name__ == "RandomRecovery"
    assert pol.ctrl.rateh == 0.8 and pol.ctrl.wasserstein_r == 0.012
    wrap.detach()


def test_p_tuned_condition_attaches_no_recovery(cfg, env):
    obs, _ = env.reset(seed=SEED)
    _, wrap = build_protagonist(env, obs, PursuitConfig(condition="P-Tuned"), TUNED, SPEC, SEED)
    assert wrap is None


def test_no_ladder_anywhere_in_the_pursuit_package():
    for f in sorted((REPO / "continuation/pursuit").glob("*.py")) + \
            sorted((REPO / "experiments/pursuit").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        src = ast.unparse(tree)
        assert "RecoveryLadder" not in src, f.name
        assert "dr_control.recovery" not in src, f.name


def test_hunter_row_uses_the_frozen_barrier_convention():
    src = HunterAugmentedSource(hunter_radius=0.3, r_robot=0.3, k_scans=5, n_rays=24,
                                lidar_range=5.0)
    assert src.hunter_row(np.zeros(2)) is None            # no state pushed -> no row
    src.push_hunter([2.0, 0.0], [-1.0, 0.0])
    h, g, dd = src.hunter_row([0.0, 0.0])
    assert h == pytest.approx(2.0 - 0.3 - 0.3)
    assert np.allclose(g, [-1.0, 0.0])                    # points away from the hunter
    assert dd == pytest.approx(-1.0)                      # closing at 1 m/s -> h decreasing


def test_hunter_row_is_appended_without_touching_the_lidar_rows(cfg, env):
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist(env, obs, cfg, TUNED, SPEC, SEED)
    p = env.agent_position
    pol.src.push(p, obs[4:28] * env.LIDAR_RANGE)
    h0, g0, d0 = super(HunterAugmentedSource, pol.src).samples(p)
    pol.src.push_hunter(env.hunter_position, env.hunter_velocity)
    h1, g1, d1 = pol.src.samples(p)
    assert len(h1) == len(h0) + 1
    assert np.array_equal(h1[:-1], h0) and np.array_equal(d1[:-1], d0)
    assert np.array_equal(g1[:, :-1], g0)
    wrap.detach()


def test_hunter_pursues_and_is_bounded(cfg):
    e = make_pursuit_env(cfg)
    obs, _ = e.reset(seed=SEED)
    h = HunterController(params=TUNED, max_speed=cfg.hunter_max_speed)
    u, info = h.act(e.hunter_position, e.hunter_lidar(), e.agent_position)
    assert np.all(np.abs(u) <= cfg.hunter_max_speed + 1e-9)
    to_target = e.agent_position - e.hunter_position
    assert float(info["u_nom"] @ to_target) > 0           # nominal action closes distance
    e.close()


def test_hunter_falls_back_to_zero_and_keeps_no_recovery(cfg):
    h = HunterController(params=TUNED, max_speed=cfg.hunter_max_speed)
    assert not hasattr(h, "recovery")
    assert h.ctrl.max_v == cfg.hunter_max_speed
    assert (h.ctrl.rateh, h.ctrl.wasserstein_r, h.ctrl.epsilon, h.ctrl.k_v) == (0.8, 0.012, 0.1, 0.10)


# ----------------------------------------------------------------- episodes / RNG / timeout
def test_episode_is_reproducible_and_rng_streams_are_independent(cfg):
    e = make_pursuit_env(cfg)
    a = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, env=e)
    b = run_episode(cfg, SEED, tuned=TUNED, spec=SPEC, env=e)
    keys = ("outcome", "steps", "goal", "captured", "min_hunter_distance", "path_length",
            "prot_n_recovery_events", "hunter_n_infeasible")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    e.close()


def test_timeout_is_reported_when_nothing_else_ends_the_episode(cfg):
    e = make_pursuit_env(cfg)
    e.reset(seed=SEED)
    e.MAX_STEPS = 3
    for i in range(3):
        _, _, term, trunc, info = e.step_pursuit(np.zeros(2), np.zeros(2))
    assert trunc and not term and info["outcome"] == "timeout"
    e.close()


def test_outcome_priority_is_collision_then_capture_then_goal():
    from continuation.pursuit.config import OUTCOME_PRIORITY
    assert OUTCOME_PRIORITY == ("protagonist_collision", "capture", "goal", "timeout")


# ----------------------------------------------------------------- frozen files
def test_no_frozen_or_historical_file_has_changed():
    from experiments.week6_continuation.protect_manifest import verify
    modified, missing = verify()
    assert not modified and not missing, (modified, missing)


def test_frozen_configs_still_match_their_hashes():
    import hashlib
    for stem in ("H2/tuned_frozen", "R3/random_frozen"):
        p = REPO / "results/week6_continuation" / f"{stem}.json"
        want = (REPO / "results/week6_continuation" / f"{stem}.sha256").read_text().split()[0]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == want


# ----------------------------------------------------------------- same-step outcome priority
def test_capture_beats_goal_on_the_same_step(env):
    """Contested step: protagonist on its goal AND hunter within capture range -> capture wins."""
    env.reset(seed=SEED)
    env.agent_position = (env.target_position - np.array([0.02, 0.0])).astype(np.float32)
    env.hunter_position = np.array(env.target_position, dtype=float) + np.array([0.4, 0.0])
    env.hunter_disabled = False
    _, _, term, _, info = env.step_pursuit(np.array([1.0, 0.0]), np.zeros(2))
    assert info["goal"] and info["captured"]          # both flags recorded
    assert info["outcome"] == "capture" and term      # priority resolves to capture


def test_protagonist_collision_beats_capture_on_the_same_step(env):
    """Own-collision outranks capture, so a contested step is never scored as a capture."""
    env.reset(seed=SEED)
    env.agent_position = np.array([0.35, 5.0], dtype=np.float32)
    env.hunter_position = np.array([0.35 + 0.5, 5.0])
    env.hunter_disabled = False
    _, _, term, _, info = env.step_pursuit(np.array([-1.0, 0.0]), np.zeros(2))
    assert info["protagonist_collision_type"] == "wall" and info["captured"]
    assert info["outcome"] == "protagonist_collision" and term


def test_every_outcome_flag_is_logged_whichever_one_wins(env):
    env.reset(seed=SEED)
    _, _, _, _, info = env.step_pursuit(np.zeros(2), np.zeros(2))
    for k in ("goal", "captured", "agent_contact", "protagonist_collision_type",
              "hunter_collision_type", "hunter_distance", "outcome", "hunter_disabled"):
        assert k in info, k


# ----------------------------------------------------------------- hunter barrier inputs
def test_hunter_lidar_is_cast_from_the_hunters_own_position(cfg, env):
    obs, _ = env.reset(seed=SEED)
    ranges_h = env.hunter_lidar()
    ranges_p = obs[4:4 + env.N_LIDAR_RAYS] * env.LIDAR_RANGE
    assert ranges_h.shape == (env.N_LIDAR_RAYS,)
    assert not np.allclose(ranges_h, ranges_p)        # different vantage point
    env.hunter_position = env.agent_position.astype(float).copy()
    assert np.allclose(env.hunter_lidar(), ranges_p, atol=1e-5)   # same point -> same scan


def test_hunter_barrier_rows_exclude_the_protagonist(cfg, env):
    """The hunter must not treat its target as an obstacle."""
    obs, _ = env.reset(seed=SEED)
    h = HunterController(params=TUNED, max_speed=cfg.hunter_max_speed)
    assert type(h.src) is not HunterAugmentedSource      # no hunter/target oracle row
    assert not hasattr(h.src, "push_hunter")
    env.hunter_position = env.agent_position.astype(float) + np.array([1.0, 0.0])
    u, info = h.act(env.hunter_position, env.hunter_lidar(), env.agent_position)
    rows = np.atleast_2d(info["xi_kept"])
    p_dir = (env.hunter_position - env.agent_position)
    p_dir = p_dir / np.linalg.norm(p_dir)
    # no kept row is the protagonist: none points from the protagonist at that exact range
    assert not any(abs(float(r[1]) - (1.0 - 0.6)) < 1e-6 and np.allclose(r[2:4], p_dir, atol=1e-6)
                   for r in rows)


def test_hunter_uses_the_shared_pre_step_state_for_relative_motion(cfg, env):
    """The hunter's nominal action points at the PRE-step protagonist position."""
    env.reset(seed=SEED)
    pre = env.pre_step_state()
    h = HunterController(params=TUNED, max_speed=cfg.hunter_max_speed)
    _, info = h.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
    want = pre["p_prot"] - pre["p_hunter"]
    got = info["u_nom"]
    assert np.allclose(got / np.linalg.norm(got), want / np.linalg.norm(want), atol=1e-9)


# ----------------------------------------------------------------- stochastic obstacle seeding
def test_p3_env_uses_the_stochastic_motion_model(cfg):
    e = make_pursuit_env(cfg)
    assert cfg.motion == "randomized"
    assert type(e.motion_model).__name__ == "SmoothStochasticMotion"
    assert e.n_dynamic_obstacles == 6 and len(e.static_obstacles) == 5
    assert e.obstacle_speed == pytest.approx(0.675)
    e.close()


def test_stochastic_obstacles_are_seed_reproducible_and_seed_dependent(cfg):
    def roll(seed, n=25):
        e = make_pursuit_env(cfg)
        e.reset(seed=seed)
        out = []
        for _ in range(n):
            e.step_pursuit(np.zeros(2), np.zeros(2))
            out.append(e.obstacle_positions.copy())
        e.close()
        return np.asarray(out)
    a, b, c = roll(11_300_000), roll(11_300_000), roll(11_300_001)
    assert np.array_equal(a, b)                      # same seed -> same obstacle trajectory
    assert not np.array_equal(a, c)                  # different seed -> different


def test_obstacle_trajectories_do_not_depend_on_either_agent(cfg):
    """Pairing requirement: obstacles must be identical whatever the two agents do."""
    def roll(u_prot, u_hunt, n=25):
        e = make_pursuit_env(cfg)
        e.reset(seed=11_300_002)
        out = []
        for _ in range(n):
            e.step_pursuit(u_prot, u_hunt)
            out.append(e.obstacle_positions.copy())
        e.close()
        return np.asarray(out)
    assert np.array_equal(roll(np.zeros(2), np.zeros(2)),
                          roll(np.array([1.0, 0.5]), np.array([-1.0, 0.3])))


# ----------------------------------------------------------------- Random recovery preserved
def test_random_recovery_is_the_frozen_algorithm_and_fires_on_infeasibility(cfg, env):
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist(env, obs, cfg, TUNED, SPEC, SEED)
    assert wrap.spec is SPEC and wrap.tau == pytest.approx(TUNED.tau_eff())
    assert wrap.max_v == pol.ctrl.max_v
    for _ in range(120):                              # drive until the QP goes infeasible
        pre = env.pre_step_state()
        pol.src.push_hunter(pre["p_hunter"], pre["v_hunter"])
        a, _ = pol.predict(obs, pre["p_prot"])
        obs, _, term, trunc, _ = env.step_pursuit(a, np.zeros(2))
        if wrap.events or term or trunc:
            break
    if wrap.events:
        e = wrap.events[0]
        assert e["speed"] in SPEC.speeds              # frozen speed ladder only
        assert e["n_candidates"] % SPEC.K == 0        # frozen candidate count
        assert e["m"] >= e["m_u0"] - 1e-9             # never worse than the u = 0 fallback
    assert all(r["mode"] in ("qp", "event", "persist") for r in wrap.records)
    wrap.detach()


def test_recovery_scores_against_the_qp_rows_including_the_hunter_row(cfg, env):
    """The hunter row reaches the recovery through the QP's own xi_kept, unmodified."""
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist(env, obs, cfg, TUNED, SPEC, SEED)
    p = env.agent_position
    for _ in range(TUNED.k_scans):                    # fill the rolling buffer as an episode does
        pol.src.push(p, obs[4:28] * env.LIDAR_RANGE)
    pol.src.push_hunter(p + np.array([0.55, 0.0]), np.array([-1.0, 0.0]))   # hunter almost on top
    h, g, dd = pol.src.samples(p)
    assert len(h) == TUNED.k_scans + 1                # 5 LiDAR rows + 1 hunter row
    assert float(np.min(h)) == pytest.approx(0.55 - 0.6)      # hunter row is the critical one
    wrap.detach()


def test_protagonist_keeps_the_frozen_predict_code_path(cfg, env):
    from dr_control.policy import DRCBFPolicy
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist(env, obs, cfg, TUNED, SPEC, SEED)
    assert type(pol).__mro__[1] is DRCBFPolicy
    assert type(pol).predict is DRCBFPolicy.predict      # predict() is not overridden
    assert pol.ctrl.n_keep == 5 and pol.ctrl.epsilon * pol.ctrl.n_keep < 1.0
    wrap.detach()
