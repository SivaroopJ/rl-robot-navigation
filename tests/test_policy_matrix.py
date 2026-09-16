"""Policy matrix: MPC pursuit (policies 3/3.5) and CD-random-CLF-DR-CBF (configs A/B/C)."""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from continuation.congestion.detector import (CAUTIOUS, CRITICAL, NORMAL, CongestionDetector,
                                              DetectorConfig, alarm_stats, replay)
from continuation.congestion.monitor import (FeasibilityMonitor, control_grid, grid_margin,
                                             keep_rows, predicted_source)
from continuation.congestion.signals import free_gap, track_signals
from continuation.congestion.wrapper import CDConfig, CongestionWrapper
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.harness import (CONTROLLER_SETUPS, attach_hunter_controllers,
                                          build_protagonist, hunter_rng, make_pursuit_env,
                                          run_episode)
from continuation.pursuit.hunter import HunterController
from continuation.pursuit.hunter_policies import (ALIASES, POLICY_IDS, VARIANTS, policy_by_id,
                                                  predict_position)
from continuation.pursuit.mpc import (MPCConfig, candidate_controls, plan, predict_protag_path,
                                      rollout)
from continuation.random_recovery import RandomRecovery, margins
from continuation.seeds import controller_rng
from continuation.sensed_obstacles import SensedObstacles
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import LidarVelocityTracker
from experiments.week6_continuation.common import load_random, load_tuned

REPO = Path(__file__).resolve().parents[1]
SEED = 11_000_003
TUNED, _ = load_tuned()
SPEC, _ = load_random()
TAU = TUNED.tau_eff()
CFG = PursuitConfig()


def _ctrl():
    return ClfCbfDrccpController(max_v=1.0, cbf_rate=TUNED.alpha, clf_rate=TUNED.clf_rate,
                                 wasserstein_r=TUNED.wasserstein_r, epsilon=TUNED.epsilon,
                                 k_v=TUNED.k_v)


class _Track:
    def __init__(self, tid, pos, vel, confirmed=True):
        self.id, self.confirmed = tid, confirmed
        self._p, self._v = np.asarray(pos, float), np.asarray(vel, float)

    @property
    def position(self):
        return self._p.copy()

    @property
    def velocity(self):
        return self._v.copy()


class _Tracker:
    use_confirmation = True

    def __init__(self, tracks):
        self.tracks = tracks

    def track_by_id(self, tid):
        return next((t for t in self.tracks if t.id == tid), None)


class _Src:
    """Minimal source with the attributes SensedObstacles / predicted_source read."""

    def __init__(self, buffer, ids, tracks):
        self.buffer = [np.asarray(b, float) for b in buffer]
        self.track_buffer = [(np.asarray(i, int), np.ones(len(i), bool)) for i in ids]
        self.tracker = _Tracker(tracks)


# =============================================================== configuration selection
def test_policy_ids_are_exact_and_aliases_map_to_identical_parameters():
    assert list(POLICY_IDS) == ["1", "1.5", "2", "2.5", "3", "3.5"]
    for old, new in ALIASES.items():
        assert replace(VARIANTS[old], name=new) == POLICY_IDS[new] == policy_by_id(old)
    assert POLICY_IDS["3"].policy == POLICY_IDS["3.5"].policy == "mpc"
    assert POLICY_IDS["3"].lidar_mode == "remove" and POLICY_IDS["3.5"].lidar_mode == "filter"
    with pytest.raises(KeyError):
        policy_by_id("4")


def test_controller_setups_assign_the_declared_controllers():
    A, B, C = (CONTROLLER_SETUPS[k] for k in "ABC")
    assert (A.protag_controller, A.hunter_controller, A.cd_mode) == \
        ("Random-CLF-DR-CBF", "Tuned-CLF-DR-CBF", "off")
    assert (B.protag_controller, B.hunter_controller, B.cd_mode) == \
        ("Random-CLF-DR-CBF", "Random-CLF-DR-CBF", "shadow")
    assert (C.protag_controller, C.hunter_controller, C.cd_mode) == \
        ("CD-Random-CLF-DR-CBF", "CD-Random-CLF-DR-CBF", "active")


def test_wrapper_order_is_cd_inside_random_for_both_agents():
    env = make_pursuit_env(CFG)
    obs, _ = env.reset(seed=SEED)
    cd = CDConfig(mode="active")
    frozen = ClfCbfDrccpController.generate_controller
    pol, wrap = build_protagonist(env, obs, CFG, TUNED, SPEC, SEED, cd=cd)
    assert wrap._orig == pol.cd._wrapped and pol.cd._orig.__func__ is frozen
    assert pol.cd.recovery is wrap and wrap.spec is SPEC
    h = HunterController(params=TUNED, max_speed=1.0)
    hcd, hwrap = attach_hunter_controllers(h, CONTROLLER_SETUPS["C"], cd, spec=SPEC, seed=SEED,
                                           tuned=TUNED, env=env, cfg=CFG)
    assert hwrap._orig == hcd._wrapped and hcd._orig.__func__ is frozen and hcd.recovery is hwrap
    h2 = HunterController(params=TUNED, max_speed=1.0)
    assert attach_hunter_controllers(h2, CONTROLLER_SETUPS["A"], None, spec=SPEC, seed=SEED,
                                     tuned=TUNED, env=env, cfg=CFG) == (None, None)
    wrap.detach(), pol.cd.detach(), env.close()


def test_hunter_rng_is_reproducible_and_independent_of_protags():
    a, b = hunter_rng(SEED).uniform(size=4), hunter_rng(SEED).uniform(size=4)
    assert np.array_equal(a, b)
    assert not np.allclose(a, controller_rng(SEED).uniform(size=4))


def test_frozen_modules_are_unmodified_in_git():
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "dr_control", "robot_env",
                        "continuation/random_recovery.py", "continuation/policy.py"], cwd=REPO)
    assert r.returncode == 0


# =============================================================== obstacle model / filtering
def test_sensed_obstacles_split_static_points_and_confirmed_tracks():
    tracks = [_Track(7, [3.0, 0.0], [1.0, 0.0]), _Track(8, [0.0, 3.0], [0, 0], confirmed=False)]
    src = _Src([[[3.0, 0.3], [0.0, 3.3], [5.0, 5.0]]], [[7, 8, -1]], tracks)
    o = SensedObstacles.from_source(src)
    assert o.static_points.tolist() == [[0.0, 3.3], [5.0, 5.0]]   # track-7 point is the track
    assert o.track_pos.tolist() == [[3.0, 0.0]]
    c = o.clearance(np.array([[[0.0, 0.0], [0.0, 0.0]]]), [0.0, 1.0])
    assert c[0, 0] == pytest.approx(2.4) and c[0, 1] == pytest.approx(3.3 - 0.3)   # track moved away


def test_protag_track_exclusion_removes_only_the_track_at_protag():
    tracks = [_Track(1, [2.0, 2.0], [0, 0]), _Track(2, [2.5, 2.0], [0, 0])]  # obstacle 0.5 m away
    src = _Src([[[9.0, 9.0]]], [[-1]], tracks)
    o = SensedObstacles.from_source(src, exclude_near=[2.0, 2.1], exclude_radius=0.35)
    assert o.n_excluded_tracks == 1 and o.track_pos.tolist() == [[2.5, 2.0]]
    assert o.static_points.tolist() == [[9.0, 9.0]]


def test_policy_35_lidar_includes_protag_and_filters_its_cbf_rows():
    env = make_pursuit_env(CFG)
    env.reset(seed=SEED)
    env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    env.hunter_position = np.array([6.0, 5.0])
    env.obstacle_positions = np.tile(np.array([[9.2, 0.8]], np.float32), (6, 1))
    v = POLICY_IDS["3.5"]
    assert v.protag_in_hunter_lidar and not POLICY_IDS["3"].protag_in_hunter_lidar
    ranges = env.hunter_lidar(include_protagonist=True)
    assert ranges[12] == pytest.approx(0.7, abs=1e-5)
    h = HunterController(params=TUNED, max_speed=1.0, filter_radius=v.filter_radius(0.3))
    h.src.push(env.hunter_position, ranges, exclude_center=env.agent_position)
    assert np.all(np.linalg.norm(h.src.buffer[0] - [5.0, 5.0], axis=1) > 0.35)
    env.close()


# =============================================================== MPC
def test_mpc_candidates_have_declared_structure_and_dimensions():
    cfg = MPCConfig()
    U, meta = candidate_controls([1.0, 1.0], [4.0, 1.0], cfg, 1.0)
    assert U.shape == (42, 10, 2) == (cfg.n_candidates, cfg.horizon_steps, 2)
    assert np.all(np.abs(U) <= 1.0) and np.all(np.linalg.norm(U, axis=-1) <= 1.0 + 1e-12)
    assert np.allclose(U[0, :5], [1.0, 0.0]) and np.allclose(U[0, 5:], [1.0, 0.0])
    assert set(np.round(np.linalg.norm(U[:, 0], axis=1), 9)) == {0.5, 1.0}
    assert np.all(U[:, :5] == U[:, :1]) and np.all(U[:, 5:] == U[:, 5:6])
    P = rollout([1.0, 1.0], U, 0.1)
    assert np.allclose(P[0, -1], [2.0, 1.0]) and P.shape == (42, 10, 2)


def test_protag_path_reuses_the_policy2_predictor_at_each_step():
    kw = dict(dt=0.1, goal_accel=1.0, v_max=1.0, world_size=10.0, body_radius=0.3)
    path = predict_protag_path([5, 5], [0.3, 0.1], [8, 8], n_steps=10, **kw)
    for k in (1, 5, 10):
        q, _ = predict_position([5, 5], [0.3, 0.1], [8, 8], horizon=k * 0.1, dt=0.1,
                                goal_accel=1.0, v_max=1.0, world_size=10.0, body_radius=0.3)
        assert np.allclose(path[k - 1], q)


def _empty_obstacles():
    return SensedObstacles(static_points=np.zeros((0, 2)), track_pos=[], track_vel=[],
                           disc_pos=[], disc_vel=[], disc_clear=[], r_robot=0.3, track_clear=0.6)


PLAN_KW = dict(dt=0.1, max_v=1.0, goal_accel=1.0, protag_v_max=1.0, world_size=10.0,
               body_radius=0.3)


def test_mpc_open_space_heads_for_protag_and_sets_gamma_to_plan_end():
    gamma, u, info = plan([2, 5], [5, 5], [0, 0], [9, 5], _empty_obstacles(), MPCConfig(),
                          **PLAN_KW)
    assert not info["fallback"] and info["n_rejected"] == 0
    assert u[0] > 0.9 and abs(u[1]) < 1e-9 and np.allclose(gamma, [3.0, 5.0])


def test_mpc_rejects_candidates_through_an_obstacle_and_predicts_capture():
    wall = np.array([[2.5 + 0.05 * i, y] for i in range(3) for y in np.linspace(4.2, 5.8, 9)])
    obst = SensedObstacles(static_points=wall, track_pos=[], track_vel=[], disc_pos=[],
                           disc_vel=[], disc_clear=[], r_robot=0.3, track_clear=0.6)
    gamma, u, info = plan([2, 5], [3.4, 5], [0, 0], [9, 5], obst, MPCConfig(), **PLAN_KW)
    assert 0 < info["n_rejected"] < 42 and not info["fallback"]
    assert abs(np.arctan2(u[1], u[0])) > np.deg2rad(20)          # goes around, not through
    _, _, near = plan([3, 5], [3.4, 5], [0, 0], [9, 5], _empty_obstacles(), MPCConfig(),
                      **PLAN_KW)
    assert near["pred_capture"]


def test_mpc_falls_back_to_direct_pursuit_when_every_candidate_is_rejected():
    ring = np.array([[2 + 0.35 * np.cos(a), 5 + 0.35 * np.sin(a)]
                     for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
    obst = SensedObstacles(static_points=ring, track_pos=[], track_vel=[], disc_pos=[],
                           disc_vel=[], disc_clear=[], r_robot=0.3, track_clear=0.6)
    gamma, u, info = plan([2, 5], [6, 8], [0, 0], [9, 5], obst, MPCConfig(), **PLAN_KW)
    assert info["fallback"] and info["n_rejected"] == 42
    assert np.allclose(gamma, [6, 8]) and np.allclose(u, np.array([4, 3]) / 5.0)


def test_qp_stays_active_under_mpc_nominal():
    """An MPC nominal driven straight into a wall point is still filtered by the frozen QP."""
    env = make_pursuit_env(CFG)
    env.reset(seed=SEED)
    env.hunter_position = np.array([2.0, 0.75])                   # near the bottom-left rectangle
    env.obstacle_positions = np.tile(np.array([[9.2, 9.2]], np.float32), (6, 1))
    h = HunterController(params=TUNED, max_speed=1.0)
    bad = lambda p, src: (p + np.array([0.0, 1.0]), np.array([0.0, 1.0]), {"mpc_ms": 0.0})
    for _ in range(5):
        u, info = h.act(env.hunter_position, env.hunter_lidar(), None, nominal=bad)
    assert np.allclose(info["u_nom"], [0.0, 1.0]) and "xi_kept" in info
    if info["status"] == "optimal":
        assert margins(info["xi_kept"], u[None, :], TUNED.alpha)[0] >= TAU - 1e-3
        assert u[1] < 1.0 - 1e-3                                  # the QP moved off the nominal
    env.close()


# =============================================================== congestion: monitor
def test_margin_sign_convention_and_grid_estimate():
    grid = control_grid(1.0)
    assert grid.shape == (81, 2) and [0, 0] in grid.tolist() and [1, 1] in grid.tolist()
    open_rows = np.array([[0.0, 3.0, 1.0, 0.0]])                  # far obstacle
    blocked = np.array([[-1.0, -0.2, 1.0, 0.0], [-1.0, -0.2, -1.0, 0.0]])   # squeezed both sides
    m_open = grid_margin(open_rows, TUNED.alpha, grid)
    m_block = grid_margin(blocked, TUNED.alpha, grid)
    assert m_open - TAU > 0 and m_block - TAU < 0
    assert m_open == pytest.approx(max(margins(open_rows, grid, TUNED.alpha)))


def test_keep_rows_matches_the_controllers_own_selection():
    ctrl = _ctrl()
    rng = np.random.default_rng(3)
    xi = np.column_stack([rng.normal(0, .3, 9), rng.uniform(.2, 2, 9), rng.normal(size=(9, 2))])
    rec = {}
    ctrl.generate_controller(np.zeros(2), np.array([3.0, 0.0]), xi, record=rec)
    assert np.array_equal(keep_rows(xi, TUNED.alpha, ctrl.n_keep), rec["xi_kept"])


def test_predicted_source_shifts_only_confirmed_track_points_and_leaves_source_untouched():
    tracker = LidarVelocityTracker()
    src = EstimatedLidarBarrierSource(r_robot=0.3, k_scans=5, tracker=tracker)
    src.buffer = [np.array([[1.0, 0.0], [0.0, 2.0]])]
    src.track_buffer = [(np.array([5, -1]), np.array([True, False]))]
    tracker.tracks = [_Track(5, [1.2, 0.0], [1.0, 0.0])]
    before = (src.buffer[0].copy(), src.n_points_no_track)
    cp = predicted_source(src, 0.5)
    assert np.allclose(cp.buffer[0], [[1.5, 0.0], [0.0, 2.0]])
    cp.samples(np.zeros(2))
    assert np.array_equal(src.buffer[0], before[0]) and src.n_points_no_track == before[1]


def test_monitor_open_versus_congested_state():
    tracker = LidarVelocityTracker()
    mon = FeasibilityMonitor(alpha=TUNED.alpha, tau=TAU, max_v=1.0)
    far = EstimatedLidarBarrierSource(r_robot=0.3, k_scans=5, tracker=tracker)
    far.buffer, far.track_buffer = [np.array([[4.0, 0.0]])], [(np.array([-1]), np.array([False]))]
    near = EstimatedLidarBarrierSource(r_robot=0.3, k_scans=5, tracker=LidarVelocityTracker())
    # one barrier row per buffered SCAN (its nearest point), so four surrounding points must sit
    # in four scans to produce four opposing rows
    near.buffer = [np.array([q]) for q in ([0.32, 0.0], [-0.32, 0.0], [0.0, 0.32], [0.0, -0.32])]
    near.track_buffer = [(np.array([-1]), np.zeros(1, bool)) for _ in range(4)]
    e_open = mon.evaluate(far, np.zeros(2), np.array([1.0, 0.0]))
    e_cong = mon.evaluate(near, np.zeros(2), np.array([1.0, 0.0]))
    assert e_open["margin"] > 0 and e_cong["margin"] < 0


# =============================================================== congestion: detector
def test_detector_states_escalation_and_hysteresis():
    d = CongestionDetector(DetectorConfig(c1=0.15, c2=0.30))
    assert d.update(1.0, 2.0) == NORMAL
    assert d.update(0.10, 2.0) == CAUTIOUS
    assert d.update(-0.01, 2.0) == CRITICAL
    assert d.update(0.03, 2.0) == CRITICAL                        # inside the hysteresis band
    assert [d.update(0.06, 2.0) for _ in range(3)] == [CRITICAL, CRITICAL, CAUTIOUS]
    assert d.update(0.19, 2.0) == CAUTIOUS                        # 0.19 < c1 + 0.05
    assert [d.update(0.5, 2.0) for _ in range(3)] == [CAUTIOUS, CAUTIOUS, NORMAL]
    assert d.update(0.5, 0.04) == CRITICAL                        # clearance alone escalates
    # 0.03 is inside Critical's hysteresis band (exit needs margin >= 0.05): it resets the hold
    seq = [d.update(0.5, 2.0) if i != 1 else d.update(0.03, 2.0) for i in range(4)]
    assert seq == [CRITICAL, CRITICAL, CRITICAL, CRITICAL]
    assert d.update(0.5, 2.0) == CAUTIOUS                         # 3rd consecutive exit step


def test_alarm_statistics_definitions():
    states = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    inf = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    al = alarm_stats(states, inf, window=10)
    assert al["warning_onsets"] == 1 and al["true_alarms"] == 1 and al["lead_steps"] == [2]
    assert al["infeasible_onsets"] == 2 and al["missed"] == 1
    assert np.array_equal(replay(DetectorConfig(), [1, .1, -1], [5, 5, 5]), [0, 1, 2])


# =============================================================== congestion: wrapper
class _FakeCtrl:
    rateh, max_v, n_keep = TUNED.alpha, 1.0, 5

    def __init__(self):
        self.prev_u = np.zeros(2)
        self.calls = []

    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        self.calls.append((np.array(p), np.array(gamma), xi, None if u_nom is None
                           else np.array(u_nom)))
        return np.zeros(2)


def _open_source():
    s = EstimatedLidarBarrierSource(r_robot=0.3, k_scans=5, tracker=LidarVelocityTracker())
    s.buffer, s.track_buffer = [np.array([[4.0, 4.0]])], [(np.array([-1]), np.array([False]))]
    return s


def test_shadow_mode_passes_the_original_arguments_through():
    ctrl, src = _FakeCtrl(), _open_source()
    cd = CongestionWrapper(ctrl, source_fn=lambda: src, cfg=CDConfig(
        mode="shadow", detector=DetectorConfig(c1=10.0, c2=10.0)), tau=TAU, n_rays=24,
        lidar_range=5.0)
    xi = np.array([[0.0, 3.0, 1.0, 0.0]])
    ctrl.generate_controller(np.zeros(2), np.array([2.0, 0.0]), xi)
    p, g, x, u = ctrl.calls[0]
    assert cd.steps[0]["state"] == CAUTIOUS and u is None and np.allclose(g, [2.0, 0.0])
    assert x is xi and cd.steps[0]["scale"] == 1.0


def test_active_caution_changes_only_nominal_and_clf_target():
    ctrl, src = _FakeCtrl(), _open_source()
    cd = CongestionWrapper(ctrl, source_fn=lambda: src, cfg=CDConfig(
        mode="active", detector=DetectorConfig(c1=10.0, c2=10.0)), tau=TAU, n_rays=24,
        lidar_range=5.0)
    xi = np.array([[0.0, 3.0, 1.0, 0.0]])
    ctrl.generate_controller(np.zeros(2), np.array([2.0, 0.0]), xi)
    p, g, x, u = ctrl.calls[0]
    k, a = cd.steps[0]["scale"], np.deg2rad(cd.steps[0]["angle"])
    assert k == 0.6 and abs(cd.steps[0]["angle"]) <= 30
    assert x is xi and ctrl.max_v == 1.0                          # rows and box untouched
    assert np.allclose(u, 0.6 * np.array([np.cos(a), np.sin(a)]))
    assert np.allclose(g, 0.6 * 2.0 * np.array([np.cos(a), np.sin(a)]))


def test_already_infeasible_state_is_critical_and_random_recovery_acts():
    ctrl, src = _ctrl(), _open_source()
    cd = CongestionWrapper(ctrl, source_fn=lambda: src, cfg=CDConfig(mode="active"), tau=TAU,
                           n_rays=24, lidar_range=5.0)
    wrap = RandomRecovery(ctrl, spec=SPEC, rng=np.random.default_rng(0), tau=TAU)
    cd.recovery = wrap
    # h >= 0 keeps the frozen objective DCP (h < 0 raises DCPError, which Random does not treat);
    # four opposing rows closing at 1 m/s make min_i CBC_i <= -0.99 < tau for every admissible u
    xi = np.array([[-1.0, 0.01, 1.0, 0.0], [-1.0, 0.01, -1.0, 0.0],
                   [-1.0, 0.01, 0.0, 1.0], [-1.0, 0.01, 0.0, -1.0]])
    rec = {}
    u = ctrl.generate_controller(np.zeros(2), np.array([3.0, 0.0]), xi, record=rec)
    assert cd.steps[0]["state"] == CRITICAL and cd.steps[0]["margin"] < 0
    assert "infeasible" in str(rec["status"]) and len(wrap.events) == 1
    assert rec["mode"] == "event"
    assert min(abs(np.linalg.norm(u) - v) for v in SPEC.speeds) < 1e-9


def test_free_gap_and_track_signals():
    ranges = np.full(24, 5.0)
    assert free_gap([0, 0], ranges, [1, 0], n_rays=24, lidar_range=5.0) == 2.0
    ranges[0] = 0.5
    assert free_gap([0, 0], ranges, [1, 0], n_rays=24, lidar_range=5.0) == 0.0
    ranges = np.full(24, 0.5)
    ranges[[23, 0, 1]] = 5.0
    assert free_gap([0, 0], ranges, [1, 0], n_rays=24, lidar_range=5.0) == \
        pytest.approx(2 * np.sin(np.deg2rad(45) / 2))
    o = SensedObstacles(static_points=np.zeros((0, 2)), track_pos=[[2.0, 0.0]],
                        track_vel=[[-1.0, 0.0]], disc_pos=[], disc_vel=[], disc_clear=[],
                        r_robot=0.3, track_clear=0.6)
    ts = track_signals([0, 0], [0, 0], o)
    assert ts["n_near"] == 1 and ts["max_closing"] == pytest.approx(1.0)
    assert ts["min_ttc"] == pytest.approx(1.4)


# =============================================================== integration
def test_config_a_matrix_path_reproduces_the_audited_episode():
    old = run_episode(CFG, SEED, tuned=TUNED, spec=SPEC, variant=VARIANTS["B1"])
    new = run_episode(CFG, SEED, tuned=TUNED, spec=SPEC, variant=POLICY_IDS["1"],
                      controllers=CONTROLLER_SETUPS["A"])
    for k, v in old.items():
        if k.endswith(("_ms_mean", "_ms_p95", "episode_time_s")) or k == "hunter_variant":
            continue
        assert new[k] == v or (v != v and new[k] != new[k]), k


def test_shadow_detection_does_not_change_the_episode():
    B = CONTROLLER_SETUPS["B"]
    kw = dict(tuned=TUNED, spec=SPEC, variant=POLICY_IDS["3.5"], trace=True)
    sh = run_episode(CFG, SEED, controllers=B, **kw)
    off = run_episode(CFG, SEED, controllers=replace(B, cd_mode="off"), **kw)
    assert [s["u_hunter"] for s in sh["trace"]] == [s["u_hunter"] for s in off["trace"]]
    assert [s["p_prot"] for s in sh["trace"]] == [s["p_prot"] for s in off["trace"]]
    assert sh["outcome"] == off["outcome"] and "hunter_cd_alarms" in sh
    assert "hunter_cd_alarms" not in off


def test_config_c_policy_3_episode_runs_with_all_diagnostics():
    thr = replace(CDConfig(), detector=DetectorConfig())
    rec = run_episode(CFG, SEED, tuned=TUNED, spec=SPEC, variant=POLICY_IDS["3"],
                      controllers=CONTROLLER_SETUPS["C"], cd_cfg=thr, trace=True, step_log=True)
    assert rec["policy_id"] == "3" and rec["controller_setup"]["cd_mode"] == "active"
    assert rec["mpc_n_steps"] == rec["steps"] and rec["mpc_config"]["n_candidates"] == 42
    assert rec["prot_cd_mode"] == rec["hunter_cd_mode"] == "active"
    for agent in ("prot", "hunter"):
        assert {"total", "qp", "recovery", "cd"} <= set(rec[f"{agent}_latency_ms"])
    assert "mpc" in rec["hunter_latency_ms"]
    log = rec["step_log"]
    assert len(log["h_qp"]) == rec["steps"] and len(log["p_cd_log"]["state"]) == rec["steps"]
    for s in rec["trace"]:
        assert s["hunter_qp_status"] != "None"                    # the QP ran every step
        assert np.all(np.abs(s["u_hunter"]) <= 1.0 + 1e-6)
    assert rec["outcome"] in ("capture", "goal", "protagonist_collision", "timeout")
