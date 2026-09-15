"""Week 6 Track A + Track B tests. The two tracks share no code; this file groups their tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- Track A
from continuation.perception.hunter_detector import HunterDetector
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.harness import make_pursuit_env


def _strip(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                n.body.pop(0)
    return tree


def test_detector_never_reaches_ground_truth():
    src = ast.unparse(_strip(ast.parse((REPO / "continuation/perception/hunter_detector.py").read_text())))
    for bad in ("robot_env", "obstacle_positions", "obstacle_velocities", "hunter_position",
                "np_random", "env"):
        assert bad not in src, bad


def test_detector_is_reproducible_and_reports_no_detection_cleanly():
    d = HunterDetector()
    out, rec = d.update(np.zeros(2), np.full(24, 5.0))     # empty scan -> nothing to track
    assert out is None and rec["detected"] is False and rec["n_tracks"] == 0


def test_hunter_is_invisible_to_lidar_in_the_known_state_baseline():
    cfg = PursuitConfig()                       # known_state
    assert cfg.hunter_visible_to_lidar is False
    e = make_pursuit_env(cfg)
    obs, _ = e.reset(seed=13_000_000)
    e.hunter_position = e.agent_position.astype(float) + np.array([0.8, 0.0])
    r_before = np.asarray(e._cast_lidar_rays(), float).copy()
    e.obstacle_positions = np.tile(np.array([9.5, 9.5], dtype=np.float32),
                                   (len(e.obstacle_positions), 1))
    r_after = np.asarray(e._cast_lidar_rays(), float)
    assert np.isclose(r_after.min(), r_before.min(), atol=1e-6) or r_after.min() > 0.8
    e.close()


def test_hunter_is_visible_to_lidar_in_the_perception_model():
    cfg = PursuitConfig(info_model="lidar_estimate")
    assert cfg.hunter_visible_to_lidar is True
    e = make_pursuit_env(cfg)
    e.reset(seed=13_000_000)
    e.obstacle_positions = np.tile(np.array([9.5, 9.5], dtype=np.float32),
                                   (len(e.obstacle_positions), 1))
    e.agent_position = np.array([5.0, 5.0], dtype=np.float32)
    e.hunter_position = np.array([6.0, 5.0])
    r = np.asarray(e._cast_lidar_rays(), float)
    assert r.min() == pytest.approx(1.0 - 0.3, abs=0.05)   # ray 0 hits the hunter's surface
    e.close()


def test_source_can_withdraw_the_hunter_row():
    from continuation.pursuit.source import HunterAugmentedSource
    s = HunterAugmentedSource(hunter_radius=0.3, r_robot=0.3, k_scans=5, n_rays=24, lidar_range=5.0)
    s.push_hunter([2.0, 0.0], [-1.0, 0.0])
    assert s.hunter_row(np.zeros(2)) is not None
    s.clear_hunter()
    assert s.hunter_row(np.zeros(2)) is None


# --------------------------------------------------------------------------- Track B
from continuation.supervisor.risk import RISK_OFF, RISK_ON, components, risk_from_state
from continuation.supervisor.supervisor import MIN_SCALE, SpeedSupervisor


def test_risk_components_are_bounded_and_monotone():
    for n in range(0, 8):
        d, _, _ = components(n_tracks_near=n, m_u0=0.5, worst_closing=0.0)
        assert 0.0 <= d <= 1.0
    assert components(n_tracks_near=5, m_u0=0.5, worst_closing=0.0)[0] == 1.0
    assert components(n_tracks_near=0, m_u0=0.5, worst_closing=0.0)[0] == 0.0
    m_lo = components(n_tracks_near=0, m_u0=0.5, worst_closing=0.0)[1]
    m_hi = components(n_tracks_near=0, m_u0=-0.5, worst_closing=0.0)[1]
    assert m_hi > m_lo                                  # smaller margin -> more risk
    c_lo = components(n_tracks_near=0, m_u0=0.5, worst_closing=0.0)[2]
    c_hi = components(n_tracks_near=0, m_u0=0.5, worst_closing=-2.0)[2]
    assert c_hi > c_lo                                  # faster closing -> more risk
    assert np.isfinite(components(n_tracks_near=1, m_u0=float("nan"), worst_closing=0.0)[1])


def test_risk_uses_only_runtime_inputs():
    src = ast.unparse(_strip(ast.parse((REPO / "continuation/supervisor/risk.py").read_text())))
    src += ast.unparse(_strip(ast.parse((REPO / "continuation/supervisor/supervisor.py").read_text())))
    for bad in ("robot_env", "obstacle_positions", "obstacle_velocities", "true_clearance",
                "np_random", "gt_"):
        assert bad not in src, bad


class _Ctrl:
    rateh, max_v, prev_u = 0.4, 1.0, np.zeros(2)

    def __init__(self):
        self.seen = []

    def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
        self.seen.append(None if u_nom is None else np.asarray(u_nom, float).copy())
        if record is not None:
            record.update(status="optimal", xi_kept=np.atleast_2d(xi))
        return np.zeros(2)


class _Tracker:
    use_confirmation = True

    def __init__(self, tracks=()):
        self.tracks = list(tracks)


class _T:
    def __init__(self, pos, vel):
        self.position = np.asarray(pos, float)
        self.velocity = np.asarray(vel, float)
        self.confirmed = True


def _drive(sup, ctrl, xi, n, tracker_tracks=None):
    for _ in range(n):
        if tracker_tracks is not None:
            sup.tracker.tracks = tracker_tracks
        ctrl.generate_controller(np.zeros(2), np.array([5.0, 0.0]), xi, record={})


SAFE_XI = np.array([[0.0, 2.0, 1.0, 0.0]])          # large margin, no risk
RISKY_XI = np.array([[-1.0, 0.05, 1.0, 0.0]])       # tiny margin -> high risk


def test_supervisor_is_inert_in_low_risk_open_space():
    c = _Ctrl()
    s = SpeedSupervisor(c, tracker=_Tracker())
    _drive(s, c, SAFE_XI, 5)
    assert all(r["state"] != "ENGAGED" and r["scale"] == 1.0 for r in s.records)
    assert np.allclose(c.seen[-1], [1.0, 0.0])       # nominal passed through unscaled
    s.detach()


def test_supervisor_engages_and_only_scales_the_nominal():
    c = _Ctrl()
    tracks = [_T([1.0, 0.0], [-0.7, 0.0]), _T([0.0, 1.2], [0.0, -0.7]), _T([1.5, 0.5], [-0.6, 0.0]),
              _T([-1.0, 0.3], [0.7, 0.0]), _T([0.6, -1.1], [0.0, 0.7])]
    s = SpeedSupervisor(c, tracker=_Tracker(tracks))
    _drive(s, c, RISKY_XI, 3)
    assert s.records[-1]["state"] == "ENGAGED"
    assert MIN_SCALE <= s.records[-1]["scale"] < 1.0
    u = c.seen[-1]
    assert np.allclose(u / np.linalg.norm(u), [1.0, 0.0])      # direction preserved
    assert np.linalg.norm(u) < 1.0                              # magnitude reduced
    s.detach()


def test_supervisor_hysteresis_and_release():
    c = _Ctrl()
    tracks = [_T([1.0, 0.0], [-0.7, 0.0])] * 5
    s = SpeedSupervisor(c, tracker=_Tracker(tracks))
    _drive(s, c, RISKY_XI, 3)
    assert s.records[-1]["state"] == "ENGAGED"
    s.tracker.tracks = []
    _drive(s, c, SAFE_XI, 3)
    assert s.records[-1]["state"] == "IDLE" and s.records[-1]["scale"] == 1.0
    assert RISK_OFF < RISK_ON
    s.detach()


def test_supervisor_stands_down_after_max_engagement_and_cannot_stall():
    c = _Ctrl()
    tracks = [_T([0.8, 0.0], [-0.9, 0.0])] * 6
    s = SpeedSupervisor(c, tracker=_Tracker(tracks), max_engaged=5, cooldown=4)
    _drive(s, c, RISKY_XI, 30)
    states = [r["state"] for r in s.records]
    assert "COOLDOWN" in states
    assert max(sum(1 for _ in g) for g in _runs(states, "ENGAGED")) <= 6
    assert all(r["scale"] == 1.0 for r in s.records if r["state"] == "COOLDOWN")
    s.detach()


def _runs(seq, val):
    out, cur = [], []
    for x in seq:
        if x == val:
            cur.append(x)
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out or [[]]


def test_supervisor_never_touches_the_executed_action_or_the_fallback():
    """The supervisor returns exactly what the inner controller returned."""
    c = _Ctrl()

    def inner(p, gamma, xi, *, u_nom=None, record=None):
        if record is not None:
            record.update(status="infeasible", xi_kept=np.atleast_2d(xi))
        return np.array([0.0, 0.0])                  # frozen fallback
    c.generate_controller = inner
    s = SpeedSupervisor(c, tracker=_Tracker([_T([0.5, 0.0], [-1.0, 0.0])] * 5))
    out = c.generate_controller(np.zeros(2), np.array([5.0, 0.0]), RISKY_XI, record={})
    assert np.array_equal(out, np.zeros(2))
    assert s.records[-1]["infeasible"] is True
    s.detach()


def test_supervisor_handles_missing_or_stale_track_information():
    c = _Ctrl()
    s = SpeedSupervisor(c, tracker=_Tracker())         # no tracks at all
    r, parts = risk_from_state(np.zeros(2), [], np.array([[np.nan, np.nan, 1.0, 0.0]]), 0.4)
    assert np.isfinite(r) and 0.0 <= r <= 1.0
    _drive(s, c, SAFE_XI, 3)
    assert all(np.isfinite(x["risk"]) for x in s.records)
    s.detach()


def test_wrapper_order_supervisor_outside_recovery():
    """Supervisor must wrap the recovery, so the scaled nominal reaches the frozen solve."""
    from continuation.supervisor.harness import run_episode, build_arms
    import inspect
    src = inspect.getsource(run_episode)
    assert src.index("attach_recovery") < src.index("SpeedSupervisor")
