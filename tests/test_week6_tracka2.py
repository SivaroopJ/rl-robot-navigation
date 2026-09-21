"""Track A continuation: unlabelled-threat experiment. Stage A (information audit) + Stage B."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.source import HunterAugmentedSource
from continuation.unlabelled_threat.harness import (ARMS, arm_config, build_protagonist_arm,
                                                    make_env, run_episode)
from continuation.unlabelled_threat.source import UnlabelledThreatSource
from experiments.week6_continuation.common import load_random, load_tuned

REPO = Path(__file__).resolve().parents[1]
TUNED, _ = load_tuned()
SPEC, _ = load_random()
SEED = 13_200_003


class _T:
    _n = 0

    def __init__(self, pos, vel, *, confirmed=True, misses=0, age=10, tid=None):
        _T._n += 1
        self.id = _T._n if tid is None else tid
        self.position = np.asarray(pos, float)
        self.velocity = np.asarray(vel, float)
        self.confirmed = confirmed
        self.misses = misses
        self.age = age


class _Tracker:
    """Minimal stand-in exposing the interface the frozen source calls."""

    use_confirmation = True
    n_coast = 5

    def __init__(self, tracks=()):
        self.tracks = list(tracks)

    def reset(self):
        self.tracks = []

    def update(self, p, ranges):
        pass

    def velocity_at(self, q):
        return np.zeros(2), False, -1          # frozen source: no track owns this surface point

    def track_by_id(self, tid):
        return next((t for t in self.tracks if t.id == tid), None)


def _src(tracks=(), **kw):
    """Default: static-map gate DISABLED, so each pre-gate rule is tested in isolation.

    The gate itself is covered by the dedicated tests at the end of this file, which pass a real
    map and the production margin.
    """
    kw.setdefault("static_obstacles", ())
    kw.setdefault("world_size", 10.0)
    kw.setdefault("static_gate_margin", -1.0)
    s = UnlabelledThreatSource(track_radius=0.3, dt=0.1, r_robot=0.3, k_scans=5, n_rays=24,
                               lidar_range=5.0, tracker=_Tracker(tracks), **kw)
    return s


# ------------------------------------------------------------------ Stage A: information audit
def _stripped(path):
    tree = ast.parse(Path(path).read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                n.body.pop(0)
    return ast.unparse(tree)


def test_unlabelled_source_cannot_read_any_ground_truth():
    src = _stripped(REPO / "continuation/unlabelled_threat/source.py")
    for bad in ("robot_env", "obstacle_positions", "obstacle_velocities", "hunter_position",
                "hunter_velocity", "push_hunter", "np_random", "env"):
        assert bad not in src, bad


def test_unlabelled_source_has_no_hunter_channel_at_all():
    s = _src()
    assert not hasattr(s, "push_hunter") and not hasattr(s, "clear_hunter")
    assert not isinstance(s, HunterAugmentedSource)
    assert s.IS_KNOWN_STATE is False


def test_unlabelled_arms_never_call_the_oracle_channel():
    """push_hunter appears in the harness only under the U0 branch."""
    tree = ast.parse((REPO / "continuation/unlabelled_threat/harness.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "push_hunter"]
    assert len(calls) == 1
    guarded = [n for n in ast.walk(tree)
               if isinstance(n, ast.If) and "U0_oracle" in ast.unparse(n.test)
               and "push_hunter" in ast.unparse(n)]
    assert guarded, "the oracle channel must be inside an explicit U0 branch"


@pytest.mark.parametrize("arm", ["U1_unlabelled", "U2_visible_only"])
def test_hidden_hunter_state_does_not_change_the_unlabelled_action(arm):
    """Stage A core check: move the hunter's HIDDEN state while holding the scan fixed.

    The scan is passed to the controller explicitly, so if the arm secretly read env state the
    action would move. It must not.
    """
    cfg = arm_config(arm)
    env = make_env(cfg)
    obs, _ = env.reset(seed=SEED)
    acts = []
    for shift in (np.zeros(2), np.array([2.0, -1.5])):
        e2 = make_env(cfg)
        o2, _ = e2.reset(seed=SEED)
        pol, wrap = build_protagonist_arm(arm, e2, o2, cfg, TUNED, SPEC, SEED)
        e2.hunter_position = e2.hunter_position + shift      # hidden state differs
        e2.hunter_velocity = np.array([0.7, 0.7])
        a, _ = pol.predict(o2, e2.agent_position)            # same obs, same pose
        acts.append(np.asarray(a, float))
        wrap.detach()
        e2.close()
    assert np.array_equal(acts[0], acts[1])
    env.close()


def test_oracle_arm_is_unchanged_and_still_privileged():
    cfg = arm_config("U0_oracle")
    assert cfg.info_model == "known_state" and cfg.hunter_visible_to_lidar is False
    env = make_env(cfg)
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist_arm("U0_oracle", env, obs, cfg, TUNED, SPEC, SEED)
    assert isinstance(pol.src, HunterAugmentedSource)
    wrap.detach()
    env.close()


# ------------------------------------------------------------------ Stage B: rows and eligibility
def test_no_eligible_tracks_gives_exactly_the_frozen_rows():
    s = _src()
    s.push(np.zeros(2), np.full(24, 2.0))
    h0, g0, d0 = s.samples(np.zeros(2))
    s2 = _src(enable_track_rows=False)
    s2.push(np.zeros(2), np.full(24, 2.0))
    h1, _, _ = s2.samples(np.zeros(2))
    assert len(h0) == len(h1) == 1          # one buffered scan -> one frozen row, no extras
    assert s.last_rows == []


def test_one_eligible_track_adds_one_row_with_the_oracle_convention():
    t = _T([2.0, 0.0], [-1.0, 0.0])
    s = _src([t])
    s.push(np.zeros(2), np.full(24, 4.0))
    h, g, d = s.samples(np.zeros(2))
    assert len(h) == len(d) == g.shape[1]
    assert h[-1] == pytest.approx(2.0 - 0.3 - 0.3)        # same h as the oracle row
    assert np.allclose(g[:, -1], [-1.0, 0.0])
    assert d[-1] == pytest.approx(-1.0)                    # closing at 1 m/s
    assert len(s.last_rows) == 1 and s.last_rows[0]["track_id"] == t.id


def test_multiple_tracks_give_one_row_each_and_no_duplicates():
    ts = [_T([2.0, 0.0], [-1.0, 0.0]), _T([0.0, 2.5], [0.0, -0.8]), _T([-3.0, 0.0], [0.9, 0.0])]
    s = _src(ts)
    s.push(np.zeros(2), np.full(24, 4.0))
    h, g, d = s.samples(np.zeros(2))
    ids = [r["track_id"] for r in s.last_rows]
    assert len(ids) == 3 and len(set(ids)) == 3            # one row per track id, no duplicates
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(d)) and np.all(np.isfinite(g))


def test_row_order_does_not_change_the_row_set():
    ts = [_T([2.0, 0.0], [-1.0, 0.0], tid=7), _T([0.0, 2.5], [0.0, -0.8], tid=9)]
    a = _src(list(ts)); a.push(np.zeros(2), np.full(24, 4.0)); a.samples(np.zeros(2))
    b = _src(list(reversed(ts))); b.push(np.zeros(2), np.full(24, 4.0)); b.samples(np.zeros(2))
    key = lambda S: sorted((r["track_id"], round(r["h_eff"], 9)) for r in S.last_rows)
    assert key(a) == key(b)


def test_unconfirmed_and_slow_and_too_stale_tracks_are_excluded():
    s = _src([_T([2.0, 0.0], [-1.0, 0.0], confirmed=False),      # not confirmed
              _T([2.0, 0.0], [0.05, 0.0]),                        # below the motion gate
              _T([2.0, 0.0], [-1.0, 0.0], misses=99)])            # beyond the coast limit
    s.push(np.zeros(2), np.full(24, 4.0))
    s.samples(np.zeros(2))
    assert s.last_rows == []


def test_coasted_track_is_used_but_its_clearance_is_deflated():
    fresh = _src([_T([2.0, 0.0], [-1.0, 0.0], misses=0)])
    stale = _src([_T([2.0, 0.0], [-1.0, 0.0], misses=3)])
    for s in (fresh, stale):
        s.push(np.zeros(2), np.full(24, 4.0))
        s.samples(np.zeros(2))
    hf = fresh.last_rows[0]["h_eff"]
    hs = stale.last_rows[0]["h_eff"]
    assert hs < hf                                   # conservative, never inflated
    assert hs == pytest.approx(hf - 3 * 0.1 * 1.0)   # clamp inactive: stays above contact


def test_track_acquisition_and_loss_changes_the_row_count():
    tracker = _Tracker()
    s = UnlabelledThreatSource(track_radius=0.3, dt=0.1, r_robot=0.3, k_scans=5, n_rays=24,
                               lidar_range=5.0, tracker=tracker, static_obstacles=(),
                               world_size=10.0, static_gate_margin=-1.0)
    s.push(np.zeros(2), np.full(24, 4.0))
    s.samples(np.zeros(2))
    assert len(s.last_rows) == 0
    tracker.tracks = [_T([2.0, 0.0], [-1.0, 0.0])]
    s.samples(np.zeros(2))
    assert len(s.last_rows) == 1
    tracker.tracks = []
    s.samples(np.zeros(2))
    assert len(s.last_rows) == 0


def test_visible_only_arm_builds_no_track_rows():
    s = _src([_T([2.0, 0.0], [-1.0, 0.0])], enable_track_rows=False)
    s.push(np.zeros(2), np.full(24, 4.0))
    h, _, _ = s.samples(np.zeros(2))
    assert s.last_rows == [] and len(h) == 1


def test_zero_distance_track_is_skipped_rather_than_dividing_by_zero():
    s = _src([_T([0.0, 0.0], [-1.0, 0.0])])
    s.push(np.zeros(2), np.full(24, 4.0))
    h, g, d = s.samples(np.zeros(2))
    assert np.all(np.isfinite(h)) and s.last_rows == []


# ------------------------------------------------------------------ Stage B: integration
@pytest.mark.parametrize("arm", list(ARMS))
def test_episode_runs_and_is_reproducible(arm):
    a = run_episode(arm, SEED, tuned=TUNED, spec=SPEC)
    b = run_episode(arm, SEED, tuned=TUNED, spec=SPEC)
    keys = ("outcome", "steps", "goal", "captured", "protagonist_collision",
            "min_hunter_distance", "prot_n_infeasible", "track_rows_mean")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    assert a["outcome"] in ("goal", "capture", "protagonist_collision", "timeout")


def test_solver_fallback_is_still_the_frozen_u_zero_under_track_rows():
    """Infeasible steps must still fall back to u = 0 and be counted."""
    arm = "U1_unlabelled"
    cfg = arm_config(arm)
    env = make_env(cfg)
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist_arm(arm, env, obs, cfg, TUNED, SPEC, SEED)
    seen_rows = 0
    for _ in range(60):
        pre = env.pre_step_state()
        a, _ = pol.predict(obs, pre["p_prot"])
        seen_rows = max(seen_rows, len(pol.src.last_rows))
        assert np.all(np.abs(a) <= 1.0 + 1e-9)
        obs, _, term, trunc, _ = env.step_pursuit(a, np.zeros(2))
        if term or trunc:
            break
    assert pol.n_solver_fail >= pol.n_infeasible >= 0
    assert seen_rows >= 1, "the pilot scenario must exercise at least one track row"
    wrap.detach()
    env.close()


def test_both_agents_still_act_from_the_same_pre_step_state():
    """Regression on the pursuit invariant, under the new source."""
    cfg = arm_config("U1_unlabelled")
    out = []
    for a_prot in (np.array([1.0, 0.0]), np.array([-1.0, 0.0])):
        env = make_env(cfg)
        env.reset(seed=SEED)
        env.agent_position = np.array([5.0, 5.0], dtype=np.float32)
        env.hunter_position = np.array([3.0, 5.0])
        _, _, _, _, info = env.step_pursuit(a_prot, np.array([1.0, 0.0]))
        out.append(info["hunter_position"].copy())
        env.close()
    assert np.array_equal(out[0], out[1])


def test_capture_and_contact_semantics_unchanged_in_the_new_arms():
    cfg = arm_config("U1_unlabelled")
    assert cfg.capture_distance(0.3) == pytest.approx(0.65)
    assert cfg.contact_distance(0.3) == pytest.approx(0.60)



# ------------------------------------------------------------------ static-map gate
from continuation.unlabelled_threat.source import STATIC_GATE_MARGIN


def test_static_gate_rejects_a_track_sitting_on_a_mapped_rectangle():
    rect = (5.0, 5.0, 1.0, 1.0)                      # centre (5,5), half-extents 1
    on_rect = _src([_T([6.0 + 0.1, 5.0], [-1.0, 0.0])], static_obstacles=[rect],
                   world_size=10.0, static_gate_margin=STATIC_GATE_MARGIN)
    on_rect.push(np.zeros(2), np.full(24, 4.0))
    on_rect.samples(np.zeros(2))
    assert on_rect.last_rows == [] and on_rect.n_gated_static == 1


def test_static_gate_keeps_a_track_in_open_space():
    rect = (5.0, 5.0, 1.0, 1.0)
    free = _src([_T([2.0, 2.0], [-1.0, 0.0])], static_obstacles=[rect],
                world_size=10.0, static_gate_margin=STATIC_GATE_MARGIN)
    free.push(np.zeros(2), np.full(24, 4.0))
    free.samples(np.zeros(2))
    assert len(free.last_rows) == 1 and free.n_gated_static == 0


def test_static_gate_rejects_a_track_hugging_the_arena_boundary():
    s = _src([_T([0.1, 5.0], [1.0, 0.0])], static_obstacles=(), world_size=10.0,
             static_gate_margin=STATIC_GATE_MARGIN)
    s.push(np.zeros(2), np.full(24, 4.0))
    s.samples(np.zeros(2))
    assert s.last_rows == [] and s.n_gated_static == 1


def test_static_gate_margin_boundary_is_exact():
    rect = (5.0, 5.0, 1.0, 1.0)                      # surface at x = 6.0
    inside = 6.0 + STATIC_GATE_MARGIN - 1e-6
    outside = 6.0 + STATIC_GATE_MARGIN + 1e-6
    a = _src([_T([inside, 5.0], [-1.0, 0.0])], static_obstacles=[rect],
             world_size=10.0, static_gate_margin=STATIC_GATE_MARGIN)
    b = _src([_T([outside, 5.0], [-1.0, 0.0])], static_obstacles=[rect],
             world_size=10.0, static_gate_margin=STATIC_GATE_MARGIN)
    for s in (a, b):
        s.push(np.zeros(2), np.full(24, 4.0))
        s.samples(np.zeros(2))
    assert a.last_rows == [] and len(b.last_rows) == 1


def test_static_gate_uses_only_the_prior_map_not_live_obstacles():
    """The gate reads `static_obstacles` passed at construction; it never sees obstacle state."""
    src_txt = _stripped(REPO / "continuation/unlabelled_threat/source.py")
    assert "obstacle_positions" not in src_txt and "obstacle_velocities" not in src_txt
    s = _src([_T([2.0, 2.0], [-1.0, 0.0])], static_obstacles=[(5.0, 5.0, 1.0, 1.0)],
             world_size=10.0, static_gate_margin=STATIC_GATE_MARGIN)
    assert s.static_obstacles == [(5.0, 5.0, 1.0, 1.0)]


def test_gate_counter_is_reported_in_row_stats():
    s = _src([_T([0.1, 5.0], [1.0, 0.0])], static_obstacles=(), world_size=10.0,
             static_gate_margin=STATIC_GATE_MARGIN)
    s.push(np.zeros(2), np.full(24, 4.0))
    s.samples(np.zeros(2))
    assert s.row_stats()["track_rows_gated_static"] == 1


def test_harness_passes_the_static_map_into_the_source():
    cfg = arm_config("U1_unlabelled")
    env = make_env(cfg)
    obs, _ = env.reset(seed=SEED)
    pol, wrap = build_protagonist_arm("U1_unlabelled", env, obs, cfg, TUNED, SPEC, SEED)
    assert len(pol.src.static_obstacles) == len(env.static_obstacles) == 5
    assert pol.src.world_size == env.WORLD_SIZE
    wrap.detach()
    env.close()



# ------------------------------------------------------------------ deflation clamp
def test_staleness_deflation_is_clamped_at_contact():
    """A coasted track must never produce h_eff < 0 while its raw clearance is non-negative.

    An h_crit < 0 makes the frozen objective non-convex (DCPError). Measured as a new failure mode
    in the first gated pilot, so the clamp is pinned by this test.
    """
    # raw h = 0.8 - 0.6 = 0.2, deflation = 5 * 0.1 * 1.0 = 0.5 -> would be -0.3 without the clamp
    s = _src([_T([0.8, 0.0], [-1.0, 0.0], misses=5)])
    s.push(np.zeros(2), np.full(24, 4.0))
    h, _, _ = s.samples(np.zeros(2))
    row = s.last_rows[0]
    assert row["h_raw"] == pytest.approx(0.2)
    assert row["h_eff"] == pytest.approx(0.0)        # clamped, not negative
    assert float(np.min(h)) >= 0.0


def test_genuinely_negative_raw_clearance_is_left_untouched():
    """Real contact must behave exactly as in the frozen system: no clamping, no masking."""
    s = _src([_T([0.5, 0.0], [-1.0, 0.0], misses=0)])   # raw h = 0.5 - 0.6 = -0.1
    s.push(np.zeros(2), np.full(24, 4.0))
    s.samples(np.zeros(2))
    row = s.last_rows[0]
    assert row["h_raw"] == pytest.approx(-0.1)
    assert row["h_eff"] == pytest.approx(-0.1)
