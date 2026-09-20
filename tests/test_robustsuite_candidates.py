"""RS Experiment 1, phase 4: the candidate arms `detour`, `yield` and `detour_yield`.

Design: MD_files/robustsuite/RS_DESIGN.md, section 7.4 as revised on 2026-09-20, and the readings
recorded for tickets 11a-11c. Two seams:
    the layer seam   a layer's see(obs) / reference(p), driven here by LiDAR cast on constructed
                     scenes with the environment's own ray caster;
    seam 1           robustsuite.harness.run_episode(arm, cell, motion, seed, config=...).
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from dr_control.drccp_controller import nominal_action
from dr_control.fast_drccp import FastClfCbfDrccpController
from dr_control.velocity_tracker import LidarVelocityTracker
from highdim import policy as HP
from optional_navigation import planner as OP
from optional_navigation.planner import CarrotFollower, StaticMapPlanner
from robot_env.lidar_core import cast_rays
from robustsuite import blocks as RB
from robustsuite import candidates as CA
from robustsuite import harness as RH
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite.candidates import _rect_distance
from robustsuite.scenario_env import RSScenarioEnv

W, R, RANGE, N = 10.0, 0.3, 5.0, 24
PED_R = 0.3


def _obs(p, rects=(), peds=()):
    """The environment's 52-d observation at `p`: LiDAR from the env's own ray caster, and the
    ground-truth block obs[28:] filled with junk no layer may read."""
    ranges = cast_rays(np.asarray(p, np.float32), n_rays=N, lidar_range=RANGE, world_size=W,
                       agent_radius=R, rects=list(rects),
                       circles=np.asarray(peds, np.float32).reshape(-1, 2),
                       circle_radius=PED_R)
    obs = np.full(52, 0.123, np.float32)
    obs[4:28] = ranges / RANGE
    return obs


def _detour(path, known=(), **cfg):
    return CA.DetourLayer(CarrotFollower(np.asarray(path, float)), list(known),
                          CA.DetourConfig(**cfg))


def _step(layer, p, rects=(), peds=()):
    layer.see(_obs(p, rects, peds))
    return layer.reference(np.asarray(p, float))


# --------------------------------------------------------------------------- the detour layer
STRAIGHT = [(1.0, 5.0), (9.0, 5.0)]
BLOCK = (5.0, 5.0, 0.2, 1.5)                      # across the straight path, unknown to the map


@pytest.mark.parametrize("K", (8, 14))
def test_a_fired_block_becomes_new_cells_within_k_steps_and_starts_a_detour(K):
    layer = _detour(STRAIGHT, K=K)
    p = (3.5, 5.0)                            # the path from progress to +2 m reaches the block
    for k in range(1, K + 1):
        _step(layer, p, [BLOCK])
        assert layer.new_cells().size > 0 if k == K else layer.new_cells().size == 0
        assert layer.active == (k == K)
    cells = layer.new_cells()
    # every new cell lies on the block's near face
    assert np.all(np.abs(cells[:, 0] - 4.8) <= 0.1)
    assert np.all(np.abs(cells[:, 1] - 5.0) <= 1.5 + 0.1)
    assert layer.detours[0]["trigger"] == "new_obstacle"


def _fire(layer, p, rects, steps=20):
    for _ in range(steps):
        _step(layer, p, rects)
        if layer.active:
            return
    raise AssertionError("no detour started")


def test_the_rejoin_point_is_past_the_new_cells_on_the_original_path():
    layer = _detour(STRAIGHT, K=8)
    _fire(layer, (3.5, 5.0), [BLOCK])
    far = layer.new_cells()[:, 0].max() + 0.05          # the far edge of the new cells
    q = np.asarray(layer.detours[0]["q"])
    assert q[1] == pytest.approx(5.0)                    # on the original path
    # 0.45 m past the far edge is the last near point; q is 0.5 m of arc beyond it
    assert q[0] == pytest.approx(far + 0.45 + 0.5, abs=0.06)
    assert layer.detours[0]["d0"] == pytest.approx(q[0] - 3.5)


@pytest.mark.parametrize("wall_y, side", [(6.2, "right"), (3.8, "left")])
def test_the_side_with_more_lidar_room_is_chosen(wall_y, side):
    shelf = (4.0, wall_y, 1.0, 0.2)                      # a known shelf beside the approach
    layer = _detour(STRAIGHT, [shelf], K=8)
    _fire(layer, (3.5, 5.0), [shelf, (5.0, 5.0, 0.2, 0.8)])
    assert layer.detours[0]["side"] == side


def _drive(layer, p, rects, peds_rel=(), steps=300):
    """Move the robot towards the layer's gamma with the frozen nominal controller."""
    p = np.asarray(p, float)
    out = []
    for _ in range(steps):
        peds = [p + np.asarray(d) for d in peds_rel]
        g = _step(layer, p, rects, peds)
        out.append((p.copy(), np.asarray(g, float), layer.state))
        p = p + 0.1 * nominal_action(p, g, 1.0)
    return out


WALL = (5.0, 5.0, 0.2, 2.0)             # known, across the path: only a stall starts a detour


def _wall_gap(p):
    return float(_rect_distance(p, [WALL])[0])


def test_a_stall_starts_a_detour_that_follows_the_wall_at_about_60_cm_and_rejoins():
    layer = _detour(STRAIGHT, [WALL], K=14)
    p = (4.3, 5.0)
    for k in range(1, 32):                               # standing still: 30 steps of history
        _step(layer, p, [WALL])
        assert layer.active == (k == 31)
    assert layer.detours[0]["trigger"] == "stall" and layer.new_cells().size == 0
    assert layer.detours[0]["q"] == pytest.approx([4.3 + 1.5, 5.0])
    run = _drive(layer, p, [WALL], steps=90)            # the first detour only
    wall = [x for x in run[8:] if x[2] == "wall"]       # settled
    assert len(wall) > 30
    gaps = [_wall_gap(q) for q, _, _ in wall if abs(q[1] - 5.0) < 1.8]   # beside the face
    assert np.median(gaps) == pytest.approx(0.6, abs=0.05)
    assert min(gaps) > 0.45
    assert layer.detours[0]["reason"] == "rejoined"
    assert run[-1][0][0] > 5.0                           # the robot is past the wall


def test_wall_following_ignores_pedestrian_hits():
    walk = lambda: _detour(STRAIGHT, [WALL], K=14)       # noqa: E731
    a, b = walk(), walk()
    for _ in range(31):
        _step(a, (4.3, 5.0), [WALL])
        _step(b, (4.3, 5.0), [WALL])
    p, beside = np.array([4.3, 5.0]), np.array([-0.55, 0.0])   # a pedestrian walking alongside
    for _ in range(40):
        g_a = _step(a, p, [WALL], [p + beside])
        g_b = _step(b, p, [WALL])
        assert a.state == b.state == "wall"
        np.testing.assert_allclose(g_a, g_b)
        p = p + 0.1 * nominal_action(p, g_b, 1.0)


def test_the_bug2_rule_leaves_on_a_clear_line_of_sight_closer_than_d0():
    layer = _detour(STRAIGHT, [WALL], K=14)
    for _ in range(31):
        _step(layer, (4.3, 5.0), [WALL])
    q, d0 = np.asarray(layer.detours[0]["q"]), layer.detours[0]["d0"]
    run = _drive(layer, (4.3, 5.0), [WALL], steps=90)
    seg = lambda p: _rect_distance(                      # noqa: E731
        p[None, :] + np.linspace(0, 1, 200)[:, None] * (q - p)[None, :], [WALL]).min()
    left = [i for i in range(1, len(run)) if run[i - 1][2] == "wall" and run[i][2] == "leave"]
    assert len(left) == 1
    p = run[left[0]][0]
    assert np.linalg.norm(p - q) < d0 and seg(p) >= 0.3
    np.testing.assert_allclose(run[left[0]][1], q)       # gamma is q itself
    for p, _, state in run[:left[0]]:                    # clearly allowed earlier -> left earlier
        if state == "wall":
            assert not (np.linalg.norm(p - q) < d0 - 0.05 and seg(p) > 0.5)


def test_the_detour_flips_side_at_200_steps_and_gives_up_at_400():
    layer = _detour(STRAIGHT, [WALL], K=14)
    p = (4.3, 5.0)                                       # the robot never moves: no way to q
    gammas = [_step(layer, p, [WALL]) for _ in range(31 + 400 + 5)]
    d = layer.detours[0]
    assert d["flipped_at"] - d["start"] == 200
    assert d["end"] - d["start"] == 400 and d["reason"] == "gave_up"
    assert d["side"] == "left"                           # the symmetric tie goes left
    t_before = gammas[d["flipped_at"] - 2] - np.asarray(p)
    t_after = gammas[d["flipped_at"] - 1] - np.asarray(p)
    assert t_before[1] > 0.5 and t_after[1] < -0.5       # along the face, then the other way
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    np.testing.assert_allclose(gammas[d["end"] - 1], follower.reference(np.asarray(p)))


def test_with_no_new_obstacle_and_no_stall_gamma_is_the_followers_step_for_step():
    path = [(1.0, 1.5), (5.0, 5.0), (9.0, 8.5)]
    layer = _detour(path, [(3.0, 5.0, 0.5, 0.5)], K=14)
    follower = CarrotFollower(np.asarray(path, float))
    p = np.array([1.0, 1.5])
    for k in range(120):
        ped = (5.0 + 3 * np.cos(k / 10), 5.0 + 3 * np.sin(k / 10))     # a walking pedestrian
        g = _step(layer, p, [(3.0, 5.0, 0.5, 0.5)], [ped])
        f = follower.reference(p)
        assert np.array_equal(g, f) and not layer.active
        p = p + 0.08 * (f - p) / max(np.linalg.norm(f - p), 1e-9)
    assert layer.detours == [] and layer.inner.progress == follower.progress


# --------------------------------------------------------------------------- the yield layer
class _Track:
    """A track of the frozen LiDAR velocity tracker, as the layer reads it."""

    def __init__(self, pos, vel, confirmed=True):
        self._p, self._v = np.asarray(pos, float), np.asarray(vel, float)
        self.confirmed = confirmed

    @property
    def position(self):
        return self._p.copy()

    @property
    def velocity(self):
        return self._v.copy()


class _Tracker:
    def __init__(self, *tracks):
        self.tracks = list(tracks)


def _yield(tracks, known=(), path=STRAIGHT, **cfg):
    return CA.YieldLayer(CarrotFollower(np.asarray(path, float)), _Tracker(*tracks),
                         StaticMapPlanner(W, R, list(known)), CA.YieldConfig(**cfg))


SHELF = (5.0, 6.6, 3.0, 0.2)                 # known and visible: less room above the robot
HEAD_ON = ((6.0, 5.0), (-0.6, 0.0))          # 1 m ahead, walking straight at the robot


def test_a_head_on_track_inside_h_and_d_yields_to_the_freer_side():
    layer = _yield([_Track(*HEAD_ON)], [SHELF])
    g = _step(layer, (5.0, 5.0), [SHELF], [HEAD_ON[0]])
    np.testing.assert_allclose(g, [5.0, 4.0])             # L = 1 m towards the open side
    assert layer.n_yield_steps == 1


@pytest.mark.parametrize("track, why", [
    (_Track((6.0, 5.0), (-0.2, 0.0)), "slower than 0.3 m/s"),
    (_Track((6.5, 5.0), (0.6, 0.0)), "walking away, already outside D"),
    (_Track((6.0, 6.5), (-0.6, 0.0)), "passing 1.5 m clear"),
    (_Track((6.0, 5.0), (-0.6, 0.0), confirmed=False), "unconfirmed"),
    (_Track((6.0, 5.0), (-0.1, 0.0), confirmed=True), "inside D only after H"),
])
def test_no_threat_leaves_the_followers_gamma(track, why):
    layer = _yield([track], [SHELF])
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    g = _step(layer, (5.0, 5.0), [SHELF], [track.position])
    np.testing.assert_allclose(g, follower.reference(np.array([5.0, 5.0])))
    assert layer.n_yield_steps == 0


def test_a_track_already_inside_d_is_a_threat_while_it_recedes():
    # the closest approach is taken over [0, H], so it includes the current distance (reading 10)
    layer = _yield([_Track((6.0, 5.0), (0.6, 0.0))], [SHELF])
    assert _step(layer, (5.0, 5.0), [SHELF], [(6.0, 5.0)])[1] == pytest.approx(4.0)


def test_the_other_side_is_taken_when_the_freer_side_point_is_not_free_on_the_known_map():
    # the known map and the LiDAR scene are set independently here, so that the freer side by
    # LiDAR is the blocked one on the map: the rule, not the scene, decides
    blocked_below = (5.0, 3.8, 3.0, 0.2)
    layer = _yield([_Track(*HEAD_ON)], [SHELF, blocked_below])
    g = _step(layer, (5.0, 5.0), [SHELF], [HEAD_ON[0]])
    np.testing.assert_allclose(g, [5.0, 6.0])             # the other side, still free at 0.3 m
    both = _yield([_Track(*HEAD_ON)], [SHELF, blocked_below, (5.0, 6.0, 3.0, 0.2)])
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    g = _step(both, (5.0, 5.0), [SHELF], [HEAD_ON[0]])
    np.testing.assert_allclose(g, follower.reference(np.array([5.0, 5.0])))
    assert both.n_yield_steps == 0


def test_the_earliest_closest_approach_decides_between_threats():
    near = _Track((5.0, 6.0), (0.0, -0.6))                # closest approach in ~1.3 s
    far = _Track((8.0, 5.0), (-0.6, 0.0))                 # head-on, but ~4.5 s away
    layer = _yield([far, near], [])
    g = _step(layer, (5.0, 5.0), [], [near.position, far.position])
    assert abs(g[1] - 5.0) < 1e-9 and abs(abs(g[0] - 5.0) - 1.0) < 1e-9   # aside from `near`


def test_the_followers_gamma_returns_when_no_threat_remains():
    tracker = _Tracker(_Track(*HEAD_ON))
    layer = _yield([], [SHELF])
    layer.tracker = tracker
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    assert _step(layer, (5.0, 5.0), [SHELF], [HEAD_ON[0]])[1] == pytest.approx(4.0)
    tracker.tracks = []
    np.testing.assert_allclose(_step(layer, (5.0, 5.0), [SHELF]),
                               follower.reference(np.array([5.0, 5.0])))
    assert layer.n_yield_steps == 1


def test_the_layer_never_changes_the_trackers_state():
    tracker = LidarVelocityTracker(r_nominal=PED_R, dt=0.1, n_rays=N, lidar_range=RANGE)
    layer = CA.YieldLayer(CarrotFollower(np.asarray(STRAIGHT, float)), tracker,
                          StaticMapPlanner(W, R, [SHELF]), CA.YieldConfig())
    p = np.array([5.0, 5.0])
    for k in range(12):                                   # a pedestrian walking at the robot
        ped = (7.0 - 0.06 * k, 5.0)
        obs = _obs(p, [SHELF], [ped])
        tracker.update(p, np.asarray(obs[4:28], float) * RANGE)
        before = deepcopy([(t.id, t.x.copy(), t.P.copy(), t.age, t.hits, t.misses,
                            t.confirmed) for t in tracker.tracks])
        layer.see(obs)
        layer.reference(p)
        after = [(t.id, t.x.copy(), t.P.copy(), t.age, t.hits, t.misses, t.confirmed)
                 for t in tracker.tracks]
        assert len(before) == len(after)
        for a, b in zip(before, after, strict=True):
            assert a[0] == b[0] and a[3:] == b[3:]
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])
    assert any(t.confirmed for t in tracker.tracks)        # the scene really did build a track


# ------------------------------------------------------------- the combination (detour_yield)
def _stack(tracks, known, path=STRAIGHT, detour=None, yld=None):
    """yield's layer over detour's layer over the follower, as `attach` composes them."""
    inner = CA.DetourLayer(CarrotFollower(np.asarray(path, float)), list(known),
                           detour or CA.DetourConfig())
    return CA.YieldLayer(inner, _Tracker(*tracks), StaticMapPlanner(W, R, list(known)),
                         yld or CA.YieldConfig())


def test_yield_overrides_the_detour_which_overrides_the_follower():
    tracker = _Tracker()
    layer = _stack([], [WALL])
    layer.tracker = tracker
    p, follower = (4.3, 5.0), CarrotFollower(np.asarray(STRAIGHT, float))
    # 1. no threat, no detour: the follower's gamma
    for _ in range(30):
        np.testing.assert_allclose(_step(layer, p, [WALL]),
                                   follower.reference(np.asarray(p, float)))
    assert not layer.inner.active
    # 2. the stall starts a detour: the detour's gamma, away from the follower's
    g_detour = _step(layer, p, [WALL])
    assert layer.inner.state == "wall"
    assert not np.allclose(g_detour, follower.reference(np.asarray(p, float)))
    assert layer.n_yield_steps == 0
    # 3. a threat arrives: yield overrides the detour, and the detour keeps running underneath
    tracker.tracks = [_Track((5.3, 5.0), (-0.6, 0.0))]
    g_yield = _step(layer, p, [WALL], [(5.3, 5.0)])
    assert layer.n_yield_steps == 1 and layer.inner.state == "wall"
    assert np.linalg.norm(np.asarray(g_yield) - np.asarray(p)) == pytest.approx(1.0)
    assert not np.allclose(g_yield, g_detour)
    # 4. the threat passes: the detour's gamma again
    tracker.tracks = []
    assert layer.inner.state == "wall"
    np.testing.assert_allclose(_step(layer, p, [WALL]), g_detour)
    assert layer.summary()["n_detours"] == 1 and layer.summary()["n_yield_steps"] == 1


# =========================================================================== seam 1
DIAG = RS.seed_block("RS_DIAG")
TIMING = {"step_time_ms_mean", "step_time_ms_median", "step_time_ms_p95", "step_time_ms_max",
          "qp_time_ms_mean", "recovery_overhead_ms_mean", "episode_time_s", "mean_step_time",
          "step_times", "random_eval_ms_mean"}
#: the labels that name the arm rather than describe the episode
LABELS = {"arm", "config", "candidate"}
BLOCK_FIRES = (RS.Cell("open_clutter", "trigger_block"), DIAG[5])  # fires at step 22
SPAWN_FIRES = (RS.Cell("open_clutter", "trigger_spawn"), DIAG[5])


def _canon(rec, drop=()):
    out = {k: v for k, v in rec.items() if k not in TIMING | set(drop)}
    out["events"] = [{k: v for k, v in e.items() if k != "t_recovery"}
                     for e in rec.get("events", [])]
    return json.dumps(out, sort_keys=True, default=float)


def _rs(arm, cell, motion, seed, max_steps=60, config=None, env=None):
    env = env or RSScenarioEnv(motion)
    env.MAX_STEPS = max_steps
    return RH.run_episode(arm, cell, motion, seed, env=env, config=config)


def test_the_candidates_are_registered_arms_with_named_configurations():
    assert set(CA.ARMS) <= set(RH.ARMS) and RH.ARMS[0] == "astar_random"
    assert CA.configure("detour") == CA.DetourConfig(K=14, L=1.0, stall=True)
    assert CA.configure("yield") == CA.YieldConfig(H=3.0, D=1.2, L=1.0)
    assert CA.configure("detour_yield") == CA.DetourYieldConfig(CA.configure("detour"),
                                                                CA.configure("yield"))
    for arm in ("detour", "yield"):
        grid = CA.search_grid(arm)
        assert len(set(grid)) == len(grid) == 8 and grid[0] == CA.configure(arm).name
        assert {CA.configure(arm, n) for n in grid} == set(
            CA.DETOUR_GRID if arm == "detour" else CA.YIELD_GRID)
    assert {(c.K, c.L, c.stall) for c in CA.DETOUR_GRID} == {
        (K, L, s) for K in (8, 14) for L in (0.6, 1.0) for s in (True, False)}
    assert {(c.H, c.D, c.L) for c in CA.YIELD_GRID} == {
        (H, D, L) for H in (2, 3) for D in (0.9, 1.2) for L in (0.6, 1.0)}
    assert CA.search_grid("detour_yield") == (CA.configure("detour_yield").name,)
    two = CA.search_grid("detour_yield", chosen=("K8_L0.6_stall_off", "H2_D0.9_L0.6"))
    assert two == (CA.configure("detour_yield").name, "K8_L0.6_stall_off+H2_D0.9_L0.6")
    for arm in CA.ARMS:
        for name in CA.search_grid(arm) + ("default",):
            CA.configure(arm, name)
        with pytest.raises(ValueError):
            CA.configure(arm, "K9_L0.6_stall_off")


def test_the_harness_takes_a_named_configuration_and_records_it():
    rec = _rs("detour", RS.Cell("open_clutter", "static"), "fixed", DIAG[0], max_steps=3,
              config="K8_L0.6_stall_off")
    assert rec["arm"] == "detour" and rec["config"] == "K8_L0.6_stall_off"
    assert rec["candidate"]["params"] == {"K": 8, "L": 0.6, "stall": False}
    base = _rs("astar_random", RS.Cell("open_clutter", "static"), "fixed", DIAG[0], max_steps=3)
    assert base["config"] is None and base["candidate"] is None
    with pytest.raises(ValueError):
        _rs("detour", RS.Cell("open_clutter", "static"), "fixed", DIAG[0], max_steps=3,
            config="nope")
    with pytest.raises(ValueError):
        _rs("astar_random", RS.Cell("open_clutter", "static"), "fixed", DIAG[0], max_steps=3,
            config="default")


@pytest.mark.parametrize("arm", CA.ARMS)
@pytest.mark.parametrize("cell", [*RS.CELLS, RS.ANCHOR], ids=lambda c: "_".join(c))
def test_every_candidate_runs_on_every_cell_and_the_anchor(arm, cell):
    rec = _rs(arm, cell, "randomized", DIAG[1], max_steps=4)
    assert rec["steps"] == 4 and rec["arm"] == arm and rec["candidate"] is not None


@pytest.mark.parametrize("arm", CA.ARMS)
def test_the_block_runner_runs_a_candidate_and_fingerprints_its_configuration(arm, tmp_path):
    cell = RS.Cell("rooms", "trigger_spawn")
    ck, tr = tmp_path / "b.jsonl", tmp_path / "b.traces.jsonl.gz"
    got = RB.run_block(arm, cell, DIAG[:2], motions=("fixed",), workers=1, checkpoint=ck,
                       traces=tr, max_steps=5, config="default")
    assert [r["seed"] for r in got["fixed"]] == DIAG[:2]
    ref = _rs(arm, cell, "fixed", DIAG[1], max_steps=5)
    assert _canon(got["fixed"][1], drop=RB.HEAVY) == _canon(ref, drop=RB.HEAVY)
    other = CA.search_grid(arm, chosen=(CA.search_grid("detour")[-1],
                                        CA.search_grid("yield")[-1]))[-1]
    assert other != CA.configure(arm).name
    with pytest.raises(SystemExit):
        RB.run_block(arm, cell, DIAG[:2], motions=("fixed",), workers=1, checkpoint=ck,
                     traces=tr, max_steps=5, config=other)
    assert RB.fingerprint("astar_random", cell, DIAG[:2], RS.MOTIONS, None, "t") == \
        RB.fingerprint("astar_random", cell, DIAG[:2], RS.MOTIONS, None, "t", config=None)


@pytest.mark.parametrize("arm", CA.ARMS)
def test_same_seed_and_configuration_reproduce_the_episode(arm):
    cell, seed = BLOCK_FIRES
    a = _rs(arm, cell, "randomized", seed, max_steps=80)
    assert _canon(a) == _canon(_rs(arm, cell, "randomized", seed, max_steps=80))


@pytest.mark.parametrize("arm", CA.ARMS)
def test_the_candidate_keeps_the_frozen_controller_qp_random_and_parameters(arm):
    params, rspec = HP.frozen_params()
    rec = _rs(arm, RS.Cell("corridors", "dynamic"), "fixed", DIAG[2], max_steps=5)
    assert rec["controller"] == "drccp" or rec["controller"] == \
        _rs("astar_random", RS.Cell("corridors", "dynamic"), "fixed", DIAG[2],
            max_steps=5)["controller"]
    assert rec["solver"] == "frozen"
    assert rec["params"] == params.as_dict() and rec["random_spec"] == rspec.as_dict()


@pytest.mark.parametrize("arm", CA.ARMS)
def test_evaluation_refuses_a_non_frozen_controller_for_a_candidate(arm, monkeypatch):
    real = RH.HP.build_hd_arm

    def with_fast(kind, env, **kw):
        pol = real(kind, env, **kw)
        c = pol.ctrl
        pol.ctrl = FastClfCbfDrccpController(cbf_rate=c.rateh, clf_rate=c.rateV,
                                             wasserstein_r=c.wasserstein_r,
                                             epsilon=c.epsilon, k_v=c.k_v, max_v=c.max_v)
        return pol
    monkeypatch.setattr(RH.HP, "build_hd_arm", with_fast)
    with pytest.raises(TypeError):
        _rs(arm, RS.Cell("open_clutter", "static"), "fixed", DIAG[0], max_steps=3)


class _NoisyTail(RSScenarioEnv):
    """Junk in the ground-truth obstacle block obs[28:] of every observation."""

    def _noisy(self, obs):
        obs = np.array(obs, copy=True)
        obs[28:] = self._junk.uniform(-1, 1, size=obs[28:].shape).astype(obs.dtype)
        return obs

    def reset(self, seed=None, options=None):
        self._junk = np.random.default_rng(11)
        obs, info = super().reset(seed=seed, options=options)
        return self._noisy(obs), info

    def step(self, action):
        obs, *rest = super().step(action)
        return (self._noisy(obs), *rest)


@pytest.mark.parametrize("arm", CA.ARMS)
@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_candidate_actions_ignore_the_ground_truth_obstacle_block(arm, motion):
    for cell, seed in (BLOCK_FIRES, SPAWN_FIRES):
        a = _rs(arm, cell, motion, seed, max_steps=80)
        b = _rs(arm, cell, motion, seed, max_steps=80, env=_NoisyTail(motion))
        assert a["trajectory"] == b["trajectory"]
        assert _canon(a) == _canon(b)


def _moved_block(sp):
    cx, cy, hw, hh = sp.block.rect
    return replace(sp, block=replace(sp.block, rect=(cx + 0.37, cy - 0.21, hw + 0.3, hh + 0.1),
                                     offset=sp.block.offset + 0.3,
                                     passage=sp.block.passage + 0.5, axis=1 - sp.block.axis))


def _moved_spawn(sp):
    s = sp.spawn
    return replace(sp, spawn=replace(
        s, start=(s.start[0] + 0.4, s.start[1] - 0.3),
        variant="head_on" if s.variant == "crossing" else "crossing",
        target=(s.target[0] + 0.2, s.target[1]), scripted=s.scripted[::-1]))


def _trigger_away(sp):
    return replace(sp, trigger=replace(sp.trigger, centre=(-50.0, -50.0), fraction=0.5,
                                       arc=0.0, radius=0.1))


@pytest.mark.parametrize("arm", CA.ARMS)
@pytest.mark.parametrize("fires, own, other", [(BLOCK_FIRES, _moved_block, _moved_spawn),
                                               (SPAWN_FIRES, _moved_spawn, _moved_block)],
                         ids=("block", "spawn"))
def test_candidate_actions_do_not_see_the_trigger_block_or_spawn_fields(arm, fires, own, other,
                                                                        monkeypatch):
    # fixed motion: pedestrians never read the layout, so the block rectangle (their reserved
    # zone) may move without changing their world
    cell, seed = fires
    real = _rs(arm, cell, "fixed", seed, max_steps=80)
    k = real["trigger_step"]
    assert real["trigger_fired"] and 1 <= k < 80
    sp = SC.generate(cell, "fixed", seed)
    # the event this cell never fires: the whole episode is unchanged
    monkeypatch.setattr(RH, "cell_spec", lambda *a, s=other(sp): s)
    assert _canon(_rs(arm, cell, "fixed", seed, max_steps=80), drop={"spec"}) == \
        _canon(real, drop={"spec"})
    # its own event moved, or its trigger moved away: identical up to the firing step
    for pert in (own(sp), _trigger_away(own(sp))):
        monkeypatch.setattr(RH, "cell_spec", lambda *a, s=pert: s)
        got = _rs(arm, cell, "fixed", seed, max_steps=80)
        assert got["trajectory"][:k + 1] == real["trajectory"][:k + 1]
        assert json.dumps(got["trace"][:k], default=float) == \
            json.dumps(real["trace"][:k], default=float)


def test_detour_calls_the_planner_only_at_reset_and_detours_round_the_fired_block(monkeypatch):
    calls = []
    real = OP.StaticMapPlanner.path

    def counted(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(OP.StaticMapPlanner, "path", counted)
    cell, seed = BLOCK_FIRES
    rec = _rs("detour", cell, "fixed", seed, max_steps=150)
    assert len(calls) == 1
    cand = rec["candidate"]
    assert cand["n_detours"] >= 1 and cand["detours"][0]["trigger"] == "new_obstacle"
    assert cand["detours"][0]["start"] > rec["trigger_step"]
    assert cand["n_new_cells"] > 0


# `yield` is not included: the frozen tracker confirms moving tracks built from static geometry
# (single LiDAR points sliding along a wall as the robot moves), so the yield layer acts even in a
# cell with no pedestrian at all. That is the mechanism as pre-registered; how often it happens is
# measured by experiments/robustsuite/rs4_candidate_checks.py and recorded as reading 11.
@pytest.mark.parametrize("arm", ("detour",))
def test_with_no_new_obstacle_and_no_stall_the_candidate_is_astar_random_step_for_step(arm):
    for cell, motion, seed in ((RS.Cell("open_clutter", "static"), "fixed", DIAG[1]),
                               (RS.Cell("rooms", "static"), "randomized", DIAG[3]),
                               (RS.ANCHOR, "fixed", DIAG[0])):
        base = _rs("astar_random", cell, motion, seed, max_steps=80)
        cand = _rs(arm, cell, motion, seed, max_steps=80)
        assert cand["candidate"].get("n_detours", 0) == 0
        assert cand["candidate"].get("n_yield_steps", 0) == 0
        assert cand["trajectory"] == base["trajectory"]
        assert _canon(cand, drop=LABELS) == _canon(base, drop=LABELS)


def test_the_harness_can_keep_the_raw_step_times_for_a_timing_sample():
    cell = RS.Cell("open_clutter", "static")
    plain = _rs("detour", cell, "fixed", DIAG[0], max_steps=5)
    assert "step_times" not in plain                       # the frozen record shape, unchanged
    env = RSScenarioEnv("fixed")
    env.MAX_STEPS = 5
    kept = RH.run_episode("detour", cell, "fixed", DIAG[0], env=env, keep_step_times=True)
    assert len(kept["step_times"]) == kept["steps"] == 5
    assert all(t > 0 for t in kept["step_times"])
    assert kept["step_time_ms_max"] == pytest.approx(1e3 * max(kept["step_times"]))
    assert _canon(kept, drop={"step_times"}) == _canon(plain)


def test_a_detour_that_gives_up_does_not_restart_on_the_same_stretch():
    """Give-up hands control back for good on that stretch (reading 17), not for one step."""
    layer = _detour(STRAIGHT, K=8, stall=False)       # trigger (a) only: no stall on top
    p = (3.5, 5.0)                                       # the robot never moves past the block
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    gammas = [_step(layer, p, [BLOCK]) for _ in range(8 + 400 + 60)]
    d = layer.detours[0]
    assert d["reason"] == "gave_up" and d["trigger"] == "new_obstacle"
    # the new cells are still there, but the layer does not start the same detour again
    assert layer.new_cells().size > 0
    assert [x for x in layer.detours if x["trigger"] == "new_obstacle"] == [d]
    assert not layer.active
    for g in gammas[d["end"]:]:
        np.testing.assert_allclose(g, follower.reference(np.asarray(p, float)))


def test_no_yield_when_neither_side_point_is_free_on_the_known_map():
    walled = [SHELF, (5.0, 3.8, 3.0, 0.2), (5.0, 6.0, 3.0, 0.2)]   # both side points blocked
    layer = _yield([_Track(*HEAD_ON)], walled)
    follower = CarrotFollower(np.asarray(STRAIGHT, float))
    g = _step(layer, (5.0, 5.0), [SHELF], [HEAD_ON[0]])
    np.testing.assert_allclose(g, follower.reference(np.array([5.0, 5.0])))
    assert layer.n_yield_steps == 0 and layer.n_no_side == 1
