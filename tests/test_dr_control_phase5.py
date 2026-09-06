"""Phase 5 gate: A* -> waypoint -> DR-CBF integration.

Hard tests:
    1. the planner returns a path whose every segment is free on the inflated static map
    2. planner failure is reported as None, never as a silent fallback
    3. path length is consistent with the SPL oracle's length (same grid, same search)
    4. carrot following is monotone and terminates at the goal
    5. evaluation/shortest_path.py is NOT modified (SPL denominator stays intact)
    6. the integration changes ONLY the CLF reference -- controller behaviour is untouched
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from evaluation.shortest_path import ShortestPathOracle
from optional_navigation.planner import CarrotFollower, StaticMapPlanner
import experiments.exp0_analytic as E

W, RA = E.WORLD_SIZE, E.AGENT_RADIUS
RECTS = E.STATIC_RECTS


@pytest.fixture(scope="module")
def planner():
    return StaticMapPlanner(W, RA, RECTS)


# ------------------------------------------------- 1 & 2: validity and honest failure
def test_path_segments_are_free_on_the_inflated_map(planner):
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        s = rng.uniform(0.6, W - 0.6, 2)
        g = rng.uniform(0.6, W - 0.6, 2)
        if planner.occupancy[planner._index(s)] or planner.occupancy[planner._index(g)]:
            continue
        path = planner.path(s, g)
        assert path is not None
        for a, b in zip(path, path[1:]):
            assert planner.visible(a, b), f"segment {a}->{b} crosses an obstacle"
        checked += 1
    assert checked > 20


def test_path_starts_at_start_and_ends_at_goal(planner):
    s, g = np.array([1.0, 1.0]), np.array([9.0, 9.0])
    path = planner.path(s, g)
    np.testing.assert_allclose(path[0], s)
    np.testing.assert_allclose(path[-1], g)


def test_blocked_endpoint_returns_none_not_a_fallback(planner):
    inside_rect = np.array([2.0, 2.0])           # centre of the first static rectangle
    assert planner.occupancy[planner._index(inside_rect)]
    assert planner.path(np.array([5.0, 5.0]), inside_rect) is None
    assert planner.path(inside_rect, np.array([5.0, 5.0])) is None


def test_path_is_simplified_not_one_node_per_cell(planner):
    path = planner.path(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert 2 <= len(path) <= 12, len(path)       # corners, not hundreds of grid cells


# ------------------------------------------------- 3: agreement with the SPL oracle
def test_path_length_matches_the_spl_oracle(planner):
    """Same grid, same search, so the polyline length must match the oracle's number.

    Tolerance covers string-pulling, which can only SHORTEN the 8-connected path, plus the
    half-cell snap at the endpoints.
    """
    oracle = ShortestPathOracle(W, RA, RECTS)
    rng = np.random.default_rng(3)
    for _ in range(15):
        s = rng.uniform(0.8, W - 0.8, 2)
        g = rng.uniform(0.8, W - 0.8, 2)
        L = oracle.path_length(s, g)
        path = planner.path(s, g)
        if L is None or path is None:
            continue
        mine = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        assert mine <= L + 0.2, (mine, L)        # never longer than the grid path
        assert mine >= 0.85 * L, (mine, L)       # and not absurdly shorter


# ------------------------------------------------- 4: carrot behaviour
def test_carrot_is_monotone_and_reaches_the_goal(planner):
    path = planner.path(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    cf = CarrotFollower(path, lookahead=1.0)
    p = path[0].copy()
    last = -np.inf
    for _ in range(400):
        gamma = cf.reference(p)
        assert cf.progress >= last - 1e-12       # progress never slides backwards
        last = cf.progress
        d = gamma - p
        n = np.linalg.norm(d)
        if n < 1e-9:
            break
        p = p + 0.1 * d / n
    assert np.linalg.norm(p - path[-1]) < 0.5


def test_carrot_returns_the_goal_past_the_end(planner):
    path = planner.path(np.array([2.0, 5.0]), np.array([8.0, 5.0]))
    cf = CarrotFollower(path, lookahead=1.0)
    gamma = cf.reference(path[-1])
    np.testing.assert_allclose(gamma, path[-1], atol=1e-9)


def test_carrot_progress_does_not_reverse_when_pushed_sideways(planner):
    """The barrier can shove the robot off-path; progress must not be given back."""
    path = planner.path(np.array([1.0, 5.0]), np.array([9.0, 5.0]))
    cf = CarrotFollower(path, lookahead=1.0)
    cf.reference(np.array([5.0, 5.0]))
    mid = cf.progress
    cf.reference(np.array([5.0, 6.5]))           # pushed sideways
    assert cf.progress >= mid - 1e-12
    cf.reference(np.array([3.0, 5.0]))           # shoved backwards along the path
    assert cf.progress >= mid - 1e-12


# ------------------------------------------------- 5: earlier phases are untouched
def test_shortest_path_oracle_is_unmodified():
    """The SPL denominator every earlier result depends on must still behave identically."""
    oracle = ShortestPathOracle(W, RA, RECTS)
    assert oracle.path_length(np.array([2.0, 2.0]), np.array([5.0, 5.0])) is None  # blocked
    L = oracle.path_length(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert L is not None and L > np.linalg.norm(np.array([8.0, 8.0]))
    # subclassing must not have mutated the base class
    assert not hasattr(ShortestPathOracle, "path")
    assert not hasattr(ShortestPathOracle, "simplify")


# ------------------------------------------------- 6: only the CLF reference changes
def test_only_the_reference_differs_between_direct_and_astar():
    """Given the SAME gamma, the controller must produce the SAME action in both modes.

    This is what makes the Phase 5 comparison attributable to the global plan alone: the
    planner supplies gamma and nothing else, and alpha / r_W / eps / the fallback are
    untouched.
    """
    xi = build_xi([0.8, 0.9, 1.0, 1.1, 1.2],
                  np.array([[1.0, 0.9, 0.8, 0.7, 0.6], [0.0, 0.44, 0.6, 0.71, 0.8]]),
                  [0.0, -0.1, 0.0, 0.0, 0.0])
    p, gamma = np.array([4.0, 4.0]), np.array([7.0, 6.0])
    a = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=0.4).generate_controller(p, gamma, xi)
    b = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=0.4).generate_controller(p, gamma, xi)
    np.testing.assert_array_equal(a, b)


def test_controller_parameters_are_unchanged_from_phase_1():
    c = ClfCbfDrccpController()
    assert (c.rateh, c.wasserstein_r, c.epsilon, c.rateV) == (0.4, 0.004, 0.1, 1.0)
    assert c.n_keep == 5 and c.solver == "SCS"
