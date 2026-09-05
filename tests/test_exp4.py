"""Experiment 4: the 19 pre-training checks from the brief, plus the ones it implies.

The single most important property here is that a step CANNOT cross a wall. The maze walls
are infinitely thin and a step spans up to a whole cell, so an endpoint-only collision test
would let the agent walk through the maze and every collision number in the experiment
would be measuring nothing. `test_no_step_ever_crosses_a_wall` is the check that would
catch that, and it is written against the maze's own wall arrays rather than against the
env's opinion of where it went.
"""
import json

import numpy as np
import pytest

from robot_env.paper_maze import (GOAL_XY, HWALL, N_CELLS, START_XY, VWALL,
                                  assert_perfect_maze, maze_statistics,
                                  open_neighbours, wall_segments)
from robot_env.point_maze import PointMazeEnv, load_exp4_config

CONFIG = "config_exp4.json"


def env(use_lidar=False, **kw):
    return PointMazeEnv(config_path=CONFIG, use_lidar=use_lidar, **kw)


def cell_of(position):
    """Grid cell containing a world point, in (row, col) with row 0 at the top."""
    col = min(int(position[0]), N_CELLS - 1)
    row = N_CELLS - 1 - min(int(position[1]), N_CELLS - 1)
    return row, col


# --------------------------------------------------------------------- 1-2: the maze
def test_the_maze_is_the_papers_perfect_maze():
    """99 passages, 1 component, 0 cycles -- the paper's 'exactly one path' invariant."""
    assert maze_statistics() == {"passages": 99, "components": 1, "cycles": 0}
    assert assert_perfect_maze()          # raises if the transcription is wrong
    assert VWALL.shape == (10, 9)
    assert HWALL.shape == (9, 10)


def test_the_maze_is_fixed_across_resets():
    """Layout must not vary with the seed -- it is a fixed map, not a generator."""
    walls = wall_segments()
    for seed in (0, 1, 7, 99):
        e = env(); e.reset(seed=seed)
        assert wall_segments() == walls
        e.close()


# ------------------------------------------------------------------ 3-5: start / goal
def _config_with(tmp_path, name, **env_overrides):
    config = json.load(open(CONFIG))
    config["environment"].update(env_overrides)
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return str(path)


def test_start_and_goal_lie_in_the_right_corner_cells(tmp_path):
    """Sampled within the paper's blue and red squares, never outside them.

    The paper samples both ("start location is sampled within the blue square ... the goal
    is randomly sampled from within the red square region"). An earlier version of this env
    pinned both to cell centres, which made a deterministic policy produce exactly ONE
    trajectory -- return std 0.000 over 100 episodes -- leaving the critic nothing to fit.
    """
    e = PointMazeEnv(config_path=CONFIG, use_lidar=False)
    assert e.start_goal_mode == "sampled"
    starts, goals = [], []
    for seed in range(60):
        e.reset(seed=seed)
        starts.append(e.agent_position.copy())
        goals.append(e.target_position.copy())
    starts, goals = np.asarray(starts), np.asarray(goals)
    # bottom-left cell and top-right cell of a 10x10 maze
    assert np.all(starts >= 0.0) and np.all(starts <= 1.0)
    assert np.all(goals >= N_CELLS - 1.0) and np.all(goals <= N_CELLS)
    # and they genuinely vary, which is the whole point
    assert starts.std(axis=0).min() > 0.05
    assert goals.std(axis=0).min() > 0.05
    e.close()


def test_fixed_mode_still_available_and_exactly_fixed(tmp_path):
    """The brief's section 5 behaviour is retained behind a flag, and still tested."""
    path = _config_with(tmp_path, "fixed.json", start_goal_mode="fixed")
    e = PointMazeEnv(config_path=path, use_lidar=False)
    for seed in (0, 1, 2, 123):
        e.reset(seed=seed)
        assert tuple(e.agent_position) == START_XY
        assert tuple(e.target_position) == GOAL_XY
    assert START_XY == (0.5, 0.5)
    assert GOAL_XY == (9.5, 9.5)
    e.close()


def test_siblings_sharing_a_seed_get_the_same_start_and_goal():
    """Sibling Rivalry requires paired rollouts from an identical (s0, g).

    Sampling must not break that: both draws come from self.np_random, so two envs given
    the same seed must agree exactly.
    """
    a = PointMazeEnv(config_path=CONFIG)
    b = PointMazeEnv(config_path=CONFIG)
    for seed in (0, 5, 41):
        a.reset(seed=seed); b.reset(seed=seed)
        assert np.allclose(a.agent_position, b.agent_position)
        assert np.allclose(a.target_position, b.target_position)
    a.close(); b.close()


def test_there_are_no_dynamic_obstacles_anywhere():
    """Experiment 4 is static. No obstacle state may exist, in the env or the observation."""
    e = env(use_lidar=True)
    obs, _ = e.reset(seed=0)
    for banned in ("obstacle_positions", "obstacle_velocities", "n_dynamic_obstacles",
                   "obstacle_speed", "motion_model"):
        assert not hasattr(e, banned), f"{banned} must not exist in the Point Maze env"
    _, _, _, _, info = e.step(np.array([0.1, 0.1]))
    assert "obstacle" not in " ".join(info.keys()).replace("collision", "")
    e.close()


# ----------------------------------------------------------- 6-9: observation shapes
@pytest.mark.parametrize("use_lidar,expected", [(False, 4), (True, 28)])
def test_observation_dimensions(use_lidar, expected):
    """PPO/PPO+SR are 4, the LiDAR arms are 4 + 24 = 28. Nothing else."""
    e = env(use_lidar=use_lidar)
    obs, _ = e.reset(seed=0)
    assert e.observation_space.shape == (expected,)
    assert obs.shape == (expected,)
    assert obs.dtype == np.float32
    e.close()


def test_observation_is_position_and_goal_not_velocity():
    """The paper's agent sees [x, y, gx, gy]. It has no velocity state at all."""
    e = env()
    obs, _ = e.reset(seed=0)
    unscale = lambda v: (v + 1.0) * N_CELLS / 2.0
    # Compared against the env's ACTUAL positions, not the cell centres: start and goal are
    # sampled per episode, so hard-coding the centres would only test the fixed mode.
    assert unscale(obs[0]) == pytest.approx(e.agent_position[0], abs=1e-4)
    assert unscale(obs[1]) == pytest.approx(e.agent_position[1], abs=1e-4)
    assert unscale(obs[2]) == pytest.approx(e.target_position[0], abs=1e-4)
    assert unscale(obs[3]) == pytest.approx(e.target_position[1], abs=1e-4)
    e.close()


def test_observation_stays_in_bounds_over_random_play():
    e = env(use_lidar=True)
    obs, _ = e.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        obs, _, term, trunc, _ = e.step(rng.uniform(-1, 1, 2))
        if term or trunc:
            obs, _ = e.reset()
    e.close()


# ------------------------------------------------------------- 10: Euclidean distance
def test_distance_to_goal_is_euclidean():
    e = env()
    e.reset(seed=0)
    goal = e.target_position.copy()
    _, _, _, _, info = e.step(np.array([1.0, 0.0]))
    expected = float(np.linalg.norm(goal - info["agent_position"]))
    assert info["distance_to_target"] == pytest.approx(expected, abs=1e-9)
    e.close()


# ------------------------------------------------- 11-12: collision penalty, no ending
def test_collision_is_penalised_and_does_not_end_the_episode():
    """The central requirement of brief section 13."""
    e = env()
    e.reset(seed=0)
    collisions = 0
    terminated_early = False
    for _ in range(50):
        _, reward, terminated, truncated, info = e.step(np.array([0.0, -1.0]))
        if info["collision"]:
            collisions += 1
            assert reward <= e.REWARD_COLLISION + 1e-9, "a collision must pay negatively"
            if terminated:
                terminated_early = True
        if terminated or truncated:
            break
    assert collisions > 0, "driving into the boundary must register collisions"
    assert not terminated_early, "a collision must NEVER terminate the episode"
    e.close()


def test_wall_hugging_is_charged_every_step():
    """An agent parked against a wall keeps paying.

    Regression: an earlier version clipped the proposed position into the arena BEFORE
    sweeping, which collapsed the move to zero once the agent rested on the border, so the
    sweep saw no motion and reported no contact -- 2 collisions across 50 steps instead of
    50. That silently understates the experiment's primary metric.
    """
    e = env()
    e.reset(seed=0)
    collisions = sum(e.step(np.array([0.0, -1.0]))[4]["collision"] for _ in range(50))
    assert collisions == 50, f"expected a charge on every step against the wall, got {collisions}"
    e.close()


# ---------------------------------------------------------- 13-14: goal reward, ending
def test_goal_gives_the_paper_reward_and_terminates():
    e = env()
    e.reset(seed=0)
    e.agent_position = np.array(GOAL_XY, dtype=np.float64) - np.array([0.2, 0.0])
    _, reward, terminated, truncated, info = e.step(np.array([0.2 / e.MAX_STEP, 0.0]))
    assert terminated and not truncated
    assert info["success"] and info["is_success"]
    assert reward == pytest.approx(e.REWARD_GOAL, abs=1e-9)   # paper Eq. 2: +1
    e.close()


def test_failure_pays_minus_distance_at_truncation():
    """Paper Eq. 2's other branch: -d(s_T, g) when the goal was not reached."""
    e = env()
    e.reset(seed=0)
    reward = terminated = truncated = None
    for _ in range(60):
        _, reward, terminated, truncated, info = e.step(np.array([0.0, 0.0]))
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert reward == pytest.approx(-info["distance_to_target"], abs=1e-9)
    e.close()


# ----------------------------------------------------------------- 15: episode length
def test_horizon_is_the_papers_fifty_steps():
    e = env()
    assert e.MAX_STEPS == 50
    e.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = e.step(np.array([0.0, 0.0]))
        steps += 1
        if terminated or truncated:
            break
    assert steps == 50
    e.close()


# --------------------------------------------------------------------- 16-17: LiDAR
def test_lidar_returns_exactly_24_values():
    e = env(use_lidar=True)
    obs, _ = e.reset(seed=0)
    rays = obs[4:]
    assert rays.shape == (24,)
    assert np.all(rays >= 0.0) and np.all(rays <= 1.0)
    e.close()


def test_lidar_measures_the_distance_to_the_nearest_wall(tmp_path):
    """Ray 0 points along +x; check it against the wall the maze says is there.

    Uses the FIXED start so the expected distances are exact numbers rather than a range --
    this is a geometry check on the ray caster, not a check on the sampling.

    From the start cell the agent can travel right through cells (9,0) and (9,1); the maze
    has a wall on the right edge of (9,2), i.e. at x = 3.0. The +x ray from x = 0.5 must
    therefore read 2.5.
    """
    path = _config_with(tmp_path, "fixed_lidar.json", start_goal_mode="fixed")
    e = PointMazeEnv(config_path=path, use_lidar=True)
    obs, _ = e.reset(seed=0)
    assert not VWALL[9, 0] and not VWALL[9, 1] and VWALL[9, 2], "maze changed; update this test"
    ray_east = obs[4] * e.LIDAR_RANGE
    assert ray_east == pytest.approx(3.0 - START_XY[0], abs=1e-3)
    # South from (0.5, 0.5) is the outer boundary at y = 0.
    ray_south = obs[4 + 18] * e.LIDAR_RANGE      # 24 rays -> index 18 is 270 degrees
    assert ray_south == pytest.approx(0.5, abs=1e-3)
    e.close()


# --------------------------------------------------- the invariant that matters most
def test_no_step_ever_crosses_a_wall():
    """Swept collision, re-checked independently of the env's own sweep.

    A step can span a whole cell and walls are infinitely thin, so endpoint-only collision
    testing would let the agent walk through the maze. Week 4 measured that exact failure
    in the other environment (8.5% of episodes) at much smaller step sizes.

    The check is deliberately independent of the env's own sweep: it re-tests the travelled
    segment against the wall list with an orientation predicate. A *proper* crossing --
    strictly through the interior of both segments -- is the violation. Merely touching a
    wall is expected and legal, because the agent is stopped exactly ON the wall it hits.

    Note a diagonal step legitimately changes BOTH cell indices at once (the action moves up
    to 1.0 in x and 1.0 in y), so a cell-adjacency test would report false violations here.
    """
    def orientation(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def properly_crosses(p0, p1, q0, q1):
        d1, d2 = orientation(q0, q1, p0), orientation(q0, q1, p1)
        d3, d4 = orientation(p0, p1, q0), orientation(p0, p1, q1)
        return d1 * d2 < 0 and d3 * d4 < 0       # strict: endpoint touches don't count

    walls = [(np.array(a, float), np.array(b, float)) for a, b in wall_segments()]
    e = env(use_lidar=True)
    rng = np.random.default_rng(0)
    crossings = 0
    for episode in range(60):
        _, _ = e.reset(seed=episode)
        previous = e.agent_position.copy()
        for _ in range(50):
            _, _, terminated, truncated, info = e.step(rng.uniform(-1, 1, 2))
            arrived = info["agent_position"]
            for wa, wb in walls:
                if properly_crosses(previous, arrived, wa, wb):
                    crossings += 1
            previous = arrived.copy()
            if terminated or truncated:
                break
    assert crossings == 0, f"{crossings} steps passed through a wall"
    e.close()


# ------------------------------------------------------------- 18: existing LiDAR test
def test_the_dynamic_obstacle_lidar_is_untouched():
    """Experiment 4 shares `cast_rays`; it must not have changed the other env's readings.

    The full guarantee lives in tests/test_week4_env.py (the scalar-reference and
    bit-identity tests). This is the cheap cross-check that the shared module is the one
    both environments actually call.
    """
    from robot_env import robot_nav_env
    from robot_env.lidar_core import cast_rays as shared
    assert robot_nav_env.cast_rays is shared


# ---------------------------------------------------- 19: SR machinery left untouched
def test_sr_terminal_modes():
    """`add` must be unchanged; `replace` must substitute rather than double-count."""
    from sibling_rivalry.reward import apply_sr_reward
    from sibling_rivalry.select import Episode

    def make(reached, final_reward):
        ep = Episode(env_index=0, pair_index=0, goal=np.array([9.5, 9.5]))
        ep.rewards = [0.0, final_reward]
        ep.positions = [np.array([0.5, 0.5]), np.array([5.0, 5.0])]
        ep.anti_goal = np.array([2.0, 2.0])
        ep.success = reached
        return ep

    d_goal = float(np.linalg.norm(np.array([5.0, 5.0]) - np.array([9.5, 9.5])))
    d_anti = float(np.linalg.norm(np.array([5.0, 5.0]) - np.array([2.0, 2.0])))
    bonus = min(0.0, -d_goal + d_anti)

    add = make(False, -d_goal)
    apply_sr_reward(add)                                    # default mode, Weeks 4-5
    assert add.rewards[-1] == pytest.approx(-d_goal + bonus)

    replace = make(False, -d_goal)
    apply_sr_reward(replace, terminal_mode="replace", terminal_baseline=-d_goal)
    assert replace.rewards[-1] == pytest.approx(bonus), "Eq. 3 must REPLACE Eq. 2's -d term"

    # Both equations agree on success: the +1 survives untouched.
    won = make(True, 1.0)
    apply_sr_reward(won, terminal_mode="replace", terminal_baseline=0.0)
    assert won.rewards[-1] == pytest.approx(1.0)


# Measured on this maze, 200 episodes per policy, in the investigation that followed the
# first pilot. An untrained (uniform-random) policy earns this much terminal reward by moving
# instead of standing still, and pays for it with this many wall contacts.
EXPLORATION_GAIN = 2.05          # Eq.2; Eq.3 measured 2.13, so Eq.2 is the binding case
CONTACTS_PER_RANDOM_EPISODE = 24.6
BREAK_EVEN_COLLISION = EXPLORATION_GAIN / CONTACTS_PER_RANDOM_EPISODE   # 0.0833


def test_config_reward_magnitudes_match_the_analysis():
    """The collision penalty must stay under the price at which exploring stops paying.

    The first pilot set it to 0.1 -- 1.2x this break-even -- and all four arms converged to
    standing still: `ppo` finished at distance 12.72 against a freeze policy's 12.68, and
    `ppo_sr_lidar` learned near-perfect wall avoidance (0.9 contacts/ep against 25 for a random
    policy) while making no navigational progress whatsoever.

    The criterion this assertion replaced was `collision * 10 == goal`, "ten collisions cost one
    goal bonus". That benchmarks against the wrong quantity: a policy that never reaches the goal
    never collects the goal bonus, so the penalty does not compete with it. What it competes with
    is the exploration gradient, and per contact that is worth only 0.083.
    """
    config = load_exp4_config(CONFIG)
    goal = config["reward"]["goal"]
    collision = abs(config["reward"]["collision"])
    horizon = config["environment"]["max_steps"]
    max_distance = float(np.hypot(N_CELLS - 1, N_CELLS - 1))

    assert goal == 1.0, "paper Eq. 2 uses +1"
    assert collision * horizon < max_distance, (
        "a collision term whose worst case exceeds the navigation signal would dominate "
        "navigation itself")
    assert collision < BREAK_EVEN_COLLISION, (
        f"collision penalty {collision} is at or above the measured break-even "
        f"{BREAK_EVEN_COLLISION:.4f}; above it the reward ranks 'freeze' above every "
        f"exploratory policy and training cannot leave the start cell")
    assert collision <= BREAK_EVEN_COLLISION / 4, (
        "leave real margin, not a hairline: the break-even was measured under one specific "
        "random policy and is not a sharp constant")


def test_collision_penalty_leaves_exploration_positive():
    """The concrete inequality, stated as returns rather than as a ratio."""
    config = load_exp4_config(CONFIG)
    collision = abs(config["reward"]["collision"])
    freeze_return = -12.68                                    # measured
    random_return = -10.63 - collision * CONTACTS_PER_RANDOM_EPISODE
    assert random_return > freeze_return, (
        f"a random walk returns {random_return:.2f} against {freeze_return:.2f} for standing "
        f"still; if this flips, an untrained policy is rewarded for not moving")
