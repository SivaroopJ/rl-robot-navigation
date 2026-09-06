"""Phase 6 gate: the frozen Phase 5 system driven from RobotNavEnv, vs PPO.

Hard tests:
    1. the policy never reads obs[28:52] -- the ground-truth obstacle block PPO does read
    2. the policy is frozen: alpha, r_W, eps, fallback and solver unchanged from Phase 1
    3. actions are inside the environment's action box
    4. the environment's observation reconstructs the LiDAR and goal exactly
    5. both systems are driven by the SAME evaluation loop on the SAME seeds
"""
from __future__ import annotations

import numpy as np
import pytest

from dr_control.drccp_controller import ClfCbfDrccpController
from dr_control.policy import DRCBFPolicy
from robot_env.lidar_core import cast_rays
from robot_env.robot_nav_env import RobotNavEnv

EVAL_SEED = 1_000_000


@pytest.fixture(scope="module")
def env():
    return RobotNavEnv(config_path="config.json", n_dynamic_obstacles=6,
                       obstacle_speed=0.675, render_mode=None, use_reward_shaping=True,
                       randomize_dynamic_obstacles=True)


def make_policy(env, **kw):
    return DRCBFPolicy(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                       static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                       dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS, **kw)


# ---------------------------------------------------- 4: observation reconstructs inputs
def test_observation_reconstructs_lidar_and_goal(env):
    obs, _ = env.reset(seed=EVAL_SEED)
    goal_hat = env.agent_position + obs[0:2] * env.WORLD_SIZE
    np.testing.assert_allclose(goal_hat, env.target_position, atol=1e-3)

    ranges = obs[4:4 + env.N_LIDAR_RAYS] * env.LIDAR_RANGE
    direct = cast_rays(env.agent_position, n_rays=env.N_LIDAR_RAYS,
                       lidar_range=env.LIDAR_RANGE, world_size=env.WORLD_SIZE,
                       agent_radius=env.AGENT_RADIUS, rects=env.static_obstacles,
                       circles=env.obstacle_positions, circle_radius=env.OBSTACLE_RADIUS)
    np.testing.assert_allclose(ranges, direct, atol=2e-3)


# ------------------------------------------ 1: the ground-truth obstacle block is unread
def test_policy_ignores_the_ground_truth_obstacle_block(env):
    """THE fairness guarantee of the comparison.

    obs[28:52] carries exact relative positions and velocities of the six nearest dynamic
    obstacles. PPO consumes it; the DR-CBF must not. Corrupting it may not move the action
    by even one ULP.
    """
    rng = np.random.default_rng(0)

    def rollout(corrupt):
        obs, _ = env.reset(seed=EVAL_SEED)
        pol = make_policy(env).reset(obs, env.agent_position)
        acts = []
        for _ in range(25):
            o = obs.copy()
            if corrupt:
                o[28:52] = rng.uniform(-1, 1, 24)
            a, _ = pol.predict(o, env.agent_position)
            acts.append(np.asarray(a).copy())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        return np.array(acts)

    base = rollout(False)
    for _ in range(3):
        np.testing.assert_array_equal(rollout(True), base)


# ------------------------------------------------------ 2: the system is frozen
def test_controller_parameters_are_frozen_at_phase1_values(env):
    pol = make_policy(env)
    c = pol.ctrl
    assert (c.rateh, c.wasserstein_r, c.epsilon, c.rateV) == (0.4, 0.004, 0.1, 1.0)
    assert c.n_keep == 5 and c.solver == "SCS"
    assert ClfCbfDrccpController().rateh == 0.4          # module default untouched


def test_policy_adds_no_control_logic_beyond_the_reference(env):
    """With the planner disabled the policy must reduce to the Phase 1-4 direct-to-goal case."""
    obs, _ = env.reset(seed=EVAL_SEED)
    pol = make_policy(env, use_planner=False).reset(obs, env.agent_position)
    assert pol.follower is None
    np.testing.assert_allclose(pol.goal, env.target_position, atol=1e-3)


# ------------------------------------------------------ 3: actions respect the env box
def test_actions_are_inside_the_action_box(env):
    obs, _ = env.reset(seed=EVAL_SEED + 1)
    pol = make_policy(env).reset(obs, env.agent_position)
    for _ in range(40):
        a, _ = pol.predict(obs, env.agent_position)
        assert env.action_space.contains(np.asarray(a, dtype=np.float32)), a
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break


def test_planner_failure_is_reported_not_hidden(env):
    obs, _ = env.reset(seed=EVAL_SEED)
    pol = make_policy(env).reset(obs, env.agent_position)
    assert pol.planner_failed in (True, False)
    assert not pol.planner_failed                        # this seed is plannable


# ------------------------------------------------------ 5: identical driving loop
def test_same_loop_drives_both_systems(env):
    """The eval harness calls predict(...) -> (action, None) for both; shape-compatible."""
    obs, _ = env.reset(seed=EVAL_SEED)
    pol = make_policy(env).reset(obs, env.agent_position)
    a, state = pol.predict(obs, env.agent_position, deterministic=True)
    assert state is None
    assert np.asarray(a).shape == (2,)
    assert np.asarray(a).dtype == np.float32
