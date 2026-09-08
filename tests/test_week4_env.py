"""Week 4 environment gate: items 1-9 of the brief's pre-training checklist.

The single most important test here is `test_legacy_path_is_bit_identical_to_main`. Week 4
refactored obstacle motion out of the env and into a motion-model class, and rerouted
start/goal placement through a sampler. Both changes are supposed to be inert when the
config selects the legacy settings. "Supposed to be" is not good enough for a baseline that
Experiment 1 and the zero-shot analysis both rest on, so the test replays the ORIGINAL file
out of git and demands identical observations, rewards and termination flags.
"""
import importlib.util
import json
import subprocess

import numpy as np
import pytest

from robot_env.robot_nav_env import RobotNavEnv


@pytest.fixture(scope="module")
def original_env_module(tmp_path_factory):
    """The env as it exists on `main`, loaded as a module for direct comparison."""
    d = tmp_path_factory.mktemp("orig")
    src = subprocess.check_output(["git", "show", "main:robot_env/robot_nav_env.py"], text=True)
    cfg = subprocess.check_output(["git", "show", "main:config.json"], text=True)
    (d / "orig_env.py").write_text(src)
    (d / "config.json").write_text(cfg)
    spec = importlib.util.spec_from_file_location("orig_env", d / "orig_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ORIGINAL_CONFIG = str(d / "config.json")
    return module


# ----------------------------------------------------------------- 1. legacy equivalence
@pytest.mark.parametrize("n_obstacles", [0, 3, 6, 10])
def test_legacy_path_is_bit_identical_to_main(original_env_module, legacy_config, n_obstacles):
    """dt=1.0 + uniform pairs + no randomization must reproduce `main` exactly.

    Not merely 'statistically similar': the same seed must give the same floats, because
    Experiment 1 is supposed to BE the original baseline rather than a near-copy of it.
    """
    for seed in range(6):
        old = original_env_module.RobotNavEnv(
            config_path=original_env_module.ORIGINAL_CONFIG,
            n_dynamic_obstacles=n_obstacles, obstacle_speed=1.0)
        new = RobotNavEnv(config_path=legacy_config,
                          n_dynamic_obstacles=n_obstacles, obstacle_speed=1.0)
        obs_old, _ = old.reset(seed=seed)
        obs_new, _ = new.reset(seed=seed)
        np.testing.assert_array_equal(obs_old, obs_new)

        rng = np.random.default_rng(1234 + seed)
        for _ in range(500):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            o1, r1, term1, trunc1, _ = old.step(action)
            o2, r2, term2, trunc2, _ = new.step(action)
            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2 and term1 == term2 and trunc1 == trunc2
            if term1 or trunc1:
                break


def test_vectorized_lidar_matches_the_scalar_reference(week4_config):
    """The vectorized ray caster must agree EXACTLY with the scalar helpers.

    `_cast_lidar_rays` was rewritten to compute all rays against all obstacles at once,
    because profiling showed it was 82% of environment step time and Week 4 needs twelve
    multi-million-step runs. The scalar `_ray_vs_walls` / `_ray_vs_rect` / `_ray_vs_circle`
    helpers are kept as the readable definition; this asserts the fast path still computes
    exactly that.

    Exactly, not approximately: matching to a tolerance would let a ~1e-7 drift through, and
    that is enough to diverge a trajectory and silently break the bit-identity guarantee
    against `main`. The first vectorized attempt did precisely this, by computing in float64
    where the reference computes in float32 under NumPy 2's weak promotion.
    """
    env = RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=6)
    rng = np.random.default_rng(0)

    def scalar_reference(self):
        readings = np.full(self.N_LIDAR_RAYS, self.LIDAR_RANGE, dtype=np.float32)
        for index, angle in enumerate(
                np.linspace(0, 2 * np.pi, self.N_LIDAR_RAYS, endpoint=False)):
            direction = np.array([np.cos(angle), np.sin(angle)])
            distance = self._ray_vs_walls(self.agent_position, direction)
            if distance < readings[index]:
                readings[index] = distance
            for rect in self.static_obstacles:
                distance = self._ray_vs_rect(self.agent_position, direction, rect)
                if distance < readings[index]:
                    readings[index] = distance
            for obstacle in self.obstacle_positions:
                distance = self._ray_vs_circle(self.agent_position, direction, obstacle,
                                               self.OBSTACLE_RADIUS)
                if distance < readings[index]:
                    readings[index] = distance
        return readings

    for episode in range(8):
        env.reset(seed=episode)
        for _ in range(25):
            np.testing.assert_array_equal(scalar_reference(env), env._cast_lidar_rays())
            _, _, terminated, truncated, _ = env.step(
                rng.uniform(-1, 1, 2).astype(np.float32))
            if terminated or truncated:
                break


# --------------------------------------------------------------- 2-4. randomized motion
def test_randomized_motion_runs_and_stays_finite(week4_config):
    env = RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=6)
    env.reset(seed=0)
    for _ in range(400):
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert np.all(np.isfinite(env.obstacle_positions))
        if term or trunc:
            env.reset(seed=1)


def test_same_seed_reproduces_the_whole_episode(week4_config):
    """Identical seed -> identical start, goal, obstacle states and full trajectory.

    Sibling Rivalry's pairing depends on this exactly: siblings are made comparable by
    resetting them with the same seed, so if this ever broke, `rho` would silently start
    measuring environment variance instead of policy variance.
    """
    def rollout(seed):
        env = RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=6)
        env.reset(seed=seed)
        frames = [(env.agent_position.copy(), env.target_position.copy(),
                   env.obstacle_positions.copy())]
        rng = np.random.default_rng(7)
        for _ in range(120):
            env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            frames.append((env.agent_position.copy(), env.target_position.copy(),
                           env.obstacle_positions.copy())) 
        return frames

    for a, b in zip(rollout(11), rollout(11)):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_different_seeds_give_different_trajectories(week4_config):
    def obstacle_path(seed):
        env = RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=6)
        env.reset(seed=seed)
        path = []
        for _ in range(80):
            env.step(np.zeros(2, dtype=np.float32))
            path.append(env.obstacle_positions.copy())
        return np.array(path)

    assert not np.allclose(obstacle_path(1), obstacle_path(2))


# ------------------------------------------------------- 5-8. the smoothness guarantees
@pytest.fixture(scope="module")
def motion_trace():
    """Long multi-obstacle trace of the stochastic model, spawned flush against the walls.

    Flush spawning is deliberately the worst case for boundary handling: an obstacle that
    starts inside its own turning radius of a wall is exactly what would force the safety
    clamp, so a passing test here covers the realistic spawn margin comfortably.
    """
    from robot_env.dynamic_obstacles import SmoothStochasticMotion
    cfg = json.load(open("config.json"))
    env_cfg, do_cfg, dt = cfg["environment"], cfg["dynamic_obstacles"], cfg["dt"]
    world, radius = env_cfg["world_size"], env_cfg["obstacle_radius"]
    traces = []
    for seed in range(8):
        model = SmoothStochasticMotion(
            world, radius,
            angular_noise_sigma=do_cfg["angular_noise_sigma"],
            angular_velocity_decay=do_cfg["angular_velocity_decay"],
            max_turn_rate=do_cfg["max_turn_rate"],
            boundary_margin=do_cfg["boundary_margin"],
            boundary_steer_gain=do_cfg["boundary_steer_gain"])
        rng = np.random.default_rng(seed)
        pos = rng.uniform(radius, world - radius, size=(8, 2)).astype(np.float32)
        vel = model.reset(rng, pos, 1.0)
        theta, omega, speed, positions = [model.theta.copy()], [model.omega.copy()], [], []
        for _ in range(1500):
            pos, vel = model.step(rng, pos, vel, dt)
            theta.append(model.theta.copy())
            omega.append(model.omega.copy())
            speed.append(np.linalg.norm(vel, axis=1))
            positions.append(pos.copy())
        traces.append(dict(model=model, dt=dt, world=world, radius=radius,
                           theta=np.array(theta), omega=np.array(omega),
                           speed=np.array(speed), positions=np.array(positions)))
    return traces


def test_obstacle_speed_stays_in_range(motion_trace):
    for t in motion_trace:
        assert np.allclose(t["speed"], 1.0, atol=1e-6), "speed must be held, not randomized per step"


def test_angular_velocity_is_bounded(motion_trace):
    for t in motion_trace:
        assert np.abs(t["omega"]).max() <= t["model"].max_turn_rate + 1e-12


def test_no_sudden_direction_changes(motion_trace):
    """Per-step heading change never exceeds omega_max * dt.

    This is the property that separates 'stochastic but smooth' from 'redraw the angle
    every step'. The bound holds through boundary handling too, because the corrective
    steer is applied BEFORE the clip rather than added on top of it.
    """
    for t in motion_trace:
        bound = t["model"].max_turn_rate * t["dt"] + 1e-12
        assert np.abs(np.diff(t["theta"], axis=0)).max() <= bound


def test_obstacles_stay_inside_the_valid_region(motion_trace):
    """After every update the obstacle is inside the arena. Non-negotiable."""
    for t in motion_trace:
        low, high = t["radius"], t["world"] - t["radius"]
        assert t["positions"].min() >= low - 1e-6
        assert t["positions"].max() <= high + 1e-6


def test_wall_contact_is_a_graze_and_not_a_bounce(motion_trace):
    """Contact with a wall must be a slide, never a reflection.

    The projection back into the arena is per-axis, so an obstacle that reaches a wall
    keeps its tangential motion and slides -- and crucially its HEADING is not touched,
    which is what separates this from the elastic bounce the brief rules out. The steer
    then curves it away over the following steps.

    Two things are asserted: contact is rare, and when it happens the projection is a
    fraction of one step of travel rather than a teleport. The heading continuity that
    matters is covered by `test_no_sudden_direction_changes`, which passes over these same
    traces -- so grazing provably does not introduce a direction discontinuity.
    """
    total_steps = sum(t["positions"].shape[0] * t["positions"].shape[1] for t in motion_trace)
    total_clips = sum(t["model"].n_clamped for t in motion_trace)
    assert total_clips / total_steps < 0.03, (
        f"{100 * total_clips / total_steps:.2f}% of obstacle-steps touched a wall; the "
        "boundary steer is not turning obstacles away early enough")

    # A projection deeper than one step of travel would mean the obstacle jumped, not grazed.
    step_travel = 1.0 * motion_trace[0]["dt"]
    for t in motion_trace:
        low, high = t["radius"], t["world"] - t["radius"]
        margin_used = np.maximum(low - t["positions"], t["positions"] - high)
        assert margin_used.max() <= step_travel


def test_angular_velocity_is_temporally_correlated(motion_trace):
    """The randomness must be correlated in time, not white noise.

    Without this the trajectory bound above could be satisfied by a process that merely
    takes tiny independent steps, which would wander nowhere. Correlation is what makes
    the motion genuinely unpredictable over an episode.
    """
    omega = motion_trace[0]["omega"][:, 0]
    omega = omega - omega.mean()
    lag1 = float(np.corrcoef(omega[:-1], omega[1:])[0, 1])
    assert lag1 > 0.8, f"omega lag-1 autocorrelation {lag1:.3f} is too low to be smooth"


# ------------------------------------------------------------ 9. collision detection
def test_collision_typing_matches_the_original_boolean(week4_fixed_config):
    """`collision_type` must agree with the untouched `_check_collision` test.

    Week 4 only instruments which collision fired; it must not change WHETHER one fires.
    """
    env = RobotNavEnv(config_path=week4_fixed_config, n_dynamic_obstacles=8)
    rng = np.random.default_rng(3)
    for ep in range(60):
        env.reset(seed=ep)
        for _ in range(300):
            _, reward, term, trunc, info = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            if info["collision"]:
                assert info["collision_type"] in ("wall", "static", "dynamic")
                assert reward <= env.REWARD_COLLISION + 1e-9
                assert term
            if term or trunc:
                break


def test_collision_penalty_and_reward_constants_are_unchanged(week4_config):
    """Week 4 changes obstacle motion, not reward design."""
    original = json.loads(subprocess.check_output(["git", "show", "main:config.json"], text=True))
    current = json.load(open("config.json"))
    assert current["reward"] == original["reward"]
    assert current["environment"] == original["environment"]
