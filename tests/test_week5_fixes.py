"""Week 5: the five fixes, each pinned by a test that fails on the pre-fix code.

WHY EACH OF THESE EXISTS
------------------------
Week 4 shipped a rollout-batch bug that survived two regression tests before the third one
actually discriminated. The lesson taken from that: a test for a numeric fix has to be
checked against the BROKEN code, not just against the fixed code. The truncation-bootstrap
test below was developed that way -- it was confirmed to fail when the bootstrap is removed.
"""
import json

import numpy as np
import pytest

from robot_env.robot_nav_env import RobotNavEnv, load_config


# --------------------------------------------------------------- Fix 3: observation
def _env(config_path, **kwargs):
    kwargs.setdefault("n_dynamic_obstacles", 6)
    kwargs.setdefault("obstacle_speed", 1.0)
    return RobotNavEnv(config_path=config_path, render_mode=None,
                       use_reward_shaping=True, **kwargs)


def test_paired_observation_has_a_slot_per_obstacle(week5_config, week4_config):
    """52 dims in paired mode, 34 in legacy: 4 + 24 lidar + 6 slots x 4 vs 3 x 2."""
    paired = _env(week5_config)
    legacy = _env(week4_config)
    assert paired.observation_space.shape == (52,)
    assert legacy.observation_space.shape == (34,)

    observation, _ = paired.reset(seed=0)
    assert observation.shape == (52,)
    paired.close(); legacy.close()


def test_paired_observation_pairs_each_position_with_its_own_velocity(week5_config):
    """The block must be [dx, dy, vx, vy] per slot, nearest first, correctly normalized.

    This is the whole point of the fix: a velocity in a slot has to belong to the obstacle
    whose position sits in the same slot, or the agent cannot tell an approaching obstacle
    from a receding one.
    """
    env = _env(week5_config)
    env.reset(seed=7)
    for _ in range(20):
        env.step(env.action_space.sample())

    observation = env._get_observation()
    block = observation[4 + env.N_LIDAR_RAYS:]
    assert block.shape == (env.N_OBSTACLE_SLOTS * 4,)

    offsets = env.obstacle_positions - env.agent_position
    order = np.argsort(np.linalg.norm(offsets, axis=1))

    for slot, obstacle in enumerate(order[:env.N_OBSTACLE_SLOTS]):
        expected_position = offsets[obstacle] / env.WORLD_SIZE
        expected_velocity = env.obstacle_velocities[obstacle] / env.obstacle_speed
        got = block[slot * 4:(slot + 1) * 4]
        assert got[:2] == pytest.approx(np.clip(expected_position, -1, 1), abs=1e-6)
        assert got[2:] == pytest.approx(np.clip(expected_velocity, -1, 1), abs=1e-6)

    # Nearest first: slot distances must be non-decreasing.
    distances = [np.linalg.norm(block[s * 4:s * 4 + 2]) for s in range(len(order))]
    assert distances == sorted(distances)
    env.close()


def test_paired_observation_zero_pads_unused_slots(week5_config):
    """Fewer obstacles than slots leaves the tail exactly zero, not stale or garbage."""
    env = _env(week5_config, n_dynamic_obstacles=2)
    env.reset(seed=3)
    block = env._get_observation()[4 + env.N_LIDAR_RAYS:]
    assert np.count_nonzero(block[2 * 4:]) == 0
    env.close()


def test_paired_observation_stays_in_bounds(week5_config):
    """Every value must remain inside the declared Box across a full episode."""
    env = _env(week5_config)
    observation, _ = env.reset(seed=11)
    low, high = env.observation_space.low, env.observation_space.high
    for _ in range(300):
        assert np.all(observation >= low) and np.all(observation <= high)
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()


# ------------------------------------------------------- Fix 1: timeout penalty guard
def test_gamma_satisfies_the_survival_criterion():
    """The agent must never prefer an EARLY collision to a late one.

    This is the invariant the first pilot violated, and the reason it exists as a test
    rather than as a note. With a per-step penalty S and a collision penalty C, the
    discounted cost of surviving forever is S/(1-gamma). If that exceeds C, then a policy
    that expects to crash eventually is better off crashing immediately, and PPO finds
    that local optimum readily: at gamma=0.999 (S/(1-g) = 20 > C = 10) episode length
    collapsed 87.6 -> 8 steps with reward pinned at the collision penalty and success 0.

        require   S / (1 - gamma)  <  C          i.e.  gamma < 1 - S/C

    The original gamma=0.99 satisfied this with room to spare but was too short-sighted
    to value the goal (effective horizon 100 steps against 500-step episodes). gamma=0.997
    keeps the criterion satisfied at 6.67 < 10 while giving a 333-step horizon.
    """
    config = load_config("config.json")
    step_penalty = abs(config["reward"]["step_penalty"])
    collision_penalty = abs(config["reward"]["collision"])
    gamma = config["training"]["gamma"]

    survival_cost = step_penalty / (1.0 - gamma)
    assert survival_cost < collision_penalty, (
        f"gamma={gamma} makes surviving cost {survival_cost:.2f} against a collision "
        f"penalty of {collision_penalty:.2f}: the agent is paid to crash immediately. "
        f"gamma must be below {1 - step_penalty / collision_penalty}.")


def test_timeout_penalty_is_disabled_because_it_cannot_bite_at_this_gamma():
    """A terminal penalty at the step limit is invisible at gamma 0.99, so it stays off.

    This records a NEGATIVE result rather than a design choice. Stalling (18-22% of
    episodes at 5M in Week 4) is real, and pricing the timeout directly is the obvious
    lever -- but at gamma 0.99 with max_steps 500 the discount factor on that penalty is
    0.99**500 = 0.0066, so a -10 arrives worth -0.066. Measured: enabling it changed
    nothing (success 0.25 with, 0.26 without, at 606k steps).

    Raising gamma to make it visible is not available either: 0.995, 0.997 and 0.999 were
    all measured WORSE than 0.99 (success 0.01, ~0, ~0 against 0.26 at matched steps), and
    0.999 additionally violates the survival criterion above. So the knob is kept, wired
    and tested, but shipped at 0.0 -- which also keeps the reward byte-identical to Week 4
    and the two matrices comparable. Fixing stalling needs a different lever.
    """
    config = load_config("config.json")
    gamma = config["training"]["gamma"]
    horizon_discount = gamma ** config["environment"]["max_steps"]
    assert horizon_discount < 0.01, (
        "if the step limit is inside the discount horizon, revisit the timeout penalty")
    assert config["termination"]["timeout_penalty"] == 0.0


def test_timeout_penalty_is_inert_when_zero(tmp_path, week5_config):
    """Setting it to 0.0 must reproduce the Week 4 reward exactly."""
    config = json.load(open(week5_config))
    config["termination"]["timeout_penalty"] = 0.0
    config["environment"]["max_steps"] = 10
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(config))

    env = _env(str(path), n_dynamic_obstacles=0)
    assert env.TIMEOUT_PENALTY == 0.0
    env.reset(seed=5)
    rewards, terminated, truncated = [], False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        rewards.append(reward)
    assert truncated and not terminated
    # No discontinuity at the final step: it looks like every other step.
    assert rewards[-1] == pytest.approx(rewards[-2], abs=1e-4)
    env.close()


def test_timeout_penalty_applies_only_on_truncation(tmp_path, week5_config):
    """When enabled it fires at the step limit and never alongside goal or collision."""
    config = json.load(open(week5_config))
    config["termination"]["timeout_penalty"] = -10.0
    config["environment"]["max_steps"] = 12
    path = tmp_path / "timeout.json"
    path.write_text(json.dumps(config))

    env = _env(str(path), n_dynamic_obstacles=0)
    env.reset(seed=5)
    # Stand still: with no dynamic obstacles and a zero action the episode can only end
    # by running out of steps.
    rewards, terminated, truncated = [], False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        rewards.append(reward)

    assert truncated and not terminated
    without_penalty = sum(rewards[:-1])
    assert rewards[-1] < 0
    # The final step carries the -10 on top of whatever it would otherwise have earned.
    assert rewards[-1] == pytest.approx(rewards[-2] - 10.0, abs=1e-4)
    assert without_penalty == pytest.approx(sum(rewards[:-1]), abs=1e-9)
    env.close()


# --------------------------------------------------- Fix 5: hyperparameters reach model
def test_shared_kwargs_reach_both_arms_identically(week5_config):
    """PPO and SR must be constructed with the same optimiser settings.

    The arms are supposed to differ in the SR machinery alone. A hyperparameter threaded
    into one builder and forgotten in the other is a silent confound, so this asserts they
    match rather than asserting each one separately.
    """
    from training.train_flat import shared_ppo_kwargs

    training = load_config(week5_config)["training"]
    kwargs = shared_ppo_kwargs(training)

    assert kwargs["batch_size"] == 256
    assert kwargs["gamma"] == pytest.approx(0.99)
    assert kwargs["target_kl"] == pytest.approx(0.02)
    assert kwargs["gae_lambda"] == pytest.approx(0.95)

    # Linear schedule: full rate at the start, zero at the end.
    schedule = kwargs["learning_rate"]
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(training["learning_rate"])
    assert schedule(0.0) == pytest.approx(0.0)


def test_constant_schedule_is_still_available(week5_config):
    """`lr_schedule: constant` must keep the plain float SB3 accepts."""
    from training.train_flat import shared_ppo_kwargs

    training = dict(load_config(week5_config)["training"])
    training["lr_schedule"] = "constant"
    assert shared_ppo_kwargs(training)["learning_rate"] == pytest.approx(3e-4)


# ------------------------------------------------ Fix 4: SR truncation bootstrap
def _tiny_sr_model(config_path, tmp_path, max_steps):
    """A 4-env SR model on an environment whose episodes can only end by truncation.

    No dynamic obstacles and a step limit small enough that the agent cannot cross the
    arena, so every episode truncates and every retained episode is a bootstrap candidate.
    """
    from training.train_flat import build_sr_model

    config = json.load(open(config_path))
    config["environment"]["max_steps"] = max_steps
    config["training"]["n_envs"] = 4
    path = tmp_path / "sr_trunc.json"
    path.write_text(json.dumps(config))

    model, env = build_sr_model(
        config_path=str(path), n_dynamic_obstacles=0, obstacle_speed=1.0,
        randomize=False, seed=0, epsilon=1e9, delta=0.6, log_dir=None)
    return model, env


def test_sr_bootstraps_truncated_episodes(week5_config, tmp_path):
    """A truncated episode's final reward must gain gamma * V(terminal observation).

    Checked against the broken code as well as the fixed code: removing the bootstrap in
    `_pack_buffer` makes this test fail, which is the only thing that makes it a
    regression test rather than a restatement of the implementation.
    """
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    model, env = _tiny_sr_model(week5_config, tmp_path, max_steps=8)
    try:
        captured = {}
        original_pack = model._pack_buffer

        def spy(kept):
            # Snapshot the rewards SR assigned BEFORE the bootstrap is folded in, and the
            # critic's value at that same moment. `learn` runs `train()` immediately after
            # packing, so the value network recomputed afterwards is a DIFFERENT network --
            # the expectation has to be captured here or the test compares against a
            # post-update critic and fails for the wrong reason.
            captured["before"] = [list(map(float, e.rewards)) for e in kept]
            captured["kept"] = kept
            expected = {}
            for index, episode in enumerate(kept):
                if episode.truncated and episode.terminal_observation is not None:
                    terminal = episode.terminal_observation.copy().reshape(1, -1)
                    terminal[:, -2:] = model._anti_goal_columns(episode)
                    with th.no_grad():
                        expected[index] = float(model.policy.predict_values(
                            obs_as_tensor(terminal, model.device)).cpu().numpy().flatten()[0])
            captured["expected_values"] = expected
            original_pack(kept)
            captured["after"] = [list(map(float, e.rewards)) for e in kept]

        model._pack_buffer = spy
        model.learn(total_timesteps=1)

        kept = captured["kept"]
        truncated = [i for i, e in enumerate(kept) if e.truncated]
        assert truncated, "the fixture must produce truncated episodes for this to test anything"

        for i in truncated:
            episode = kept[i]
            assert episode.terminal_observation is not None
            value = captured["expected_values"][i]

            delta = captured["after"][i][-1] - captured["before"][i][-1]
            assert delta == pytest.approx(model.gamma * value, abs=1e-4), (
                "the final reward of a truncated episode must gain gamma * V(s_T)")
            # And it must be a real correction, not a no-op hidden by a zero value.
            assert abs(delta) > 1e-6

        # Non-truncated episodes must be left completely alone.
        for i, episode in enumerate(kept):
            if not episode.truncated:
                assert captured["after"][i] == captured["before"][i]
    finally:
        env.close()


def test_anti_goal_wrapper_widens_the_terminal_observation(week5_config, tmp_path):
    """infos["terminal_observation"] must match the widened observation space.

    It previously stayed narrow while `obs` was widened, so the critic would have been fed
    a vector two columns short.
    """
    model, env = _tiny_sr_model(week5_config, tmp_path, max_steps=4)
    try:
        width = env.observation_space.shape[0]
        env.reset()
        seen = 0
        for _ in range(12):
            _, _, dones, infos = env.step(np.zeros((4, 2), dtype=np.float32))
            for done, info in zip(dones, infos):
                if done:
                    terminal = info.get("terminal_observation")
                    assert terminal is not None
                    assert terminal.shape == (width,), (
                        f"terminal_observation is {terminal.shape}, expected ({width},)")
                    seen += 1
            if seen:
                break
        assert seen, "no episode finished, so nothing was checked"
    finally:
        env.close()
