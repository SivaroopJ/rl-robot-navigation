"""Sibling Rivalry gate: the selection rule, the reward, and the asymmetric critic.

The selection rule is the part of SR most easily got subtly wrong, and a mistake in it does
not crash -- it quietly turns SR into ordinary PPO with a strange reward, which would then
be reported as "SR does not help". These tests check it against the paper's pseudocode
directly, with hand-built episodes, so a regression is caught here rather than inferred
from a null result weeks later.
"""
import numpy as np
import pytest

from sibling_rivalry.reward import apply_sr_reward, sr_terminal_bonus
from sibling_rivalry.select import Episode, relabel_and_select


def make_episode(env_index, pair_index, terminal, goal, *, success=False, n_steps=3):
    episode = Episode(env_index=env_index, pair_index=pair_index,
                      goal=np.asarray(goal, dtype=float), success=success)
    episode.positions = [np.zeros(2), np.asarray(terminal, dtype=float)]
    episode.rewards = [0.0] * n_steps
    episode.observations = [np.zeros(36, dtype=np.float32)] * n_steps
    episode.actions = [np.zeros(2, dtype=np.float32)] * n_steps
    episode.log_probs = [0.0] * n_steps
    return episode


# ------------------------------------------------------------------ the selection rule
def test_farther_sibling_is_always_retained():
    """tau_f is kept unconditionally -- the push away from a shared attractor."""
    goal = [10.0, 0.0]
    a = make_episode(0, 0, [1.0, 0.0], goal)      # closer
    b = make_episode(1, 0, [0.0, 9.0], goal)      # farther, and far from a
    kept, stats = relabel_and_select([a, b], epsilon=0.1, delta=0.5)
    assert b in kept
    assert a not in kept, "rho >> epsilon and d_c > delta, so the closer sibling is dropped"
    assert stats[0].kept_farther and not stats[0].kept_closer


def test_closer_sibling_is_kept_when_siblings_agree():
    """rho < epsilon means the pair agree, so pushing away from it would be noise."""
    goal = [10.0, 0.0]
    a = make_episode(0, 0, [1.0, 0.0], goal)
    b = make_episode(1, 0, [1.05, 0.0], goal)
    kept, stats = relabel_and_select([a, b], epsilon=1.0, delta=0.01)
    assert len(kept) == 2 and stats[0].kept_closer


def test_closer_sibling_is_kept_when_it_succeeded():
    """d_c < delta protects a policy that has genuinely reached the goal."""
    goal = [10.0, 0.0]
    a = make_episode(0, 0, [9.9, 0.0], goal, success=True)
    b = make_episode(1, 0, [0.0, 0.0], goal)
    kept, stats = relabel_and_select([a, b], epsilon=0.001, delta=0.5)
    assert len(kept) == 2 and stats[0].kept_closer


def test_relabeling_is_mutual():
    """Each sibling's anti-goal is the OTHER's terminal state -- never its own."""
    goal = [10.0, 0.0]
    a = make_episode(0, 0, [1.0, 2.0], goal)
    b = make_episode(1, 0, [3.0, 4.0], goal)
    relabel_and_select([a, b], epsilon=100.0, delta=0.5)
    np.testing.assert_allclose(a.anti_goal, [3.0, 4.0])
    np.testing.assert_allclose(b.anti_goal, [1.0, 2.0])


def test_roles_follow_distance_to_goal_not_env_index():
    goal = [10.0, 0.0]
    a = make_episode(0, 0, [9.0, 0.0], goal)     # closer, but lower env index
    b = make_episode(1, 0, [0.0, 0.0], goal)
    relabel_and_select([a, b], epsilon=100.0, delta=0.5)
    assert a.role == "closer" and b.role == "farther"


def test_unpaired_episodes_are_an_error():
    """A pair that lost a member means pairing broke; rho would be meaningless."""
    goal = [10.0, 0.0]
    with pytest.raises(ValueError, match="expected exactly 2"):
        relabel_and_select([make_episode(0, 0, [1.0, 0.0], goal)], epsilon=1.0, delta=0.5)


def test_epsilon_of_infinity_accepts_everything():
    """The paper's own 'eps-inf' ablation: SR degenerates to keeping every episode.

    Pinned because it is the failure mode an uncalibrated epsilon produces, and the one
    that would make an SR arm indistinguishable from PPO.
    """
    goal = [10.0, 0.0]
    episodes = []
    for pair in range(20):
        episodes.append(make_episode(2 * pair, pair, [pair * 0.4, 0.0], goal))
        episodes.append(make_episode(2 * pair + 1, pair, [0.0, pair * 0.4], goal))
    kept, _ = relabel_and_select(episodes, epsilon=np.inf, delta=0.5)
    assert len(kept) == len(episodes)


# ------------------------------------------------------------------------- the reward
def test_sr_bonus_is_never_positive():
    """The clamp at zero is the paper's, and it is load-bearing.

    Without it, ending far from the anti-goal would be PAID for, making the anti-goal a
    second attractor rather than a repulsor.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        terminal, goal, anti = rng.uniform(0, 10, 2), rng.uniform(0, 10, 2), rng.uniform(0, 10, 2)
        assert sr_terminal_bonus(terminal, goal, anti, reached=False) <= 0.0


def test_successful_episodes_get_no_sr_term():
    """Otherwise success would be worth less on the SR arm than on the PPO arm."""
    assert sr_terminal_bonus([1, 1], [1, 1], [9, 9], reached=True) == 0.0


def test_missing_anti_goal_raises_rather_than_degrading_silently():
    with pytest.raises(ValueError, match="anti-goal"):
        sr_terminal_bonus([0, 0], [1, 1], None, reached=False)


def test_reward_is_applied_only_to_the_final_transition():
    """The per-step reward must stay exactly the environment's."""
    goal = [10.0, 0.0]
    episode = make_episode(0, 0, [1.0, 0.0], goal, n_steps=5)
    episode.rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    episode.anti_goal = np.array([2.0, 0.0])
    bonus = apply_sr_reward(episode)
    assert episode.rewards[:4] == [1.0, 2.0, 3.0, 4.0]
    assert episode.rewards[4] == pytest.approx(5.0 + bonus)
    assert bonus < 0.0


# ------------------------------------------------------------------ asymmetric critic
def test_actor_is_identical_in_width_to_the_ppo_arm():
    """The SR actor must see exactly the env observation, so the arms differ only by SR."""
    import torch as th
    from gymnasium import spaces
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy, widen_observation_space

    base = spaces.Box(low=-1, high=1, shape=(34,), dtype=np.float32)
    widened = widen_observation_space(base)
    assert widened.shape == (36,)

    policy = AntiGoalActorCriticPolicy(
        widened, spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        lambda _: 3e-4, net_arch=[256, 256], policy_obs_dim=34)
    assert policy.mlp_extractor.policy_net[0].in_features == 34
    assert policy.mlp_extractor.value_net[0].in_features == 36


def test_actions_ignore_the_anti_goal_columns():
    """Changing only the anti-goal must not change the action distribution.

    This is what makes the placeholder zeros used during rollout collection safe: the
    actions actually taken cannot depend on a value that was not known at the time.
    """
    import torch as th
    from gymnasium import spaces
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy, widen_observation_space

    base = spaces.Box(low=-1, high=1, shape=(34,), dtype=np.float32)
    policy = AntiGoalActorCriticPolicy(
        widen_observation_space(base),
        spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        lambda _: 3e-4, net_arch=[64, 64], policy_obs_dim=34)
    policy.set_training_mode(False)

    obs = th.zeros(4, 36)
    obs[:, :34] = th.randn(4, 34)
    other = obs.clone()
    other[:, 34:] = th.randn(4, 2)

    with th.no_grad():
        a = policy.get_distribution(obs[:, :34]).distribution.mean
        b = policy.get_distribution(other[:, :34]).distribution.mean
        va = policy.predict_values(obs)
        vb = policy.predict_values(other)
    th.testing.assert_close(a, b)
    assert not th.allclose(va, vb), "the critic MUST respond to the anti-goal"


def test_sr_model_survives_a_save_load_round_trip(tmp_path, week4_config):
    """A saved SR model must load and act without an environment attached.

    This is a regression test for a real failure the pipeline smoke run caught: training
    worked, but every saved SR model was unloadable, so the bug only surfaced at evaluation
    time -- after the expensive part. Three separate things broke, and each is asserted:

      * `__init__` had REQUIRED keyword arguments, but SB3 reconstructs a model by calling
        the class with only policy and env;
      * the sibling-pairing check read `n_envs`, which does not exist when loading with no
        environment;
      * `predict()` routes through SB3's `get_distribution`, which bypasses the
        `extract_features` override and handed the 36-wide observation to the 34-wide actor.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_env.robot_nav_env import RobotNavEnv
    from sibling_rivalry.ppo_sr import SiblingRivalryPPO
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy
    from sibling_rivalry.vec_env import AntiGoalObsWrapper

    def make():
        return RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=3)

    env = AntiGoalObsWrapper(DummyVecEnv([make, make]))
    model = SiblingRivalryPPO(
        policy=AntiGoalActorCriticPolicy, env=env, n_steps=64, batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32], policy_obs_dim=env.policy_obs_dim),
        sr_epsilon=1.5, sr_delta=0.6, verbose=0)
    path = tmp_path / "sr_model"
    model.save(str(path))
    env.close()

    restored = SiblingRivalryPPO.load(str(path), device="cpu")
    assert restored.sr_epsilon == 1.5 and restored.sr_delta == 0.6

    observation = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    action, _ = restored.predict(observation, deterministic=True)
    assert action.shape == (2,) and np.all(np.isfinite(action))


def test_training_refuses_without_a_calibrated_epsilon(week4_config):
    """An SR arm must not train on a default epsilon.

    epsilon now defaults to None so saved models can be reconstructed; the guard therefore
    has to live at the point of use. Without it, a model built for evaluation and then
    handed to `learn()` would silently train with no acceptance rule at all.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_env.robot_nav_env import RobotNavEnv
    from sibling_rivalry.ppo_sr import SiblingRivalryPPO
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy
    from sibling_rivalry.vec_env import AntiGoalObsWrapper

    def make():
        return RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=3)

    env = AntiGoalObsWrapper(DummyVecEnv([make, make]))
    model = SiblingRivalryPPO(
        policy=AntiGoalActorCriticPolicy, env=env, n_steps=64, batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32], policy_obs_dim=env.policy_obs_dim),
        verbose=0)
    with pytest.raises(ValueError, match="sr_epsilon"):
        model.learn(total_timesteps=64)
    env.close()


def test_sr_collects_the_same_batch_size_as_stock_ppo(week4_config):
    """SR must update on n_steps * n_envs transitions, like stock PPO.

    Regression test for a bug that survived every earlier check and only surfaced after
    three hours of training. Stable-Baselines3 passes `n_rollout_steps = n_steps`, meaning
    it PER ENVIRONMENT; this collector counts total transitions, so without scaling by
    n_envs the SR arm updated on 1/8 of PPO's batch.

    Nothing crashed. The arm simply took 1294 noisy updates instead of 184, approx_kl rose
    0.030 -> 0.710, clip_fraction 0.231 -> 0.68, and the action std collapsed 0.308 -> 0.078
    -- a result that reads as "Sibling Rivalry destabilises PPO" while actually being an
    accounting error.

    `n_steps` is deliberately large here. A first version of this test used n_steps=64 with
    4 envs, and PASSED against the buggy code: one batch of complete episodes already
    overshot both the correct target (256) and the broken one (64), so the two were
    indistinguishable. The target must exceed what a single batch of episodes yields for the
    assertion to have any power.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_env.robot_nav_env import RobotNavEnv
    from sibling_rivalry.ppo_sr import SiblingRivalryPPO
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy
    from sibling_rivalry.vec_env import AntiGoalObsWrapper

    n_envs, n_steps = 4, 512
    target = n_steps * n_envs

    def make():
        return RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=3)

    env = AntiGoalObsWrapper(DummyVecEnv([make] * n_envs))
    model = SiblingRivalryPPO(
        policy=AntiGoalActorCriticPolicy, env=env, n_steps=n_steps, batch_size=64,
        policy_kwargs=dict(net_arch=[32, 32], policy_obs_dim=env.policy_obs_dim),
        sr_epsilon=1.0, sr_delta=0.6, verbose=0)

    # learn(total_timesteps=1) performs exactly ONE collect_rollouts: the first rollout
    # already carries num_timesteps past 1, so the loop exits. Measuring after a full
    # learn() would instead sum several rollouts and hide an undersized one -- which is how
    # the first two versions of this test passed against the buggy code.
    model.learn(total_timesteps=1, progress_bar=False)
    collected = model.num_timesteps
    buffered = model.rollout_buffer.buffer_size
    env.close()

    # Whole episodes are never cut mid-way, so the collector overshoots rather than landing
    # exactly on the target. It must never UNDERSHOOT: that is the bug.
    assert collected >= target, (
        f"one rollout collected {collected} transitions against a target of {target}; SR "
        "would train on a smaller batch than the PPO arm and the comparison would be unfair")
    assert collected < 4 * target, (
        f"one rollout collected {collected} against a target of {target} -- the overshoot "
        "from finishing whole episodes should be bounded")
    # The buffer holds only the episodes the acceptance rule retained, so it is smaller than
    # the number of steps taken -- but it must still be a substantial fraction of them.
    assert 0.2 * target <= buffered <= collected
