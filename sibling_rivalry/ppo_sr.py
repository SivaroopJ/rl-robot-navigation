"""SiblingRivalryPPO: Stable-Baselines3 PPO with paired episodic rollout collection.

WHY collect_rollouts HAD TO BE REPLACED
---------------------------------------
Stock PPO collects a fixed-length slice of `n_steps` transitions per environment and cuts
episodes wherever the slice ends. Sibling Rivalry cannot work with that, for two reasons
that are both about the TERMINAL state:

  * The anti-goal IS a terminal state. A half-collected episode has none, so it can be
    neither relabeled nor rewarded.
  * The acceptance rule keeps or discards WHOLE episodes. Discarding is not expressible as
    a mask over a fixed-shape buffer without leaving holes that GAE would read across.

So this collector runs the vector env until every sub-environment has finished at least one
episode and the transition budget is met, applies `relabel_and_select`, adds the SR terminal
payout, and only then packs the survivors into a rollout buffer sized to fit them.

SIBLING PAIRING
---------------
Sub-environments 2i and 2i+1 form pair i and are reset with the SAME seed. Because
RobotNavEnv derives everything from `self.np_random` -- start, goal, obstacle spawn points,
initial headings, and the entire Ornstein-Uhlenbeck noise stream -- the two siblings face a
byte-identical world and diverge ONLY through the actions sampled from the stochastic
policy. That is the paper's requirement, and it is what makes `rho` a measure of policy
variance. Had the siblings been allowed to see different obstacle trajectories, `rho` would
have mixed policy variance with environment variance and `epsilon` would have been
calibrated against a quantity that is not the one the rule is about.

`_assert_pairs_match` checks this every rollout rather than trusting it.

THE ENVIRONMENT-STEP ACCOUNTING
-------------------------------
`num_timesteps` advances by every step actually taken in the environment, including steps
belonging to episodes the acceptance rule later discarded. Both arms therefore spend the
same environment budget, which is the x-axis of every learning curve. SR simply trains on
fewer of those steps -- that is the algorithm, not a handicap.

GAE AND TRUNCATION
------------------
The buffer is packed as a single column of concatenated episodes with `episode_starts`
marking the boundaries, so SB3's GAE recursion resets at every boundary and never reads
across episodes. Truncated episodes ARE bootstrapped: `gamma * V(s_T)` is folded into the
final reward, which is what stock SB3 PPO does inside its own `collect_rollouts`.

This reverses an earlier decision, and the reason is worth recording. The original code
followed ppo-nav, which sets `bootstrap_on_truncation = False` on EVERY arm, on the
argument that a bootstrapped truncation blends a learned value into the one transition the
SR payout lives on. That argument is sound only when both arms agree. Here the control arm
is stock SB3 PPO, which bootstraps -- so not bootstrapping in SR made the two arms
incomparable rather than making SR purer. It was invisible while timeouts were ~0% of
episodes, and became a live confound in Week 4's 5M matrix where 18-22% of episodes hit the
step limit: SR was charged the full pessimism of a stall and PPO was not.

Bootstrapping is also the correct treatment of a time-limit truncation in its own right
(Pardo et al., "Time Limits in Reinforcement Learning") -- the episode did not end, the
clock ran out. The SR payout stays separable: `apply_sr_reward` adds it, and the bootstrap
is applied afterwards on top, so neither overwrites the other.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor

from sibling_rivalry.reward import apply_sr_reward
from sibling_rivalry.vec_env import set_pair_seeds
from sibling_rivalry.select import Episode, euclid, relabel_and_select


class SiblingRivalryPPO(PPO):
    """PPO whose rollouts are paired complete episodes, relabeled by Sibling Rivalry."""

    def __init__(self, *args, sr_epsilon: float | None = None,
                 sr_delta: float | None = None,
                 sr_use_anti_goal: bool = True, sr_episode_seed_base: int = 0,
                 sr_terminal_mode: str = "add",
                 world_size: float = 10.0, **kwargs):
        """
        Parameters
        sr_epsilon: float
            Sibling-agreement threshold. Calibrate it with `sibling_rivalry.calibrate`;
            a value transplanted from another environment is meaningless because rho scales
            with the arena and the episode horizon.
        sr_delta: float
            Success radius, set to the env's TARGET_RADIUS so SR's success set and the
            environment's termination test agree.
        sr_use_anti_goal: bool
            False runs the acceptance rule but zeroes the anti-goal, which is the paper's
            own ablation ("noag") isolating relabeling from selection.
        sr_episode_seed_base: int
            Offset for the per-episode seeds that make siblings identical.
        world_size: float
            Used only to normalize the anti-goal into the observation's [-1, 1] scale.
        """
        super().__init__(*args, **kwargs)
        # These default to None so `SiblingRivalryPPO.load()` can reconstruct the object.
        # Stable-Baselines3 rebuilds a model by calling the class with only policy/env and
        # then restoring __dict__, so a REQUIRED keyword here would make every saved SR
        # model unloadable -- which is exactly how the evaluator first failed. Training
        # still refuses to start without them; the check moved to `collect_rollouts`.
        self.sr_epsilon = None if sr_epsilon is None else float(sr_epsilon)
        self.sr_delta = None if sr_delta is None else float(sr_delta)
        self.sr_use_anti_goal = bool(sr_use_anti_goal)
        # "add" (Weeks 4-5) bolts the SR payout onto the env's dense reward. "replace"
        # (Experiment 4) substitutes it for the env's terminal -d(s_T, g), which is what
        # the paper's Eq. 3 actually does to Eq. 2. See sibling_rivalry/reward.py.
        self.sr_terminal_mode = str(sr_terminal_mode)
        self.world_size = float(world_size)

        # Guarded on `self.env`: SB3 loads a model for evaluation with no environment
        # attached, in which case `n_envs` does not exist yet and there is nothing to pair.
        if getattr(self, "env", None) is not None:
            if self.n_envs % 2 != 0 or self.n_envs < 2:
                raise ValueError(
                    f"Sibling Rivalry pairs sub-envs 2i and 2i+1, so n_envs must be even "
                    f"and >= 2; got {self.n_envs}. An odd count would silently drop a "
                    "sibling."
                )
            self.n_pairs = self.n_envs // 2
        else:
            self.n_pairs = 0
        self._episode_seed = int(sr_episode_seed_base)
        #: Rolling diagnostics, drained into the logger each rollout.
        self.last_pair_stats = []

    # ------------------------------------------------------------------ helpers
    def _next_pair_seeds(self):
        """One fresh seed per pair; both siblings of a pair get the same one."""
        seeds = []
        for _ in range(self.n_pairs):
            seeds.append(self._episode_seed)
            self._episode_seed += 1
        return seeds

    def _reset_pairs(self, env):
        """Reset every sub-env, giving both members of a pair an identical seed.

        `env.seed()` is deliberately NOT used: it hands sub-env i the seed `base + i`, so
        siblings would face different worlds and `rho` would stop measuring policy variance.
        """
        seeds = self._next_pair_seeds()
        per_env = [seeds[i // 2] for i in range(self.n_envs)]
        set_pair_seeds(env, per_env)
        return env.reset(), per_env

    @staticmethod
    def _assert_pairs_match(positions, goals):
        """Both siblings must start from the same (s0, g). Checked, not assumed."""
        for pair in range(len(positions) // 2):
            a, b = 2 * pair, 2 * pair + 1
            if not (np.allclose(positions[a], positions[b])
                    and np.allclose(goals[a], goals[b])):
                raise AssertionError(
                    f"sibling pair {pair} did not start from the same (s0, g): "
                    f"s0 {positions[a]} vs {positions[b]}, g {goals[a]} vs {goals[b]}. "
                    "Sibling Rivalry requires it (Trott et al. Algorithm 1, line 2)."
                )

    def _anti_goal_columns(self, episode):
        """The anti-goal, normalized to the observation's scale."""
        if not self.sr_use_anti_goal or episode.anti_goal is None:
            return np.zeros(2, dtype=np.float32)
        return np.clip(np.asarray(episode.anti_goal, dtype=np.float32) / self.world_size,
                       -1.0, 1.0)

    # ------------------------------------------------------------------ collection
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Collect paired complete episodes, relabel them, and pack the survivors."""
        assert self._last_obs is not None
        if self.sr_epsilon is None or self.sr_delta is None:
            raise ValueError(
                "training a Sibling Rivalry arm needs sr_epsilon and sr_delta. Calibrate "
                "epsilon with `python -m sibling_rivalry.calibrate` first; an uncalibrated "
                "value makes the acceptance rule either vacuous or destabilising."
            )
        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        # Stable-Baselines3 passes `n_rollout_steps = self.n_steps`, which it means PER
        # ENVIRONMENT: stock PPO fills a (n_steps, n_envs) buffer, i.e. n_steps * n_envs
        # transitions per policy update. This collector counts TOTAL transitions, so the
        # target has to be scaled or SR silently updates on 1/n_envs of PPO's batch.
        #
        # It did, and the effect was severe rather than subtle. At n_envs=8, SR updated on
        # ~2300 transitions against PPO's 16384 and took 1294 noisy updates instead of 184;
        # approx_kl went 0.030 -> 0.710, clip_fraction 0.231 -> 0.68, and the action std
        # collapsed 0.308 -> 0.078. That looks exactly like "Sibling Rivalry destabilises
        # training" and is nothing of the sort -- it is rollout accounting.
        target_transitions = int(n_rollout_steps) * int(self.n_envs)
        completed, steps_taken = [], 0
        # In-flight episode per sub-env, plus the pair index it belongs to.
        obs = self._last_obs
        open_episodes = None

        while steps_taken < target_transitions:
            # A fresh synchronized reset per batch of pairs. Resetting all sub-envs
            # together is what keeps siblings aligned; letting them auto-reset
            # independently would desynchronize the pairs after the first episode.
            obs, _ = self._reset_pairs(env)
            starts = np.array(env.get_attr("agent_position"))
            goals = np.array(env.get_attr("target_position"))
            self._assert_pairs_match(starts, goals)

            pair_base = len(completed) // 2
            open_episodes = [
                Episode(env_index=i, pair_index=pair_base + i // 2, goal=goals[i].copy())
                for i in range(self.n_envs)
            ]
            for i in range(self.n_envs):
                open_episodes[i].positions.append(starts[i].copy())
            alive = np.ones(self.n_envs, dtype=bool)

            while alive.any():
                with th.no_grad():
                    tensor_obs = obs_as_tensor(obs, self.device)
                    # The policy slices the actor's view itself; see
                    # AntiGoalActorCriticPolicy.get_distribution.
                    distribution = self.policy.get_distribution(tensor_obs)
                    actions = distribution.get_actions(deterministic=False)
                    log_probs = distribution.log_prob(actions)
                actions_np = actions.cpu().numpy()
                clipped = np.clip(actions_np, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped)
                self.num_timesteps += int(alive.sum())
                steps_taken += int(alive.sum())

                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                for i in range(self.n_envs):
                    if not alive[i]:
                        continue
                    episode = open_episodes[i]
                    episode.observations.append(obs[i].copy())
                    episode.actions.append(actions_np[i].copy())
                    episode.log_probs.append(float(log_probs[i].item()))
                    episode.rewards.append(float(rewards[i]))
                    episode.positions.append(np.asarray(infos[i]["agent_position"], float))
                    if dones[i]:
                        episode.success = bool(infos[i].get("is_success", False))
                        episode.collision_type = infos[i].get("collision_type")
                        episode.truncated = bool(
                            infos[i].get("TimeLimit.truncated", False)
                            or (not episode.success and episode.collision_type is None))
                        if episode.truncated:
                            terminal = infos[i].get("terminal_observation")
                            if terminal is not None:
                                episode.terminal_observation = np.asarray(
                                    terminal, dtype=np.float32).copy()
                        completed.append(episode)
                        alive[i] = False
                obs = new_obs

        self._last_obs = obs

        # ---- Sibling Rivalry proper
        kept, stats = relabel_and_select(
            completed, epsilon=self.sr_epsilon, delta=self.sr_delta)
        for episode in kept:
            # Under "replace" the env's terminal payout has to be removed before SR's is
            # put in its place. The env pays -d(s_T, g) on a truncation and +1 on success;
            # both equations agree on success, so only the truncation branch is undone.
            baseline = 0.0
            if self.sr_terminal_mode == "replace" and not episode.success:
                baseline = -euclid(episode.terminal(), episode.goal)
            apply_sr_reward(episode, terminal_mode=self.sr_terminal_mode,
                            terminal_baseline=baseline)
        self.last_pair_stats = stats

        self._pack_buffer(kept)
        self._log_sr_diagnostics(completed, kept, stats)
        callback.on_rollout_end()
        return True

    def _pack_buffer(self, kept):
        """Pack retained episodes into a fresh single-column buffer and run GAE.

        A new buffer is built each iteration because the retained transition count varies
        with the acceptance rate; a fixed-size buffer would either truncate an episode --
        destroying the terminal payout SR's whole signal lives on -- or leave holes.
        """
        n_transitions = sum(len(e) for e in kept)
        if n_transitions == 0:
            raise RuntimeError(
                "Sibling Rivalry retained no transitions this iteration. tau_f is always "
                "kept, so this means no episode completed at all."
            )

        buffer = RolloutBuffer(
            buffer_size=n_transitions, observation_space=self.observation_space,
            action_space=self.action_space, device=self.device,
            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=1,
        )

        n_bootstrapped = 0
        for episode in kept:
            anti_goal = self._anti_goal_columns(episode)
            observations = np.asarray(episode.observations, dtype=np.float32)
            observations[:, -2:] = anti_goal          # the placeholder zeros, filled in
            with th.no_grad():
                values = self.policy.predict_values(
                    obs_as_tensor(observations, self.device)).cpu().numpy().flatten()

            # Truncation bootstrap, matching what stock SB3 PPO does in its own
            # collect_rollouts. An episode that hit the step limit did not end -- the
            # clock ran out -- so its final reward carries gamma * V(s_T) for the
            # continuation it never got to play. Without this, SR pays the full pessimism
            # of a stall while the PPO arm does not, and the two arms stop being
            # comparable the moment stalling becomes common (Week 4's 5M matrix: 18-22%
            # of episodes). Applied AFTER apply_sr_reward, so the SR terminal payout and
            # the bootstrap compose rather than overwrite.
            if episode.truncated and episode.terminal_observation is not None:
                terminal_obs = episode.terminal_observation.copy().reshape(1, -1)
                terminal_obs[:, -2:] = anti_goal
                with th.no_grad():
                    terminal_value = float(self.policy.predict_values(
                        obs_as_tensor(terminal_obs, self.device)).cpu().numpy().flatten()[0])
                episode.rewards[-1] = float(episode.rewards[-1]) + self.gamma * terminal_value
                n_bootstrapped += 1

            for t in range(len(episode)):
                buffer.add(
                    observations[t].reshape(1, -1),
                    np.asarray(episode.actions[t], dtype=np.float32).reshape(1, -1),
                    np.array([episode.rewards[t]], dtype=np.float32),
                    np.array([1.0 if t == 0 else 0.0], dtype=np.float32),
                    th.tensor([values[t]], device=self.device),
                    th.tensor([episode.log_probs[t]], device=self.device),
                )

        # Every packed episode is complete, so the buffer-level call needs no bootstrap of
        # its own -- truncated episodes have already had gamma * V(s_T) folded into their
        # final reward above, which is exactly how SB3 PPO handles the same case.
        buffer.compute_returns_and_advantage(
            last_values=th.zeros(1, 1, device=self.device), dones=np.ones(1, dtype=bool))
        self.rollout_buffer = buffer
        self.logger.record("sr/n_truncated_bootstrapped", n_bootstrapped)

    def _log_sr_diagnostics(self, completed, kept, stats):
        """Record the numbers that reveal whether SR is actually doing anything."""
        if not stats:
            return
        rho = np.array([s.rho for s in stats])
        self.logger.record("sr/n_pairs", len(stats))
        self.logger.record("sr/n_episodes", len(completed))
        self.logger.record("sr/n_retained", len(kept))
        self.logger.record("sr/accept_rate_closer",
                           float(np.mean([s.kept_closer for s in stats])))
        self.logger.record("sr/rho_mean", float(rho.mean()))
        self.logger.record("sr/rho_p50", float(np.percentile(rho, 50)))
        self.logger.record("sr/rho_p90", float(np.percentile(rho, 90)))
        self.logger.record("sr/epsilon", self.sr_epsilon)
        self.logger.record("sr/success_rate",
                           float(np.mean([e.success for e in completed])))
