"""Measure the sibling-distance distribution, so epsilon is calibrated rather than guessed.

WHY epsilon CANNOT BE TRANSPLANTED
----------------------------------
The acceptance rule keeps the closer sibling when `rho = ||s_T^c - s_T^f|| < epsilon`.
`rho` is a distance in world units, so its scale is set by the arena, the episode horizon
and how much the current policy's stochasticity spreads two rollouts apart. A value tuned
on one environment says nothing about another: too large and every pair is accepted, which
reduces Sibling Rivalry to ordinary PPO; too small and the closer sibling is almost never
retained, which destabilises training by pushing the policy away from the goal region even
after it has learned to reach it.

ppo-nav measured 2.25 for PointMaze against a 5.39-unit straight-line distance. This arena
stratifies pairs to a ~6.9-unit mean over a 10x10 map with a different horizon, so the
number has to be measured here.

WHAT IS MEASURED
----------------
`rho` under an UNTRAINED policy, which is the regime the rule has to be sane in first --
early training is when both siblings are diffuse and when a badly set epsilon does the most
damage. The reported percentile is the recommended epsilon: at the default 50th, roughly
half of pairs retain both siblings, so the rule is active in both directions rather than
saturated at accept-everything or reject-everything.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sibling_rivalry.select import euclid


def measure_rho(model, env, *, n_pairs: int, seed_base: int = 0):
    """Roll out sibling pairs under `model`'s current policy and return their rho values.

    Parameters
    model: SiblingRivalryPPO
    env: VecEnv
        Must be the anti-goal-wrapped vector env the model was built on.
    n_pairs: int
        Number of sibling pairs to roll out.

    Returns
    rho: np.ndarray
        Sibling terminal-state distances.
    d_goal: np.ndarray
        Terminal distance to goal, one per episode, for context.
    """
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    from sibling_rivalry.vec_env import set_pair_seeds

    n_envs = env.num_envs
    rho, d_goal = [], []
    pairs_done = 0
    seed = seed_base

    while pairs_done < n_pairs:
        per_env = [seed + i // 2 for i in range(n_envs)]
        seed += n_envs // 2
        set_pair_seeds(env, per_env)
        obs = env.reset()
        goals = np.array(env.get_attr("target_position"))
        terminal = [None] * n_envs
        alive = np.ones(n_envs, dtype=bool)

        while alive.any():
            with th.no_grad():
                dist = model.policy.get_distribution(obs_as_tensor(obs, model.device))
                actions = dist.get_actions(deterministic=False).cpu().numpy()
            obs, _, dones, infos = env.step(
                np.clip(actions, env.action_space.low, env.action_space.high))
            for i in range(n_envs):
                if alive[i] and dones[i]:
                    terminal[i] = np.asarray(infos[i]["agent_position"], dtype=float)
                    alive[i] = False

        for pair in range(n_envs // 2):
            a, b = 2 * pair, 2 * pair + 1
            rho.append(euclid(terminal[a], terminal[b]))
            d_goal.append(euclid(terminal[a], goals[a]))
            d_goal.append(euclid(terminal[b], goals[b]))
            pairs_done += 1

    return np.array(rho), np.array(d_goal)


def summarize(rho, d_goal, percentile=50.0):
    """Recommended epsilon and the distribution it came from."""
    return {
        "n_pairs": int(len(rho)),
        "rho_mean": float(rho.mean()),
        "rho_percentiles": {str(p): float(np.percentile(rho, p))
                            for p in (10, 25, 50, 75, 90)},
        "d_goal_mean": float(d_goal.mean()),
        "recommended_epsilon": float(np.percentile(rho, percentile)),
        "recommended_at_percentile": float(percentile),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obstacles", type=int, default=6)
    parser.add_argument("--obstacle-speed", type=float, default=1.0)
    parser.add_argument("--obstacle-speed-range", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="calibrate against a sampled speed range. rho's scale depends "
                             "on the environment, so a matrix run at a different obstacle "
                             "speed needs its own epsilon.")
    parser.add_argument("--randomize", action="store_true",
                        help="calibrate on the randomized-obstacle env (Exp 2/3)")
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--percentile", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from training.train_flat import build_sr_model  # imported late: heavy dependencies

    model, env = build_sr_model(
        config_path=args.config, n_dynamic_obstacles=args.n_obstacles,
        obstacle_speed=args.obstacle_speed,
        obstacle_speed_range=args.obstacle_speed_range,
        randomize=args.randomize, seed=args.seed, epsilon=1.0)
    rho, d_goal = measure_rho(model, env, n_pairs=args.n_pairs, seed_base=10_000)
    report = summarize(rho, d_goal, args.percentile)
    report["randomize"] = args.randomize
    report["n_obstacles"] = args.n_obstacles
    report["obstacle_speed"] = args.obstacle_speed
    report["obstacle_speed_range"] = args.obstacle_speed_range
    env.close()

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
