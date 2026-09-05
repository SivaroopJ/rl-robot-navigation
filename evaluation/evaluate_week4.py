"""Week 4 evaluation: paired, seed-matched, with the metrics the brief asks for.

WHAT IS NEW AGAINST `evaluation/evaluate.py`
--------------------------------------------
The existing evaluator records success, collision count, episode length and final distance.
Week 4 additionally needs return, path length, SPL, and a breakdown of WHICH kind of
obstacle was hit -- the last one because the research question is specifically about
dynamic obstacles, and the committed results cannot distinguish a wall from a moving
obstacle. Terminal positions and collision locations are also recorded, because the
terminal-state density and collision heatmaps are computed from them.

The existing evaluator is left untouched so the e1-e4 results stay reproducible. It also
has a latent bug that would have bitten here: it constructs the env without passing
`use_reward_shaping`, so the flag always defaults True. That is harmless while no
reward-based metric is recorded and silently wrong the moment return is, which is why this
module passes the flag explicitly.

PAIRING, AND WHY THE EVALUATION SEEDS ARE DISJOINT FROM TRAINING
----------------------------------------------------------------
Every arm is evaluated on the same seed block, so for a given episode index PPO and PPO+SR
face an identical start, goal, obstacle spawn and Ornstein-Uhlenbeck noise stream. The
comparison is therefore paired, and the difference between arms is the policy alone.

The block starts at `EVAL_SEED_BASE`, far from the seeds training consumes (training draws
`config.seed + rank`, and the SR arms consume a counter from 0 upward). The existing
evaluator reuses seeds 0..199, which overlap the training range -- fine when nothing is
claimed about generalization, not fine for Week 4, where Experiments 2 and 3 rest on being
tested on trajectory realizations that were never trained on.
"""
from __future__ import annotations

import json
import os

import numpy as np

from evaluation.shortest_path import ShortestPathOracle
from robot_env.robot_nav_env import RobotNavEnv, load_config

#: Disjoint from every seed training consumes. See the module docstring.
EVAL_SEED_BASE = 1_000_000


def _distance_band(distance, bins):
    for index, (low, high) in enumerate(bins):
        if low <= distance <= high:
            return index
    return None


def load_policy(model_path, sr):
    """Load a PPO or PPO+SR policy for evaluation.

    The SR policy was trained on a 36-wide observation whose last two columns carry the
    anti-goal. At evaluation there is no sibling and therefore no anti-goal -- but the
    ACTOR never reads those columns (see `sibling_rivalry/policy.py`), so padding them with
    zeros reproduces the acting policy exactly rather than approximating it.
    """
    if sr:
        from sibling_rivalry.ppo_sr import SiblingRivalryPPO
        return SiblingRivalryPPO.load(model_path, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(model_path, device="cpu")


def evaluate_model(model_path, *, sr, randomize, n_dynamic_obstacles=6,
                   obstacle_speed=1.0, n_eval_episodes=200, config_path="config.json",
                   deterministic=True, out_path=None, label=""):
    """Evaluate one policy and return the full metric record.

    Parameters
    sr: bool
        Whether `model_path` is a Sibling Rivalry model (changes loading and obs padding).
    randomize: bool
        Which environment to evaluate ON. Independent of how the model was TRAINED, which
        is what makes the zero-shot analysis a call to this same function.
    """
    config = load_config(config_path)
    env = RobotNavEnv(config_path=config_path, n_dynamic_obstacles=n_dynamic_obstacles,
                      obstacle_speed=obstacle_speed, render_mode=None,
                      use_reward_shaping=True,
                      randomize_dynamic_obstacles=randomize)
    model = load_policy(model_path, sr)
    oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    bins = config["start_goal"]["distance_bins"]
    pad = 0 if not sr else int(model.policy.anti_goal_dim)

    episodes = []
    for index in range(n_eval_episodes):
        seed = EVAL_SEED_BASE + index
        observation, _ = env.reset(seed=seed)
        start = env.agent_position.copy()
        goal = env.target_position.copy()
        shortest = oracle.path_length(start, goal)

        trajectory = [start.copy()]
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            model_obs = observation if pad == 0 else np.concatenate(
                [observation, np.zeros(pad, dtype=observation.dtype)])
            action, _ = model.predict(model_obs, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            trajectory.append(env.agent_position.copy())

        trajectory = np.asarray(trajectory)
        path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
        success = bool(info["success"])
        collision_type = info.get("collision_type")
        straight = float(np.linalg.norm(goal - start))

        # SPL uses the true shortest path (see evaluation/shortest_path.py). An
        # unreachable pair contributes None rather than a fabricated score.
        if not success or shortest is None:
            spl = 0.0 if shortest is not None else None
        else:
            spl = float(shortest / max(path_length, shortest, 1e-9))

        episodes.append({
            "episode": index,
            "seed": seed,
            "outcome": "SUCCESS" if success else ("TIMEOUT" if truncated else "COLLISION"),
            "success": success,
            "collision": bool(collision_type is not None and not success),
            "collision_type": None if success else collision_type,
            "timeout": bool(truncated and not success),
            "steps": steps,
            "return": total_reward,
            "path_length": path_length,
            "straight_line_distance": straight,
            "shortest_path": shortest,
            "spl": spl,
            "final_distance": float(info["distance_to_target"]),
            "distance_band": _distance_band(straight, bins),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "terminal_position": trajectory[-1].tolist(),
            "collision_position": (trajectory[-1].tolist()
                                   if collision_type is not None and not success else None),
        })
    env.close()

    record = summarize(episodes)
    record.update({
        "label": label, "model_path": model_path, "sr": sr,
        "evaluated_on_randomized": randomize,
        "n_dynamic_obstacles": n_dynamic_obstacles, "obstacle_speed": obstacle_speed,
        "n_episodes": n_eval_episodes, "deterministic": deterministic,
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_seeds": [EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval_episodes - 1],
        "distance_bins": bins,
        "episodes": episodes,
    })
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(record, handle, indent=2)
    return record


def summarize(episodes):
    """Aggregate metrics, overall and per start-goal distance band."""
    n = len(episodes)
    if n == 0:
        return {}

    def rate(key):
        return float(np.mean([bool(e[key]) for e in episodes]))

    def mean(key):
        values = [e[key] for e in episodes if e[key] is not None]
        return float(np.mean(values)) if values else None

    types = [e["collision_type"] for e in episodes if e["collision"]]
    summary = {
        "success_rate": rate("success"),
        "collision_rate": rate("collision"),
        "timeout_rate": rate("timeout"),
        "wall_collisions": int(sum(t == "wall" for t in types)),
        "static_collisions": int(sum(t == "static" for t in types)),
        "dynamic_collisions": int(sum(t == "dynamic" for t in types)),
        "avg_episode_length": mean("steps"),
        "avg_path_length": mean("path_length"),
        "mean_return": mean("return"),
        "avg_final_distance": mean("final_distance"),
        "mean_spl": mean("spl"),
        "n_success": int(sum(e["success"] for e in episodes)),
        "n_collision": int(sum(e["collision"] for e in episodes)),
        "n_timeout": int(sum(e["timeout"] for e in episodes)),
    }

    # Per-band breakdown: the point of stratifying start/goal pairs is being able to ask
    # whether an effect lives on the long crossings rather than in the average.
    bands = {}
    for episode in episodes:
        bands.setdefault(episode["distance_band"], []).append(episode)
    summary["by_distance_band"] = {
        str(band): {
            "n": len(group),
            "success_rate": float(np.mean([e["success"] for e in group])),
            "collision_rate": float(np.mean([e["collision"] for e in group])),
            "dynamic_collision_rate": float(np.mean(
                [e["collision_type"] == "dynamic" for e in group])),
            "avg_episode_length": float(np.mean([e["steps"] for e in group])),
        }
        for band, group in sorted(bands.items(), key=lambda kv: (kv[0] is None, kv[0]))
    }
    return summary


def paired_comparison(record_a, record_b):
    """Per-episode paired differences between two arms evaluated on the same seeds.

    Both records must come from the same seed block, which `evaluate_model` guarantees.
    A paired test is the right one here: the episodes are matched by construction, so the
    variance from scenario difficulty cancels instead of swamping the policy effect.
    """
    episodes_a = {e["seed"]: e for e in record_a["episodes"]}
    episodes_b = {e["seed"]: e for e in record_b["episodes"]}
    shared = sorted(set(episodes_a) & set(episodes_b))
    if len(shared) != len(episodes_a) or len(shared) != len(episodes_b):
        raise ValueError(
            "the two records were not evaluated on the same seeds, so they cannot be "
            "compared pairwise"
        )

    success_a = np.array([episodes_a[s]["success"] for s in shared], dtype=float)
    success_b = np.array([episodes_b[s]["success"] for s in shared], dtype=float)
    collision_a = np.array([episodes_a[s]["collision"] for s in shared], dtype=float)
    collision_b = np.array([episodes_b[s]["collision"] for s in shared], dtype=float)

    return {
        "n_paired_episodes": len(shared),
        "success_rate_a": float(success_a.mean()),
        "success_rate_b": float(success_b.mean()),
        "success_delta": float(success_b.mean() - success_a.mean()),
        "collision_rate_a": float(collision_a.mean()),
        "collision_rate_b": float(collision_b.mean()),
        "collision_delta": float(collision_b.mean() - collision_a.mean()),
        # McNemar counts: episodes where exactly one arm succeeded. These, not the marginal
        # rates, are what a paired significance test on a binary outcome actually uses.
        "only_a_success": int(np.sum((success_a == 1) & (success_b == 0))),
        "only_b_success": int(np.sum((success_a == 0) & (success_b == 1))),
        "both_success": int(np.sum((success_a == 1) & (success_b == 1))),
        "neither_success": int(np.sum((success_a == 0) & (success_b == 0))),
    }
