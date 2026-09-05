"""Experiment 4 training: four arms differing only in SR on/off and LiDAR on/off.

THE FOUR ARMS

    arm            SR    LiDAR   obs   what it isolates
    ppo            no    no       4    the baseline; the paper's PPO arm (Eq. 2)
    ppo_lidar      no    yes     28    LiDAR alone
    ppo_sr         yes   no       4    SR alone; the paper's PPO+SR arm (Eq. 3)
    ppo_sr_lidar   yes   yes     28    the primary question: does LiDAR add anything to SR

Everything else -- maze, start, goal, delta, reward magnitudes, horizon, PPO
hyperparameters, action space, physics -- is shared by construction, because all four read
the same `config_exp4.json` and differ only in the two booleans above.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from robot_env.point_maze import PointMazeEnv, load_exp4_config

CONFIG_PATH = "config_exp4.json"

ARMS = {
    "ppo":          dict(sr=False, lidar=False),
    "ppo_lidar":    dict(sr=False, lidar=True),
    "ppo_sr":       dict(sr=True,  lidar=False),
    "ppo_sr_lidar": dict(sr=True,  lidar=True),
}


def pin_torch_threads(n_threads=1):
    """See training/train_flat.py. Four concurrent jobs x 16 torch threads on 16 cores was
    measured at a 3.4x slowdown in Week 5; one thread per job is the fix."""
    n_threads = max(1, int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)
    return n_threads


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(config_path, use_lidar, rank, seed):
    """One sub-environment. Deliberately NOT wrapped in `Monitor` here.

    `VecMonitor` wraps the vector env below and records episode statistics once. Wrapping
    each sub-env in `Monitor` as well makes both layers write an `episode` key and the outer
    one then reports statistics the inner one already consumed -- the repo's own
    `robot_env/env_factory.py` uses VecMonitor alone for exactly this reason.
    """
    def factory():
        env = PointMazeEnv(config_path=config_path, use_lidar=use_lidar)
        env.reset(seed=seed + rank)
        return env
    return factory


def create_envs(config, config_path, use_lidar, seed, monitor_dir=None):
    n_envs = int(config["training"]["n_envs"])
    factories = [make_env(config_path, use_lidar, i, seed) for i in range(n_envs)]
    vec = SubprocVecEnv(factories)
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        vec = VecMonitor(vec, filename=os.path.join(monitor_dir, "monitor"),
                         info_keywords=("is_success", "collision_count"))
    return vec


def shared_ppo_kwargs(training):
    """Optimiser settings common to all four arms, so they cannot drift apart."""
    learning_rate = training["learning_rate"]
    if training.get("lr_schedule", "constant") == "linear":
        initial = float(learning_rate)
        learning_rate = lambda progress_remaining: progress_remaining * initial
    return dict(
        learning_rate=learning_rate,
        n_steps=training["n_steps"],
        batch_size=training["batch_size"],
        n_epochs=training["n_epochs"],
        gamma=training["gamma"],
        gae_lambda=training.get("gae_lambda", 0.95),
        ent_coef=training["ent_coef"],
        target_kl=training.get("target_kl"),
    )


def build_model(arm, seed, *, config_path=CONFIG_PATH, epsilon=None, log_dir=None):
    """The model and its env for one arm. SR and LiDAR are the only branches."""
    spec = ARMS[arm]
    config = load_exp4_config(config_path)
    training = config["training"]
    env = create_envs(config, config_path, spec["lidar"], seed, monitor_dir=log_dir)

    if not spec["sr"]:
        model = PPO(policy=training["policy"], env=env, seed=seed,
                    policy_kwargs={"net_arch": training["policy_kwargs"]["net_arch"]},
                    tensorboard_log=log_dir, verbose=1, **shared_ppo_kwargs(training))
        return model, env

    from sibling_rivalry.policy import AntiGoalActorCriticPolicy
    from sibling_rivalry.ppo_sr import SiblingRivalryPPO
    from sibling_rivalry.vec_env import AntiGoalObsWrapper

    env = AntiGoalObsWrapper(env)
    if epsilon is None:
        epsilon = config.get("sr", {}).get("epsilon")
    if epsilon is None:
        raise ValueError(
            "the SR arms need an epsilon. Run sibling_rivalry.calibrate against the Point "
            "Maze first and pass --epsilon; rho scales with the arena and horizon, so a "
            "value from the dynamic-obstacle environment is meaningless here."
        )
    model = SiblingRivalryPPO(
        policy=AntiGoalActorCriticPolicy, env=env, seed=seed,
        policy_kwargs=dict(net_arch=training["policy_kwargs"]["net_arch"],
                           policy_obs_dim=env.policy_obs_dim),
        tensorboard_log=log_dir, verbose=1,
        sr_epsilon=epsilon,
        sr_delta=config["sr"]["delta"],
        sr_use_anti_goal=config["sr"]["use_anti_goal"],
        # Experiment 4 uses the paper's Eq. 3 in place of Eq. 2, not on top of it.
        sr_terminal_mode=config["sr"].get("terminal_mode", "replace"),
        world_size=float(PointMazeEnv(config_path=config_path).WORLD_SIZE),
        **shared_ppo_kwargs(training),
    )
    return model, env


def train_arm(arm, seed, *, total_timesteps=None, config_path=CONFIG_PATH, epsilon=None,
              results_root="results/experiment4", tag=""):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    spec = ARMS[arm]
    config = load_exp4_config(config_path)
    total_timesteps = int(total_timesteps or config["training"]["total_timesteps"])
    threads = pin_torch_threads(config["training"].get("torch_threads", 1))
    set_global_seeds(seed)

    run_tag = f"{arm}_seed{seed}{tag}"
    out_dir = os.path.join(results_root, arm, f"seed{seed}")
    log_dir = os.path.join("logs", "experiment4", run_tag)
    model_dir = os.path.join("models", "experiment4", run_tag)
    for path in (out_dir, log_dir, model_dir):
        os.makedirs(path, exist_ok=True)

    model, env = build_model(arm, seed, config_path=config_path, epsilon=epsilon,
                             log_dir=log_dir)
    checkpoint = CheckpointCallback(
        save_freq=max(1, 200_000 // config["training"]["n_envs"]),
        save_path=model_dir, name_prefix="ppo_sr" if spec["sr"] else "ppo")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint, progress_bar=False)
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)

    run_config = {
        "arm": arm, "seed": seed, "sr": spec["sr"], "lidar": spec["lidar"],
        "git_commit": git_commit(), "total_timesteps": total_timesteps, "tag": tag,
        "environment": config["environment"], "reward": config["reward"],
        "ppo": config["training"],
        "ppo_effective": {
            "batch_size": int(model.batch_size), "n_steps": int(model.n_steps),
            "gamma": float(model.gamma), "gae_lambda": float(model.gae_lambda),
            "target_kl": None if model.target_kl is None else float(model.target_kl),
            "observation_dim": int(model.observation_space.shape[0]),
            "torch_threads": threads,
        },
        "sr_hyperparameters": ({"epsilon": epsilon, "delta": config["sr"]["delta"],
                                "terminal_mode": config["sr"].get("terminal_mode")}
                               if spec["sr"] else None),
        "model_dir": model_dir, "log_dir": log_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as handle:
        json.dump(run_config, handle, indent=2)
    env.close()
    return {"model": final_path + ".zip", "arm": arm, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--results-root", default="results/experiment4")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    print(json.dumps(train_arm(
        args.arm, args.seed, total_timesteps=args.timesteps, config_path=args.config,
        epsilon=args.epsilon, results_root=args.results_root, tag=args.tag), indent=2))


if __name__ == "__main__":
    main()
