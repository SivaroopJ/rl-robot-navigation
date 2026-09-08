"""Week 4 training: one arm, one config, one seed, no curriculum.

WHY FLAT AND NOT THE EXISTING CURRICULUM
----------------------------------------
The repo's baseline trains through five stages of increasing obstacle count. That is a
sensible way to get a single strong policy, and a poor way to compare two algorithms: the
staged schedule interacts with the algorithm under test (a curriculum stage boundary
changes the task distribution at the same moment SR's rho distribution is shifting), and
its retry-on-threshold logic gives the two arms different effective budgets. Week 4 trains
every arm flat at one configuration for a fixed step budget, so the only difference between
arms is Sibling Rivalry.

WHY EVERY ARM RETRAINS FROM SCRATCH
-----------------------------------
No existing checkpoint can be reused. Week 4 changes dt and the start/goal distribution, so
the task is not the one those policies were trained on; and the SR arm's critic reads two
extra inputs, so its parameter shapes differ. Warm-starting would confound the comparison
rather than save time.

RUN TAGGING
-----------
`run_tag` is threaded into the model, log and best-model directories. The existing
`training/train.py` keys `best_model_dir` on the obstacle configuration alone, which is why
its shaping and no-shaping runs silently overwrote each other's best models. With four arms
and three seeds that bug would corrupt the whole comparison, so Week 4 never shares a
directory between runs.
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from robot_env.env_factory import create_eval_env, create_parallel_envs
from robot_env.robot_nav_env import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def pin_torch_threads(n_threads=1):
    """Confine this process's torch intra-op parallelism to `n_threads`.

    The matrix runs four training jobs at once, each of which already owns eight
    SubprocVecEnv worker processes. Torch defaults to one intra-op thread per core -- 16
    here -- so four jobs ask for 64 threads on 16 cores and spend their time in the
    scheduler.

    This did not bite before Week 5 because `batch_size` was 64: the per-minibatch matmuls
    were below the size at which torch bothers to parallelise, so each job quietly stayed
    single-threaded. Raising `batch_size` to 256 pushed them over that threshold, and the
    "free speedup" turned into a measured 3.4x SLOWDOWN (623 fps -> 187 fps at 4-way
    concurrency, 47 threads and 340% CPU per job). Parallelism across runs and across envs
    is already saturating the machine; parallelism inside the backward pass on a
    75k-parameter MLP is pure contention.

    Set before any model is built. `OMP_NUM_THREADS` is exported too so the env worker
    processes inherit it rather than each spawning their own OpenMP pool.
    """
    n_threads = max(1, int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)
    return n_threads

#: The four Week 4 arms. `sr` and `randomize` are the only two axes; everything else --
#: hyperparameters, architecture, budget, env count, evaluation -- is shared by construction.
ARMS = {
    "ppo_fixed":       dict(sr=False, randomize=False, experiment="experiment1", label="ppo"),
    "ppo_sr_fixed":    dict(sr=True,  randomize=False, experiment="experiment1", label="ppo_sr"),
    "ppo_random":      dict(sr=False, randomize=True,  experiment="experiment2", label="ppo_randomized"),
    "ppo_sr_random":   dict(sr=True,  randomize=True,  experiment="experiment3", label="ppo_sr_randomized"),
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sr_model(*, config_path, n_dynamic_obstacles, obstacle_speed, randomize, seed,
                   epsilon, delta=None, use_anti_goal=True, log_dir=None,
                   obstacle_speed_range=None):
    """A SiblingRivalryPPO and the anti-goal-wrapped vector env it trains on.

    Split out from `train_arm` because `sibling_rivalry.calibrate` needs exactly this
    object -- an untrained SR model on the right environment -- to measure the rho
    distribution before a real epsilon is known.
    """
    from sibling_rivalry.ppo_sr import SiblingRivalryPPO
    from sibling_rivalry.policy import AntiGoalActorCriticPolicy
    from sibling_rivalry.vec_env import AntiGoalObsWrapper

    config = load_config(config_path)
    training = config["training"]
    env = AntiGoalObsWrapper(create_parallel_envs(
        config, config_path, n_dynamic_obstacles, obstacle_speed=obstacle_speed,
        obstacle_speed_range=obstacle_speed_range,
        randomize_dynamic_obstacles=randomize, monitor_dir=log_dir))

    model = SiblingRivalryPPO(
        policy=AntiGoalActorCriticPolicy,
        env=env,
        seed=seed,
        **shared_ppo_kwargs(training),
        policy_kwargs=dict(net_arch=training["policy_kwargs"]["net_arch"],
                           policy_obs_dim=env.policy_obs_dim),
        tensorboard_log=log_dir,
        verbose=1,
        sr_epsilon=epsilon,
        sr_delta=delta if delta is not None else config["environment"]["target_radius"],
        sr_use_anti_goal=use_anti_goal,
        world_size=config["environment"]["world_size"],
    )
    return model, env


def shared_ppo_kwargs(training):
    """PPO settings common to both arms, so they can never drift apart.

    Both `build_sr_model` and `build_ppo_model` take their optimiser settings from here.
    The comparison is only meaningful if the two arms differ in the SR machinery alone,
    and a hyperparameter added to one builder but not the other is exactly the kind of
    silent divergence that would break that.

    `learning_rate` may be a float or, when `lr_schedule` is "linear", the callable SB3
    expects: it is passed `progress_remaining`, which falls 1 -> 0 over training.
    """
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


def build_ppo_model(*, config_path, n_dynamic_obstacles, obstacle_speed, randomize, seed,
                    log_dir=None, obstacle_speed_range=None):
    """Stock SB3 PPO on the unmodified observation -- the control arm."""
    config = load_config(config_path)
    training = config["training"]
    env = create_parallel_envs(
        config, config_path, n_dynamic_obstacles, obstacle_speed=obstacle_speed,
        obstacle_speed_range=obstacle_speed_range,
        randomize_dynamic_obstacles=randomize, monitor_dir=log_dir)
    model = PPO(
        policy=training["policy"],
        env=env,
        seed=seed,
        policy_kwargs={"net_arch": training["policy_kwargs"]["net_arch"]},
        tensorboard_log=log_dir,
        verbose=1,
        **shared_ppo_kwargs(training),
    )
    return model, env


def train_arm(arm, seed, *, total_timesteps=None, n_dynamic_obstacles=6,
              obstacle_speed=1.0, config_path=CONFIG_PATH, epsilon=None,
              results_root="results/week4", eval_freq=25_000, n_eval_episodes=20,
              obstacle_speed_range=None, tag=""):
    """Train one arm at one seed.

    `obstacle_speed_range` samples the episode's obstacle speed uniformly from (low, high)
    instead of holding it fixed, and takes priority over `obstacle_speed`. Observations
    always normalise by the range MIDPOINT, so the policy sees a stable scale regardless of
    the sampled speed.

    `tag` is appended to every model, log and results path. A second matrix at different
    obstacle speeds is a different experiment, and must not overwrite the first.
    """
    """Train one arm at one seed and write its model, logs and run configuration."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    spec = ARMS[arm]
    config = load_config(config_path)
    total_timesteps = int(total_timesteps or config["training"]["total_timesteps"])
    torch_threads = pin_torch_threads(config["training"].get("torch_threads", 1))
    set_global_seeds(seed)

    run_tag = f"{arm}_seed{seed}{tag}"
    out_dir = os.path.join(results_root, spec["experiment"], spec["label"], f"seed{seed}")
    log_dir = os.path.join("logs", "week4", run_tag)
    model_dir = os.path.join("models", "week4", run_tag)
    for path in (out_dir, log_dir, model_dir):
        os.makedirs(path, exist_ok=True)

    if spec["sr"]:
        if epsilon is None:
            epsilon = config.get("sr", {}).get("epsilon")
        if epsilon is None:
            raise ValueError(
                "the SR arms need an epsilon. Run sibling_rivalry.calibrate first and pass "
                "--epsilon (or set sr.epsilon in config.json); an uncalibrated value would "
                "make the acceptance rule either vacuous or destabilising."
            )
        model, env = build_sr_model(
            config_path=config_path, n_dynamic_obstacles=n_dynamic_obstacles,
            obstacle_speed=obstacle_speed, obstacle_speed_range=obstacle_speed_range,
            randomize=spec["randomize"], seed=seed,
            epsilon=epsilon, delta=config["sr"]["delta"],
            use_anti_goal=config["sr"]["use_anti_goal"], log_dir=log_dir)
        eval_env = None      # the SR observation is wider; EvalCallback would mismatch
    else:
        model, env = build_ppo_model(
            config_path=config_path, n_dynamic_obstacles=n_dynamic_obstacles,
            obstacle_speed=obstacle_speed, obstacle_speed_range=obstacle_speed_range,
            randomize=spec["randomize"], seed=seed, log_dir=log_dir)
        eval_env = create_eval_env(
            config_path, n_dynamic_obstacles, obstacle_speed=obstacle_speed,
            obstacle_speed_range=obstacle_speed_range,
            randomize_dynamic_obstacles=spec["randomize"])

    callbacks = [CheckpointCallback(save_freq=max(eval_freq, 1), save_path=model_dir,
                                    name_prefix="ppo_sr" if spec["sr"] else "ppo")]
    if eval_env is not None:
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=os.path.join(model_dir, "best"),
            log_path=log_dir, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
            deterministic=True, verbose=1))

    run_config = {
        "arm": arm, "seed": seed, "sr": spec["sr"], "randomize": spec["randomize"],
        "git_commit": git_commit(), "git_branch": git_branch(),
        "total_timesteps": total_timesteps,
        "n_dynamic_obstacles": n_dynamic_obstacles, "obstacle_speed": obstacle_speed,
        "obstacle_speed_range": obstacle_speed_range, "tag": tag,
        "dt": config["dt"], "start_goal": config["start_goal"],
        "dynamic_obstacles": config["dynamic_obstacles"],
        "reward": config["reward"], "environment": config["environment"],
        "observation": config.get("observation", {}),
        "termination": config.get("termination", {}),
        "ppo": config["training"],
        # What the model was ACTUALLY constructed with, read back off the object rather
        # than copied from config. These diverge the moment a builder forgets to thread a
        # key through, which is precisely the failure this records.
        "ppo_effective": {
            "batch_size": int(model.batch_size),
            "n_steps": int(model.n_steps),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
            "target_kl": (None if model.target_kl is None else float(model.target_kl)),
            "learning_rate_at_start": float(model.lr_schedule(1.0)),
            "learning_rate_at_end": float(model.lr_schedule(0.0)),
            "observation_dim": int(model.observation_space.shape[0]),
            "torch_threads": torch_threads,
        },
        "sr_hyperparameters": ({"epsilon": epsilon, "delta": config["sr"]["delta"],
                                "use_anti_goal": config["sr"]["use_anti_goal"]}
                               if spec["sr"] else None),
        "model_dir": model_dir, "log_dir": log_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as handle:
        json.dump(run_config, handle, indent=2)

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    env.close()
    if eval_env is not None:
        eval_env.close()
    return {"run_config": run_config, "model_path": final_path + ".zip"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=6)
    parser.add_argument("--obstacle-speed", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--results-root", default="results/week4")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--obstacle-speed-range", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="sample the episode obstacle speed from [LOW, HIGH]; "
                             "takes priority over --obstacle-speed")
    parser.add_argument("--tag", default="",
                        help="suffix for model/log paths, so a second matrix at different "
                             "settings cannot overwrite the first")
    args = parser.parse_args()

    result = train_arm(
        args.arm, args.seed, total_timesteps=args.timesteps,
        n_dynamic_obstacles=args.n_obstacles, obstacle_speed=args.obstacle_speed,
        config_path=args.config, epsilon=args.epsilon,
        results_root=args.results_root, eval_freq=args.eval_freq,
        obstacle_speed_range=args.obstacle_speed_range, tag=args.tag)
    print(json.dumps({"model": result["model_path"],
                      "arm": args.arm, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
