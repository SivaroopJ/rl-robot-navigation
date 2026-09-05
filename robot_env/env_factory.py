"""
Environment creation module for RobotNavEnv training.
This module contains factory functions used to create vectorized training environments and evaluation environments for RL experiments.
"""
import os
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from robot_env.robot_nav_env import RobotNavEnv


def make_env(config_path, n_dynamic_obstacles, obstacle_speed=None, obstacle_speed_range = None, rank=0, seed=0, use_reward_shaping=True,
             randomize_dynamic_obstacles=None):
    """
    Factory function used by SubprocVecEnv to create parallel RobotNavEnv instances. Each environment receives a different random seed.

    Parameters
    config_path: str
        Path to config.json.
    n_dynamic_obstacles: int
        Number of dynamic obstacles in the environment.
    obstacle_speed: float or None
        Fixed obstacle speed. Mutually exclusive with obstacle_speed_range.
    obstacle_speed_range: tuple of (float, float) or None
        (low, high) range for per-episode speed sampling. When provided, obstacle_speed is ignored.
    rank: int
        Index of this environment in the parallel pool (used for seeding).
    seed: int
        Base random seed. Each environment gets seed + rank.
    use_reward_shaping: bool
        If True, adds progress reward toward the target.
    randomize_dynamic_obstacles: bool or None
        Week 4. True selects smoothly stochastic obstacle motion, False the original
        constant-velocity bounce, None the value in config.json.

    Returns
    callable
        A no-argument function that returns an initialized RobotNavEnv.
    """
    def environment_factory():
        env = RobotNavEnv(
            config_path=config_path,
            n_dynamic_obstacles=n_dynamic_obstacles,
            obstacle_speed=obstacle_speed,
            obstacle_speed_range=obstacle_speed_range,
            render_mode=None,
            use_reward_shaping=use_reward_shaping,
            randomize_dynamic_obstacles=randomize_dynamic_obstacles
        )
        env.reset(seed=seed + rank)
        return env

    return environment_factory


def create_parallel_envs(config, config_path, n_dynamic_obstacles, obstacle_speed=None, obstacle_speed_range=None, use_reward_shaping=True,
                         randomize_dynamic_obstacles=None, monitor_dir=None):
    """
    Create a vectorized environment with multiple parallel instances.

    Wraps SubprocVecEnv with VecMonitor. info_keywords=("is_success",) is required so that EvalCallback records per-episode success 
    rates in evaluations.npz, which is used by the curriculum threshold logic.

    Parameters
    config: dict
        Full configuration dictionary.
    config_path: str
        Path to config.json, passed to each environment instance.
    n_dynamic_obstacles: int
        Number of dynamic obstacles in each environment.
    obstacle_speed: float or None
        Fixed obstacle speed. Ignored when obstacle_speed_range is provided.
    obstacle_speed_range: tuple of (float, float) or None
        Per-episode speed sampling range.
    use_reward_shaping: bool
        If True, adds progress reward toward the target.
        If False, only goal, collision and step penalty rewards are used.

    Returns
    VecMonitor
        Vectorized and monitored parallel environment.
    """
    base_seed = config.get("seed", 0) 

    number_of_envs = config["training"]["n_envs"]
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    env_factory_list = []
    for i in range(number_of_envs):
        env_factory_list.append(make_env(config_path = config_path, n_dynamic_obstacles = n_dynamic_obstacles, obstacle_speed = obstacle_speed,
                                        obstacle_speed_range = obstacle_speed_range, rank=i, use_reward_shaping = use_reward_shaping, seed=base_seed,
                                        randomize_dynamic_obstacles=randomize_dynamic_obstacles))

    parallel_env = SubprocVecEnv(env_factory_list)
    monitor_root = monitor_dir if monitor_dir is not None else log_dir
    os.makedirs(monitor_root, exist_ok=True)
    monitored_env = VecMonitor(parallel_env, filename=os.path.join(monitor_root, "monitor"), info_keywords=("is_success",),)

    return monitored_env


def create_eval_env(config_path, n_dynamic_obstacles, obstacle_speed=None, obstacle_speed_range=None, use_reward_shaping=True,
                    randomize_dynamic_obstacles=None):
    """
    Create a single environment used for periodic evaluation during training.

    Parameters
    config_path: str
        Path to config.json.
    n_dynamic_obstacles: int
        Number of dynamic obstacles.
    obstacle_speed: float or None
        Fixed speed used for eval. When obstacle_speed_range is provided, this is ignored and the range midpoint is used as the fixed
         eval speed so that eval episodes are comparable across curriculum stages.
    obstacle_speed_range: tuple of (float, float) or None
    use_reward_shaping: bool
        If True, adds progress reward toward the target.
        If False, only goal, collision and step penalty rewards are used.

    Returns
    VecMonitor
        Vectorized and monitored single environment instance for evaluation.
    """
    # Eval always uses a fixed speed (range midpoint) for consistent comparison.
    if obstacle_speed_range is not None:
        eval_speed = float(np.mean(obstacle_speed_range))
        speed_range_arg = None
    else:
        eval_speed = obstacle_speed
        speed_range_arg = None


    env_factory = make_env(
        config_path=config_path,
        n_dynamic_obstacles=n_dynamic_obstacles,
        obstacle_speed=eval_speed,
        obstacle_speed_range=speed_range_arg,
        rank=0,
        use_reward_shaping=use_reward_shaping,
        randomize_dynamic_obstacles=randomize_dynamic_obstacles
    )
    eval_env = DummyVecEnv([env_factory])


    return VecMonitor(eval_env, info_keywords=("is_success",))
