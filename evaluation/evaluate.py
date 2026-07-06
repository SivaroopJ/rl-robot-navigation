"""
Evaluation module for a trained PPO agent in RobotNavEnv. Loads a trained model, evaluates it over a fixed number of episodes, collects 
aggregate and per-episode metrics and saves evaluation results to a JSON file.

Trajectory collection: during each episode the agent's position is recorded at every step. After evaluation, plot_evaluation_trajectories()
converts the raw data and calls visualization.plot_trajectories (plot_trajectory for individual episodes, plot_multiple_trajectories for the
overview plot).
"""

import os
import json
import numpy as np
from stable_baselines3 import PPO
from robot_env.robot_nav_env import RobotNavEnv
from visualization.plot_trajectories import plot_trajectory, plot_multiple_trajectories

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config(config_path):
    """
    Load configuration from a JSON file.

    Parameters
    config_path: str
        Path to the JSON configuration file.

    Returns
    dict
        Full configuration dictionary.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def plot_evaluation_trajectories(episode_trajectories, world_size, target_radius, n_dynamic_obstacles, obstacle_speed, experiment=None, 
                                 label=None, n_traj_single=3, n_traj_overview=50):
    """
    Generate and save trajectory plots from a completed evaluation run.

    Produces two kinds of output:
      - One overview PNG with up to n_traj_overview episodes overlaid on a single map, colour-coded by outcome (green = success, 
      red = collision, orange = timeout).
      - Up to n_traj_single individual PNGs, one representative episode per outcome type.

    All files are written under results/figures/trajectories/<experiment>/ (or results/figures/trajectories/ when experiment is None).

    Parameters
    episode_trajectories: list of dict
        Raw trajectory data collected during evaluate(). Each dict has keys:
          "trajectory" - list of [x, y] lists, one per step (including step 0)
          "goal_pos" - [x, y] list, target position at episode start
          "static_obstacles" - list of (cx, cy, hw, hh) tuples, same as env.static_obstacles
          "outcome" - str, one of "success" | "collision" | "timeout"
    world_size: float
        Side length of the square world (env.WORLD_SIZE).
    target_radius: float
        Visual radius of the goal marker (env.TARGET_RADIUS).
    n_dynamic_obstacles: int
        Number of dynamic obstacles used during evaluation (for file naming).
    obstacle_speed: float
        Obstacle speed used during evaluation (for file naming).
    experiment: str or None
        Experiment subfolder name (e.g. "e1"). Used for directory creation and plot titles.
    label: str or None
        Optional suffix that distinguishes variants (e.g. "shaping"). Used for file naming and plot titles.
    n_traj_single: int
        Maximum number of individual trajectory plots to save (one per distinct outcome). Defaults to 3 (one success, one collision, 
        one timeout where available).
    n_traj_overview: int
        Maximum number of episodes drawn in the overview plot. Passed to plot_multiple_trajectories(max_to_plot=...).
    """
    # Build output directory
    if experiment:
        traj_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures", "trajectories", experiment)
    else:
        traj_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures", "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    # Build shared file-name tag that mirrors the results JSON name
    base_tag = f"obs{n_dynamic_obstacles}_spd{obstacle_speed}"
    if label:
        base_tag += f"_{label}"

    # Build shared title suffixes
    obs_label = f"obs={n_dynamic_obstacles}, spd={obstacle_speed}"
    if experiment:
        exp_label = f"[{experiment}]"
    else:
        exp_label = f""
    if label:
        lbl_label = f"[{label}]" 
    else:
        lbl_label = f""

    # Convert list-of-lists back to numpy arrays expected by the plotting helpers
    traj_data_np = []
    for ep in episode_trajectories:
        trajectory_np = []
        for p in ep["trajectory"]:
            trajectory_np.append(np.array(p))
        traj_data_np.append({
            "trajectory": trajectory_np,
            "goal_pos": np.array(ep["goal_pos"]),
            "static_obstacles": ep["static_obstacles"],
            "outcome": ep["outcome"],
        })

    # Overview: all episodes overlaid on one plot
    overview_title = f"Agent trajectories — {obs_label}{exp_label}{lbl_label}"
    overview_path = os.path.join(traj_dir, f"trajectories_overview_{base_tag}.png")

    plot_multiple_trajectories(
        episodes=traj_data_np,
        world_size=world_size,
        target_radius=target_radius,
        title=overview_title,
        max_to_plot=n_traj_overview,
        save_path=overview_path,
    )
    print(f"Trajectory overview saved to: {overview_path}")

    # Individual plots: one representative episode per distinct outcome
    seen_outcomes = {}
    for ep in traj_data_np:
        oc = ep["outcome"]
        if oc not in seen_outcomes:
            seen_outcomes[oc] = ep
        if len(seen_outcomes) == n_traj_single:
            break  # one example per outcome found; stop early

    for oc, ep in seen_outcomes.items():
        single_title = f"Trajectory ({oc}) — {obs_label}{exp_label}{lbl_label}"
        single_path = os.path.join(traj_dir, f"trajectory_{oc}_{base_tag}.png")
        plot_trajectory(
            trajectory=ep["trajectory"],
            goal_pos=ep["goal_pos"],
            static_obstacles=ep["static_obstacles"],
            world_size=world_size,
            outcome=oc,
            target_radius=target_radius,
            title=single_title,
            save_path=single_path,
        )
        print(f"Single trajectory [{oc}] saved to: {single_path}")

def evaluate(model_path, n_dynamic_obstacles=None, obstacle_speed=None, n_eval_episodes=None, experiment = None, label = None, 
             plot_trajectories=True, n_traj_single=3, n_traj_overview=50):
    """
    Evaluate a trained PPO agent over a fixed number of episodes.

    Runs the agent deterministically (no exploration) and collects per-episode metrics. Records the agent's position at every step so that 
    trajectories can be visualized. Prints a summary at the end.

    Parameters
    model_path: str
        Path to the saved model (.zip file).
    n_dynamic_obstacles: int or None
        Number of dynamic obstacles. Overrides config if provided.
    obstacle_speed: float or None
        Speed of dynamic obstacles. Overrides config if provided.
    n_eval_episodes: int or None
        Number of evaluation episodes. Overrides config if provided.
    experiment: str or None
        Experiment name used for saving results (e.g. "e1").
    label: str or None
        Optional label appended to the results filename to distinguish between variants evaluated on the same configuration.
    plot_trajectories: bool
        If True (default), calls plot_evaluation_trajectories() after evaluation to generate and save trajectory plots. Set
        to False to skip all trajectory plotting.
    n_traj_single: int
        Forwarded to plot_evaluation_trajectories(). Maximum individual plots saved, one per distinct outcome. Defaults to 3.
        Default 3 (one success, one collision, one timeout where available).
    n_traj_overview: int
        Maximum number of episodes shown in the multi-trajectory overview plot. Passed directly to plot_multiple_trajectories
        (max_to_plot=...).

    Returns
    dict
        Dictionary with aggregated evaluation results:
        - 'success_rate'(float): fraction of successful episodes
        - 'avg_episode_length'(float): average number of steps per episode
        - 'avg_collision_count'(float): average collisions per episode
        - 'n_success'(int): number of successful episodes
        - 'n_collision'(int): number of episodes ending in collision
        - 'n_truncated'(int): number of truncated episodes
        - 'n_episodes'(int): total episodes evaluated
        - 'episodes'(list): detailed per-episode evaluation results
    """
    config = load_config(CONFIG_PATH)

    # Apply overrides or use config defaults
    if n_dynamic_obstacles is None:
        n_dynamic_obstacles = config["environment"]["n_dynamic_obstacles"]
    if obstacle_speed is None:
        obstacle_speed = config["environment"]["obstacle_speed"]
    if n_eval_episodes is None:
        n_eval_episodes = config["evaluation"]["n_eval_episodes"]

    deterministic = config["evaluation"]["deterministic"]

    print("PPO Evaluation — RobotNavEnv")
    print(f"Model: {model_path}")
    print(f"Dynamic obstacles: {n_dynamic_obstacles}")
    print(f"Obstacle speed: {obstacle_speed}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Deterministic: {deterministic}")
    print()

    # Load trained model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}")

    eval_env = RobotNavEnv(
        config_path=CONFIG_PATH,
        n_dynamic_obstacles=n_dynamic_obstacles,
        obstacle_speed=obstacle_speed,
        render_mode=None
    )

    model = PPO.load(model_path, env=eval_env)
    print("Model loaded successfully.")
    print()

    # Metrics accumulators 
    number_of_successes = 0
    number_of_collisions = 0
    number_of_truncations = 0
    total_steps = 0
    total_collisions = 0

    # Per-episode metrics for JSON export and analysis
    episode_results = []

    # Per-episode trajectory data for visualization
    # Each entry: {"trajectory", "goal_pos", "static_obstacles", "outcome"}
    episode_trajectories = []

    for episode_index in range(n_eval_episodes):
        observation, info = eval_env.reset(seed=episode_index)
        episode_step_count = 0
        episode_collision_count = 0
        terminated = False
        truncated = False

        # trajectory recording: capture the starting position after reset
        episode_trajectory = [eval_env.agent_position.copy()]
        episode_goal_pos = eval_env.target_position.copy()
        episode_static_obstacles = list(eval_env.static_obstacles)

        while not terminated and not truncated:
            # Run agent deterministically — no random exploration
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            episode_step_count += 1

            # trajectory recording: capture position after every step
            episode_trajectory.append(eval_env.agent_position.copy())

            # Count collisions: episode ended due to collision
            if terminated and not info["success"]:
                episode_collision_count += 1

        # Classify episode outcome
        if info["success"]:
            number_of_successes += 1
            outcome = "SUCCESS"
            traj_outcome = "success"
        elif truncated:
            number_of_truncations += 1
            outcome = "TRUNCATED"
            traj_outcome = "timeout"

        else:
            number_of_collisions  += 1
            outcome = "COLLISION"
            traj_outcome = "collision"

        total_steps += episode_step_count
        total_collisions += episode_collision_count

        episode_results.append({
            "episode": episode_index + 1,
            "outcome": outcome,
            "steps": episode_step_count,
            "collisions": episode_collision_count,
            "final_dist": info["distance_to_target"]
        })

        # Store trajectory data (convert numpy arrays to lists for JSON serializability)
        episode_trajectory_lists = []
        for pos in episode_trajectory:
            episode_trajectory_lists.append(pos.tolist())
        episode_trajectories.append({
            "trajectory": episode_trajectory_lists,
            "goal_pos": episode_goal_pos.tolist(),
            "static_obstacles": episode_static_obstacles,
            "outcome": traj_outcome,
        })

        # Print progress every 50 episodes
        if (episode_index + 1) % 50 == 0:
            current_success_rate = number_of_successes / (episode_index + 1) * 100
            print(f"Episode {episode_index + 1:>4} / {n_eval_episodes}")
            print(f"Success rate so far: {current_success_rate:.1f}%")

    eval_env.close()

    # Compute summary metrics 
    success_rate = number_of_successes / n_eval_episodes
    avg_episode_length = total_steps / n_eval_episodes
    avg_collision_count = total_collisions / n_eval_episodes

    results = {
        "success_rate": success_rate,
        "avg_episode_length": avg_episode_length,
        "avg_collision_count": avg_collision_count,
        "n_success": number_of_successes,
        "n_collision": number_of_collisions,
        "n_truncated": number_of_truncations,
        "n_episodes": n_eval_episodes,
        "episodes": episode_results
    }

    # Print summary 
    print()
    print("Evaluation Results")
    print(f"Episodes evaluated: {n_eval_episodes}")
    print(f"Successful: {number_of_successes} ({success_rate * 100:.1f}%)")
    print(f"Collisions: {number_of_collisions} ({number_of_collisions / n_eval_episodes * 100:.1f}%)")
    print(f"Truncated: {number_of_truncations} ({number_of_truncations / n_eval_episodes * 100:.1f}%)")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Average collisions/episode: {avg_collision_count:.3f}")

    # Save evaluation results to JSON file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    if experiment is not None:
        results_dir = os.path.join(results_dir, experiment)
    os.makedirs(results_dir, exist_ok=True)
    
    if label:
        results_filename = f"eval_obs{n_dynamic_obstacles}_spd{obstacle_speed}_{label}.json"
    else:
        results_filename = f"eval_obs{n_dynamic_obstacles}_spd{obstacle_speed}.json"    
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, "w", encoding="utf-8") as results_file:        
        json.dump(results, results_file, indent=4, ensure_ascii=False)    
    print(f"Results saved to: {results_path}")

    # Trajectory visualization
    if plot_trajectories and episode_trajectories:
        # Derive world-size and target-radius from config (env already closed)
        world_size = config["environment"]["world_size"]
        target_radius = config["environment"]["target_radius"]

        plot_evaluation_trajectories(episode_trajectories=episode_trajectories, world_size= world_size, target_radius=target_radius, 
                                     n_dynamic_obstacles=n_dynamic_obstacles, obstacle_speed=obstacle_speed, experiment=experiment, 
                                     label=label, n_traj_single=n_traj_single, n_traj_overview=n_traj_overview)
            
    return results