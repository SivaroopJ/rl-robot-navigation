"""Week 4 figures: coverage diagnostics, obstacle trajectories, curves, and heatmaps.

TWO THINGS THE EXISTING PLOTS DO THAT THESE MUST NOT
-----------------------------------------------------
`visualization/visualize.py` hardcodes `ax.set_ylim(-11, -2)` on all four of its curve
functions. Sibling Rivalry's anti-goal payout pushes returns below -11, so inheriting that
limit would silently clip the SR arm off the bottom of its own plot. Nothing here sets a
fixed y-limit.

The existing metric plots also show a single run. Week 4 has three seeds per arm, and a
PPO-vs-SR difference of a few points is not interpretable without spread, so every curve
here is a mean with a +-1 standard deviation band across seeds.

TERMINAL-STATE DENSITY IS NOT A COLLISION HEATMAP
-------------------------------------------------
They are generated separately and must stay separate. The terminal-state density covers
EVERY episode -- successes, collisions and timeouts alike -- and is the plot that speaks to
Sibling Rivalry's motivation: if SR is escaping a local optimum, its terminal states should
be less concentrated away from the goal than PPO's. The collision heatmap covers only
failures and says where the environment is dangerous. Reading one as the other would turn
"SR fails in different places" into "SR explores better", which the data would not support.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = "results/week4"
ARM_COLOURS = {"ppo": "#3b6ea5", "ppo_sr": "#c8642a",
               "ppo_randomized": "#3b6ea5", "ppo_sr_randomized": "#c8642a"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _draw_arena(ax, env):
    """Walls and static rectangles, so every spatial plot shares one frame of reference."""
    for cx, cy, hw, hh in env.static_obstacles:
        ax.add_patch(plt.Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh,
                                   facecolor="#9a9a9a", edgecolor="none", zorder=2))
    ax.set_xlim(0, env.WORLD_SIZE)
    ax.set_ylim(0, env.WORLD_SIZE)
    ax.set_aspect("equal")


# ------------------------------------------------------- start/goal coverage diagnostics
def plot_start_goal_coverage(config_path="config.json",
                             out_dir=os.path.join(RESULTS_ROOT, "start_goal_coverage"),
                             n_samples=6000):
    """Evidence that the stratified sampler covers the map and spans the distance range.

    Plotted against the ORIGINAL uniform scheme, because the claim being supported is a
    comparative one: the point is not that the new distribution is reasonable in isolation
    but that it fixes a specific defect in the old one.
    """
    from robot_env.robot_nav_env import RobotNavEnv, load_config
    from robot_env.start_goal import StartGoalSampler

    config = load_config(config_path)
    env_cfg, sg = config["environment"], config["start_goal"]
    env = RobotNavEnv(config_path=config_path, n_dynamic_obstacles=0)
    sampler = StartGoalSampler(
        env_cfg["world_size"], max(env_cfg["agent_radius"] * 2, 0.5),
        env_cfg["static_obstacles"], region_grid=sg["region_grid"],
        region_free_threshold=sg["region_free_threshold"],
        distance_bins=sg["distance_bins"], min_separation=sg["min_separation"])

    rng = np.random.default_rng(0)
    starts, goals, distances = [], [], []
    for _ in range(n_samples):
        s, g = sampler.sample(rng)
        starts.append(s); goals.append(g)
        distances.append(float(np.linalg.norm(g - s)))
    starts, goals, distances = np.array(starts), np.array(goals), np.array(distances)

    # The legacy distribution, for comparison.
    legacy = []
    for _ in range(n_samples):
        a = env._random_free_position()
        b = env._random_free_position(exclude=[a], min_dist=2.0)
        legacy.append(float(np.linalg.norm(b - a)))
    legacy = np.array(legacy)

    written = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, points, title in ((axes[0], starts, "Start positions"),
                              (axes[1], goals, "Goal positions")):
        ax.hexbin(points[:, 0], points[:, 1], gridsize=22, cmap="viridis",
                  extent=(0, env.WORLD_SIZE, 0, env.WORLD_SIZE), zorder=1)
        _draw_arena(ax, env)
        for edge in np.linspace(sampler.low, sampler.high, sampler.region_grid + 1):
            ax.axvline(edge, color="w", lw=0.5, alpha=0.4, zorder=3)
            ax.axhline(edge, color="w", lw=0.5, alpha=0.4, zorder=3)
        ax.set_title(f"{title} (n={n_samples})")
    fig.suptitle("Stratified sampling covers every usable region of the map")
    written.append(_save(fig, os.path.join(out_dir, "region_coverage.png")))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(legacy, bins=60, alpha=0.55, label=f"original uniform (mean {legacy.mean():.2f})",
            color="#9a9a9a", density=True)
    ax.hist(distances, bins=60, alpha=0.7, label=f"Week 4 stratified (mean {distances.mean():.2f})",
            color="#3b6ea5", density=True)
    for low, high in sampler.distance_bins:
        ax.axvline(low, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(sampler.distance_bins[-1][1], color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("start-goal straight-line distance")
    ax.set_ylabel("density")
    ax.set_title("Equal mass per distance band, and no more very short episodes")
    ax.legend()
    written.append(_save(fig, os.path.join(out_dir, "distance_distribution.png")))

    summary = {
        "n_samples": n_samples,
        "stratified": {"mean": float(distances.mean()), "median": float(np.median(distances)),
                       "min": float(distances.min()), "max": float(distances.max())},
        "legacy_uniform": {"mean": float(legacy.mean()), "median": float(np.median(legacy)),
                           "min": float(legacy.min()),
                           "fraction_under_4": float((legacy < 4).mean())},
        "band_shares": {f"{lo}-{hi}": float(((distances >= lo) & (distances <= hi)).mean())
                        for lo, hi in sampler.distance_bins},
        "fallback_rate": sampler.fallback_rate,
        "usable_regions": len(sampler.usable_regions),
        "total_regions": sampler.region_grid ** 2,
    }
    path = os.path.join(out_dir, "coverage_summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  wrote {path}")
    return written


# ------------------------------------------------------------- obstacle trajectory demo
def plot_obstacle_trajectories(config_path="config.json",
                               out_dir=os.path.join(RESULTS_ROOT, "trajectory_examples"),
                               n_steps=400, n_random=3):
    """The deterministic path beside three randomized ones, plus the smoothness traces.

    This is the figure that has to carry the claim "stochastic but smooth". A path plot
    alone cannot: a smooth-looking curve could still have been produced by a bounded but
    uncorrelated process. The omega and heading traces underneath are what distinguish
    them, which is why they are on the same figure rather than in an appendix.
    """
    from robot_env.dynamic_obstacles import DeterministicMotion, SmoothStochasticMotion
    from robot_env.robot_nav_env import RobotNavEnv, load_config

    config = load_config(config_path)
    do_cfg, dt = config["dynamic_obstacles"], config["dt"]
    env = RobotNavEnv(config_path=config_path, n_dynamic_obstacles=0)
    world, radius = env.WORLD_SIZE, env.OBSTACLE_RADIUS

    def trace(model, seed):
        rng = np.random.default_rng(seed)
        pos = np.array([[world * 0.3, world * 0.3]], dtype=np.float32)
        vel = model.reset(rng, pos, 1.0)
        path, omega, theta = [pos[0].copy()], [], []
        for _ in range(n_steps):
            pos, vel = model.step(rng, pos, vel, dt)
            path.append(pos[0].copy())
            omega.append(float(getattr(model, "omega", [0.0])[0]))
            theta.append(float(getattr(model, "theta", [0.0])[0]))
        return np.array(path), np.array(omega), np.array(theta)

    def stochastic():
        return SmoothStochasticMotion(
            world, radius, angular_noise_sigma=do_cfg["angular_noise_sigma"],
            angular_velocity_decay=do_cfg["angular_velocity_decay"],
            max_turn_rate=do_cfg["max_turn_rate"],
            boundary_margin=do_cfg["boundary_margin"],
            boundary_steer_gain=do_cfg["boundary_steer_gain"])

    runs = [("original (deterministic)", *trace(DeterministicMotion(world, radius), 0))]
    for k in range(n_random):
        runs.append((f"randomized, seed {k + 1}", *trace(stochastic(), k + 1)))

    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 7.5))
    for col, (label, path, omega, theta) in enumerate(runs):
        ax = axes[0, col]
        ax.plot(path[:, 0], path[:, 1], lw=1.3,
                color="#9a9a9a" if col == 0 else "#c8642a", zorder=4)
        ax.plot(*path[0], "o", color="k", ms=5, zorder=5)
        _draw_arena(ax, env)
        ax.set_title(label, fontsize=10)

        ax = axes[1, col]
        if col == 0:
            ax.text(0.5, 0.5, "constant heading,\ndiscontinuous flip at each wall",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    color="#555555")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.plot(omega, lw=0.9, color="#3b6ea5", label=r"$\omega$")
            ax.axhline(do_cfg["max_turn_rate"], color="k", ls="--", lw=0.7,
                       label=r"$\pm\omega_{max}$")
            ax.axhline(-do_cfg["max_turn_rate"], color="k", ls="--", lw=0.7)
            ax.set_xlabel("step"); ax.set_ylabel(r"$\omega$ (rad/s)")
            ax.legend(fontsize=8, loc="upper right")
        max_turn = np.abs(np.diff(theta)).max() if len(theta) > 1 else 0.0
        if col > 0:
            ax.set_title(rf"max $|\Delta\theta|$ = {max_turn:.4f} rad/step "
                         rf"(bound {do_cfg['max_turn_rate'] * dt:.4f})", fontsize=8)
    fig.suptitle("Randomized obstacle motion is stochastic but smooth: "
                 "the turn rate stays bounded and is correlated in time")
    fig.tight_layout()
    return [_save(fig, os.path.join(out_dir, "obstacle_trajectories.png"))]


# ---------------------------------------------------------------------- learning curves
def _load_monitor_curves(log_glob):
    """Episode return / length / success against timestep, from SB3 monitor CSVs."""
    import pandas as pd
    curves = []
    for path in sorted(glob.glob(log_glob)):
        try:
            frame = pd.read_csv(path, skiprows=1)
        except Exception:
            continue
        if frame.empty:
            continue
        frame["timestep"] = frame["l"].cumsum()
        curves.append(frame)
    return curves


def plot_learning_curves(arms, seeds, out_dir=os.path.join(RESULTS_ROOT, "plots"),
                         bins=40):
    """Mean +- std across seeds, identical axes, PPO against PPO+SR.

    Returns the figures written; an arm with no logs is skipped rather than plotted as a
    flat line, so a missing run is visible as a missing curve.
    """
    metrics = [("r", "episode return"), ("l", "episode length"),
               ("is_success", "success rate")]
    written = []
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2))
    for ax, (column, title) in zip(axes, metrics):
        for arm in arms:
            series = []
            for seed in seeds:
                curves = _load_monitor_curves(
                    os.path.join("logs", "week4", f"{arm}_seed{seed}", "*monitor*.csv"))
                for frame in curves:
                    if column not in frame:
                        continue
                    edges = np.linspace(0, frame["timestep"].max(), bins + 1)
                    idx = np.digitize(frame["timestep"], edges) - 1
                    binned = [frame[column][idx == b].mean() for b in range(bins)]
                    series.append((edges[:-1], np.array(binned, dtype=float)))
            if not series:
                continue
            length = min(len(s[1]) for s in series)
            stack = np.array([s[1][:length] for s in series], dtype=float)
            x = series[0][0][:length]
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            colour = ARM_COLOURS.get(arm.replace("_fixed", "").replace("_random", ""), None)
            ax.plot(x, mean, label=f"{arm} (n={len(series)})", color=colour)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=colour)
        ax.set_xlabel("environment steps")
        ax.set_title(title)
        ax.legend(fontsize=8)
    # Deliberately no set_ylim: SR's anti-goal payout drives returns below the range the
    # existing plots hardcode, and clipping it would hide the arm's actual behaviour.
    fig.tight_layout()
    written.append(_save(fig, os.path.join(out_dir, "learning_curves.png")))
    return written


# ------------------------------------------------- terminal-state and collision heatmaps
def _episodes_from(paths):
    episodes = []
    for path in sorted(paths):
        episodes.extend(json.load(open(path))["episodes"])
    return episodes


def plot_terminal_and_collision_maps(pairs, out_dir=os.path.join(RESULTS_ROOT, "plots"),
                                     config_path="config.json"):
    """Terminal-state density and collision locations, on a shared colour normalization.

    Shared normalization is what makes the two panels comparable; normalizing each panel to
    its own maximum would make a diffuse distribution and a concentrated one look alike.

    Parameters
    pairs: list of (label, glob) for the arms to compare.
    """
    from robot_env.robot_nav_env import RobotNavEnv
    env = RobotNavEnv(config_path=config_path, n_dynamic_obstacles=0)
    written = []

    for kind, key, directory in (("terminal-state density", "terminal_position", "terminal_heatmaps"),
                                 ("collision locations", "collision_position", "collision_heatmaps")):
        panels = []
        for label, pattern in pairs:
            episodes = _episodes_from(glob.glob(pattern))
            points = np.array([e[key] for e in episodes if e.get(key) is not None])
            if len(points):
                panels.append((label, points))
        if not panels:
            continue

        grids = []
        edges = np.linspace(0, env.WORLD_SIZE, 41)
        for _, points in panels:
            grid, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[edges, edges])
            grids.append(grid)
        vmax = max(g.max() for g in grids) or 1.0

        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8),
                                 squeeze=False)
        for ax, (label, points), grid in zip(axes[0], panels, grids):
            mesh = ax.pcolormesh(edges, edges, grid.T, cmap="magma", vmin=0, vmax=vmax,
                                 zorder=1)
            _draw_arena(ax, env)
            ax.set_title(f"{label}  (n={len(points)})", fontsize=10)
            fig.colorbar(mesh, ax=ax, fraction=0.046)
        fig.suptitle(f"{kind} — shared colour scale")
        fig.tight_layout()
        written.append(_save(fig, os.path.join(RESULTS_ROOT, directory,
                                               f"{directory}.png")))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", nargs="+",
                        default=["coverage", "obstacles", "curves", "heatmaps"],
                        choices=["coverage", "obstacles", "curves", "heatmaps"])
    parser.add_argument("--arms", nargs="+", default=["ppo_random", "ppo_sr_random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    if "coverage" in args.what:
        plot_start_goal_coverage()
    if "obstacles" in args.what:
        plot_obstacle_trajectories()
    if "curves" in args.what:
        plot_learning_curves(args.arms, args.seeds)
    if "heatmaps" in args.what:
        plot_terminal_and_collision_maps([
            ("PPO (randomized)", os.path.join(RESULTS_ROOT, "experiment2",
                                              "ppo_randomized", "eval_seed*.json")),
            ("PPO + SR (randomized)", os.path.join(RESULTS_ROOT, "experiment3",
                                                   "ppo_sr_randomized", "eval_seed*.json")),
        ])


if __name__ == "__main__":
    main()
