"""Experiment 4 figures: the maze, the LiDAR probe, trajectories, heatmaps, curves.

TWO THINGS THAT ARE EASY TO CONFLATE AND MUST NOT BE
----------------------------------------------------
The brief asks for both a COLLISION HEATMAP and a TERMINAL-STATE DISTRIBUTION, and they
answer different questions:

    collision heatmap            "where does the robot hit walls?"
    terminal-state distribution  "where do trajectories end up?"

They are produced by separate functions here and are never merged. The terminal-state plot
is the paper's own diagnostic (Trott et al. Figure 3, right) -- it is how you SEE a policy
stuck in a local optimum, because its terminal states pile up somewhere that is not the
goal.

SHARED NORMALISATION IS NOT OPTIONAL
------------------------------------
Every four-panel comparison uses one set of bins and one colour scale across all four arms.
Normalising each panel independently makes the densities look similar no matter how
different they are, which would make the central comparison of the experiment
(PPO+SR vs PPO+SR+LiDAR) unreadable while appearing to show it.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_env.lidar_core import cast_rays, ray_angles
from robot_env.paper_maze import (GOAL_XY, N_CELLS, START_XY, wall_rects,
                                  wall_segments)

ARM_TITLES = {
    "ppo": "PPO",
    "ppo_lidar": "PPO + LiDAR",
    "ppo_sr": "PPO + SR",
    "ppo_sr_lidar": "PPO + SR + LiDAR",
}
#: Panel order for every 2x2 comparison, so the figures are directly comparable to the
#: brief's layout: PPO / PPO+LiDAR on top, the SR pair beneath.
PANEL_ORDER = ("ppo", "ppo_lidar", "ppo_sr", "ppo_sr_lidar")


def draw_maze(ax, *, linewidth=2.0, colour="black"):
    """The maze walls, start cell and goal cell. Shared by every figure."""
    for (x0, y0), (x1, y1) in wall_segments():
        ax.plot([x0, x1], [y0, y1], color=colour, lw=linewidth, solid_capstyle="round",
                zorder=3)
    ax.add_patch(plt.Rectangle((START_XY[0] - 0.5, START_XY[1] - 0.5), 1, 1,
                               color="#8c8cf0", alpha=0.85, zorder=1))
    ax.add_patch(plt.Rectangle((GOAL_XY[0] - 0.5, GOAL_XY[1] - 0.5), 1, 1,
                               color="#f89a9a", alpha=0.85, zorder=1))
    ax.set_xlim(-0.25, N_CELLS + 0.25)
    ax.set_ylim(-0.25, N_CELLS + 0.25)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


#: The five situations the brief asks the LiDAR to be verified in. Chosen by inspecting
#: the maze arrays, and each is asserted to be the situation it claims in
#: `plot_lidar_probe` -- a label that quietly stopped being true would defeat the check.
PROBE_POSITIONS = [
    ("open corridor",      (1.5, 0.5)),
    ("beside a vertical wall",   (2.9, 0.5)),
    ("beside a horizontal wall", (0.5, 0.1)),
    ("in a corner",        (0.15, 0.15)),
    ("narrow passage",     (0.5, 4.5)),
]


def plot_lidar_probe(out_path, n_rays=24, lidar_range=5.0):
    """Draw the 24 rays and their endpoints at five representative positions.

    This is the experiment's central sanity check: Experiment 4 exists to test whether
    local wall geometry helps, so the rays must demonstrably measure wall geometry. Each
    panel prints the shortest ray, which is the number the policy would actually react to.
    """
    rects = wall_rects()
    angles = ray_angles(n_rays)
    fig, axes = plt.subplots(1, len(PROBE_POSITIONS),
                             figsize=(4.0 * len(PROBE_POSITIONS), 4.4), dpi=130)
    for ax, (label, position) in zip(axes, PROBE_POSITIONS):
        draw_maze(ax, linewidth=1.6)
        rays = cast_rays(np.asarray(position), n_rays=n_rays, lidar_range=lidar_range,
                         world_size=float(N_CELLS), agent_radius=0.0,
                         rects=rects, circles=(), circle_radius=0.0)
        for angle, distance in zip(angles, rays):
            end = (position[0] + distance * np.cos(angle),
                   position[1] + distance * np.sin(angle))
            ax.plot([position[0], end[0]], [position[1], end[1]],
                    color="#d1701f", lw=0.9, alpha=0.85, zorder=4)
            ax.plot(*end, marker="o", ms=2.4, color="#d1701f", zorder=5)
        ax.plot(*position, marker="o", ms=7, color="#12305e", zorder=6)
        ax.set_title(f"{label}\nnearest wall {rays.min():.2f}  farthest {rays.max():.2f}",
                     fontsize=10)
    fig.suptitle("Experiment 4 — 24-ray LiDAR against the maze walls", fontsize=13, y=1.06)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_maze(out_path):
    """The bare maze, for the results document."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    draw_maze(ax)
    ax.set_title("Trott et al. 2D Point Maze — recovered from Figure 3\n"
                 "10x10 cells, perfect maze (99 passages, 1 component, 0 cycles)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _density(points, bins):
    grid, _, _ = np.histogram2d(
        [p[0] for p in points] or [0], [p[1] for p in points] or [0],
        bins=[bins, bins], range=[[0, N_CELLS], [0, N_CELLS]])
    return grid.T if points else np.zeros((bins, bins))


def plot_density_comparison(points_by_arm, out_path, *, title, bins=40, cmap="magma"):
    """One 2x2 panel per arm, ONE colour scale across all four.

    `points_by_arm` maps arm name -> list of (x, y). Used for both the collision heatmap
    and the terminal-state distribution; the caller decides which points to pass, and the
    title says which question the figure answers.
    """
    grids = {arm: _density(points_by_arm.get(arm, []), bins) for arm in PANEL_ORDER}
    vmax = max((g.max() for g in grids.values()), default=1.0) or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=130)
    for ax, arm in zip(axes.flat, PANEL_ORDER):
        ax.imshow(grids[arm], origin="lower", extent=[0, N_CELLS, 0, N_CELLS],
                  cmap=cmap, vmin=0, vmax=vmax, zorder=0, interpolation="nearest")
        draw_maze(ax, linewidth=1.3, colour="#6f6f6f")
        n = len(points_by_arm.get(arm, []))
        ax.set_title(f"{ARM_TITLES[arm]}   (n = {n})", fontsize=12)
    fig.suptitle(f"{title}\nshared bins and one colour scale across all four panels",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectories(trajectories, out_path, *, title):
    """Representative rollouts. `trajectories` maps arm -> list of dicts.

    Each dict carries `path` (T+1, 2), `outcome`, and `collisions` (list of (x, y)).
    Failures are included deliberately: reporting only successful rollouts alongside an
    aggregate success rate misrepresents the aggregate.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=130)
    for ax, arm in zip(axes.flat, PANEL_ORDER):
        draw_maze(ax, linewidth=1.3)
        for record in trajectories.get(arm, []):
            path = np.asarray(record["path"])
            colour = {"SUCCESS": "#1f7a4d", "TIMEOUT": "#b8860b"}.get(
                record["outcome"], "#b23a2c")
            ax.plot(path[:, 0], path[:, 1], color=colour, lw=1.8, alpha=0.9, zorder=4)
            if record.get("collisions"):
                bumps = np.asarray(record["collisions"])
                ax.plot(bumps[:, 0], bumps[:, 1], "x", color="#b23a2c", ms=5, mew=1.4,
                        zorder=5)
        ax.set_title(ARM_TITLES[arm], fontsize=12)
    fig.suptitle(f"{title}\ngreen = reached, amber = ran out of steps, "
                 "red = failed;  x = wall contact", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_learning_curves(curves, out_path, metrics=("success_rate", "collision_rate",
                                                    "mean_return", "episode_length")):
    """Mean +- std across seeds, four metrics, all arms overlaid on shared axes.

    Deliberately sets no fixed ylim: `visualize.py` hardcodes `set_ylim(-11, -2)` on its
    curve functions, which would clip Experiment 4's returns (bounded by +1 and -12.7)
    straight off the plot.
    """
    colours = {"ppo": "#4c72b0", "ppo_lidar": "#2f7a4f",
               "ppo_sr": "#c44e52", "ppo_sr_lidar": "#8d2f63"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2), dpi=130)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for arm in PANEL_ORDER:
            series = curves.get(arm, {}).get(metric)
            if not series:
                continue
            steps = np.asarray(series["steps"])
            mean, std = np.asarray(series["mean"]), np.asarray(series["std"])
            ax.plot(steps, mean, color=colours[arm], lw=1.8, label=ARM_TITLES[arm])
            ax.fill_between(steps, mean - std, mean + std, color=colours[arm], alpha=0.18)
        ax.set_xlabel("environment steps")
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
