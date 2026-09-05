"""Experiment 4: the continuous 2D Point Maze of Trott et al. (NeurIPS 2019).

WHAT THIS IS NOT
----------------
It is NOT the dynamic-obstacle environment. `RobotNavEnv` models a 10x10 arena with six
moving circular obstacles, an agent with velocity state, and a six-term shaped reward.
Experiment 4 is a different question on a different world: a STATIC maze whose only
obstacles are walls, where the thing being measured is whether 24-ray LiDAR reduces wall
collisions. There is no obstacle state in this env at all -- no positions, no velocities,
no IDs -- so the observation cannot accidentally carry dynamic-obstacle features.

The two environments share exactly one thing: `robot_env.lidar_core.cast_rays`, so the
LiDAR under test is the same code, unchanged, that Weeks 4-5 validated.

THE PAPER'S SPECIFICATION (NeurIPS supplement, "2D point maze navigation")
--------------------------------------------------------------------------
  * 10x10 environment of pseudo-randomly connected 1x1 squares, every pair of squares
    joined by exactly one path (see `paper_maze.py`, which re-checks that at import).
  * Continuous.
  * "The agent sees as input its 2D coordinates and well as the 2D goal coordinates" --
    so the observation is [x, y, gx, gy]. There is NO velocity: the paper's agent has no
    velocity state, because the action IS the displacement.
  * "The agent takes an action in a 2D space that controls the direction and magnitude of
    the step it takes."
  * Horizon 50 steps (Figure 3 caption).

WHY MAX_STEP IS 1.0
-------------------
Measured on the actual maze: the unique start->goal route is 22 moves against a 12.73-unit
straight line. Over a 50-step horizon the agent must therefore average 0.44 world units per
step just to arrive. MAX_STEP = 1.0 (one cell) leaves 2.3x slack, which is enough for
detours and imperfect steering without making the task trivially short.

WHY COLLISION IS SWEPT AND NOT TESTED AT THE ENDPOINT
-----------------------------------------------------
A step of up to 1.0 units crosses an entire cell, and the walls are infinitely thin. An
endpoint-only test -- "is the arrival position inside a wall?" -- would answer "no" for
almost every wall crossing, so the agent would pass straight through the maze and the
experiment would measure nothing. Week 4 measured exactly this failure mode in the other
environment (8.5% of episodes contained an undetected obstacle pass-through) at far smaller
step sizes. Here it would be the common case, not the tail.

So every step intersects the segment s_t -> s_{t+1} against every wall segment and stops
the agent just short of the nearest crossing.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robot_env.lidar_core import cast_rays
from robot_env.paper_maze import (GOAL_CELL, GOAL_XY, N_CELLS, START_CELL, START_XY,
                                  cell_bounds, wall_rects, wall_segments)


def load_exp4_config(config_path="config_exp4.json"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment 4 config not found: {config_path}")
    with open(path) as handle:
        return json.load(handle)


class PointMazeEnv(gym.Env):
    """The paper's maze, with LiDAR as the one switchable input.

    `use_lidar` is the ONLY thing that differs between the LiDAR and non-LiDAR arms. The
    maze, start, goal, reward, horizon, physics and action space are identical, which is
    what makes the ablation an ablation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config_path="config_exp4.json", use_lidar=False, render_mode=None):
        super().__init__()
        config = load_exp4_config(config_path)
        env_config = config["environment"]
        reward_config = config["reward"]

        self.use_lidar = bool(use_lidar)
        self.render_mode = render_mode

        self.WORLD_SIZE = float(N_CELLS)
        self.MAX_STEP = float(env_config["max_step"])
        self.MAX_STEPS = int(env_config["max_steps"])
        self.DELTA = float(env_config["goal_radius"])
        self.N_LIDAR_RAYS = int(env_config["n_lidar_rays"])
        self.LIDAR_RANGE = float(env_config["lidar_range"])

        # "sampled" is the paper's own setup: "start location is sampled within the blue
        # square ... the goal is randomly sampled from within the red square region".
        # "fixed" pins both to cell centres, which the Experiment 4 brief asked for -- but
        # measured, that makes a deterministic policy produce ONE trajectory (return std
        # 0.000 over 100 episodes), leaving the critic almost nothing to fit. Both modes
        # are kept and tested; the default follows the paper.
        self.start_goal_mode = str(env_config.get("start_goal_mode", "sampled"))
        if self.start_goal_mode not in ("sampled", "fixed"):
            raise ValueError("start_goal_mode must be 'sampled' or 'fixed', "
                             f"got {self.start_goal_mode!r}")
        self.CELL_MARGIN = float(env_config.get("cell_margin", 0.15))

        self.REWARD_GOAL = float(reward_config["goal"])
        self.REWARD_COLLISION = float(reward_config["collision"])

        # Geometry, built once. `_wall_rects` feeds the shared ray caster; `_seg_*` feed
        # the swept collision test.
        self._wall_rects = wall_rects()
        segments = np.asarray(wall_segments(), dtype=np.float64)      # (W, 2, 2)
        self._seg_a = segments[:, 0, :]                               # (W, 2)
        self._seg_ab = segments[:, 1, :] - segments[:, 0, :]          # (W, 2)

        # Placeholders until the first reset. Kept as attributes because
        # `sibling_rivalry.ppo_sr` reads them through `VecEnv.get_attr` to check that
        # sibling rollouts really do share (s0, g).
        self.target_position = np.array(GOAL_XY, dtype=np.float64)
        self.agent_position = np.array(START_XY, dtype=np.float64)

        obs_dim = 4 + (self.N_LIDAR_RAYS if self.use_lidar else 0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.step_count = 0
        self.collision_count = 0

    # ------------------------------------------------------------------ observation
    def _get_observation(self):
        """[x, y, gx, gy] scaled to [-1, 1], plus the rays when LiDAR is on.

        Positions are divided by the maze size and shifted to [-1, 1] so every channel
        shares one scale; ray distances are divided by LIDAR_RANGE for the same reason.
        """
        scale = lambda p: (2.0 * np.asarray(p, dtype=np.float64) / self.WORLD_SIZE) - 1.0
        parts = [scale(self.agent_position), scale(self.target_position)]
        if self.use_lidar:
            rays = cast_rays(
                self.agent_position,
                n_rays=self.N_LIDAR_RAYS,
                lidar_range=self.LIDAR_RANGE,
                world_size=self.WORLD_SIZE,
                agent_radius=0.0,          # the paper's agent is a POINT
                rects=self._wall_rects,
                circles=(),                # no dynamic obstacles in Experiment 4
                circle_radius=0.0,
            )
            parts.append(np.asarray(rays, dtype=np.float64) / self.LIDAR_RANGE)
        observation = np.concatenate(parts).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

    # ------------------------------------------------------------------- collision
    def _sweep(self, start, end):
        """Advance from `start` toward `end`, stopping just short of the first wall.

        Returns (position, collided). Vectorised over all 85 wall segments at once: for
        each wall, solve start + t*(end-start) = a + u*(b-a) and keep the smallest t in
        [0, 1] whose u also lies in [0, 1].
        """
        move = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        if not np.any(move):
            return np.asarray(end, dtype=np.float64), False

        ap = self._seg_a - np.asarray(start, dtype=np.float64)        # (W, 2)
        cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        denominator = cross(move[None, :], self._seg_ab)              # (W,)
        parallel = np.abs(denominator) < 1e-12
        safe = np.where(parallel, 1.0, denominator)
        t = cross(ap, self._seg_ab) / safe
        u = cross(ap, move[None, :]) / safe

        hit = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        if not hit.any():
            return np.asarray(start, dtype=np.float64) + move, False

        # Back off a hair so the agent never rests exactly on the wall, which would make
        # the next sweep start with t = 0 and pin it there forever.
        t_hit = float(t[hit].min())
        backed_off = max(0.0, t_hit - 1e-9)
        return np.asarray(start, dtype=np.float64) + backed_off * move, True

    def _sample_in_cell(self, cell):
        """Uniform point inside a cell, inset by CELL_MARGIN so it never starts on a wall."""
        x_low, x_high, y_low, y_high = cell_bounds(*cell, margin=self.CELL_MARGIN)
        return np.array([self.np_random.uniform(x_low, x_high),
                         self.np_random.uniform(y_low, y_high)], dtype=np.float64)

    # ----------------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both draws come from self.np_random, so `set_pair_seeds` still gives sibling
        # rollouts a byte-identical (s0, g) -- SR's requirement is untouched by sampling.
        if self.start_goal_mode == "sampled":
            self.agent_position = self._sample_in_cell(START_CELL)
            self.target_position = self._sample_in_cell(GOAL_CELL)
        else:
            self.agent_position = np.array(START_XY, dtype=np.float64)
            self.target_position = np.array(GOAL_XY, dtype=np.float64)
        self.step_count = 0
        self.collision_count = 0
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        proposed = self.agent_position + action * self.MAX_STEP

        # The proposed position is deliberately NOT clipped to the arena first. Clipping it
        # collapses the move to zero once the agent is resting ON the border, and a
        # zero-length sweep reports no contact -- so an agent parked against the outer wall
        # stopped being charged for it (measured: 2 collisions across 50 steps of driving
        # into the boundary, where interior walls correctly charged all 50). That silently
        # understates wall-hugging on the exact metric Experiment 4 exists to measure. The
        # border is in the wall-segment list, so the sweep handles it like any other wall.
        self.agent_position, collided = self._sweep(self.agent_position, proposed)
        # Numerical guard only, applied AFTER the sweep so it cannot mask a contact.
        self.agent_position = np.clip(self.agent_position, 0.0, self.WORLD_SIZE)
        self.step_count += 1
        if collided:
            self.collision_count += 1

        distance = float(np.linalg.norm(self.target_position - self.agent_position))
        reached = distance <= self.DELTA
        truncated = self.step_count >= self.MAX_STEPS and not reached

        # Collisions are charged every step and NEVER end the episode (brief section 13):
        # the agent must be free to keep trying, or the experiment would be measuring
        # "how soon does it crash" instead of "how often".
        reward = self.REWARD_COLLISION if collided else 0.0

        # Terminal payout, the paper's Eq. 2. The SR arms REPLACE the -d branch with the
        # anti-goal term (Eq. 3); see sibling_rivalry/reward.py. Paying only at the end is
        # the paper's own formulation, not a dense shaping term.
        terminated = bool(reached)
        if terminated:
            reward += self.REWARD_GOAL
        elif truncated:
            reward += -distance

        info = {
            "distance_to_target": distance,
            "step": self.step_count,
            "success": reached,
            "is_success": reached,
            "collision": bool(collided),
            # `sibling_rivalry` reads this key; the maze has only one kind of obstacle.
            "collision_type": "wall" if collided else None,
            "collision_count": self.collision_count,
            "agent_position": self.agent_position.copy(),
        }
        return self._get_observation(), reward, terminated, truncated, info

    def close(self):
        return None
