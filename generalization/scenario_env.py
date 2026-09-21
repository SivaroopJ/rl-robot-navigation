"""Week 6 / scenario environment. A SUBCLASS of RobotNavEnv -- robot_env/ is never edited.

The frozen sha256 manifest covers robot_env/ and evaluation/, and every Phase 1-7 result
depends on that gate staying green, so behaviour the canonical environment does not have is
added here by override instead. Same pattern Phase 7 used for Phase7Policy and
ProjectionCappedSource.

WHAT THIS ADDS, and nothing else:
    fixed start/goal      F8 -- declared pairs instead of the stratified sampler
    reactive walls        F10 -- a declared rectangle activates when the robot enters a
                          declared trigger disc. Deterministic, identical for both arms.
    scripted trajectories F12, X1-X4 -- obstacle motion from a declared spec rather than the
                          deterministic-bounce or OU models.

WHAT IT DOES NOT ADD:
    Moving goals. F11 is DROPPED (spec decision 4). `DRCBFPolicy.reset()` fixes the goal and
    the A* path and `predict()` never re-reads obs[0:2], so a moving goal is invisible to both
    arms. That is a documented boundary-of-applicability limitation established by code
    inspection, not an experiment, and no moving-goal code exists here.

    Replanning. Neither arm replans when a reactive wall appears (spec 1.4). A* runs once at
    policy reset and CarrotFollower.progress is monotone non-decreasing. The wall is visible
    only through LiDAR, to both arms equally.

FAIRNESS. Nothing here reads or depends on which controller is driving. Triggers are geometric
and deterministic, scripted motion is a pure function of step index, and the RNG stream is the
env's own. Two runs with the same seed and different controllers see the same environment.
"""
from __future__ import annotations

import numpy as np

from robot_env.robot_nav_env import RobotNavEnv


class ScriptedMotion:
    """Obstacle motion from a declared spec. Same interface as the frozen motion models.

    Each obstacle is a dict:
        {"start": [x, y], "vel": [vx, vy]}                     straight, no bounce
        {"start": [x, y], "centre": [cx, cy], "omega": w}      circular about a centre
        {"start": [x, y], "waypoints": [[x, y], ...], "speed": s}   patrol, ping-pong

    Deterministic and independent of the RNG, so the same spec yields the same track under both
    arms. `np_random` is accepted to match the frozen signature and is not used.
    """

    def __init__(self, spec, *, world_size, obstacle_radius):
        self.spec = [dict(s) for s in spec]
        self.low = float(obstacle_radius)
        self.high = float(world_size) - float(obstacle_radius)
        self.t = 0

    def reset(self, np_random, positions, speed):
        self.t = 0
        for i, s in enumerate(self.spec[:len(positions)]):
            positions[i] = np.asarray(s["start"], dtype=positions.dtype)
        vel = np.zeros_like(positions)
        for i, s in enumerate(self.spec[:len(positions)]):
            vel[i] = self._velocity(i, s, positions[i])
        return vel

    def _velocity(self, i, s, p):
        if "vel" in s:
            return np.asarray(s["vel"], dtype=float)
        if "centre" in s:
            c = np.asarray(s["centre"], dtype=float)
            r = p - c
            w = float(s.get("omega", 0.5))
            return np.array([-w * r[1], w * r[0]])
        if "waypoints" in s:
            wps = np.asarray(s["waypoints"], dtype=float).reshape(-1, 2)
            k = s.setdefault("_k", 0)
            d = s.setdefault("_dir", 1)
            v = wps[k] - p
            n = float(np.linalg.norm(v))
            if n < 0.05:
                if len(wps) == 1:
                    # A single waypoint is a destination, not a patrol: hold on arrival.
                    # X2 and X3 use this to close a passage and then stay closed.
                    return np.zeros(2)
                k += d
                if k >= len(wps):
                    k, d = len(wps) - 2, -1
                elif k < 0:
                    k, d = 1, 1
                s["_k"], s["_dir"] = k, d
                v = wps[k] - p
                n = float(np.linalg.norm(v))
            return (v / max(n, 1e-9)) * float(s.get("speed", 0.675))
        return np.zeros(2)

    def step(self, np_random, positions, velocities, dt):
        self.t += 1
        for i, s in enumerate(self.spec[:len(positions)]):
            velocities[i] = self._velocity(i, s, positions[i])
        positions += velocities * dt
        np.clip(positions, self.low, self.high, out=positions)
        return positions, velocities


class ScenarioEnv(RobotNavEnv):
    """RobotNavEnv plus declared start/goal, reactive walls and scripted obstacle motion."""

    def __init__(self, *args, start_goal=None, reactive_walls=None, scripted=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.declared_start_goal = start_goal          # (start_xy, goal_xy) or None
        self.declared_walls = [dict(w) for w in (reactive_walls or [])]
        self._base_static = [tuple(o) for o in self.static_obstacles]
        self.wall_events = []
        if scripted is not None:
            self.motion_model = ScriptedMotion(
                scripted, world_size=self.WORLD_SIZE, obstacle_radius=self.OBSTACLE_RADIUS)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        # RESTORE BEFORE super().reset(), not after. `super().reset()` samples obstacle
        # positions with `_random_free_position`, which rejects candidates against
        # `self.static_obstacles` and therefore consumes a DIFFERENT number of RNG draws when a
        # wall from the previous episode is still present. Arms A and B fire walls at different
        # steps, so restoring afterwards let the previous episode's wall change THIS episode's
        # obstacle layout, and the two arms diverged. Caught by the paired fairness assertion
        # (`F10a_behind seed 8000046: obstacles differ`), which is why it exists.
        self.static_obstacles = [tuple(o) for o in self._base_static]
        self.wall_events = []
        for w in self.declared_walls:
            w["_fired"] = False
        obs, info = super().reset(seed=seed, options=options)

        if self.declared_start_goal is not None:
            start, goal = self.declared_start_goal
            self.agent_position = np.asarray(start, dtype=np.float32).copy()
            self.target_position = np.asarray(goal, dtype=np.float32).copy()
            # Push any obstacle that the declared pair now overlaps, using the env's own RNG
            # so the layout stays a deterministic function of the seed.
            excl = [self.agent_position, self.target_position]
            for i in range(len(self.obstacle_positions)):
                d = min(np.linalg.norm(self.obstacle_positions[i] - q) for q in excl)
                if d < 1.5:
                    self.obstacle_positions[i] = self._random_free_position(
                        exclude=excl + [p for j, p in enumerate(self.obstacle_positions)
                                        if j != i],
                        min_dist=1.5)
            self.initial_distance = float(
                np.linalg.norm(self.target_position - self.agent_position))
            self.previous_distance = self.initial_distance
            obs = self._get_observation()
            info = dict(info)
            info["distance_to_target"] = self.initial_distance
        return obs, info

    # ------------------------------------------------------------------ step
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        fired = self._fire_walls()
        if fired:
            # The wall exists from this step on: re-read the observation so LiDAR sees it
            # immediately rather than a step late. Both arms get the identical timing.
            obs = self._get_observation()
            info = dict(info)
            info["wall_activated"] = fired
        return obs, reward, terminated, truncated, info

    def _fire_walls(self):
        """Activate any declared wall whose trigger disc now contains the robot."""
        fired = []
        for w in self.declared_walls:
            if w.get("_fired"):
                continue
            c = np.asarray(w["trigger"], dtype=float)
            if float(np.linalg.norm(self.agent_position - c)) <= float(w.get("radius", 0.5)):
                w["_fired"] = True
                rect = tuple(float(v) for v in w["rect"])
                self.static_obstacles = list(self.static_obstacles) + [rect]
                d_wall = self._distance_to_rect(self.agent_position, rect)
                ev = {"name": w.get("name", "wall"), "step": int(self.step_count),
                      "rect": list(rect), "distance_at_activation": d_wall}
                self.wall_events.append(ev)
                fired.append(ev)
        return fired

    @staticmethod
    def _distance_to_rect(p, rect):
        cx, cy, hw, hh = rect
        qx = float(np.clip(p[0], cx - hw, cx + hw))
        qy = float(np.clip(p[1], cy - hh, cy + hh))
        return float(np.linalg.norm(np.asarray(p, dtype=float) - np.array([qx, qy])))
