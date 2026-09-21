"""PursuitEnv: RobotNavEnv + a hunter, with a strictly simultaneous update.

FROZEN ENV IS NOT EDITED. This subclass adds hunter state, the hunter's own LiDAR, capture and
hunter-collision detection, and a step entry point that takes BOTH actions. The protagonist's own
motion, the dynamic obstacles, the reward and every existing collision check are the frozen
`RobotNavEnv.step()` call, unchanged.

SIMULTANEITY (plan section 5). `step_pursuit(u_prot, u_hunter)` requires both actions to have been
computed from the same pre-step state; the harness guarantees that and `pre_step_state()` is what it
reads. Inside one call the frozen `super().step(u_prot)` runs first, but the hunter's action was
already fixed beforehand, so the hunter can never react to the protagonist's updated position, nor
the protagonist to the hunter's. `assert_simultaneous()` and the tests pin this.

EPISODE LAYOUT IS PRESERVED. The hunter is drawn from `np_random` AFTER the protagonist's start,
goal and every obstacle, so for a given seed the protagonist faces exactly the navigation
experiment's layout and obstacle trajectories. A test asserts this against the plain RobotNavEnv.

OUTCOMES. Priority is own-collision -> capture -> goal -> timeout (audit section 7). Every flag is
recorded regardless of which one won, so another convention can be recomputed offline. Capture
(d <= 0.65) is NOT a collision; agent-agent physical contact (d < 0.60) is recorded separately.

HUNTER CRASH RULE, pre-declared. A hunter that hits a wall, a rectangle or a moving obstacle is
DISABLED at the point of impact: it stops moving, its velocity is zeroed and it can no longer
capture. The episode CONTINUES so the protagonist's own outcome stays measurable, and the crash is
reported both as an episode-level hunter-collision rate and with the step at which it happened. The
alternative (ending the episode on a hunter crash) would silently convert protagonist successes into
truncated episodes, which is why it is not used.
"""
from __future__ import annotations

import numpy as np

from robot_env.lidar_core import cast_rays
from robot_env.robot_nav_env import RobotNavEnv

from continuation.pursuit.config import PursuitConfig


class PursuitEnv(RobotNavEnv):
    def __init__(self, *, pursuit: PursuitConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pursuit = pursuit or PursuitConfig()
        self.hunter_position = None
        self.hunter_velocity = np.zeros(2, dtype=float)
        self.hunter_disabled = False
        self.hunter_collision_type = None
        self.hunter_collision_step = None
        self.capture_distance = self.pursuit.capture_distance(self.AGENT_RADIUS)
        self.contact_distance = self.pursuit.contact_distance(self.AGENT_RADIUS)

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)      # frozen layout, untouched
        exclude = [self.agent_position, self.target_position] + list(self.obstacle_positions)
        pos = None
        for _ in range(50):                     # redraw until far enough from the protagonist
            cand = self._random_free_position(exclude=exclude, min_dist=1.5)
            if np.linalg.norm(cand - self.agent_position) >= self.pursuit.hunter_start_min_dist:
                pos = cand
                break
        if pos is None:                          # documented deterministic fallback
            pos = cand
        self.hunter_position = np.asarray(pos, dtype=float).reshape(2)
        self.hunter_velocity = np.zeros(2, dtype=float)
        self.hunter_disabled = False
        self.hunter_collision_type = None
        self.hunter_collision_step = None
        return obs, info

    # ------------------------------------------------------------------ sensing
    def hunter_lidar(self, include_protagonist=False):
        """The hunter's own 24-ray scan, same sensor model, cast from the hunter's position.

        Default (`include_protagonist=False`) is the audited scan: the protagonist is not rendered,
        so rays pass through it. With True the protagonist's body (radius AGENT_RADIUS) is rendered
        too, as a second cast combined by per-ray minimum; a ray that does not hit the protagonist
        returns exactly the default reading. Observation only: the world is not modified.
        """
        base = np.asarray(cast_rays(
            self.hunter_position, n_rays=self.N_LIDAR_RAYS, lidar_range=self.LIDAR_RANGE,
            world_size=self.WORLD_SIZE, agent_radius=self.pursuit.hunter_radius,
            rects=self.static_obstacles, circles=self.obstacle_positions,
            circle_radius=self.OBSTACLE_RADIUS), dtype=float)
        if not include_protagonist:
            return base
        prot = np.asarray(cast_rays(
            self.hunter_position, n_rays=self.N_LIDAR_RAYS, lidar_range=self.LIDAR_RANGE,
            world_size=self.WORLD_SIZE, agent_radius=self.pursuit.hunter_radius,
            circles=np.asarray(self.agent_position, dtype=np.float32).reshape(1, 2),
            circle_radius=self.AGENT_RADIUS), dtype=float)
        return np.minimum(base, prot)

    def _cast_lidar_rays(self):
        """Frozen scan, plus the hunter as one more disc when Track A's perception model is on.

        The frozen `RobotNavEnv._cast_lidar_rays` excludes the hunter; this override is in OUR
        subclass and changes nothing for the known-state baseline, where
        `hunter_visible_to_lidar` is False and the frozen implementation is called unchanged.
        """
        if not self.pursuit.hunter_visible_to_lidar or self.hunter_position is None:
            return super()._cast_lidar_rays()
        circles = np.vstack([np.asarray(self.obstacle_positions, dtype=np.float32),
                             np.asarray(self.hunter_position, dtype=np.float32).reshape(1, 2)])
        return cast_rays(self.agent_position, n_rays=self.N_LIDAR_RAYS,
                         lidar_range=self.LIDAR_RANGE, world_size=self.WORLD_SIZE,
                         agent_radius=self.AGENT_RADIUS, rects=self.static_obstacles,
                         circles=circles, circle_radius=self.OBSTACLE_RADIUS)

    def pre_step_state(self):
        """The one shared world state both controllers must read (plan section 5, step 1)."""
        return {"p_prot": self.agent_position.copy(), "v_prot": self.agent_velocity.copy(),
                "p_hunter": self.hunter_position.copy(),
                "v_hunter": self.hunter_velocity.copy(),
                "obstacles": self.obstacle_positions.copy(),
                "obstacle_velocities": self.obstacle_velocities.copy(),
                "step": int(self.step_count)}

    # ------------------------------------------------------------------ the step
    def _hunter_collision_type(self):
        """Hunter vs dynamic circles / static rectangles, same geometry as the protagonist's."""
        r = self.pursuit.hunter_radius
        for q in self.obstacle_positions:
            if np.linalg.norm(self.hunter_position - q) < r + self.OBSTACLE_RADIUS:
                return "dynamic"
        for cx, cy, hw, hh in self.static_obstacles:
            closest = np.array([np.clip(self.hunter_position[0], cx - hw, cx + hw),
                                np.clip(self.hunter_position[1], cy - hh, cy + hh)])
            if np.linalg.norm(self.hunter_position - closest) < r:
                return "static"
        return None

    def step_pursuit(self, u_prot, u_hunter):
        """Apply both actions simultaneously.

        u_prot: protagonist action in ENV units (u / MAX_SPEED, per-axis in [-1, 1]).
        u_hunter: hunter velocity in m/s, per-axis clipped to the hunter's own speed bound.
        """
        if self.hunter_position is None:
            raise RuntimeError("call reset() before step_pursuit()")
        r = self.pursuit.hunter_radius
        v_h = np.clip(np.asarray(u_hunter, dtype=float).reshape(2),
                      -self.pursuit.hunter_max_speed, self.pursuit.hunter_max_speed)

        # 1) protagonist + dynamic obstacles: the FROZEN step, verbatim
        obs, reward, terminated, truncated, info = super().step(u_prot)

        # 2) hunter, integrated with the action fixed from the same pre-step state.
        #    A disabled (crashed) hunter never moves again.
        if self.hunter_disabled:
            v_h = np.zeros(2)
            hunter_wall = False
        else:
            new_h = self.hunter_position + v_h * self.dt
            hunter_wall = bool(new_h[0] < r or new_h[0] > self.WORLD_SIZE - r
                               or new_h[1] < r or new_h[1] > self.WORLD_SIZE - r)
            self.hunter_position = np.clip(new_h, r, self.WORLD_SIZE - r)
        self.hunter_velocity = v_h

        # 3) outcomes, evaluated after both agents and the obstacles have moved
        d = float(np.linalg.norm(self.agent_position - self.hunter_position))
        hunter_coll = None
        if not self.hunter_disabled:
            hunter_coll = "wall" if hunter_wall else self._hunter_collision_type()
            if hunter_coll is not None:          # crash: disable in place, episode continues
                self.hunter_disabled = True
                self.hunter_collision_type = hunter_coll
                self.hunter_collision_step = int(self.step_count)
                self.hunter_velocity = np.zeros(2)
        captured = (d <= self.capture_distance) and not self.hunter_disabled
        contact = d < self.contact_distance
        prot_coll = info.get("collision_type") if info.get("collision") else None
        goal = bool(info.get("success"))
        timeout = bool(truncated)

        if prot_coll is not None:
            outcome = "protagonist_collision"
        elif captured:
            outcome = "capture"
        elif goal:
            outcome = "goal"
        elif timeout:
            outcome = "timeout"
        else:
            outcome = None
        terminated = bool(prot_coll is not None or captured or goal)

        info.update({
            "outcome": outcome,
            "goal": goal, "captured": captured, "agent_contact": contact,
            "protagonist_collision_type": prot_coll,
            "hunter_collision_type": hunter_coll,
            "hunter_disabled": bool(self.hunter_disabled),
            "hunter_collision_step": self.hunter_collision_step,
            "hunter_position": self.hunter_position.copy(),
            "hunter_velocity": self.hunter_velocity.copy(),
            "hunter_distance": d,
            "capture_distance": self.capture_distance,
        })
        return obs, reward, terminated, truncated, info
