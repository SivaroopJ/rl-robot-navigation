"""Phase 6: drive the frozen Phase 5 system from RobotNavEnv's observation.

FROZEN. This adapter adds no control logic. alpha = 0.4, r_W = 0.004, eps = 0.1, the u = 0
fallback, the CBF formulation, the LiDAR pipeline, the velocity estimator, the A* planner and
the carrot follower are all exactly as validated in Phases 1-5. All this file does is unpack
the environment's 52-d observation and hand the pieces to the existing components.

WHAT IS READ FROM THE OBSERVATION, AND WHAT IS NOT
---------------------------------------------------
    obs[0:2]    goal offset / WORLD_SIZE      USED
    obs[2:4]    own velocity / MAX_SPEED      available, unused
    obs[4:28]   24 LiDAR ranges / LIDAR_RANGE USED
    obs[28:52]  exact relative position AND velocity of the 6 nearest dynamic obstacles
                                              NEVER READ  <-- PPO does read this

The last row is the central information asymmetry of the comparison and it is enforced here
the same way it was in Phases 2-4: this class never indexes past 28, and
`tests/test_dr_control_phase6.py::test_policy_ignores_the_ground_truth_obstacle_block`
asserts that perturbing obs[28:52] leaves the action bit-identical.

EGO POSE
--------
Supplied by the caller, as in Phases 2-5 and as in the paper (which localises with Hector
SLAM). This is information PPO does NOT get, and the report states it as such rather than
treating the asymmetry as one-directional.
"""
from __future__ import annotations

import numpy as np

from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import LidarVelocityTracker
from optional_navigation.planner import CarrotFollower, StaticMapPlanner

N_RAYS = 24


class DRCBFPolicy:
    """A* -> carrot -> LiDAR DR-CBF, with an SB3-shaped predict() so the same evaluation
    loop can drive it and a PPO checkpoint identically."""

    def __init__(self, *, world_size, agent_radius, static_obstacles, max_speed, dt,
                 lidar_range=5.0, n_rays=N_RAYS, alpha=0.4, lookahead=1.0,
                 use_planner=True, r_nominal=0.3):
        self.world_size = float(world_size)
        self.agent_radius = float(agent_radius)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.lidar_range = float(lidar_range)
        self.n_rays = int(n_rays)
        self.lookahead = float(lookahead)
        self.use_planner = bool(use_planner)
        self.r_nominal = float(r_nominal)
        self.alpha = float(alpha)

        self.planner = StaticMapPlanner(world_size, agent_radius, static_obstacles)
        self.ctrl = ClfCbfDrccpController(max_v=self.max_speed, cbf_rate=self.alpha)
        self.src = None
        self.follower = None
        self.goal = None
        self.planner_failed = False
        # per-episode diagnostics, metrics only
        self.n_infeasible = 0
        self.n_solver_fail = 0
        self.n_steps = 0
        self.last_step_infeasible = False
        self.u_dev = []
        self.min_cbc_solved = np.inf
        self.solve_times = []

    # ------------------------------------------------------------------ episode setup
    def reset(self, obs, ego_pose):
        """Plan once from the current pose to the goal implied by the observation."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        p = np.asarray(ego_pose, dtype=float).reshape(2)
        self.goal = p + obs[0:2] * self.world_size

        tracker = LidarVelocityTracker(r_nominal=self.r_nominal, dt=self.dt,
                                       n_rays=self.n_rays, lidar_range=self.lidar_range)
        self.src = EstimatedLidarBarrierSource(r_robot=self.agent_radius, k_scans=5,
                                               n_rays=self.n_rays,
                                               lidar_range=self.lidar_range, tracker=tracker)
        self.ctrl.reset()
        self.planner_failed = False
        self.follower = None
        if self.use_planner:
            path = self.planner.path(p, self.goal)
            if path is None:
                self.planner_failed = True          # reported, never silently patched
            else:
                self.follower = CarrotFollower(path, lookahead=self.lookahead)

        self.n_infeasible = self.n_solver_fail = self.n_steps = 0
        self.last_step_infeasible = False
        self.u_dev = []
        self.min_cbc_solved = np.inf
        self.solve_times = []
        return self

    # ------------------------------------------------------------------ per step
    def predict(self, obs, ego_pose, deterministic=True):
        """(action, None), matching SB3's signature so the eval loop is shared.

        `action` is in the environment's units: u / MAX_SPEED, clipped to the action box.
        """
        obs = np.asarray(obs, dtype=float).reshape(-1)
        p = np.asarray(ego_pose, dtype=float).reshape(2)
        ranges = obs[4:4 + self.n_rays] * self.lidar_range      # nothing past index 28

        self.src.push(p, ranges)
        h, g, dd = self.src.samples(p)
        xi = build_xi(h, g, dd)

        gamma = self.follower.reference(p) if self.follower is not None else self.goal
        info = {}
        u = self.ctrl.generate_controller(p, gamma, xi, record=info)

        self.n_steps += 1
        status = info.get("status")
        self.last_step_infeasible = status is not None and "infeasible" in str(status)
        if status != "optimal":
            self.n_solver_fail += 1
            if self.last_step_infeasible:
                self.n_infeasible += 1
        else:
            cbc = float(np.min(xi @ np.array([1.0, self.alpha, u[0], u[1]])))
            self.min_cbc_solved = min(self.min_cbc_solved, cbc)
        st = info.get("total_time")
        if st == st and st is not None:
            self.solve_times.append(float(st))
        nom = info.get("u_nom")
        if nom is not None:
            self.u_dev.append(float(np.linalg.norm(np.asarray(u) - np.asarray(nom))))

        action = np.clip(np.asarray(u, dtype=float) / self.max_speed, -1.0, 1.0)
        return action.astype(np.float32), None
