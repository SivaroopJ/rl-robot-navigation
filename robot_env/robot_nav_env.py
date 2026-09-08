"""
Custom Gymnasium environment for mobile robot navigation in a 2D space with static and dynamic obstacles.

Observation space (v2):
    [dx, dy, vx, vy, l1...l24, ovx1, ovy1, ovx2, ovy2, ovx3, ovy3]
    - dx, dy: relative position of the target (normalized by world size)
    - vx, vy: current agent velocity (normalized by max speed)
    - l1...l24:  24 lidar readings at 15 degree intervals (normalized by lidar range)
    - ovx1..ovy3: velocity of the 3 nearest dynamic obstacles (normalized by obstacle speed)
                  zero-padded when fewer than 3 obstacles are present


Action space:
    [vx, vy] in [-1, 1]^2 - continuous velocity vector

Reward:
    +10.00: reaching the target
    -10.00: collision with any obstacle
    +0.3 * delta_d: progress toward the target (if use_reward_shaping=True)
    -0.02: penality per timestep
    proximity penalty: smooth penalty scaling from 0 at danger_zone_radius to proximity_penalty_scale at contact distance 
                       (if use_reward_shaping=True)

"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_env.lidar_core import cast_rays
import json
from pathlib import Path

from robot_env.dynamic_obstacles import build_motion_model
from robot_env.start_goal import StartGoalSampler

def load_config(config_path = "config.json"):
    """
    Load configuration parameters from a JSON file.

    Parameters:
    config_path: str
        Path to the JSON configuration file

    Returns:
    config: dict
        Dictionary containing all configuration parameters grouped by section
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, 'r') as f:
        return json.load(f)

class RobotNavEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path = "config.json", n_dynamic_obstacles = None, obstacle_speed = None, obstacle_speed_range = None, 
                 render_mode = None, use_reward_shaping = True, randomize_dynamic_obstacles = None):   
        """
        Initialize the RobotNavEnv environment. Loads all parameters from config.json.

        Parameters:
        config_path: str
            Path to the JSON configuration file
        n_dynamic_obstacles: int or None
            Number of moving obstacles
        obstacle_speed: float or None
            Fixed speed of moving obstacles per timestep. Mutually exclusive with obstacle_speed_range.
        obstacle_speed_range: tuple/list of (float, float) or None
            (low, high) range from which obstacle speed is sampled uniformly at the start of each episode. When set, obstacle_speed 
            is ignored. self.obstacle_speed is set to mean(range) and used for observation normalization so the agent receives a 
            consistent scale regardless of the sampled episode speed.
        render_mode: str or None
            Rendering mode
        use_reward_shaping: bool
            If True, adds progress reward toward the target.
            If False, only goal, collision and step penalty rewards aare used.
        randomize_dynamic_obstacles: bool or None
            Week 4. True gives every dynamic obstacle a smoothly stochastic heading
            (Ornstein-Uhlenbeck steering, see robot_env/dynamic_obstacles.py); False keeps
            the original constant-velocity billiard motion. None falls back to
            config["dynamic_obstacles"]["randomize"]. Obstacle count, radius and speed are
            unaffected either way -- only the DIRECTION process changes.
        """     
        super().__init__()

        config = load_config(config_path)
        environment_config = config["environment"]
        reward_config = config["reward"]
        self.start_goal_config = config.get("start_goal", {})
        self.dynamic_obstacle_config = config.get("dynamic_obstacles", {})

        # Integration timestep. The original env had none: it advanced positions by the
        # raw per-step velocity, so the agent crossed the 10-unit arena in ~5 steps and
        # every recorded episode in results/e*/ lasted 4-6 steps. dt scales the agent and
        # the obstacles by the SAME factor, so their speed ratio, the radii and the arena
        # are all preserved and only the temporal resolution changes. dt = 1.0 reproduces
        # the original environment exactly.
        self.dt = float(config.get("dt", 1.0))

        # Environment parameters
        self.WORLD_SIZE = environment_config["world_size"]
        self.AGENT_RADIUS = environment_config["agent_radius"]
        self.OBSTACLE_RADIUS = environment_config["obstacle_radius"]
        self.TARGET_RADIUS = environment_config["target_radius"]
        self.MAX_SPEED = environment_config["max_speed"]
        self.MAX_STEPS = environment_config["max_steps"]
        self.N_LIDAR_RAYS = environment_config["n_lidar_rays"]
        self.LIDAR_RANGE = environment_config["lidar_range"]

        if n_dynamic_obstacles is not None:
            self.n_dynamic_obstacles = n_dynamic_obstacles
        else:
            self.n_dynamic_obstacles = environment_config["n_dynamic_obstacles"]

         # Speed configuration: range takes priority over fixed speed.
        # self.obstacle_speed is always a float used for observation normalisation.
        if obstacle_speed_range is not None:
            low, high = float(obstacle_speed_range[0]), float(obstacle_speed_range[1])
            if low > high:
                raise ValueError(f"obstacle_speed_range low ({low}) must be <= high ({high})")
            self.obstacle_speed_range = (low, high)
            self.obstacle_speed = (low + high) / 2.0  # normalisation anchor
        else:
            self.obstacle_speed_range = None
            if obstacle_speed is not None:
                self.obstacle_speed = float(obstacle_speed)
            else:
                self.obstacle_speed = float(environment_config["obstacle_speed"])


        # Static obstacles (list of x,y, half_width, half_height)
        self.static_obstacles = []
        for obstacle in environment_config["static_obstacles"]:
            self.static_obstacles.append(tuple(obstacle))
        
        # Reward parameters
        self.REWARD_GOAL = reward_config["goal"]
        self.REWARD_COLLISION = reward_config["collision"]
        self.REWARD_PROGRESS = reward_config["progress_scale"]
        self.REWARD_STEP = reward_config["step_penalty"]
        self.DANGER_ZONE_RADIUS = reward_config["danger_zone_radius"]
        self.PROXIMITY_PENALTY_SCALE = reward_config["proximity_penalty_scale"]
        self.ACTION_SMOOTHING_SCALE = reward_config.get("action_smoothing_scale", 0.02)

        # Number of nearest obstacle velocities to include in observation
        self.N_OBSTACLE_VELOCITIES = 3

        # Week 5. "legacy" is the original block of N_OBSTACLE_VELOCITIES velocities with
        # NO positions attached -- the agent was told how fast the nearest obstacles were
        # moving but never where they were, and had to bind a velocity to a bearing
        # through a distance ranking that reorders discontinuously whenever two obstacles
        # swap places. "paired" gives each slot its own (relative position, velocity), so
        # a slot is self-describing. The ranking is still by distance and can still
        # permute; carrying the position is what makes the permutation survivable rather
        # than destroying the binding.
        observation_config = config.get("observation", {})
        self.obstacle_observation_mode = observation_config.get("obstacle_mode", "legacy")
        if self.obstacle_observation_mode not in ("legacy", "paired"):
            raise ValueError(
                f"observation.obstacle_mode must be 'legacy' or 'paired', "
                f"got {self.obstacle_observation_mode!r}")
        self.N_OBSTACLE_SLOTS = int(observation_config.get("n_obstacle_slots", 6))

        # Week 5. Terminal penalty applied when an episode ends by hitting the step limit.
        # Defaults to 0.0, which leaves every reward identical to Week 4 -- it exists so
        # that stalling can be priced without a code change if raising gamma proves
        # insufficient. See MD_files/week5/.
        self.TIMEOUT_PENALTY = float(config.get("termination", {}).get("timeout_penalty", 0.0))

        self.render_mode = render_mode
        self.use_reward_shaping = use_reward_shaping

        # Observation space: [dx, dy, vx, vy] + N_LIDAR_RAYS + the obstacle block, whose
        # width depends on the mode selected above.
        if self.obstacle_observation_mode == "paired":
            obstacle_block = self.N_OBSTACLE_SLOTS * 4      # (rel_x, rel_y, vx, vy) each
        else:
            obstacle_block = self.N_OBSTACLE_VELOCITIES * 2
        observation_size = 4 + self.N_LIDAR_RAYS + obstacle_block
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape = (observation_size,), dtype = np.float32)

        # Action space
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)

        # Internal state
        self.agent_position = None
        self.agent_velocity = None
        self.target_position = None
        self.obstacle_positions = None
        self.obstacle_velocities = None
        self.step_count = None
        self.previous_distance = None
        self.previous_action = None
        self._episode_speed = self.obstacle_speed  # updated each reset when using range
        self.renderer = None

        # Week 4. Which obstacle motion model this env runs. Deterministic is the original
        # behaviour and is what Experiment 1 uses; the stochastic model is Experiments 2/3.
        if randomize_dynamic_obstacles is None:
            self.randomize_dynamic_obstacles = bool(
                self.dynamic_obstacle_config.get("randomize", False))
        else:
            self.randomize_dynamic_obstacles = bool(randomize_dynamic_obstacles)
        self.motion_model = build_motion_model(
            self.dynamic_obstacle_config,
            randomize=self.randomize_dynamic_obstacles,
            world_size=self.WORLD_SIZE,
            obstacle_radius=self.OBSTACLE_RADIUS,
        )

        # Week 4. Start/goal sampling. "uniform" deliberately leaves `self.start_goal_sampler`
        # as None and keeps reset() on its original `_random_free_position` calls, so the
        # legacy path stays bit-identical rather than merely equivalent.
        self.wall_margin = max(self.AGENT_RADIUS * 2, 0.5)
        if self.start_goal_config.get("sampling", "uniform") == "stratified":
            self.start_goal_sampler = StartGoalSampler(
                self.WORLD_SIZE,
                self.wall_margin,
                self.static_obstacles,
                region_grid=self.start_goal_config.get("region_grid", 4),
                region_free_threshold=self.start_goal_config.get("region_free_threshold", 0.25),
                distance_bins=self.start_goal_config.get("distance_bins"),
                min_separation=self.start_goal_config.get("min_separation", 4.0),
                max_attempts=self.start_goal_config.get("max_attempts", 200),
            )
        else:
            self.start_goal_sampler = None

        #: Straight-line start-goal distance for the current episode. Recorded so the
        #: evaluator can report every metric stratified by distance band and compute SPL.
        self.initial_distance = None

    def reset(self, seed = None, options = None):
        """
        Reset the environment to a new random initial state. Places the agent, target and all dynamic obstacles at random positions.
        Each dynamic obstacle gets a random initial velocity direction.

        When obstacle_speed_range is set, the episode speed is sampled uniformly from that range each reset. The sampled speed is 
        stored in self._episode_speed and used only for physics; observation normalisation always uses self.obstacle_speed (the range
         midpoint) so the agent sees a consistent value scale across episodes.

        Parameters:
        seed: int or None
            Random seed for reproducibility

        Returns:
        observation: np.ndarray (34,)
            The initial observation vector for the new episode
        info: dict
            Empty dictionary
        """
        super().reset(seed = seed)

        # Sample episode speed from range, or use fixed speed
        if self.obstacle_speed_range is not None:
            low, high = self.obstacle_speed_range
            self._episode_speed = float(self.np_random.uniform(low, high))
        else:
            self._episode_speed = self.obstacle_speed

        # Place agent and target. The stratified sampler spreads pairs over map regions
        # and over distance bands; the uniform branch is the original code untouched.
        if self.start_goal_sampler is not None:
            self.agent_position, self.target_position = self.start_goal_sampler.sample(
                self.np_random)
        else:
            self.agent_position = self._random_free_position()
            self.target_position = self._random_free_position(exclude=[self.agent_position], min_dist=2.0)
        self.agent_velocity = np.zeros(2, dtype=np.float32)

        excluded = [self.agent_position, self.target_position]
        obstacle_list = []
        for _ in range(self.n_dynamic_obstacles):
            obstacle_list.append(self._random_free_position(exclude=excluded, min_dist=1.5))

        self.obstacle_positions = np.array(obstacle_list, dtype=np.float32).reshape(-1, 2)

        # Initial obstacle velocities come from the motion model. For the deterministic
        # model this is the original uniform-heading draw, unchanged.
        self.obstacle_velocities = self.motion_model.reset(
            self.np_random, self.obstacle_positions, self._episode_speed)

        self.initial_distance = float(
            np.linalg.norm(self.target_position - self.agent_position))
        self.step_count = 0
        self.previous_distance  = np.linalg.norm(self.target_position - self.agent_position)
        self.previous_action = np.zeros(2, dtype=np.float32)

        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute one timestep in the environment.
        Applies the action to move the agent, moves dynamics obstacles,computes the reward and checks for termination conditions.

        Parameters:
        action: np.ndarray (2,)
            Velocity command [vx, vy] in [-1, 1]^2 

        Returns:
        observation: np.ndarray (34,)
            Updated observation vector after the step
        reward: float 
            Reward received for this timestep
        terminated: bool
            True if episode ended(goal reached or collision)
        truncated: bool
            True if the episode reached the maximum number of steps
        info: dict
            Dictionary with following information:
            - distance_to_target (float): current distance to target
            - step (int): current step count
            - success (bool): whether the agent reached the target
            - is_success (bool): same as success; required by SB3 EvalCallback to record per-episode success in evaluations.npz
        """
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.agent_velocity = action * self.MAX_SPEED
        new_position = self.agent_position + self.agent_velocity * self.dt

        # Check wall collision before clamping
        wall_collision = (
            new_position[0] < self.AGENT_RADIUS or
            new_position[0] > self.WORLD_SIZE - self.AGENT_RADIUS or
            new_position[1] < self.AGENT_RADIUS or
            new_position[1] > self.WORLD_SIZE - self.AGENT_RADIUS
        )

        self.agent_position = np.clip(
            new_position,
            self.AGENT_RADIUS,
            self.WORLD_SIZE - self.AGENT_RADIUS
        )

        # Move dynamic obstacles. The deterministic model reproduces the original
        # constant-velocity bounce; the stochastic one steers smoothly (see
        # robot_env/dynamic_obstacles.py). Collision mechanics below are unchanged either way.
        self.obstacle_positions, self.obstacle_velocities = self.motion_model.step(
            self.np_random, self.obstacle_positions, self.obstacle_velocities, self.dt)

        # Reward 
        reward = self.REWARD_STEP

        # Progress reward
        current_distance = np.linalg.norm(self.target_position - self.agent_position)
        if self.use_reward_shaping:
            reward += self.REWARD_PROGRESS * (self.previous_distance - current_distance)
            
            # Proximity penalty: smooth quadratic penalty when inside danger zone
            if self.n_dynamic_obstacles > 0:
                dists_to_obs = np.linalg.norm( self.obstacle_positions - self.agent_position, axis=1) - self.OBSTACLE_RADIUS - self.AGENT_RADIUS
                clearance = float(np.min(dists_to_obs))
                if clearance < self.DANGER_ZONE_RADIUS:
                    t = max(0.0, 1.0 - clearance / self.DANGER_ZONE_RADIUS)
                    reward -= self.PROXIMITY_PENALTY_SCALE * (t ** 2)

            # Action smoothing penalty: discourages sudden direction changes
            action_delta = action - self.previous_action
            reward -= self.ACTION_SMOOTHING_SCALE * float(np.dot(action_delta, action_delta))

        self.previous_action = action.copy()
        self.previous_distance = current_distance

        terminated = False

        goal_reached = current_distance <self.TARGET_RADIUS

        # Which kind of collision fired. The TEST and the PENALTY are unchanged -- this
        # only records information the env already computed but previously discarded, so
        # that Week 4 can report wall / static / dynamic collisions separately. The
        # research question is about dynamic obstacles specifically, and the committed
        # results could not distinguish them.
        collision_type = None
        if wall_collision:
            collision_type = "wall"
        else:
            collision_type = self._check_collision_type()

        if goal_reached:
            reward += self.REWARD_GOAL
            terminated = True
        elif collision_type is not None:
            reward += self.REWARD_COLLISION
            terminated = True

        truncated = self.step_count >= self.MAX_STEPS

        # Week 5 guard, inert while TIMEOUT_PENALTY is 0.0. Running out of steps is a
        # failure to arrive, and while gamma alone should make stalling unattractive this
        # prices it directly if it does not. Applied only when the episode did not already
        # end for a real reason, so it can never stack with the goal or collision terms.
        if truncated and not terminated and self.TIMEOUT_PENALTY:
            reward += self.TIMEOUT_PENALTY

        info = {
            "distance_to_target": float(current_distance),
            "step": self.step_count,
            "success": goal_reached,
            "is_success": goal_reached,
            "collision": bool(collision_type is not None and not goal_reached),
            "collision_type": None if goal_reached else collision_type,
            "agent_position": self.agent_position.copy(),
            "initial_distance": self.initial_distance,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Build the normalized observation vector for the current state.

        Obstacle velocities are normalised by self.obstacle_speed (the fixed speed or the midpoint of the speed range), not by the 
        episode speed. This keeps the observation scale consistent across episodes when using speed randomisation, so the agent 
        receives comparable signals regardless of the sampled speed.

        Parameters:
        None

        Returns:
        observation: np.ndarray
            Normalized observation vector. In the legacy obstacle mode:
            [dx, dy, vx, vy, l1...l24, ovx1, ovy1, ovx2, ovy2, ovx3, ovy3]
            Total size: 4 + N_LIDAR_RAYS + N_OBSTACLE_VELOCITIES * 2 = 34
            In the paired mode (see _paired_obstacle_block):
            [dx, dy, vx, vy, l1...l24, odx1, ody1, ovx1, ovy1, odx2, ...]
            Total size: 4 + N_LIDAR_RAYS + N_OBSTACLE_SLOTS * 4 = 52
        """
        relative_target = (self.target_position - self.agent_position) / self.WORLD_SIZE
        normalized_velocity = self.agent_velocity / self.MAX_SPEED
        normalized_lidar_readings = self._cast_lidar_rays() / self.LIDAR_RANGE
        
        # Velocities of the N_OBSTACLE_VELOCITIES nearest dynamic obstacles. Normalised by self.obstacle_speed (range midpoint or 
        # fixed speed) so the scale is stable across episodes. Zero-padded when fewer obstacles are present.
        if self.obstacle_observation_mode == "paired":
            obstacle_block = self._paired_obstacle_block()
        else:
            obstacle_block = np.zeros(self.N_OBSTACLE_VELOCITIES * 2, dtype=np.float32)
            if self.n_dynamic_obstacles > 0 and self.obstacle_speed > 0:
                distances = np.linalg.norm(self.obstacle_positions - self.agent_position, axis=1)
                n_nearest = min(self.N_OBSTACLE_VELOCITIES, self.n_dynamic_obstacles)
                nearest_idx = np.argsort(distances)[:n_nearest]
                nearest_velocities = self.obstacle_velocities[nearest_idx] / self.obstacle_speed
                obstacle_block[:n_nearest * 2] = nearest_velocities.flatten()

        observation = np.concatenate([relative_target, normalized_velocity, normalized_lidar_readings, obstacle_block]).astype(np.float32)

        return np.clip(observation, -1.0, 1.0)

    def _paired_obstacle_block(self):
        """Per obstacle slot: relative position then velocity, nearest obstacle first.

        Layout is [dx_1, dy_1, vx_1, vy_1, dx_2, ...], length N_OBSTACLE_SLOTS * 4, zero
        padded when there are fewer obstacles than slots. Positions are normalised by
        WORLD_SIZE and velocities by self.obstacle_speed, matching how the target offset
        and the legacy velocity block are already scaled, so the whole vector stays in
        [-1, 1] and no term dominates the input scale.

        WHY POSITION AND VELOCITY TRAVEL TOGETHER
        -----------------------------------------
        A velocity on its own is close to useless for avoidance: "something nearby is
        moving left" does not say whether it is moving into the agent's path or away from
        it. The pairing is the whole point of this block, so the two are written into
        adjacent columns of the same slot rather than into two separate blocks that the
        network would have to learn to align.
        """
        block = np.zeros(self.N_OBSTACLE_SLOTS * 4, dtype=np.float32)
        if self.n_dynamic_obstacles <= 0:
            return block

        offsets = self.obstacle_positions - self.agent_position          # (N, 2)
        distances = np.linalg.norm(offsets, axis=1)
        n_used = min(self.N_OBSTACLE_SLOTS, self.n_dynamic_obstacles)
        nearest_idx = np.argsort(distances)[:n_used]

        normalized_offsets = offsets[nearest_idx] / self.WORLD_SIZE
        if self.obstacle_speed > 0:
            normalized_velocities = self.obstacle_velocities[nearest_idx] / self.obstacle_speed
        else:
            # Stationary obstacles: the velocity columns are genuinely zero, which is
            # different from "unknown" and is exactly what the agent should see.
            normalized_velocities = np.zeros_like(normalized_offsets)

        slots = np.concatenate([normalized_offsets, normalized_velocities], axis=1)  # (n, 4)
        block[:n_used * 4] = slots.reshape(-1)
        return block

    def _cast_lidar_rays(self):
        """Distance to the nearest obstacle or wall along each of N_LIDAR_RAYS directions.

        VECTORIZED, AND WHY THAT MATTERED
        ---------------------------------
        The original implementation looped in Python over rays x obstacles, calling
        `_ray_vs_rect` and `_ray_vs_circle` once per pair. Profiling the Week 4 environment
        showed this single method accounting for **82% of all step time** (13.8s of 16.9s
        over 4000 steps), at 1.07 million scalar helper calls. Week 4 needs twelve training
        runs of a few million steps each, so that overhead was the difference between
        roughly 17 hours and roughly 5.

        This computes all rays against all obstacles at once with numpy. The scalar helpers
        below are KEPT: they are the readable reference this was derived from, and
        `tests/test_week4_env.py::test_vectorized_lidar_matches_the_scalar_reference`
        asserts the two agree exactly, so the optimization cannot silently drift from the
        definition. The legacy bit-identity test against `main` covers the same ground from
        the other direction.

        Returns
        readings: np.ndarray (N_LIDAR_RAYS,) float32
            Distance along each ray, in [0, LIDAR_RANGE].
        """
        # The body of this method now lives in `robot_env/lidar_core.py` so that
        # Experiment 4's Point Maze can cast the SAME rays rather than growing a second
        # implementation. Pure extraction -- no arithmetic changed, no operation reordered.
        # `test_vectorized_lidar_matches_the_scalar_reference` and
        # `test_legacy_path_is_bit_identical_to_main` both still pass, from opposite
        # directions, which is what makes that claim checkable rather than asserted.
        return cast_rays(
            self.agent_position,
            n_rays=self.N_LIDAR_RAYS,
            lidar_range=self.LIDAR_RANGE,
            world_size=self.WORLD_SIZE,
            agent_radius=self.AGENT_RADIUS,
            rects=self.static_obstacles,
            circles=self.obstacle_positions,
            circle_radius=self.OBSTACLE_RADIUS,
        )

    def _ray_vs_walls(self, ray_origin, ray_direction):
        """
        Compute the distance from a ray origin to the nearest world boundary wall. Uses parametric ray-boundary intersection: finds 
        the t value at which the ray hits each wall and returns the smallest positive t.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray (agent position)
        ray_direction : np.ndarray (2,)
            Unit direction vector of the ray

        Returns
        float
            Distance to the nearest wall. Returns LIDAR_RANGE if no intersection is found within range.
        """
        intersection_distances = []
        for dimension in range(2):
            # Skip dimensions where the ray is nearly parallel to the wall
            if abs(ray_direction[dimension]) > 1e-9:
                if ray_direction[dimension] > 0:
                    # Ray points toward the far wall (high boundary)
                    t = (self.WORLD_SIZE - self.AGENT_RADIUS - ray_origin[dimension]) / ray_direction[dimension]
                else:
                    # Ray points toward the near wall (low boundary)
                    t = (self.AGENT_RADIUS - ray_origin[dimension]) / ray_direction[dimension]
                if t > 0:
                    intersection_distances.append(t)
        if len(intersection_distances) > 0:
            return min(intersection_distances)  
        else:
            return self.LIDAR_RANGE

    def _ray_vs_rect(self, ray_origin, ray_direction, rect):
        """
        Compute the distance from a ray to an axis-aligned rectangular obstacle.
        Uses the slab method (AABB ray intersection): computes entry and exit t values for each axis and checks for overlap.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray
        ray_direction: np.ndarray of shape (2,)
            Unit direction vector of the ray
        rect: tuple of (float, float, float, float)
            Obstacle defined as (center_x, center_y, half_width, half_height)

        Returns
        float
            Distance to the rectangle along the ray. Returns LIDAR_RANGE if the ray does not intersect the rectangle.
        """
        center_x, center_y, half_width, half_height = rect

        # Axis-aligned bounding box boundaries
        bounds = [
            (center_x - half_width, center_x + half_width), # x-axis slab
            (center_y - half_height, center_y + half_height)  # y-axis slab
        ]

        t_enter = -np.inf
        t_exit = np.inf

        for dimension, (low_bound, high_bound) in enumerate(bounds):
            ray_component = ray_direction[dimension]
            origin_component = ray_origin[dimension]

            if abs(ray_component) < 1e-9:
                # Ray is parallel to this slab — check if origin is inside
                if origin_component < low_bound or origin_component > high_bound:
                    return self.LIDAR_RANGE
            else:
                t1 = (low_bound - origin_component) / ray_component
                t2 = (high_bound - origin_component) / ray_component

                slab_enter = min(t1, t2)
                slab_exit = max(t1, t2)

                if slab_enter > t_enter:
                    t_enter = slab_enter
                if slab_exit < t_exit:
                    t_exit = slab_exit

        # Valid intersection: entry before exit and exit in front of ray
        if t_enter <= t_exit and t_exit > 0:
            if t_enter > 0:
                intersection_t = t_enter
            else:
                intersection_t = t_exit
            return float(np.clip(intersection_t, 0, self.LIDAR_RANGE))

        return self.LIDAR_RANGE

    def _ray_vs_circle(self, ray_origin, ray_direction, circle_center, circle_radius):
        """
        Compute the distance from a ray to a circular obstacle. Solves the quadratic equation derived from substituting the parametric
         ray equation into the circle equation.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray
        ray_direction: np.ndarray (2,)
            Unit direction vector of the ray
        circle_center: np.ndarray (2,)
            Center position of the circular obstacle
        circle_radius: float
            Radius of the circular obstacle

        Returns
        float
            Distance to the circle along the ray. Returns LIDAR_RANGE if the ray does not intersect the circle.
        """
        # Vector from circle center to ray origin
        origin_to_center = ray_origin - circle_center

        # Quadratic coefficients: at^2 + bt + c = 0 (a=1 since direction is unit)
        b = 2 * np.dot(origin_to_center, ray_direction)
        c = np.dot(origin_to_center, origin_to_center) - circle_radius ** 2
        discriminant = b ** 2 - 4 * c

        # No real solution means no intersection
        if discriminant < 0:
            return self.LIDAR_RANGE

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / 2.0
        t2 = (-b + sqrt_discriminant) / 2.0

        # Pick the smallest positive t (nearest intersection in front of ray)
        if t1 > 0:
            nearest_t = t1
        else:
            nearest_t = t2

        if nearest_t > 0:
            return float(np.clip(nearest_t, 0, self.LIDAR_RANGE))
        else:
            return self.LIDAR_RANGE

    def _check_collision_type(self):
        """Which kind of obstacle the agent is touching, or None.

        Exactly the geometry `_check_collision` has always used -- the circles are tested
        before the rectangles and the same radii and margins apply -- but it reports WHICH
        family matched instead of collapsing both to a bool. `_check_collision` is kept
        below as a wrapper over this so no existing caller changes behaviour.

        Returns
        str or None
            "dynamic" for a moving circular obstacle, "static" for a rectangle, None for
            no contact.
        """
        combined_radius = self.AGENT_RADIUS + self.OBSTACLE_RADIUS
        for obstacle_position in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_position)
            if distance < combined_radius:
                return "dynamic"

        for center_x, center_y, half_width, half_height in self.static_obstacles:
            closest_x = np.clip(self.agent_position[0], center_x - half_width, center_x + half_width)
            closest_y = np.clip(self.agent_position[1], center_y - half_height, center_y + half_height)
            closest_point = np.array([closest_x, closest_y])
            distance = np.linalg.norm(self.agent_position - closest_point)
            if distance < self.AGENT_RADIUS:
                return "static"

        return None

    def _check_collision(self):
        """
        Check whether the agent is currently colliding with anything. Tests collision against world boundary walls, all dynamic circular
        obstacles and all static rectangular obstacles.

        Parameters
        None

        Returns
        bool
            True if a collision is detected, False otherwise.
        """
        return self._check_collision_type() is not None

    def _random_free_position(self, exclude=None, min_dist=1.0):
        """
        Sample a random position that is free of walls, static obstacles and sufficiently far from a list of excluded positions.
        Attempts up to 200 random samples before falling back to the arena center if no valid position is found.

        Parameters
        exclude: list of np.ndarray or None
            List of positions that the new position must be at least min_dist away from
        min_dist: float
            Minimum allowed distance from each excluded position

        Returns
        position: np.ndarray (2,)
            A valid free position in the arena
        """
        wall_margin = max(self.AGENT_RADIUS * 2, 0.5)

        for _ in range(200):
            candidate = self.np_random.uniform(wall_margin, self.WORLD_SIZE - wall_margin, size=2).astype(np.float32)

            # Check minimum distance from all excluded positions
            too_close_to_excluded = False
            if exclude is not None:
                for excluded_position in exclude:
                    distance = np.linalg.norm(candidate - excluded_position)
                    if distance < min_dist:
                        too_close_to_excluded = True
                        break
            if too_close_to_excluded:
                continue

            # Check that candidate is not inside any static obstacle
            inside_obstacle = False
            for center_x, center_y, half_width, half_height in self.static_obstacles:
                x_overlap = abs(candidate[0] - center_x) < half_width + wall_margin
                y_overlap = abs(candidate[1] - center_y) < half_height + wall_margin
                if x_overlap and y_overlap:
                    inside_obstacle = True
                    break
            if inside_obstacle:
                continue

            return candidate

        # Fallback: return arena center if no valid position found after 200 attempts
        return np.array([self.WORLD_SIZE / 2.0, self.WORLD_SIZE / 2.0], dtype=np.float32)

    def render(self):
        """
        Render the current environment state using pygame.
        Draws the arena, static obstacles (grey rectangles), the target(green circle), dynamic obstacles (orange circles) and the agent
        (blue circle).

        Parameters
        None

        Returns
        None
            When render_mode is 'human' (draws to screen)
        np.ndarray of shape (HEIGHT, WIDTH, 3)
            RGB image array when render_mode is 'rgb_array'
        """
        if self.render_mode is None:
            return

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. ""Install it with: pip install pygame")

        PIXELS_PER_UNIT = 60
        WINDOW_SIZE = int(self.WORLD_SIZE * PIXELS_PER_UNIT)

        # Initialize pygame and create display surface on first call
        if self.renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self.renderer = pygame.display.set_mode(
                    (WINDOW_SIZE, WINDOW_SIZE)
                )
                pygame.display.set_caption("RL Robot Navigation")
            else:
                self.renderer = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

        surface = self.renderer
        surface.fill((240, 240, 240))

        def world_to_pixels(world_position):
            """Convert world coordinates to pixel coordinates."""
            pixel_x = int(world_position[0] * PIXELS_PER_UNIT)
            pixel_y = int((self.WORLD_SIZE - world_position[1]) * PIXELS_PER_UNIT)
            return (pixel_x, pixel_y)

        # Draw static rectangular obstacles (grey)
        for center_x, center_y, half_width, half_height in self.static_obstacles:
            rect_pixel_x = int((center_x - half_width) * PIXELS_PER_UNIT)
            rect_pixel_y = int((self.WORLD_SIZE - center_y - half_height) * PIXELS_PER_UNIT)
            rect_width = int(2 * half_width  * PIXELS_PER_UNIT)
            rect_height = int(2 * half_height * PIXELS_PER_UNIT)
            rect = pygame.Rect(rect_pixel_x, rect_pixel_y, rect_width, rect_height)
            pygame.draw.rect(surface, (100, 100, 100), rect)

        # Draw target (green circle)
        pygame.draw.circle(surface, (80, 180, 80), world_to_pixels(self.target_position), int(self.TARGET_RADIUS * PIXELS_PER_UNIT))

        # Draw dynamic obstacles (orange circles)
        for obstacle_position in self.obstacle_positions:
            pygame.draw.circle(surface,(220, 100, 50), world_to_pixels(obstacle_position), int(self.OBSTACLE_RADIUS * PIXELS_PER_UNIT))

        # Draw agent (blue circle)
        pygame.draw.circle(surface, (50, 120, 210), world_to_pixels(self.agent_position), int(self.AGENT_RADIUS * PIXELS_PER_UNIT))

        if self.render_mode == "human":
            pygame.display.flip()
            pygame.time.Clock().tick(self.metadata["render_fps"])
        else:
            raw_pixels = pygame.surfarray.array3d(surface)
            return np.transpose(raw_pixels, axes=(1, 0, 2))

    def close(self):
        """
        Clean up resources when the environment is no longer needed. Shuts down the pygame display if it was initialized.

        Parameters
        None
        """
        if self.renderer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self.renderer = None