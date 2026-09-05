"""
Motion models for RobotNavEnv's dynamic obstacles.

WHAT THE ORIGINAL MOTION ACTUALLY IS
------------------------------------
Week 4's brief describes the existing obstacles as following "fixed/deterministic
trajectories". They do not. `RobotNavEnv.reset` draws a uniform initial heading per
obstacle and `step` then advances each by a constant velocity, reflecting it elastically
off the world boundary (the original `_bounce_obstacles`, now folded into this module).
Obstacles pass straight THROUGH the static
rectangles and through each other; only the four walls turn them.

So the trajectory is a random-heading billiard path: piecewise linear, deterministic
given the seed, with a discontinuous heading flip at each wall. `DeterministicMotion`
reproduces that exactly -- it is a lift of the original code, not a reinterpretation --
so Experiment 1 remains the original baseline.

WHAT "RANDOMIZED" MEANS HERE, AND WHY IT IS NOT JUST NOISE
----------------------------------------------------------
`SmoothStochasticMotion` replaces the constant heading with a heading driven by an
Ornstein-Uhlenbeck angular velocity:

    omega <- (1 - lambda*dt) * omega + sigma * sqrt(dt) * N(0, 1)
    omega <- clip(omega, -omega_max, +omega_max)
    theta <- theta + omega * dt
    x, y  <- x + v*cos(theta)*dt,  y + v*sin(theta)*dt

The point of the OU term rather than a fresh angle per step is TEMPORAL CORRELATION.
Redrawing theta each step would give a direction that is unpredictable but also
physically absurd -- the obstacle would jitter in place rather than travel. OU gives
a heading that wanders smoothly: unpredictable over the length of an episode, but with
a bounded turn rate at every step, which is what "stochastic but smooth" has to mean for
the agent to have anything learnable to react to.

Speed is NOT randomized per step. It is drawn once per episode from `speed_range` (which
defaults to the degenerate [1.0, 1.0], i.e. the original fixed speed) and held, so this
class changes the DIRECTION process only. Obstacle count, radius and the arena are
untouched.

BOUNDARY HANDLING IS A STEER, NOT A BOUNCE
------------------------------------------
An elastic reflection would reintroduce exactly the discontinuity the OU process exists
to avoid: one step the obstacle travels +x, the next -x. Instead, within `boundary_margin`
of a wall a corrective term is added to omega BEFORE the clip, proportional to how far
into the margin the obstacle has come and to how much its heading points outward. The
obstacle therefore curves away over several steps, and `|omega| <= omega_max` still holds
at every step because the steer is clipped along with everything else.

A hard position clamp remains as a safety net. It is a bug if it ever fires, and
`tests/test_week4_obstacles.py` asserts it does not: if the margin and turn rate are ever
configured so an obstacle cannot turn in time, that must fail loudly rather than silently
reintroduce a discontinuity.

RANDOMNESS SOURCE
-----------------
Every draw goes through the `np_random` handed in by the env, which is Gymnasium's
seeded generator. Nothing here touches global `np.random`, so `reset(seed=k)` reproduces
a trajectory exactly and different seeds diverge -- the property Sibling Rivalry's sibling
pairing depends on.
"""
from __future__ import annotations

import numpy as np


class DeterministicMotion:
    """The original behaviour: constant velocity, elastic reflection off the walls.

    Lifted verbatim from the original `RobotNavEnv._bounce_obstacles`, which this class
    replaced, so Experiment 1 is unchanged.
    """

    #: Reported in `info` and in run configs so results are never ambiguous about which
    #: motion model produced them.
    name = "deterministic"

    def __init__(self, world_size, obstacle_radius, **_ignored):
        self.world_size = float(world_size)
        self.obstacle_radius = float(obstacle_radius)
        self.low = self.obstacle_radius
        self.high = self.world_size - self.obstacle_radius
        self.n_clamped = 0

    def reset(self, np_random, positions, speed):
        """Assign initial velocities. Mirrors the original uniform-heading draw.

        Parameters
        np_random: np.random.Generator
        positions: np.ndarray (N, 2)
        speed: float
            The episode's obstacle speed.

        Returns
        velocities: np.ndarray (N, 2) float32
        """
        n = len(positions)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        angles = np_random.uniform(0, 2 * np.pi, n)
        return np.column_stack(
            [np.cos(angles) * speed, np.sin(angles) * speed]
        ).astype(np.float32)

    def step(self, np_random, positions, velocities, dt):
        """Advance one timestep and reflect off the boundary.

        Note the original code advanced by the raw velocity; `dt` multiplies it here so
        the same model serves both the legacy `dt=1.0` setting and Week 4's `dt=0.1`.
        """
        positions += velocities * dt
        for i in range(len(positions)):
            for axis in range(2):
                if positions[i, axis] < self.low:
                    positions[i, axis] = self.low
                    velocities[i, axis] *= -1
                elif positions[i, axis] > self.high:
                    positions[i, axis] = self.high
                    velocities[i, axis] *= -1
        return positions, velocities


class SmoothStochasticMotion:
    """Ornstein-Uhlenbeck steering: unpredictable heading, bounded turn rate.

    State per obstacle is (position, heading theta, speed v, angular velocity omega).
    `velocities` stays the public interface -- it is derived from (v, theta) after every
    step -- so the observation, which feeds the nearest obstacles' velocities to the
    policy, needs no change at all.
    """

    name = "smooth_stochastic"

    def __init__(self, world_size, obstacle_radius, *, angular_noise_sigma=0.35,
                 angular_velocity_decay=1.0, max_turn_rate=0.5, boundary_margin=2.0,
                 boundary_steer_gain=2.0, steer_full_error=np.pi / 4,
                 steer_full_depth=0.35,
                 randomize_initial_heading=True, **_ignored):
        """
        Parameters
        world_size, obstacle_radius: float
            Arena geometry, unchanged from the original env.
        angular_noise_sigma: float
            OU driving noise. With decay lambda the stationary std of omega is
            sigma / sqrt(2*lambda); the default 0.35 against lambda 1.0 gives 0.247,
            about half of max_turn_rate, so the clip is approached but rarely hit.
        angular_velocity_decay: float
            OU mean reversion lambda, in 1/time. The correlation time is 1/lambda, so the
            default 1.0 at dt=0.1 correlates the heading over ~10 steps.
        max_turn_rate: float
            omega_max, in rad/time. The turning radius is speed / max_turn_rate, so the
            default 0.5 against speed 1.0 curves at radius 2.0 in a 10-unit arena.
        boundary_margin: float
            Distance from a wall at which the corrective steer begins. Must exceed the
            turning radius or an obstacle cannot turn away in time.
        boundary_steer_gain: float
            Strength of that steer, in units of max_turn_rate at full encroachment.
        steer_full_error: float
            Heading error at which the corrective steer saturates. Smaller means a more
            decisive turn away from the wall, which lets `boundary_margin` stay small
            enough to leave the arena's interior usable.
        steer_full_depth: float
            Fraction of the margin at which the steer reaches full authority. The margin
            must exceed the turning radius for avoidance to be possible at all, and this
            keeps most of it in reserve.
        randomize_initial_heading: bool
            When False the initial heading is zero rather than uniform. Only useful for
            tests that need a fixed starting direction.
        """
        self.world_size = float(world_size)
        self.obstacle_radius = float(obstacle_radius)
        self.sigma = float(angular_noise_sigma)
        self.decay = float(angular_velocity_decay)
        self.max_turn_rate = float(max_turn_rate)
        self.boundary_margin = float(boundary_margin)
        self.boundary_steer_gain = float(boundary_steer_gain)
        self.steer_full_error = float(steer_full_error)
        self.steer_full_depth = float(steer_full_depth)
        self.randomize_initial_heading = bool(randomize_initial_heading)

        self.low = self.obstacle_radius
        self.high = self.world_size - self.obstacle_radius

        #: Non-zero means the safety clamp fired, i.e. the steer failed to turn an
        #: obstacle in time. Asserted zero by the tests.
        self.n_clamped = 0

        self.theta = np.zeros(0, dtype=np.float64)
        self.omega = np.zeros(0, dtype=np.float64)
        self.speed = np.zeros(0, dtype=np.float64)

    def reset(self, np_random, positions, speed):
        """Initialise heading, angular velocity and speed for a new episode."""
        n = len(positions)
        if n == 0:
            self.theta = np.zeros(0)
            self.omega = np.zeros(0)
            self.speed = np.zeros(0)
            return np.zeros((0, 2), dtype=np.float32)

        if self.randomize_initial_heading:
            self.theta = np_random.uniform(0, 2 * np.pi, n)
        else:
            self.theta = np.zeros(n)

        # An obstacle spawned closer to a wall than its turning radius cannot steer away
        # in time if it also happens to start pointing outward, and the safety clamp would
        # fire on the first few steps -- reintroducing exactly the discontinuity this class
        # exists to avoid. Rather than restrict where obstacles may spawn (which would make
        # the randomized env differ from the deterministic one in initial POSITIONS as well
        # as motion, muddying the Exp1-vs-Exp2 comparison), bias only the initial HEADING:
        # inside the margin the obstacle starts aimed into the interior, give or take a
        # right angle, so it is already travelling away before the steer has to work.
        inward = self._inward_direction(positions)
        magnitude = np.linalg.norm(inward, axis=1)
        near = magnitude > 1e-12
        if np.any(near):
            aim = np.arctan2(inward[near, 1], inward[near, 0])
            spread = (np.pi / 2.0) * (1.0 - np.clip(magnitude[near], 0.0, 1.0))
            self.theta[near] = aim + np_random.uniform(-1.0, 1.0, int(near.sum())) * spread
        # Start from the OU stationary distribution rather than from zero, so the first
        # few steps of an episode are not systematically straighter than the rest.
        stationary_std = self.sigma / np.sqrt(2.0 * self.decay) if self.decay > 0 else 0.0
        self.omega = np.clip(
            np_random.normal(0.0, stationary_std, n),
            -self.max_turn_rate, self.max_turn_rate,
        )
        self.speed = np.full(n, float(speed))
        return self._velocities()

    def _velocities(self):
        return np.column_stack(
            [self.speed * np.cos(self.theta), self.speed * np.sin(self.theta)]
        ).astype(np.float32)

    def _inward_direction(self, positions):
        """Depth-weighted inward normal of whichever walls are within the margin.

        Zero-length outside the margin, growing to unit length per axis at the wall. Both
        `reset` (to aim the initial heading) and `_boundary_steer` (to curve the obstacle
        back) read it, so the two cannot drift apart.
        """
        inward = np.zeros((len(positions), 2))
        if self.boundary_margin <= 0.0:
            return inward
        for axis in range(2):
            depth_low = np.clip(
                (self.low + self.boundary_margin - positions[:, axis]) / self.boundary_margin,
                0.0, 1.0)
            depth_high = np.clip(
                (positions[:, axis] - (self.high - self.boundary_margin)) / self.boundary_margin,
                0.0, 1.0)
            inward[:, axis] = depth_low - depth_high
        return inward

    def _boundary_steer(self, positions):
        """Corrective angular velocity that curves an obstacle back into the interior.

        Zero outside the margin. Inside it, the correction scales with encroachment depth
        and acts only on the component of heading pointing OUT of the arena, so an
        obstacle already turning away is not fought.
        """
        n = len(positions)
        if n == 0 or self.boundary_margin <= 0.0:
            return np.zeros(n)

        inward = self._inward_direction(positions)
        magnitude = np.linalg.norm(inward, axis=1)
        steer = np.zeros(n)
        active = magnitude > 1e-12
        if not np.any(active):
            return steer

        desired = np.arctan2(inward[active, 1], inward[active, 0])
        # Signed smallest angle from current heading to the inward direction.
        error = np.arctan2(
            np.sin(desired - self.theta[active]), np.cos(desired - self.theta[active])
        )
        # The response saturates once the heading is more than `steer_full_error` away
        # from safety, rather than scaling with the raw angle. A proportional-to-error
        # steer is far too gentle near a wall: an obstacle heading almost straight at it
        # has a SMALL error against the inward normal only when it is already turning, so
        # the correction arrived late and the margin had to be made huge to compensate --
        # which in a 10-unit arena left almost no interior and made every obstacle loop.
        # Depth saturates at `steer_full_depth` of the way into the margin, so the steer
        # is at full authority with most of the margin still in hand. Ramping it over the
        # whole margin was the other half of the problem: at margin entry the correction
        # was ~0, so the effective margin was far smaller than the configured one.
        steer[active] = (
            self.boundary_steer_gain
            * self.max_turn_rate
            * np.clip(magnitude[active] / self.steer_full_depth, 0.0, 1.0)
            * np.clip(error / self.steer_full_error, -1.0, 1.0)
        )
        return steer

    def step(self, np_random, positions, velocities, dt):
        """Advance one timestep of the OU steering process."""
        n = len(positions)
        if n == 0:
            return positions, velocities

        noise = np_random.normal(0.0, 1.0, n)
        omega = (1.0 - self.decay * dt) * self.omega + self.sigma * np.sqrt(dt) * noise
        omega = omega + self._boundary_steer(positions)
        # The clip is applied AFTER the steer, so the bounded-turn-rate guarantee covers
        # boundary handling too rather than being a property of the free-flight term only.
        self.omega = np.clip(omega, -self.max_turn_rate, self.max_turn_rate)

        self.theta = self.theta + self.omega * dt
        positions[:, 0] += self.speed * np.cos(self.theta) * dt
        positions[:, 1] += self.speed * np.sin(self.theta) * dt

        # Safety net only. A firing clamp means the steer could not turn the obstacle in
        # time, which is a configuration bug -- see the module docstring.
        clamped = np.clip(positions, self.low, self.high)
        if not np.array_equal(clamped, positions):
            self.n_clamped += 1
            positions[:] = clamped

        return positions, self._velocities()


def build_motion_model(config, *, randomize, world_size, obstacle_radius):
    """Select and construct the motion model for an episode.

    Parameters
    config: dict
        The `dynamic_obstacles` section of config.json.
    randomize: bool
        True selects `SmoothStochasticMotion`, False the original `DeterministicMotion`.

    Returns
    DeterministicMotion or SmoothStochasticMotion
    """
    if not randomize:
        return DeterministicMotion(world_size, obstacle_radius)
    return SmoothStochasticMotion(
        world_size,
        obstacle_radius,
        angular_noise_sigma=config.get("angular_noise_sigma", 0.35),
        angular_velocity_decay=config.get("angular_velocity_decay", 1.0),
        max_turn_rate=config.get("max_turn_rate", 0.5),
        boundary_margin=config.get("boundary_margin", 2.0),
        boundary_steer_gain=config.get("boundary_steer_gain", 2.0),
        steer_full_error=config.get("steer_full_error", float(np.pi / 4)),
        steer_full_depth=config.get("steer_full_depth", 0.35),
        randomize_initial_heading=config.get("randomize_initial_heading", True),
    )
