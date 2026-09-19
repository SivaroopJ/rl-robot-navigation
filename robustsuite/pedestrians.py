"""Pedestrians: routes drawn by the generator, and the motion model the scenario env runs.

ROUTES (RS_DESIGN.md 4.4). Each pedestrian has a start and a cycle of 8 waypoints, all uniform in
free space at clearance R_C (0.45 m). Its legs are grid paths on the frozen oracle's occupancy at
that inflation, simplified to line segments that keep the clearance. The route is one polyline:
the entry leg start -> w1, then the loop w1 -> ... -> w8 -> w1, walked in order and repeated.
Starts are >= 1.5 m from the robot start and >= 1.0 m from the goal. Routes are part of the
episode spec, so they are fixed by the seed and identical for every arm and both motions.

MOTION (4.4). Same interface as the frozen motion models: reset(np_random, positions, speed) and
step(np_random, positions, velocities, dt), mutating `positions` in place.
    fixed       the pedestrian walks its polyline at 0.675 m/s exactly. No RNG draw.
    randomized  the heading is perturbed by M0's randomized OU process (the frozen
                `SmoothStochasticMotion` update of the angular velocity: noise sigma 0.35, decay
                0.8, clipped at the max turn rate 0.7 rad/s), plus steering towards the route with
                gain 4.0 / s on the heading error to the route point LOOKAHEAD (1 m) ahead of the
                pedestrian's progress. The turn-rate clip bounds the perturbation, not the
                steering (ticket 05's reading of RS_DESIGN 4.4; clipping both left pedestrians
                unable to follow their routes round corners).
                A noisy step that would end within 0.3 m of a static obstacle or of the outer wall
                is replaced by the exact leg step: one step towards the route point one step
                ahead, else one step back towards the pedestrian's own route point; if neither is
                clear it holds for the step. Progress is the pedestrian's projection onto its
                route, searched over the 1.5 m ahead of the current progress (never backwards).
                One normal draw per pedestrian per step from the env's RNG, independent of the
                robot.
Pedestrians pass through each other and ignore the robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsuite.geometry import WORLD, arc_lengths, clearance, points_at

SPEED = 0.675                  # M0's obstacle speed, below the robot's 1 m/s
RADIUS = 0.3
N_WAYPOINTS = 8
START_FROM_ROBOT, START_FROM_GOAL = 1.5, 1.0
#: Waypoint-cycle draws per pedestrian before the family draw fails.
ROUTE_TRIES = 20
#: Candidate points per waypoint draw (rejection sampling in free space).
POINT_TRIES = 2000

# randomized motion: M0's randomized OU values (config.json dynamic_obstacles)
OU_SIGMA, OU_DECAY, MAX_TURN = 0.35, 0.8, 0.7
STEER_GAIN = 4.0              # 1/s, on the heading error to the route point LOOKAHEAD ahead
#: How far ahead along its route a randomized pedestrian steers (the robot's carrot is 1 m too).
LOOKAHEAD = 1.0
#: A noisy step must keep the pedestrian's centre this far from obstacles and the outer wall.
STEP_CLEARANCE = RADIUS


@dataclass(frozen=True)
class Route:
    start: tuple
    waypoints: tuple            # the 8-waypoint cycle
    entry: tuple                # polyline start -> w1
    loop: tuple                 # polyline w1 -> ... -> w8 -> w1
    speed: float = SPEED

    def as_dict(self):
        return {"start": list(self.start), "waypoints": [list(w) for w in self.waypoints],
                "entry": [list(p) for p in self.entry], "loop": [list(p) for p in self.loop],
                "speed": self.speed}


def _free_point(rng, layout, radius, keep_away=()):
    for _ in range(POINT_TRIES):
        q = rng.uniform(radius, WORLD - radius, size=2)
        if clearance(q, layout) >= radius and all(
                np.linalg.norm(q - np.asarray(c)) >= d for c, d in keep_away):
            return tuple(float(v) for v in q)
    return None


def draw_route(rng, layout, grid, radius, robot_start, goal):
    """One pedestrian route, or None if ROUTE_TRIES waypoint cycles all fail."""
    start = _free_point(rng, layout, radius,
                        ((robot_start, START_FROM_ROBOT), (goal, START_FROM_GOAL)))
    if start is None:
        return None
    for _ in range(ROUTE_TRIES):
        wps = [_free_point(rng, layout, radius) for _ in range(N_WAYPOINTS)]
        if any(w is None for w in wps):
            continue
        entry = grid.route(start, wps[0], layout, radius)
        legs = [grid.route(wps[i], wps[(i + 1) % N_WAYPOINTS], layout, radius)
                for i in range(N_WAYPOINTS)]
        if entry is None or any(leg is None for leg in legs):
            continue
        entry = entry[0]
        loop = list(legs[0][0])
        for leg, _ in legs[1:]:
            loop += leg[1:]
        return Route(start, tuple(wps), tuple(entry), tuple(loop))
    return None


class _Track:
    """Arc-length walk along a route: the entry once, then the loop forever. Vectorised."""

    def __init__(self, route):
        self.entry = np.asarray(route.entry, dtype=float)
        self.loop = np.asarray(route.loop, dtype=float)
        self.c_entry, self.c_loop = arc_lengths(self.entry), arc_lengths(self.loop)
        self.l_entry, self.l_loop = float(self.c_entry[-1]), float(self.c_loop[-1])

    def points(self, s):
        """Route points at arc lengths s (array)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        on_entry = s <= self.l_entry
        out = np.empty((len(s), 2))
        if on_entry.any():
            out[on_entry] = points_at(self.entry, s[on_entry], self.c_entry)
        if (~on_entry).any():
            u = (s[~on_entry] - self.l_entry) % self.l_loop
            out[~on_entry] = points_at(self.loop, u, self.c_loop)
        return out

    def point(self, s):
        return self.points([s])[0]

    def tangent(self, s, h=1e-3):
        d = self.point(s + h) - self.point(s)
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else np.zeros(2)


#: Progress is re-projected onto the route over this window ahead of the current progress
#: (metres, never backwards), at PROJECTION_STEP resolution.
PROJECTION_WINDOW, PROJECTION_STEP = 1.5, 0.01
_AHEAD = np.arange(0.0, PROJECTION_WINDOW + PROJECTION_STEP / 2, PROJECTION_STEP)


class PedestrianMotion:
    """The pedestrian motion model for one episode's routes (see the module docstring)."""

    name = "pedestrians"

    def __init__(self, routes, layout, *, randomized, world_size=WORLD):
        self.routes = list(routes)
        self.layout = [tuple(r) for r in layout]
        self.randomized = bool(randomized)
        self.world = float(world_size)
        self.tracks = [_Track(r) for r in self.routes]
        self.s = np.zeros(len(self.routes))
        self.theta = np.zeros(len(self.routes))
        self.omega = np.zeros(len(self.routes))
        #: each pedestrian walks at its route's speed (0.675 m/s, RS_DESIGN 4.4)
        self.speeds = np.array([r.speed for r in self.routes])
        self.n_fallback = 0             # noisy steps replaced by a step back towards the route
        self.n_hold = 0                 # steps held because no fallback step was clear either

    def reset(self, np_random, positions, speed):
        """Place every pedestrian at its route start, heading along the route. `speed` (the
        env's episode speed) is accepted for the frozen interface; each route carries its own.
        Randomized: the OU angular velocity starts from its stationary distribution, as M0's
        does (one normal draw per pedestrian)."""
        n = len(self.routes)
        self.s[:] = 0.0
        self.n_fallback = self.n_hold = 0
        self.omega = np.zeros(n)
        if self.randomized and n:
            std = OU_SIGMA / np.sqrt(2.0 * OU_DECAY)
            self.omega = np.clip(np_random.normal(0.0, std, n), -MAX_TURN, MAX_TURN)
        vel = np.zeros((n, 2), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            positions[i] = tr.point(0.0)
            t = tr.tangent(0.0)
            self.theta[i] = float(np.arctan2(t[1], t[0]))
            vel[i] = self.speeds[i] * t
        return vel

    def _clear(self, q):
        return clearance(q, self.layout, self.world) >= STEP_CLEARANCE

    @staticmethod
    def _towards(p, target, step):
        d = target - p
        dist = float(np.linalg.norm(d))
        return p.copy() if dist == 0 else p + d * min(1.0, step / dist)

    def step(self, np_random, positions, velocities, dt):
        n = len(self.routes)
        if n == 0:
            return positions, velocities
        if not self.randomized:
            for i, tr in enumerate(self.tracks):
                self.s[i] += self.speeds[i] * dt
                q = tr.point(self.s[i])
                velocities[i] = (q - positions[i]) / dt
                positions[i] = q
            return positions, velocities

        noise = np_random.normal(0.0, 1.0, n)
        for i, tr in enumerate(self.tracks):
            step = self.speeds[i] * dt
            p = positions[i].astype(float)
            here, target, carrot = tr.points([self.s[i], self.s[i] + step,
                                              self.s[i] + LOOKAHEAD])
            want = float(np.arctan2(carrot[1] - p[1], carrot[0] - p[0]))
            err = float(np.arctan2(np.sin(want - self.theta[i]), np.cos(want - self.theta[i])))
            omega = (1.0 - OU_DECAY * dt) * self.omega[i] + OU_SIGMA * np.sqrt(dt) * noise[i]
            self.omega[i] = float(np.clip(omega, -MAX_TURN, MAX_TURN))
            self.theta[i] += (self.omega[i] + STEER_GAIN * err) * dt
            q = p + step * np.array([np.cos(self.theta[i]), np.sin(self.theta[i])])
            if not self._clear(q):
                # the exact leg step: towards the route point one step ahead, else back towards
                # the pedestrian's own route point; hold only if neither is clear
                self.n_fallback += 1
                for aim in (target, here):
                    q = self._towards(p, aim, step)
                    if self._clear(q):
                        break
                else:
                    self.n_hold += 1
                    q = p.copy()
                d = q - p
                if np.linalg.norm(d) > 0:
                    self.theta[i] = float(np.arctan2(d[1], d[0]))
            ahead = self.s[i] + _AHEAD
            self.s[i] = float(ahead[np.argmin(np.linalg.norm(tr.points(ahead) - q, axis=1))])
            velocities[i] = (q - p) / dt
            positions[i] = q
        return positions, velocities
