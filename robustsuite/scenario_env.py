"""The RS scenario environment: a SUBCLASS of the canonical RobotNavEnv (RS_DESIGN.md section 3).

robot_env/ is frozen (hash manifest), so everything the suite adds is layered on by override, the
pattern of generalization/scenario_env.py.

TWO MODES, chosen per episode by `install(spec)` before `reset()`:
    spec None     the M0 anchor. reset() and step() are the canonical ones, untouched: M0's map,
                  its stratified start/goal sampler and its six dynamic obstacles under the motion
                  model of the motion condition. Bit-identical to exp6.make_env (tested).
    EpisodeSpec   a generated cell. At reset the spec's layout replaces the static obstacles, its
                  start and goal replace the sampled ones, and its active pedestrians (none in the
                  static condition) become the dynamic obstacles, driven by
                  robustsuite.pedestrians.PedestrianMotion for this env's motion condition on
                  the spec's `pedestrian_layout` (the layout plus the reserved block zone, in
                  every condition). In `trigger_block` the block is armed, in `trigger_spawn`
                  the spawn (below).

The layout is installed BEFORE super().reset(), so every draw the canonical reset makes (the
stratified sampler, which only ever sees M0's map, and the empty obstacle draw) is a function of
the seed alone. The start and goal the sampler returns are then overwritten by the spec's. So in
a generated cell the env RNG has already made the sampler's draws when the episode starts;
randomized pedestrian noise continues from there, still a function of the seed alone.

THE TRIGGER (RS_DESIGN 4.6). It is checked once per step, after the robot has moved and before
the pedestrians move, the collision check and the observation: the frozen step() calls the motion
model at exactly that point, so the pedestrian model is wrapped by `_ScenarioMotion`. It fires on
the first step on which the robot's centre is inside the trigger disc, at most once.
    trigger_block  the block is added to `static_obstacles`, so the collision check and the LiDAR
                   of that same step already see it. The robot cannot be touching it then (the
                   generator keeps it >= 0.8 m from the trigger centre).
    trigger_spawn  the spawned pedestrian appears at its start and takes its first step at once,
                   as the last obstacle slot: `obstacle_positions` and `n_dynamic_obstacles` grow
                   by one, so it is in the collision check, the LiDAR and the observation from
                   that step on, and in none of them before. It walks `spec.spawn.route` on the
                   static layout (it ignores the block zone, RS_DESIGN 14.3). Its randomized noise
                   comes from its OWN generator, SeedSequence([episode seed, SPAWN_STREAM]),
                   never from the env RNG: the firing step depends on the robot, so drawing from
                   the env RNG would make the regular pedestrians' later noise depend on the robot
                   (researcher decision, 14.3). The regular pedestrians stay exactly those of the
                   family's dynamic cell.

The block arrives as a NEW `static_obstacles` list, never an in-place append, so a policy that
kept a reference to the list it was built from still holds the layout alone. The known static map
a policy is built from is `static_obstacles` right after reset, which is the spec's layout:
nothing triggered is in it.

`scenario_events` is the per-episode event log: "trigger_fired" (step, robot position), then
"block_added" (step, rectangle) or "spawned" (step, start, variant). It is measurement for the
record and never reaches a policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_env.robot_nav_env import RobotNavEnv
from robustsuite import seeds as RS
from robustsuite.pedestrians import PedestrianMotion
from robustsuite.scenario import N_SLOTS

#: exp6.make_env's arguments (the M0 environment of HD Experiment 1), reused for every cell; the
#: six obstacle slots are the generator's N_SLOTS.
PED_SPEED = 0.675
#: The spawned pedestrian's generator is SeedSequence([episode seed, SPAWN_STREAM]).
SPAWN_STREAM = 1


@dataclass
class _Spawned:
    """The fired spawn: its own motion model and generator, and its one-row state arrays."""
    model: PedestrianMotion
    rng: np.random.Generator
    positions: np.ndarray
    velocities: np.ndarray


class _ScenarioMotion:
    """The regular pedestrians' model, with the env's trigger check run first on every step and,
    once a spawn has fired, the spawned pedestrian stepped after them as the last slot."""

    def __init__(self, model, check):
        self.model, self.check = model, check
        self.name = model.name
        self.spawned = None             # a _Spawned once the spawn has fired

    def reset(self, np_random, positions, speed):
        return self.model.reset(np_random, positions, speed)

    def add_spawn(self, model, rng, speed):
        pos = np.zeros((1, 2), dtype=np.float32)
        self.spawned = _Spawned(model, rng, pos, model.reset(rng, pos, speed))

    def step(self, np_random, positions, velocities, dt):
        self.check()
        n = len(self.model.routes)
        pos, vel = self.model.step(np_random, positions[:n], velocities[:n], dt)
        if self.spawned is None:
            return pos, vel
        sw = self.spawned
        sw.positions, sw.velocities = sw.model.step(sw.rng, sw.positions, sw.velocities, dt)
        return (np.concatenate([pos, sw.positions]).astype(np.float32),
                np.concatenate([vel, sw.velocities]).astype(np.float32))


class RSScenarioEnv(RobotNavEnv):
    """RobotNavEnv plus an installed EpisodeSpec. With no spec installed it is M0, unchanged."""

    def __init__(self, motion, config_path="config.json"):
        RS.check_motion(motion)
        super().__init__(config_path=config_path, n_dynamic_obstacles=N_SLOTS,
                         obstacle_speed=PED_SPEED, render_mode=None, use_reward_shaping=True,
                         randomize_dynamic_obstacles=(motion == "randomized"))
        self.motion = motion
        #: M0's static map, restored for every anchor episode.
        self.m0_static = [tuple(o) for o in self.static_obstacles]
        self._m0_n_dynamic = self.n_dynamic_obstacles
        self._m0_motion = self.motion_model
        self.spec = None
        self.scenario_events = []
        self._armed = False             # a trigger still to fire this episode
        self._seed = None               # the episode seed, for the spawn's own generator

    def install(self, spec):
        """The spec for the next reset(); None for the M0 anchor."""
        self.spec = spec

    @property
    def trigger_step(self):
        """The step on which the trigger fired this episode, or None."""
        return next((e["step"] for e in self.scenario_events if e["event"] == "trigger_fired"),
                    None)

    def reset(self, seed=None, options=None):
        self.scenario_events = []
        self._armed = False
        self._seed = seed
        if self.spec is None:
            self.static_obstacles = list(self.m0_static)
            self.n_dynamic_obstacles = self._m0_n_dynamic
            self.motion_model = self._m0_motion
            return super().reset(seed=seed, options=options)

        self.static_obstacles = [tuple(float(v) for v in r) for r in self.spec.layout]
        self.n_dynamic_obstacles = 0
        # an empty model for the canonical reset: it draws nothing, so the previous episode's
        # model can never touch this episode's RNG stream
        self.motion_model = PedestrianMotion((), self.spec.layout, randomized=False)
        _, info = super().reset(seed=seed, options=options)
        self.agent_position = np.asarray(self.spec.start, dtype=np.float32).copy()
        self.target_position = np.asarray(self.spec.goal, dtype=np.float32).copy()
        peds = self.spec.active_pedestrians
        self.motion_model = _ScenarioMotion(
            PedestrianMotion(peds, self.spec.pedestrian_layout,
                             randomized=self.motion == "randomized", world_size=self.WORLD_SIZE),
            self._check_trigger)
        self._armed = (self.spec.active_block is not None
                       or self.spec.active_spawn is not None)
        self.n_dynamic_obstacles = len(peds)
        self.obstacle_positions = np.zeros((len(peds), 2), dtype=np.float32)
        self.obstacle_velocities = self.motion_model.reset(
            self.np_random, self.obstacle_positions, self._episode_speed)
        self.initial_distance = float(np.linalg.norm(self.target_position - self.agent_position))
        self.previous_distance = self.initial_distance
        return self._get_observation(), info

    def _check_trigger(self):
        if not self._armed:
            return
        trig = self.spec.trigger
        if np.linalg.norm(self.agent_position - np.asarray(trig.centre)) > trig.radius:
            return
        self._armed = False
        step = int(self.step_count)
        self.scenario_events.append({"event": "trigger_fired", "step": step,
                                     "position": [float(v) for v in self.agent_position]})
        if self.spec.active_block is not None:
            block = tuple(float(v) for v in self.spec.active_block)
            self.static_obstacles = self.static_obstacles + [block]
            self.scenario_events.append({"event": "block_added", "step": step,
                                         "block": list(block)})
        spawn = self.spec.active_spawn
        if spawn is not None:
            entropy = None if self._seed is None else [int(self._seed), SPAWN_STREAM]
            model = PedestrianMotion([spawn.route], self.spec.layout,
                                     randomized=self.motion == "randomized",
                                     world_size=self.WORLD_SIZE)
            rng = np.random.default_rng(np.random.SeedSequence(entropy))
            self.motion_model.add_spawn(model, rng, self._episode_speed)
            self.n_dynamic_obstacles += 1
            self.scenario_events.append({"event": "spawned", "step": step,
                                         "position": [float(v) for v in spawn.start],
                                         "variant": spawn.variant})
