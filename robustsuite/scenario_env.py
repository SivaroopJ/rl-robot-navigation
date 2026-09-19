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
                  every condition). In `trigger_block` the block is armed (below).

The layout is installed BEFORE super().reset(), so every draw the canonical reset makes (the
stratified sampler, which only ever sees M0's map, and the empty obstacle draw) is a function of
the seed alone. The start and goal the sampler returns are then overwritten by the spec's. So in
a generated cell the env RNG has already made the sampler's draws when the episode starts;
randomized pedestrian noise continues from there, still a function of the seed alone.

THE TRIGGER (RS_DESIGN 4.6). It is checked once per step, after the robot has moved and before
the pedestrians move, the collision check and the observation: the frozen step() calls the motion
model at exactly that point, so the pedestrian model is wrapped by `_Triggered`. On the first step
on which the robot's centre is inside the trigger disc, the block is added to `static_obstacles`,
so the collision check and the LiDAR of that same step already see it. The robot cannot be
touching it then (the generator keeps it >= 0.8 m from the trigger centre).

The block arrives as a NEW `static_obstacles` list, never an in-place append, so a policy that
kept a reference to the list it was built from still holds the layout alone. The known static map
a policy is built from is `static_obstacles` right after reset, which is the spec's layout:
nothing triggered is in it.

`scenario_events` is the per-episode event log: "trigger_fired" (step, robot position) and
"block_added" (step, rectangle). It is measurement for the record and never reaches a policy.
"""
from __future__ import annotations

import numpy as np

from robot_env.robot_nav_env import RobotNavEnv
from robustsuite import seeds as RS
from robustsuite.pedestrians import PedestrianMotion

#: exp6.make_env's arguments (the M0 environment of HD Experiment 1), reused for every cell.
N_SLOTS, PED_SPEED = 6, 0.675


class _Triggered:
    """The pedestrian model, with the env's trigger check run first on every step."""

    def __init__(self, model, check):
        self.model, self.check = model, check
        self.name = model.name

    def reset(self, np_random, positions, speed):
        return self.model.reset(np_random, positions, speed)

    def step(self, np_random, positions, velocities, dt):
        self.check()
        return self.model.step(np_random, positions, velocities, dt)


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
        self._armed = None              # the block still to fire this episode, if any

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
        self._armed = None
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
        self.motion_model = _Triggered(
            PedestrianMotion(peds, self.spec.pedestrian_layout,
                             randomized=self.motion == "randomized", world_size=self.WORLD_SIZE),
            self._check_trigger)
        self._armed = self.spec.active_block
        self.n_dynamic_obstacles = len(peds)
        self.obstacle_positions = np.zeros((len(peds), 2), dtype=np.float32)
        self.obstacle_velocities = self.motion_model.reset(
            self.np_random, self.obstacle_positions, self._episode_speed)
        self.initial_distance = float(np.linalg.norm(self.target_position - self.agent_position))
        self.previous_distance = self.initial_distance
        return self._get_observation(), info

    def _check_trigger(self):
        if self._armed is None:
            return
        trig = self.spec.trigger
        if np.linalg.norm(self.agent_position - np.asarray(trig.centre)) > trig.radius:
            return
        block, self._armed = tuple(float(v) for v in self._armed), None
        self.static_obstacles = self.static_obstacles + [block]
        self.scenario_events += [
            {"event": "trigger_fired", "step": int(self.step_count),
             "position": [float(v) for v in self.agent_position]},
            {"event": "block_added", "step": int(self.step_count), "block": list(block)}]
