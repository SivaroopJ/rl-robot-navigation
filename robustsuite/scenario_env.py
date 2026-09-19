"""The RS scenario environment: a SUBCLASS of the canonical RobotNavEnv (RS_DESIGN.md section 3).

robot_env/ is frozen (hash manifest), so everything the suite adds is layered on by override, the
pattern of generalization/scenario_env.py.

TWO MODES, chosen per episode by `install(spec)` before `reset()`:
    spec None     the M0 anchor. reset() and step() are the canonical ones, untouched: M0's map,
                  its stratified start/goal sampler and its six dynamic obstacles under the motion
                  model of the motion condition. Bit-identical to exp6.make_env (tested).
    EpisodeSpec   a generated cell. At reset the spec's layout replaces the static obstacles and
                  its start and goal replace the sampled ones. The static condition has no dynamic
                  obstacles (pedestrians arrive with ticket 05, triggers with 06 and 07).

The layout is installed BEFORE super().reset(), so every draw the canonical reset makes (the
stratified sampler, which only ever sees M0's map, and the empty obstacle draw) is a function of
the seed alone. The start and goal the sampler returns are then overwritten by the spec's. So in
a generated cell the env RNG has already made the sampler's draws when the episode starts;
pedestrian noise (ticket 05) continues from there, still a function of the seed alone.

`scenario_events` is the per-episode event log (trigger fired, block added, spawn). It is empty
until the triggered conditions exist. It is measurement for the record and never reaches a policy.

The known static map a policy is built from is `static_obstacles` right after reset, which is the
spec's layout: nothing triggered is in it.
"""
from __future__ import annotations

import numpy as np

from robot_env.robot_nav_env import RobotNavEnv
from robustsuite import seeds as RS

#: exp6.make_env's arguments (the M0 environment of HD Experiment 1), reused for every cell.
N_SLOTS, PED_SPEED = 6, 0.675


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
        self.spec = None
        self.scenario_events = []

    def install(self, spec):
        """The spec for the next reset(); None for the M0 anchor."""
        self.spec = spec

    def reset(self, seed=None, options=None):
        self.scenario_events = []
        if self.spec is None:
            self.static_obstacles = list(self.m0_static)
            self.n_dynamic_obstacles = self._m0_n_dynamic
            return super().reset(seed=seed, options=options)

        self.static_obstacles = [tuple(float(v) for v in r) for r in self.spec.layout]
        self.n_dynamic_obstacles = 0
        _, info = super().reset(seed=seed, options=options)
        self.agent_position = np.asarray(self.spec.start, dtype=np.float32).copy()
        self.target_position = np.asarray(self.spec.goal, dtype=np.float32).copy()
        self.initial_distance = float(np.linalg.norm(self.target_position - self.agent_position))
        self.previous_distance = self.initial_distance
        return self._get_observation(), info
