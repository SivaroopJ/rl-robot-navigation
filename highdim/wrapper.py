"""Seam 2: the PPO environment wrapper for HD Experiment 1 (HD_DESIGN.md sections 4, 9).

The safety filter is part of the environment PPO trains in. PPO's action is the SUBGOAL action;
the executed velocity is produced inside step() by the frozen Random-CLF-DR-CBF and never leaves
the wrapper except as a diagnostic.

    observation  obs[0:28] of the canonical M0 env (goal offset, own velocity, 24 LiDAR rays);
                 the ground-truth obstacle block obs[28:52] is dropped
    action       Box(-1, 1, 2): the subgoal action, mapped by highdim.subgoal to gamma
    reset        the env (its first reset uses the wrapper's HD_TRAIN seed, later resets continue
                 its own np_random), then a FRESH ppo_random policy (planning off, empty map)
                 with Random recovery; the controller RNG is drawn from the wrapper's own
                 generator (self.np_random, seeded by the same first reset)
    step         set the subgoal, run the frozen predict with the full observation and the env's
                 ego position, step the env with the resulting velocity. With hold k > 1 the
                 decision is held for k control steps (gamma frozen in world coordinates, the QP
                 runs every step) and the k rewards are summed. Reward and termination are the
                 env's own, on the executed velocity
    info         "diagnostics": one record per control step (DIAGNOSTIC_KEYS)

Only the filtered PPO arm exists here; the filter-bypass arm (ppo_unfiltered) is evaluation-only
and cannot be built by this wrapper. The solver is "frozen" unless the trainer passes the solver
selected by the fast-solver check (highdim.train).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from continuation.harness import StepRecorder
from experiments.exp6_ppo_comparison import make_env
from highdim import policy as HP
from highdim.harness import TraceRecorder
from highdim.ppo import action_space, observation_space
from highdim.subgoal import check_hold, policy_obs

DIAGNOSTIC_KEYS = ("p", "gamma", "V", "u_nom", "u_exec", "u_dev", "qp_status", "random_event",
                   "min_cbc", "a_raw", "a_disc", "reward")


class HDSubgoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, condition, *, seed, solver="frozen", hold=1):
        if condition not in ("fixed", "randomized"):
            raise ValueError(condition)
        if solver not in HP.SOLVERS:
            raise ValueError(solver)
        self.condition, self.solver, self.hold = condition, solver, check_hold(hold)
        self.env = make_env(condition == "randomized")
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.params, self.spec = HP.frozen_params()
        self._first_seed = int(seed)
        self._seeded = False
        self.policy = self.subgoal = self.recovery = None
        self._obs = None
        self._recorders = ()

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._seeded:
            seed = self._first_seed
        super().reset(seed=seed)
        self._seeded = True
        for r in self._recorders:
            r.detach()
        obs, info = self.env.reset(seed=seed)
        pol = HP.build_hd_arm("ppo_random", self.env, params=self.params, solver=self.solver)
        pol.reset(obs, self.env.agent_position)
        self.subgoal = HP.attach_subgoal(pol)
        rng = np.random.default_rng(int(self.np_random.integers(2 ** 63)))
        self.recovery = HP.attach_random(pol, params=self.params, spec=self.spec, rng=rng)
        self._steprec = StepRecorder(pol.ctrl, pol.ctrl.rateh)
        self._trace = TraceRecorder(pol.ctrl)
        self._recorders = (self._trace, self._steprec, self.recovery)
        self.policy, self._obs = pol, obs
        return policy_obs(obs), info

    def step(self, action):
        total, diags = 0.0, []
        term = trunc = False
        info = {}
        for k in range(self.hold):
            p = self.env.agent_position.copy()
            if k == 0:
                self.subgoal.decide(action, p, self.hold)
            act, _ = self.policy.predict(self._obs, self.env.agent_position, deterministic=True)
            self._obs, r, term, trunc, info = self.env.step(act)
            total += float(r)
            diags.append(self._diagnostic(p, float(r)))
            if term or trunc:
                break
        info = dict(info)
        info["diagnostics"] = diags
        return policy_obs(self._obs), total, term, trunc, info

    def _diagnostic(self, p, reward):
        t, m = self._trace.steps[-1], self._steprec.steps[-1]
        nom = t["u_nom"]
        return {"p": [float(x) for x in p], **t, "min_cbc": m["m"],
                "u_dev": (None if nom is None else
                          float(np.linalg.norm(np.subtract(t["u_exec"], nom)))),
                "a_raw": [float(x) for x in self.subgoal.a_raw],
                "a_disc": [float(x) for x in self.subgoal.a_disc], "reward": reward}

    def close(self):
        self.env.close()
