"""Track B harness: the four factorial arms, on the canonical M0 with stochastic obstacles.

Arms (never pooled across controller families):
    O-off  frozen Original CLF-DR-CBF (reference params), no supervisor
    O-on   the same + SpeedSupervisor
    R-off  frozen Random-CLF-DR-CBF (tuned + frozen random recovery), no supervisor
    R-on   the same + SpeedSupervisor
Wrapper order is recorder -> supervisor -> recovery -> frozen QP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from continuation.harness import StepRecorder, infeasible_runs
from continuation.params import REFERENCE, DRCBFParams
from continuation.policy import attach_recovery, build_arm
from continuation.random_recovery import RandomSpec
from continuation.seeds import controller_rng
from continuation.supervisor.supervisor import SpeedSupervisor
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import episode_record, make_env, true_clearance

ARMS = ("O-off", "O-on", "R-off", "R-on")


@dataclass(frozen=True)
class TrackBArm:
    name: str
    family: str            # "original" | "random"
    supervisor: bool
    params: DRCBFParams = REFERENCE
    random: RandomSpec | None = None

    def describe(self):
        return {"name": self.name, "family": self.family, "supervisor": self.supervisor,
                "params": self.params.as_dict(),
                "random": self.random.as_dict() if self.random else None}


def build_arms(tuned, spec):
    return [TrackBArm("O-off", "original", False),
            TrackBArm("O-on", "original", True),
            TrackBArm("R-off", "random", False, params=tuned, random=spec),
            TrackBArm("R-on", "random", True, params=tuned, random=spec)]


def run_episode(arm: TrackBArm, seed, *, env=None, oracle=None):
    own = env is None
    env = env or make_env(True)
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    if arm.family == "original":
        pol = build_arm("original", env)
    else:
        pol = build_arm("random", env, params=arm.params)
    pol.reset(obs, env.agent_position)
    rec_wrap = attach_recovery("random", pol, params=arm.params, random_spec=arm.random,
                               rng=controller_rng(seed)) if arm.family == "random" else None
    sup = SpeedSupervisor(pol.ctrl, tracker=pol.src.tracker) if arm.supervisor else None
    recd = StepRecorder(pol.ctrl, pol.ctrl.rateh)

    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        t0 = time.perf_counter()
        action, _ = pol.predict(obs, env.agent_position, deterministic=True)
        times.append(time.perf_counter() - t0)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
    ep_time = time.perf_counter() - t_ep
    recd.detach()
    if sup is not None:
        sup.detach()
    if rec_wrap is not None:
        rec_wrap.detach()
    if own:
        env.close()

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    S = recd.steps
    inf = [s["infeasible"] for s in S]
    runs = infeasible_runs(inf)
    t = np.asarray(times) * 1e3
    rec = episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info,
                         extra={"clearances": clear, "step_times": times,
                                "planner_failed": int(pol.planner_failed),
                                "n_infeasible": pol.n_infeasible,
                                "n_solver_fail": pol.n_solver_fail,
                                "collided_after_infeasible": 0,
                                "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
                                "min_cbc_solved": float("nan")})
    rec.update({
        "arm": arm.name, "family": arm.family, "supervisor": arm.supervisor,
        "outcome": outcome, "start": [float(x) for x in start],
        "goal_xy": [float(x) for x in goal],
        "frac_infeasible": sum(inf) / max(steps, 1),
        "n_infeasible_runs": len(runs),
        "infeasible_run_len_mean": float(np.mean([d for _, d in runs])) if runs else 0.0,
        "infeasible_run_len_max": int(max((d for _, d in runs), default=0)),
        "n_fallback": int(sum(1 for s in S if not s["optimal"])),
        "n_recovery_events": len(rec_wrap.events) if rec_wrap is not None else 0,
        "recovery_fraction": ((sum(1 for r in rec_wrap.records if r["mode"] in ("event", "persist"))
                               / max(steps, 1)) if rec_wrap is not None else 0.0),
        "step_ms_mean": float(t.mean()), "step_ms_p95": float(np.percentile(t, 95)),
        "episode_time_s": ep_time,
    })
    if sup is not None:
        s = sup.summary()
        rec.update({f"sup_{k}": v for k, v in s.items()})
        # a trigger is "followed by restored feasibility" if no infeasible step occurs while engaged
        eng = [r for r in sup.records if r["state"] == "ENGAGED"]
        rec["sup_engaged_steps"] = len(eng)
        rec["sup_engaged_fraction"] = len(eng) / max(steps, 1)
        rec["sup_unnecessary_fraction"] = (float(np.mean([r["risk"] < 0.5 for r in eng]))
                                           if eng else float("nan"))
    else:
        rec.update({"sup_engaged_steps": 0, "sup_engaged_fraction": 0.0,
                    "sup_unnecessary_fraction": float("nan")})
    return rec
