"""Paired episode harness for the Week 6 Continuation, on the canonical Phase-6 M0 environment.

Environment, metrics and headline statistics are exp6's OWN functions, imported unchanged:
make_env, episode_record, summarise, true_clearance (and mcnemar / paired_ci in stats.py). The
per-episode loop is exp6.run_drcbf / exp8.run_updated_drcbf's loop, with the policy supplied by
`continuation.policy` and a metrics-only StepRecorder wrapped outermost around the controller.

PAIRING. Every arm runs each (condition, seed) from `env.reset(seed=...)`. The obstacle motion
model reads only env.np_random and obstacle state, never the agent (audit A9), so every arm
faces the same start, goal, obstacle spawn and obstacle trajectory. The random arm's own RNG is
`seeds.controller_rng(seed)`, independent of env.np_random. `run_block` asserts identical
start/goal across arms for every (condition, seed).

GROUND TRUTH is used ONLY by true_clearance and the env's own outcome flags, both of which are
measurement, never fed to any controller.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import episode_record, make_env, true_clearance

from continuation.params import REFERENCE, DRCBFParams
from continuation.policy import attach_recovery, build_arm
from continuation.random_recovery import RandomSpec, margins
from continuation.seeds import controller_rng

CONDITIONS = ("fixed", "randomized")
MAP_ID = "M0_canonical(config.json, 6 dyn obstacles @0.675, 5 rects, max_steps 500)"
COLLISION_WINDOW = 10          # steps (1 s): event-level "collision after infeasibility"
RESULTS = Path("results/week6_continuation")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                                  # original | c0b | tuned | ladder | random
    params: DRCBFParams = REFERENCE
    random: RandomSpec | None = None

    def tau(self):
        return self.params.tau_eff()

    def describe(self):
        return {"name": self.name, "kind": self.kind, "params": self.params.as_dict(),
                "t1_cap": 0.96 if self.kind == "c0b" else None,
                "recovery": {"original": "none (frozen u=0)", "tuned": "none (frozen u=0)",
                             "c0b": "LADDER (historical)", "ladder": "LADDER",
                             "random": "random"}[self.kind],
                "random": self.random.as_dict() if self.random else None}


# --------------------------------------------------------------------------- recorder
class StepRecorder:
    """Metrics-only outermost wrapper. Never changes the action it passes through."""

    def __init__(self, ctrl, alpha):
        self.ctrl = ctrl
        self.alpha = float(alpha)
        self.steps = []
        self._inner = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._inner

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        t0 = time.perf_counter()
        u = self._inner(p, gamma, xi, u_nom=u_nom, record=rec)
        call = time.perf_counter() - t0
        kept = np.atleast_2d(np.asarray(rec.get("xi_kept", xi), float))
        status = str(rec.get("status"))
        uu = np.asarray(u, float).reshape(2)
        m_exec, m_u0 = margins(kept, np.vstack([uu, np.zeros(2)]), self.alpha)
        qp_t = rec.get("total_time")
        self.steps.append({
            "status": status, "infeasible": "infeasible" in status, "optimal": status == "optimal",
            "mode": rec.get("mode", rec.get("rung", "qp")),
            "recovered": bool(rec.get("recovered", False)) or rec.get("mode") in ("event", "persist"),
            "m": float(m_exec), "m_u0": float(m_u0), "h_crit": float(kept[0, 1]),
            "delta": rec.get("delta"), "qp_time": float(qp_t) if qp_t == qp_t and qp_t is not None else float("nan"),
            "call_time": call, "t_recovery": float(rec.get("t_recovery", 0.0)),
            "u_norm": float(np.linalg.norm(uu)),
        })
        return u


def tier_of(m, tau):
    return 0 if m >= tau else (1 if m >= 0.0 else 2)


def infeasible_runs(flags):
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return runs


# --------------------------------------------------------------------------- one episode
def run_episode(arm: ArmSpec, condition, seed, *, env=None, oracle=None):
    own_env = env is None
    env = env if env is not None else make_env(condition == "randomized")
    oracle = oracle or ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    t_ep = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = build_arm(arm.kind, env, params=arm.params)
    pol.reset(obs, env.agent_position)
    rng = controller_rng(seed) if arm.kind == "random" else None
    wrap = attach_recovery(arm.kind, pol, params=arm.params, random_spec=arm.random, rng=rng)
    alpha = pol.ctrl.rateh
    recd = StepRecorder(pol.ctrl, alpha)

    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    infeasible_before_end = False
    while not (term or trunc):
        t0 = time.perf_counter()
        action, _ = pol.predict(obs, env.agent_position, deterministic=True)
        times.append(time.perf_counter() - t0)
        infeasible_before_end = pol.last_step_infeasible
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
    ep_time = time.perf_counter() - t_ep
    recd.detach()
    if wrap is not None:
        wrap.detach()
    if own_env:
        env.close()

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    extra = {"clearances": clear, "step_times": times,
             "planner_failed": int(pol.planner_failed),
             "n_infeasible": pol.n_infeasible, "n_solver_fail": pol.n_solver_fail,
             "collided_after_infeasible": int(outcome == "collision" and infeasible_before_end),
             "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
             "min_cbc_solved": (float(pol.min_cbc_solved)
                                if np.isfinite(pol.min_cbc_solved) else float("nan"))}
    rec = episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info, extra=extra)
    rec.update(continuation_metrics(arm, recd.steps, wrap, outcome, steps, times, ep_time))
    rec.update({"arm": arm.name, "kind": arm.kind, "condition": condition,
                "start": [float(x) for x in start], "goal": [float(x) for x in goal]})
    return rec


def continuation_metrics(arm, S, wrap, outcome, steps, times, ep_time):
    tau = arm.tau() if arm.kind != "c0b" else 0.04
    H = arm.random.H if arm.kind == "random" else 1
    inf = [s["infeasible"] for s in S]
    runs = infeasible_runs(inf)
    coll_step = steps - 1 if outcome == "collision" else None     # action index that hit
    # triggers: steps where the recovery mechanism (or the u=0 fallback) takes over
    if arm.kind == "random":
        triggers = [e["t"] for e in wrap.events]
    else:
        triggers = [t for t, f in enumerate(inf) if f]
    ok = fail = 0
    for t in triggers:
        end = t + H                                    # first step after the window
        if coll_step is not None and t <= coll_step < end:
            fail += 1
        elif end < len(S):
            ok, fail = (ok + 1, fail) if S[end]["optimal"] else (ok, fail + 1)
        elif outcome == "success":
            ok += 1
        # else: episode ended by timeout inside the window -> undetermined, not counted
    ev_coll = sum(1 for (s0, _) in runs
                  if coll_step is not None and s0 <= coll_step < s0 + COLLISION_WINDOW)
    inf_steps = [s for s in S if s["infeasible"]]
    tiers = Counter(tier_of(s["m"], tau) for s in inf_steps)
    rec_steps = [s for s in S if s["recovered"]]
    deltas = [s["delta"] for s in S if s["optimal"] and s["delta"] == s["delta"]
              and s["delta"] is not None]
    qp_t = [s["qp_time"] for s in S if s["qp_time"] == s["qp_time"]]
    extra_t = [s["call_time"] - s["qp_time"] for s in inf_steps if s["qp_time"] == s["qp_time"]]
    t = np.asarray(times) * 1e3
    out = {
        "tau_eff": tau, "frac_infeasible": sum(inf) / max(steps, 1),
        "n_infeasible_events": len(runs),
        "infeasible_duration_mean": float(np.mean([d for _, d in runs])) if runs else float("nan"),
        "infeasible_duration_max": int(max((d for _, d in runs), default=0)),
        "n_recovery_steps": len(rec_steps), "recovery_fraction": len(rec_steps) / max(steps, 1),
        "n_triggers": len(triggers), "n_trigger_success": ok, "n_trigger_fail": fail,
        "n_events_coll_within_window": ev_coll,
        "tier_counts": {str(k): v for k, v in sorted(tiers.items())},
        "cbf_min_margin": float(min((s["m"] for s in S), default=float("nan"))),
        "m_u0_infeasible_mean": (float(np.mean([s["m_u0"] for s in inf_steps]))
                                 if inf_steps else float("nan")),
        "m_exec_infeasible_mean": (float(np.mean([s["m"] for s in inf_steps]))
                                   if inf_steps else float("nan")),
        "m_recovery_mean": (float(np.mean([s["m"] for s in rec_steps]))
                            if rec_steps else float("nan")),
        "clf_slack_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "action_norm_mean": float(np.mean([s["u_norm"] for s in S])) if S else float("nan"),
        "step_time_ms_mean": float(t.mean()), "step_time_ms_median": float(np.median(t)),
        "step_time_ms_p95": float(np.percentile(t, 95)), "step_time_ms_max": float(t.max()),
        "qp_time_ms_mean": float(np.mean(qp_t)) * 1e3 if qp_t else float("nan"),
        "recovery_overhead_ms_mean": float(np.mean(extra_t)) * 1e3 if extra_t else float("nan"),
        "episode_time_s": ep_time,
        "collision_step": coll_step,
    }
    if arm.kind in ("ladder", "c0b"):
        out["rungs"] = dict(Counter(r["rung"] for r in wrap.records if r["infeasible"]))
    if arm.kind == "random":
        E = wrap.events
        out.update({
            "n_recovery_events": len(E),
            "recovery_speed_mean": float(np.mean([e["speed"] for e in E])) if E else float("nan"),
            "recovery_accept_frac": float(np.mean([e["accepted"] for e in E])) if E else float("nan"),
            "candidates_mean": float(np.mean([e["n_candidates"] for e in E])) if E else float("nan"),
            "random_eval_ms_mean": (float(np.mean([e["t_recovery"] for e in E])) * 1e3
                                    if E else float("nan")),
            "events": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()
                        if k != "m_by_speed"} | {"m_by_speed": [round(x, 6) for x in e["m_by_speed"]]}
                       for e in E],
        })
    return out


# --------------------------------------------------------------------------- blocks
_W = {}


def _worker(job):
    arm, cond, seed = job
    if cond not in _W:
        env = make_env(cond == "randomized")
        _W[cond] = (env, ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS,
                                            env.static_obstacles))
    env, oracle = _W[cond]
    return arm.name, cond, seed, run_episode(arm, cond, seed, env=env, oracle=oracle)


def _arms_fingerprint(arms, seeds, conditions):
    blob = json.dumps({"arms": [a.describe() for a in arms], "seeds": list(seeds),
                       "conditions": list(conditions)}, sort_keys=True, default=_js)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_block(arms, seeds, *, conditions=CONDITIONS, workers=8, progress=None,
              checkpoint=None, maxtasksperchild=40):
    """{arm_name: {condition: [records sorted by seed]}}; asserts pairing across arms.

    checkpoint: optional .jsonl path. Every finished episode is appended as it arrives; a rerun
    with the IDENTICAL arm/seed/condition fingerprint resumes, skipping finished episodes.
    Episodes are deterministic given the seed (verified by the equivalence gate and
    test_random_arm_episode_is_bit_reproducible), so a resumed block equals an uninterrupted
    one except for wall-clock timing fields. Worker processes are recycled every
    `maxtasksperchild` episodes to bound memory.
    """
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    out = {a.name: {c: {} for c in conditions} for a in arms}
    fp = _arms_fingerprint(arms, seeds, conditions)
    ck = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists():
            lines = checkpoint.read_text().splitlines()
            head = json.loads(lines[0])
            if head.get("fingerprint") != fp:
                raise SystemExit(f"{checkpoint} belongs to a different block definition")
            for ln in lines[1:]:
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue                      # a torn final line from a kill
                out[row["arm"]][row["cond"]][row["seed"]] = row["rec"]
            print(f"  resuming: {sum(len(v) for d in out.values() for v in d.values())} "
                  f"episodes already in {checkpoint}", flush=True)
        else:
            checkpoint.write_text(json.dumps({"fingerprint": fp}) + "\n")
        ck = open(checkpoint, "a")
    jobs = [(a, c, s) for c in conditions for s in seeds for a in arms
            if s not in out[a.name][c]]
    t0 = time.time()
    if workers <= 1:
        it = map(_worker, jobs)
        pool = None
    else:
        pool = get_context("fork").Pool(workers, maxtasksperchild=maxtasksperchild)
        it = pool.imap_unordered(_worker, jobs, chunksize=1)
    try:
        for i, (name, cond, seed, rec) in enumerate(it, 1):
            out[name][cond][seed] = rec
            if ck is not None:
                ck.write(json.dumps({"arm": name, "cond": cond, "seed": seed, "rec": rec},
                                    default=_js) + "\n")
                ck.flush()
            if progress and (i % progress == 0 or i == len(jobs)):
                print(f"  {i}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if ck is not None:
            ck.close()
    res = {n: {c: [out[n][c][s] for s in seeds] for c in conditions} for n in names}
    for c in conditions:
        for i, s in enumerate(seeds):
            ref = res[names[0]][c][i]
            for n in names[1:]:
                r = res[n][c][i]
                assert r["seed"] == s and r["start"] == ref["start"] and r["goal"] == ref["goal"], \
                    f"pairing broken at {c}/{s}/{n}"
    return res


# --------------------------------------------------------------------------- manifest / io
def manifest_rows(res, arms, stage):
    by = {a.name: a for a in arms}
    rows = []
    for name, conds in res.items():
        a = by[name]
        for cond, recs in conds.items():
            for r in recs:
                rows.append({
                    "stage": stage, "controller": name, "kind": a.kind,
                    "hyperparameters": a.params.as_dict(), "map": MAP_ID,
                    "episode_seed": r["seed"], "motion_model": cond,
                    "recovery_mode": a.describe()["recovery"],
                    "recovery_speeds": list(a.random.speeds) if a.random else None,
                    "K": a.random.K if a.random else None, "H": a.random.H if a.random else None,
                    "outcome": r["outcome"], "collision_type": r["collision_type"],
                    "success": r["success"], "collision": r["collision"], "timeout": r["timeout"],
                    "n_infeasible": r["n_infeasible"], "frac_infeasible": r["frac_infeasible"],
                    "n_infeasible_events": r["n_infeasible_events"],
                    "n_triggers": r["n_triggers"], "n_trigger_success": r["n_trigger_success"],
                    "recovery_fraction": r["recovery_fraction"],
                    "recovery_speed_mean": r.get("recovery_speed_mean"),
                    "tier_counts": r["tier_counts"], "min_clearance": r["min_clearance"],
                    "path_length": r["path_length"], "spl": r["spl"], "steps": r["steps"],
                    "step_time_ms_mean": r["step_time_ms_mean"],
                    "step_time_ms_p95": r["step_time_ms_p95"],
                    "episode_time_s": r["episode_time_s"],
                })
    return rows


def write_new(path, obj):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; continuation results are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r, default=_js) + "\n" for r in obj))
    else:
        path.write_text(json.dumps(obj, indent=1, default=_js))
    return path


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
