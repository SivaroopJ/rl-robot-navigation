"""Experiment 6 -- PPO baselines vs the frozen A* -> waypoint -> LiDAR DR-CBF system.

PROTOCOL
    Environment      the real RobotNavEnv from config.json -- not the Phase 1-5 standalone
                     harness. 6 dynamic obstacles, obstacle_speed = 0.675, dt = 0.1,
                     max_steps = 500, the 5 fixed rectangles, stratified start/goal. This is
                     exactly what run_week6_matrix.sh used for the published week7 numbers.
    Seeds            EVAL_SEED_BASE + 0..199, 200 paired episodes. Reserved and untouched
                     since Phase 1; every tuning decision used DEV_SEED_BASE = 20_000.
    PPO              the twelve 10M-step checkpoints (tag _v4), deterministic actions,
                     NOT retrained. Fixed-motion arms vs deterministic obstacles, randomized
                     arms vs OU obstacles, matching how each was trained.
    DR-CBF           frozen Phase 5 system. alpha = 0.4, r_W = 0.004, eps = 0.1, u = 0
                     fallback, LiDAR pipeline and velocity estimator all unchanged.

INFORMATION ASYMMETRY -- IT RUNS IN BOTH DIRECTIONS, AND BOTH ARE REPORTED
    PPO reads obs[28:52]: exact relative position AND velocity of the six nearest dynamic
    obstacles, plus 10M steps of training on this task. The DR-CBF reads none of that.
    The DR-CBF instead gets ego pose in the world frame and a prior static map for A*.
    Neither system is simply "better informed"; the report states both directions.

    The main DR-CBF arm uses the LiDAR velocity ESTIMATOR. An oracle-velocity arm may be run
    as a clearly labelled DIAGNOSTIC and is never the headline comparison.

COLLISION ATTRIBUTION
    Collisions are split by whether the QP was feasible on the step before impact, because
    Phases 1.5 and 3 showed the u = 0 fallback is itself a failure mode against moving
    obstacles. A collision after an infeasible step is a different event from a collision
    the controller chose.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from dr_control.policy import DRCBFPolicy
from evaluation.evaluate_week4 import EVAL_SEED_BASE, load_policy
from evaluation.shortest_path import ShortestPathOracle
from robot_env.robot_nav_env import RobotNavEnv

ARM_SPEC = {                      # arm -> (is_sr, trained_on_randomized)
    "ppo_fixed": (False, False), "ppo_sr_fixed": (True, False),
    "ppo_random": (False, True), "ppo_sr_random": (True, True),
}
TAG = "_v4"                       # the 10M-step matrix
SEEDS = (0, 1, 2)


def make_env(randomize, config_path="config.json", obstacle_speed=0.675):
    return RobotNavEnv(config_path=config_path, n_dynamic_obstacles=6,
                       obstacle_speed=obstacle_speed, render_mode=None,
                       use_reward_shaping=True, randomize_dynamic_obstacles=randomize)


def model_path(arm, seed):
    return Path("models") / "week4" / f"{arm}_seed{seed}{TAG}" / "final_model.zip"


def episode_record(env, seed, oracle, outcome, steps, traj, start, goal, info,
                   extra=None):
    traj = np.asarray(traj)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    shortest = oracle.path_length(start, goal)
    ctype = info.get("collision_type")
    rec = {
        "seed": seed, "outcome": outcome, "steps": steps,
        "success": int(outcome == "success"),
        "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "collision_type": ctype if outcome == "collision" else None,
        "path_length": path_len,
        "shortest_path": float(shortest) if shortest is not None else None,
        "spl": (float(shortest / max(path_len, shortest, 1e-9))
                if (outcome == "success" and shortest is not None) else 0.0),
        "min_clearance": float(np.min(extra["clearances"])) if extra else float("nan"),
        "mean_step_time": float(np.mean(extra["step_times"])) if extra else float("nan"),
    }
    if extra:
        rec.update({k: v for k, v in extra.items()
                    if k not in ("clearances", "step_times")})
    return rec


def true_clearance(env):
    """Ground-truth clearance -- MEASUREMENT ONLY, identical for both systems."""
    p = env.agent_position
    vals = [p[0] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[0],
            p[1] - env.AGENT_RADIUS, env.WORLD_SIZE - env.AGENT_RADIUS - p[1]]
    for cx, cy, hw, hh in env.static_obstacles:
        q = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
        vals.append(float(np.linalg.norm(p - q)) - env.AGENT_RADIUS)
    if len(env.obstacle_positions):
        d = np.linalg.norm(env.obstacle_positions - p[None, :], axis=1)
        vals.append(float(d.min()) - env.AGENT_RADIUS - env.OBSTACLE_RADIUS)
    return float(min(vals))


# --------------------------------------------------------------------------- PPO
def run_ppo(arm, seed, *, episodes, randomize, oracle):
    env = make_env(randomize)
    path = model_path(arm, seed)
    is_sr = ARM_SPEC[arm][0]
    model = load_policy(str(path), is_sr)
    pad = int(model.policy.anti_goal_dim) if is_sr else 0

    recs, load_failures = [], 0
    for i in range(episodes):
        s = EVAL_SEED_BASE + i
        obs, _ = env.reset(seed=s)
        start, goal = env.agent_position.copy(), env.target_position.copy()
        traj, clear, times = [start.copy()], [true_clearance(env)], []
        term = trunc = False
        info = {}
        steps = 0
        while not (term or trunc):
            mo = obs if pad == 0 else np.concatenate([obs, np.zeros(pad, obs.dtype)])
            t0 = time.perf_counter()
            action, _ = model.predict(mo, deterministic=True)
            times.append(time.perf_counter() - t0)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            traj.append(env.agent_position.copy())
            clear.append(true_clearance(env))
        outcome = ("success" if info.get("success") else
                   "collision" if info.get("collision") else "timeout")
        recs.append(episode_record(
            env, s, oracle, outcome, steps, traj, start, goal, info,
            extra={"clearances": clear, "step_times": times,
                   "planner_failed": 0, "n_infeasible": 0, "n_solver_fail": 0,
                   "collided_after_infeasible": 0, "mean_u_dev": float("nan"),
                   "min_cbc_solved": float("nan")}))
    env.close()
    return recs, load_failures


# --------------------------------------------------------------------------- DR-CBF
def run_drcbf(*, episodes, randomize, oracle, use_planner=True, label="drcbf"):
    env = make_env(randomize)
    pol = DRCBFPolicy(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                      static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                      dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS,
                      use_planner=use_planner)
    recs = []
    for i in range(episodes):
        s = EVAL_SEED_BASE + i
        obs, _ = env.reset(seed=s)
        start, goal = env.agent_position.copy(), env.target_position.copy()
        pol.reset(obs, env.agent_position)
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
        outcome = ("success" if info.get("success") else
                   "collision" if info.get("collision") else "timeout")
        recs.append(episode_record(
            env, s, oracle, outcome, steps, traj, start, goal, info,
            extra={"clearances": clear, "step_times": times,
                   "planner_failed": int(pol.planner_failed),
                   "n_infeasible": pol.n_infeasible,
                   "n_solver_fail": pol.n_solver_fail,
                   "collided_after_infeasible": int(outcome == "collision"
                                                    and infeasible_before_end),
                   "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
                   "min_cbc_solved": (float(pol.min_cbc_solved)
                                      if np.isfinite(pol.min_cbc_solved) else float("nan"))}))
    env.close()
    return recs


# --------------------------------------------------------------------------- stats
def summarise(recs, label):
    t = Counter(r["collision_type"] for r in recs if r["collision_type"])
    n = len(recs)
    fin = lambda k: [r[k] for r in recs if r[k] == r[k]]
    return {
        "arm": label, "episodes": n,
        "success_rate": float(np.mean([r["success"] for r in recs])),
        "collision_rate": float(np.mean([r["collision"] for r in recs])),
        "dynamic_collision_rate": t.get("dynamic", 0) / n,
        "static_collision_rate": t.get("static", 0) / n,
        "wall_collision_rate": t.get("wall", 0) / n,
        "timeout_rate": float(np.mean([r["timeout"] for r in recs])),
        "min_clearance_mean": float(np.mean(fin("min_clearance"))),
        "min_clearance_worst": float(np.min(fin("min_clearance"))),
        "mean_spl": float(np.mean([r["spl"] for r in recs])),
        "mean_path_length": float(np.mean([r["path_length"] for r in recs])),
        "mean_steps": float(np.mean([r["steps"] for r in recs])),
        "planner_failures": int(sum(r["planner_failed"] for r in recs)),
        "n_infeasible": int(sum(r["n_infeasible"] for r in recs)),
        "n_solver_fail": int(sum(r["n_solver_fail"] for r in recs)),
        "collisions_after_infeasible": int(sum(r["collided_after_infeasible"] for r in recs)),
        "collisions_while_feasible": int(sum(r["collision"] for r in recs)
                                         - sum(r["collided_after_infeasible"] for r in recs)),
        "mean_u_dev": float(np.mean(fin("mean_u_dev"))) if fin("mean_u_dev") else float("nan"),
        "min_cbc_solved": (float(np.min(fin("min_cbc_solved")))
                           if fin("min_cbc_solved") else float("nan")),
        "mean_step_time_ms": float(np.mean(fin("mean_step_time"))) * 1e3,
    }


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return {"only_a": n10, "only_b": n01, "p": float(p)}


def paired_ci(a, b, n_boot=10000, seed=12345):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def compare(x, y, label):
    out = {}
    for key in ("success", "collision"):
        a = [r[key] for r in x]
        b = [r[key] for r in y]
        out[f"{label} [{key}]"] = {**mcnemar(a, b), **paired_ci(a, b)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--motion", choices=["fixed", "randomized", "both"], default="both")
    ap.add_argument("--oracle-diagnostic", action="store_true",
                    help="add a clearly-labelled oracle-velocity DR-CBF arm (NOT the main "
                         "comparison)")
    ap.add_argument("--out", default="results/phase6/exp6_ppo_comparison.json")
    args = ap.parse_args()

    oracle = ShortestPathOracle(10.0, 0.3, __import__("json").load(
        open("config.json"))["environment"]["static_obstacles"])

    conditions = (["fixed", "randomized"] if args.motion == "both" else [args.motion])
    report = {"config": vars(args), "eval_seed_base": EVAL_SEED_BASE,
              "tag": TAG, "conditions": {}}

    for cond in conditions:
        randomize = cond == "randomized"
        arms = ([a for a in ARM_SPEC if ARM_SPEC[a][1] == randomize])
        block = {"arms": {}, "paired": {}}

        dr = run_drcbf(episodes=args.episodes, randomize=randomize, oracle=oracle)
        block["arms"]["DR-CBF (LiDAR, A*)"] = {"summary": summarise(dr, "DR-CBF"),
                                               "episodes": dr}

        for arm in arms:
            for sd in args.seeds:
                recs, _ = run_ppo(arm, sd, episodes=args.episodes,
                                  randomize=randomize, oracle=oracle)
                lbl = f"{arm}_seed{sd}"
                block["arms"][lbl] = {"summary": summarise(recs, lbl), "episodes": recs}
                block["paired"].update(compare(dr, recs, f"DR-CBF vs {lbl}"))

        report["conditions"][cond] = block

        print(f"\n{'=' * 108}\n### condition = {cond} obstacles "
              f"(speed 0.675, {args.episodes} paired episodes from {EVAL_SEED_BASE})")
        keys = ["success_rate", "collision_rate", "dynamic_collision_rate",
                "static_collision_rate", "wall_collision_rate", "timeout_rate",
                "min_clearance_mean", "min_clearance_worst", "mean_spl",
                "collisions_while_feasible", "collisions_after_infeasible",
                "n_infeasible", "n_solver_fail", "planner_failures", "mean_u_dev",
                "min_cbc_solved", "mean_step_time_ms"]
        labels = list(block["arms"])
        print(f"{'metric':28s}" + "".join(f"{l[:16]:>18s}" for l in labels))
        for k in keys:
            row = [block["arms"][l]["summary"].get(k) for l in labels]
            print(f"{k:28s}" + "".join(
                f"{v:>18.4g}" if isinstance(v, (int, float)) else f"{'-':>18s}"
                for v in row))
        print("\npaired vs DR-CBF (positive diff = DR-CBF higher):")
        for k, v in block["paired"].items():
            sig = "SIGNIFICANT" if v["p"] < 0.05 else "not resolved"
            print(f"   {k:44s} diff={v['diff']:+.4f} "
                  f"CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.5f}  {sig}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
