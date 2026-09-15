"""P2 - pursuit with the canonical static obstacles only (plan section 6).

Protagonist controller unchanged, hunter pursuit law unchanged. Verifies that the hunter's safety
filter prevents rectangle collisions in feasible situations, and logs the cases where safety
constraints stop the hunter from making progress.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from continuation.pursuit.config import PursuitConfig, pursuit_seeds
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.harness import run_episode
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/pursuit/P2")


def make_static_env(cfg):
    """Canonical map, canonical rectangles, NO dynamic obstacles."""
    return PursuitEnv(pursuit=cfg, config_path="config.json", n_dynamic_obstacles=0,
                      obstacle_speed=0.675, render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=(cfg.motion == "randomized"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hunter-speed", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    cfg = PursuitConfig(condition="P-Random", hunter_max_speed=a.hunter_speed)
    env = make_static_env(cfg)
    assert len(env.static_obstacles) == 5 and env.n_dynamic_obstacles == 0
    t0 = time.time()
    recs = [run_episode(cfg, s, tuned=tuned, spec=spec, env=env)
            for s in pursuit_seeds("P2_static", a.episodes)]
    env.close()

    m = lambda k: float(np.mean([r[k] for r in recs]))
    summ = {
        "episodes": len(recs),
        "goal_rate": m("goal"), "capture_rate": m("captured"), "timeout_rate": m("timeout"),
        "protagonist_collision_rate": m("protagonist_collision"),
        "protagonist_static_collision_rate": float(np.mean(
            [r["protagonist_collision_type"] == "static" for r in recs])),
        "protagonist_wall_collision_rate": float(np.mean(
            [r["protagonist_collision_type"] == "wall" for r in recs])),
        "hunter_collision_rate": m("hunter_collision"),
        "hunter_static_collision_rate": float(np.mean(
            [r["hunter_collision_type"] == "static" for r in recs])),
        "hunter_wall_collision_rate": float(np.mean(
            [r["hunter_collision_type"] == "wall" for r in recs])),
        "hunter_frac_nominal_unsafe_mean": m("hunter_frac_nominal_unsafe"),
        "hunter_frac_infeasible_mean": m("hunter_frac_infeasible"),
        "hunter_n_fallback_total": int(sum(r["hunter_n_fallback"] for r in recs)),
        "hunter_mean_u_dev": m("hunter_mean_u_dev"),
        "min_hunter_distance_mean": m("min_hunter_distance"),
        "min_clearance_mean": m("min_clearance"),
        "prot_frac_infeasible_mean": m("prot_frac_infeasible"),
        "wall_s": time.time() - t0,
    }
    # "Safety constraints stopped the hunter making progress": it fell back to u = 0, or its
    # nominal pursuit action was unsafe, while it never got near the protagonist.
    stalled = [r for r in recs if r["hunter_n_fallback"] > 0 and not r["captured"]]
    summ["hunter_stalled_episodes"] = len(stalled)
    checks = {
        "hunter_avoids_rectangles": summ["hunter_static_collision_rate"] == 0.0,
        "protagonist_still_reaches_goal": summ["goal_rate"] > 0.0,
        "capture_and_collision_stay_distinct": all(
            not (r["captured"] and r["protagonist_collision"]) for r in recs),
    }
    out = {"label": "Pursuit P2 / static-obstacle pursuit", "config": cfg.as_dict(),
           "tuned": tf["params"], "random": rf["spec"],
           "seeds": [recs[0]["seed"], recs[-1]["seed"]],
           "summary": summ, "checks": checks, "records": recs}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"P2_hunter{a.hunter_speed:g}{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; results are never overwritten")
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summ, indent=1))
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("wrote", p)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
