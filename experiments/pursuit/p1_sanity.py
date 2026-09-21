"""P1 - obstacle-free pursuit sanity check (plan section 6).

One protagonist, one hunter, NO obstacles (static rectangles emptied, 0 dynamic obstacles).
Validates mechanics only: that the hunter closes when geometrically able, that capture fires at the
declared threshold, and that the protagonist can still reach its goal. Not a performance result.
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

OUT = Path("results/pursuit/P1")


def make_free_env(cfg):
    """Canonical env with every obstacle removed."""
    env = PursuitEnv(pursuit=cfg, config_path="config.json", n_dynamic_obstacles=0,
                     obstacle_speed=0.675, render_mode=None, use_reward_shaping=True,
                     randomize_dynamic_obstacles=(cfg.motion == "randomized"))
    env.static_obstacles = []
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hunter-speed", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    cfg = PursuitConfig(condition="P-Random", hunter_max_speed=a.hunter_speed)
    env = make_free_env(cfg)
    t0 = time.time()
    recs = [run_episode(cfg, s, tuned=tuned, spec=spec, env=env)
            for s in pursuit_seeds("P1_sanity", a.episodes)]
    env.close()

    n = len(recs)
    summ = {
        "episodes": n,
        "goal_rate": float(np.mean([r["goal"] for r in recs])),
        "capture_rate": float(np.mean([r["captured"] for r in recs])),
        "timeout_rate": float(np.mean([r["timeout"] for r in recs])),
        "protagonist_collision_rate": float(np.mean([r["protagonist_collision"] for r in recs])),
        "hunter_collision_rate": float(np.mean([r["hunter_collision"] for r in recs])),
        "min_hunter_distance_mean": float(np.mean([r["min_hunter_distance"] for r in recs])),
        "hunter_start_distance_mean": float(np.mean([r["hunter_start_distance"] for r in recs])),
        "closed_distance_frac": float(np.mean([r["min_hunter_distance"] < r["hunter_start_distance"]
                                               for r in recs])),
        "hunter_frac_nominal_unsafe_mean": float(np.mean([r["hunter_frac_nominal_unsafe"]
                                                          for r in recs])),
        "hunter_frac_materially_filtered_mean": float(
            np.mean([r["hunter_frac_materially_filtered"] for r in recs])),
        "protagonist_wall_collision_rate": float(np.mean(
            [r["protagonist_collision_type"] == "wall" for r in recs])),
        "protagonist_obstacle_collision_rate": float(np.mean(
            [r["protagonist_collision_type"] in ("static", "dynamic") for r in recs])),
        "hunter_frac_infeasible_mean": float(np.mean([r["hunter_frac_infeasible"] for r in recs])),
        "prot_frac_infeasible_mean": float(np.mean([r["prot_frac_infeasible"] for r in recs])),
        "wall_s": time.time() - t0,
    }
    # mechanical checks, not performance claims
    checks = {
        "capture_threshold_respected": all(
            (r["min_hunter_distance"] <= cfg.capture_distance(0.3) + 1e-9) == bool(r["captured"])
            for r in recs),
        "hunter_closes_when_able": summ["closed_distance_frac"] >= 0.8,
        "protagonist_can_still_reach_goal": summ["goal_rate"] > 0.0,
        # Walls still exist in an obstacle-free arena, and evading into one is a real outcome.
        # What must be zero is collisions with obstacles that are not present.
        "no_obstacle_collisions_without_obstacles": (
            summ["protagonist_obstacle_collision_rate"] == 0.0
            and summ["hunter_collision_rate"] == 0.0),
    }
    out = {"label": "Pursuit P1 / obstacle-free sanity check",
           "config": cfg.as_dict(), "tuned": tf["params"], "random": rf["spec"],
           "seeds": [recs[0]["seed"], recs[-1]["seed"]],
           "summary": summ, "checks": checks, "records": recs}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"P1_hunter{a.hunter_speed:g}{a.tag}.json"
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
