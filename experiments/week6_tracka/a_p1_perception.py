"""Track A / A-P1 - controlled perception tests for the LiDAR hunter detector.

Scripted scenarios in the canonical map. The robot and the hunter are placed by hand, the scan is
cast by the env, and the detector sees ONLY that scan. Ground truth is used solely to score the
estimate afterwards. Falsification gate: if identification is unreliable here, the pursuit
experiment cannot be run and we stop and report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuation.perception.hunter_detector import HunterDetector
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.harness import make_pursuit_env

OUT = Path("results/week6_tracka/A_P1")
ASSOC_TOL = 0.6          # an estimate within this distance of the true hunter counts as correct


def scenario(env, det, steps, hunter_traj, obstacle_traj=None, robot_traj=None):
    rows = []
    for k in range(steps):
        env.hunter_position = np.asarray(hunter_traj(k), float)
        if obstacle_traj is not None:
            env.obstacle_positions = np.asarray(obstacle_traj(k), dtype=np.float32)
        if robot_traj is not None:
            env.agent_position = np.asarray(robot_traj(k), dtype=np.float32)
        ranges = np.asarray(env._cast_lidar_rays(), float)
        est, rec = det.update(env.agent_position, ranges)
        d_true = float(np.linalg.norm(env.hunter_position - env.agent_position))
        # is the hunter geometrically visible at all? (nearest ray hit near its true bearing)
        row = {"k": k, "true_dist": d_true, "detected": bool(est is not None),
               "n_tracks": rec["n_tracks"]}
        if est is not None:
            err = float(np.linalg.norm(est["position"] - env.hunter_position))
            row.update(pos_err=err, correct=bool(err <= ASSOC_TOL), stale=est["stale"],
                       track_id=est["track_id"], score=est["mean_score"],
                       vel_err=float(np.linalg.norm(est["velocity"] -
                                                    (np.asarray(hunter_traj(k), float) -
                                                     np.asarray(hunter_traj(max(k - 1, 0)), float)) / env.dt)))
        rows.append(row)
    return rows


def summarise(rows, warmup=5):
    r = [x for x in rows if x["k"] >= warmup]
    det = [x for x in r if x["detected"]]
    tp = [x for x in det if x.get("correct")]
    fp = [x for x in det if not x.get("correct")]
    miss = [x for x in r if not x["detected"]]
    gaps, run = [], 0
    for x in r:
        if not x["detected"]:
            run += 1
        elif run:
            gaps.append(run); run = 0
    if run:
        gaps.append(run)
    return {
        "steps_scored": len(r),
        "detection_rate": len(det) / max(len(r), 1),
        "identification_precision": len(tp) / max(len(det), 1),
        "false_positive_rate": len(fp) / max(len(r), 1),
        "miss_rate": len(miss) / max(len(r), 1),
        "pos_err_mean": float(np.mean([x["pos_err"] for x in tp])) if tp else float("nan"),
        "pos_err_p95": float(np.percentile([x["pos_err"] for x in tp], 95)) if tp else float("nan"),
        "vel_err_mean": float(np.mean([x["vel_err"] for x in tp])) if tp else float("nan"),
        "max_gap_steps": max(gaps) if gaps else 0,
        "n_gaps": len(gaps),
        "reacquire_steps_mean": float(np.mean(gaps)) if gaps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = PursuitConfig(info_model="lidar_estimate")
    env = make_pursuit_env(cfg)
    env.reset(seed=13_000_000)
    far = np.tile(np.array([9.5, 9.5], dtype=np.float32), (len(env.obstacle_positions), 1))
    rect = env.static_obstacles[0]
    out = {}

    def fresh(robot):
        env.reset(seed=13_000_000)
        env.agent_position = np.asarray(robot, dtype=np.float32)
        env.obstacle_positions = far.copy()
        return HunterDetector(dt=env.dt, n_rays=env.N_LIDAR_RAYS, lidar_range=env.LIDAR_RANGE)

    # S1 hunter stationary and visible (no motion signature -> should NOT be claimed as hunter)
    det = fresh([5.0, 5.0])
    out["S1_stationary_visible"] = summarise(scenario(
        env, det, 40, lambda k: [7.0, 5.0]))
    # S2 hunter approaching at a known velocity
    det = fresh([5.0, 5.0])
    out["S2_approaching_0.6ms"] = summarise(scenario(
        env, det, 40, lambda k: [7.5 - 0.06 * k, 5.0]))
    # S3 hunter behind a rectangle (occluded), then emerging
    cx, cy, hw, hh = rect
    det = fresh([cx - hw - 2.5, cy])
    out["S3_occluded_then_emerging"] = summarise(scenario(
        env, det, 60, lambda k: [cx + hw + 1.0 - 0.05 * k, cy + (0.0 if k < 30 else 0.03 * (k - 30))]))
    # S4 approaching hunter with the six dynamic obstacles present and moving (ambiguity)
    env.reset(seed=13_000_001)
    env.agent_position = np.asarray([5.0, 5.0], dtype=np.float32)
    obs0 = np.asarray(env.obstacle_positions, float).copy()
    vel0 = np.asarray(env.obstacle_velocities, float).copy()
    det = HunterDetector(dt=env.dt, n_rays=env.N_LIDAR_RAYS, lidar_range=env.LIDAR_RANGE)
    out["S4_with_moving_obstacles"] = summarise(scenario(
        env, det, 40, lambda k: [7.5 - 0.06 * k, 5.0],
        obstacle_traj=lambda k: obs0 + vel0 * (k * env.dt)))
    # S5 moving robot, moving hunter, moving obstacles (closest to the real loop)
    env.reset(seed=13_000_002)
    obs0 = np.asarray(env.obstacle_positions, float).copy()
    vel0 = np.asarray(env.obstacle_velocities, float).copy()
    det = HunterDetector(dt=env.dt, n_rays=env.N_LIDAR_RAYS, lidar_range=env.LIDAR_RANGE)
    out["S5_all_moving"] = summarise(scenario(
        env, det, 50, lambda k: [8.0 - 0.07 * k, 5.0 + 0.02 * k],
        obstacle_traj=lambda k: obs0 + vel0 * (k * env.dt),
        robot_traj=lambda k: [4.0 + 0.02 * k, 5.0]))
    env.close()

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"a_p1_perception{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text(json.dumps({"assoc_tol": ASSOC_TOL, "scenarios": out}, indent=1, default=float))
    for k, v in out.items():
        print(f"{k:28s} det={v['detection_rate']:.2f} prec={v['identification_precision']:.2f} "
              f"fp={v['false_positive_rate']:.2f} poserr={v['pos_err_mean']:.3f} "
              f"gapmax={v['max_gap_steps']}")
    print("wrote", p)


if __name__ == "__main__":
    main()
