"""RS Experiment 1, phase 2 follow-up: how the baseline's pedestrian collisions happen (ticket 10).

Evidence for the candidate list (RS_DESIGN 7.4), read from the finished RS2 diagnostic. Every
pedestrian collision of RS2 (`collision_dynamic` and `collision_spawned`, the anchor included) is
replayed exactly (rs2_diagnostic._replay, checked against the stored trajectory) to recover the
pedestrian that was hit. Measured over the last 0.5 s before contact:

    robot_speed       the robot's speed
    robot_toward      the robot's velocity towards the pedestrian (< 0: backing away)
    robot_cos_toward  cosine between the robot's velocity and its line to the pedestrian
                      (near -1: backing away along that line)
    ped_toward        the pedestrian's velocity towards the robot
    cos_head          cosine between the pedestrian's heading and its line to the robot
    steps_within_2m   how many steps before contact the pedestrian had been within 2 m
                      (centres), contiguously; the contact step itself not counted
    infeasible_last5  infeasible QP steps among the last 5

A spawned pedestrian exists only from its firing step, so its window can be shorter; the
velocities then use what exists.

No new episode is run; ground truth is used only to describe finished episodes.

    python -m experiments.robustsuite.rs2_collision_probe --workers 14
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.robustsuite import rs2_diagnostic as D
from robustsuite import diagnostic as RD
from robustsuite import seeds as RS

#: The probe's output, next to the RS2 results it reads.
OUT = D.RESULTS / "collision_probe.json"
#: The window the velocities and the infeasible count are measured over, in steps (0.5 s).
WINDOW = 5
#: The env's step, in seconds.
DT = 0.1
#: "Near": centre distance at which the approach is counted.
NEAR = 2.0
#: The failure modes that are pedestrian collisions.
PED_MODES = ("collision_dynamic", "collision_spawned")


def collisions():
    """[(cell, motion, record, trace row, view)] of every pedestrian collision in RS2."""
    prov = json.loads((D.RESULTS / "provenance.json").read_text())
    out = []
    for c in [*RS.CELLS, RS.ANCHOR]:
        recs, rows = D.run([c], prov["seeds"], D.RESULTS, workers=1, max_steps=None,
                           tag=prov["commit"], report_only=True)[c]
        for m in RS.MOTIONS:
            for r in recs[m]:
                row = rows[(m, r["seed"])]
                v = RD.episode_view(r, row)
                if v["mode"] in PED_MODES:
                    out.append((c, m, r, row, v))
    return out


def measure(traj, ped, trace):
    """The approach measurements for one collision; `ped` is the hit pedestrian's track."""
    traj, ped = np.asarray(traj, float), np.asarray(ped, float)
    w = min(WINDOW, len(ped) - 1)
    vr = (traj[-1] - traj[-1 - w]) / (w * DT)
    vp = (ped[-1] - ped[-1 - w]) / (w * DT)
    d = ped[-1] - traj[-1]
    d /= np.linalg.norm(d)
    dist = np.linalg.norm(ped - traj, axis=1)
    k = len(dist) - 1
    while k > 0 and dist[k - 1] < NEAR:
        k -= 1
    speed = float(np.linalg.norm(vr))
    return {"robot_speed": speed, "robot_toward": float(vr @ d),
            "robot_cos_toward": float(vr @ d / speed) if speed > 1e-9 else 0.0,
            "ped_toward": float(-vp @ d),
            "cos_head": float((-vp @ d) / max(np.linalg.norm(vp), 1e-9)),
            "steps_within_2m": int(len(dist) - 1 - k),
            "infeasible_last5": sum("infeasible" in s["qp_status"] for s in trace[-WINDOW:])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cols = collisions()
    jobs = [(c, m, r["seed"], None, row["trajectory"]) for c, m, r, row, _ in cols]
    with ProcessPoolExecutor(args.workers) as ex:
        tracks = list(ex.map(D._replay, jobs))
    rows = []
    for (c, m, r, row, v), tr in zip(cols, tracks, strict=True):
        # a spawned pedestrian exists only from its firing step: align both tracks at the end
        ped = [p[v["hit_slot"]] for p in tr if len(p) > v["hit_slot"]]
        rows.append({"cell": D.cell_name(c), "motion": m, "seed": r["seed"], "mode": v["mode"],
                     **measure(row["trajectory"][-len(ped):], ped, row["trace"])})
    col = lambda k: np.array([x[k] for x in rows], float)         # noqa: E731
    summary = {"n": len(rows),
               **{k: np.percentile(col(k), [10, 25, 50, 75, 90]).round(3).tolist()
                  for k in ("robot_speed", "robot_toward", "robot_cos_toward", "ped_toward",
                            "cos_head", "steps_within_2m")},
               "backing_away_speed": np.percentile(
                   col("robot_speed")[col("robot_toward") < 0], [25, 50, 75]).round(3).tolist(),
               "backing_away_along_line": float(np.mean(col("robot_cos_toward") < -0.7)),
               "robot_slow": float(np.mean(col("robot_speed") < 0.2)),
               "robot_backing_away": float(np.mean(col("robot_toward") < 0)),
               "ped_head_on": float(np.mean(col("cos_head") > 0.7)),
               "infeasible_in_last5": float(np.mean(col("infeasible_last5") > 0))}
    OUT.write_text(json.dumps({"summary": summary, "collisions": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
