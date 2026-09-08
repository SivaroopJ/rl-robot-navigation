"""Experiment 5 -- Phase 5 gate: A* -> waypoint -> DR-CBF, no governor.

WHAT IS BEING TESTED
    Whether the validated LOCAL controller integrates with a global planner to give
    end-to-end navigation. The only difference from Phase 4 is the CLF reference gamma:

        reference = "direct"   gamma = the goal            (Phases 1-4 behaviour)
        reference = "astar"    gamma = the pure-pursuit carrot on the A* polyline

    alpha = 0.4, r_W = 0.004, eps = 0.1, the u = 0 fallback, the CBF formulation, the solver
    configuration, the LiDAR pipeline and the velocity estimator are all UNCHANGED. The
    planner sees only the static map; dynamic obstacles remain the DR-CBF's job, from LiDAR.

    No reference governor and no occupancy mapping: both are later steps.

PAIRING
    Obstacle trajectories are pre-computed per seed exactly as in Phase 4, so `direct` and
    `astar` face identical worlds and can be compared seed by seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dr_control.cbf_sources import AnalyticBarrierSource
from dr_control.diagnostics import EpisodeDiagnostics, aggregate
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.dynamic_cbf import ray_owners
from optional_navigation.planner import CarrotFollower, StaticMapPlanner
import experiments.exp0_analytic as E
from experiments.exp3_dynamic_oracle import collision_type
from experiments.exp4_dynamic_estimated import (
    ALPHA, BIND_TOL, TAU, make_source, mcnemar, paired_ci, precompute_trajectory, scan)

LOOKAHEAD = 1.0
STUCK_WINDOW, STUCK_DIST = 50, 0.25       # 5 s of sim time with < 0.25 m of travel


def run_episode(*, seed, arm, reference, motion, planner, tracker_kw=None):
    tracker_kw = dict(tracker_kw or {})
    p, goal, P, V = precompute_trajectory(seed, motion)
    start = p.copy()

    # ---- global plan over the KNOWN STATIC MAP (dynamic obstacles excluded by design) ----
    plan_ok, path, follower = True, None, None
    if reference == "astar":
        path = planner.path(start, goal)
        if path is None:
            plan_ok = False                        # reported, never silently patched over
        else:
            follower = CarrotFollower(path, lookahead=LOOKAHEAD)

    truth = AnalyticBarrierSource(world_size=E.WORLD_SIZE, r_robot=E.AGENT_RADIUS,
                                  static_rects=E.STATIC_RECTS)
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=ALPHA)
    src = make_source(arm, tracker_kw)
    diag = EpisodeDiagnostics()

    traj = [p.copy()]
    ctype, outcome, steps = None, "timeout", 0
    stuck = False

    for t in range(E.MAX_STEPS):
        steps = t + 1
        circles, vels = P[t], V[t]
        ranges = scan(p, circles)

        if arm == "estimated":
            src.push(p, ranges)
            h, g, dd = src.samples(p)
        else:
            owners, _ = ray_owners(p, rects=E.STATIC_RECTS, circles=circles,
                                   circle_radius=E.OBSTACLE_RADIUS, n_rays=24,
                                   lidar_range=5.0, world_size=E.WORLD_SIZE,
                                   agent_radius=E.AGENT_RADIUS)
            src.push(p, ranges, owners)
            h, g, dd = src.samples(p, vels)

        # ---- THE ONLY PHASE 5 CHANGE ----------------------------------------------------
        gamma = follower.reference(p) if follower is not None else goal

        xi = build_xi(h, g, dd)
        info = {}
        u = ctrl.generate_controller(p, gamma, xi, record=info)
        diag.record(info=info, u=u, xi=xi, cbf_rate=ALPHA,
                    h_true=truth.true_clearance(p, circles, E.OBSTACLE_RADIUS))

        p_new = p + np.asarray(u) * E.DT
        ctype = collision_type(p_new, P[t + 1])
        p = np.clip(p_new, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)
        traj.append(p.copy())

        if ctype is not None:
            outcome = "collision"
            break
        if float(np.linalg.norm(goal - p)) < E.TARGET_RADIUS:
            outcome = "success"
            break
        if t >= STUCK_WINDOW:
            if float(np.linalg.norm(p - traj[t - STUCK_WINDOW])) < STUCK_DIST:
                stuck = True                       # still runs to the step limit; recorded

    traj = np.asarray(traj)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    shortest = planner.path_length(start, goal)
    return diag.summary(
        seed=seed, arm=arm, reference=reference, motion=motion, outcome=outcome, steps=steps,
        success=int(outcome == "success"), collision=int(outcome == "collision"),
        timeout=int(outcome == "timeout"),
        collision_type=ctype if outcome == "collision" else None,
        planner_ok=int(plan_ok), planner_failed=int(not plan_ok),
        n_waypoints=(len(path) if path is not None else 0),
        stuck=int(stuck), path_length=path_len,
        shortest_path=(float(shortest) if shortest is not None else float("nan")),
        spl=(float(shortest / max(path_len, shortest, 1e-9))
             if (outcome == "success" and shortest is not None) else 0.0),
    )


def summarise(records, label):
    s = aggregate(records)
    types = Counter(r["collision_type"] for r in records if r["collision_type"])
    s.update(arm=label, episodes=len(records),
             planner_failure_rate=float(np.mean([r["planner_failed"] for r in records])),
             success_rate=float(np.mean([r["success"] for r in records])),
             collision_rate=float(np.mean([r["collision"] for r in records])),
             timeout_rate=float(np.mean([r["timeout"] for r in records])),
             stuck_rate=float(np.mean([r["stuck"] for r in records])),
             collisions_dynamic=types.get("dynamic", 0),
             collisions_static=types.get("static", 0),
             collisions_wall=types.get("wall", 0),
             mean_spl=float(np.mean([r["spl"] for r in records])),
             mean_waypoints=float(np.mean([r["n_waypoints"] for r in records])))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--motion", choices=["cv", "ou"], default="cv")
    ap.add_argument("--arms", nargs="+", default=["oracle", "estimated"])
    ap.add_argument("--references", nargs="+", default=["direct", "astar"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    planner = StaticMapPlanner(E.WORLD_SIZE, E.AGENT_RADIUS, E.STATIC_RECTS)
    report = {"config": vars(args), "alpha": ALPHA, "lookahead": LOOKAHEAD, "arms": {}}
    outcomes = {}

    for arm in args.arms:
        for ref in args.references:
            label = f"{arm}/{ref}"
            recs = [run_episode(seed=i, arm=arm, reference=ref, motion=args.motion,
                                planner=planner) for i in range(args.episodes)]
            report["arms"][label] = {"summary": summarise(recs, label), "episodes": recs}
            outcomes[label] = recs

    # paired: astar vs direct on identical seeds, per dh_dt arm
    report["paired"] = {}
    for arm in args.arms:
        a, b = f"{arm}/astar", f"{arm}/direct"
        if a in outcomes and b in outcomes:
            for key in ("collision", "success", "timeout"):
                x = [r[key] for r in outcomes[a]]
                y = [r[key] for r in outcomes[b]]
                report["paired"][f"{arm}: astar vs direct [{key}]"] = {
                    **mcnemar(x, y), **paired_ci(x, y)}

    labels = list(report["arms"])
    keys = ["planner_failure_rate", "success_rate", "collision_rate", "collisions_static",
            "collisions_dynamic", "collisions_wall", "timeout_rate", "stuck_rate",
            "min_h_true", "min_h_true_min", "min_cbc_solved", "frac_infeasible",
            "n_infeasible", "n_solver_fail", "n_dcp_error", "mean_u_dev", "mean_spl",
            "mean_waypoints", "steps"]
    print(f"motion = {args.motion}   episodes = {args.episodes}   lookahead = {LOOKAHEAD}\n")
    print(f"{'metric':24s}" + "".join(f"{l:>18s}" for l in labels))
    for k in keys:
        row = [report["arms"][l]["summary"].get(k) for l in labels]
        if all(v is None for v in row):
            continue
        print(f"{k:24s}" + "".join(
            f"{v:>18.4g}" if isinstance(v, (int, float)) else f"{'-':>18s}" for v in row))

    print("\npaired (astar vs direct, identical seeds):")
    for k, v in report["paired"].items():
        verdict = "SIGNIFICANT" if v["p"] < 0.05 else "not resolved"
        print(f"   {k:44s} diff={v['diff']:+.4f} CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f" p={v['p']:.5f}  {verdict}")

    out = Path(args.out or f"results/phase5/exp5_{args.motion}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
