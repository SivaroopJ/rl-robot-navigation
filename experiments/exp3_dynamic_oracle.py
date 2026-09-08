"""Experiment 3 -- Phase 3 gate: dynamic obstacles, ORACLE velocity.

WHAT IS BEING TESTED
    Whether the DR-CBF handles the dh_dt term correctly, in isolation. Obstacle POSITIONS
    come from the 24-ray LiDAR exactly as in Phase 2; obstacle VELOCITY comes from simulator
    ground truth, bound to LiDAR returns by GROUND-TRUTH IDENTITY (dr_control.dynamic_cbf.
    ray_owners), so association error cannot contaminate the dh_dt measurement.

    alpha = 0.4, r_W = 0.004, eps = 0.1 and the u = 0 infeasibility fallback are UNCHANGED
    from Phase 1, as instructed.

ARMS -- identical except for the dh_dt channel
    oracle        dh_dt = -grad_h . v_true(owner)
    zero          dh_dt = 0                        (blind to obstacle motion)
    conservative  dh_dt = -v_assumed               (worst case, no identity used)

EXPECTED FAILURE MODE
    The `zero` arm should collide markedly more with DYNAMIC obstacles specifically. If it
    does not, dh_dt is not entering the constraint correctly and the dynamic-obstacle story
    is unsupported -- which is the whole reason this phase exists.

SECONDARY MEASUREMENT (never in the control path)
    Nearest-neighbour association is run alongside and compared against true ownership, so
    the error it WOULD have introduced is quantified for Phase 4.
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
from dr_control.dynamic_cbf import (
    WALL, DynamicLidarBarrierSource, associate_nearest, ray_owners)
from robot_env.lidar_core import cast_rays
import experiments.exp0_analytic as E

ALPHA, V_ASSUMED = 0.4, 0.75


def scan(p, circles):
    return cast_rays(p, n_rays=24, lidar_range=5.0, world_size=E.WORLD_SIZE,
                     agent_radius=E.AGENT_RADIUS, rects=E.STATIC_RECTS,
                     circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=E.OBSTACLE_RADIUS)


def collision_type(p_new, circles):
    """Which family was hit, in RobotNavEnv's own precedence order.

    The env tests the wall BEFORE clamping, then circles, then rectangles.
    """
    lo, hi = E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS
    if p_new[0] < lo or p_new[0] > hi or p_new[1] < lo or p_new[1] > hi:
        return "wall"
    q = np.clip(p_new, lo, hi)
    circles = np.asarray(circles, float).reshape(-1, 2)
    if len(circles) and np.min(np.linalg.norm(circles - q[None, :], axis=1)) \
            < E.AGENT_RADIUS + E.OBSTACLE_RADIUS:
        return "dynamic"
    for cx, cy, hw, hh in E.STATIC_RECTS:
        closest = np.array([np.clip(q[0], cx - hw, cx + hw), np.clip(q[1], cy - hh, cy + hh)])
        if np.linalg.norm(q - closest) < E.AGENT_RADIUS:
            return "static"
    return None


def run_episode(*, seed, dh_dt_mode, n_circles=6):
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    p, goal, circles, vels = E.sample_scenario(rng, n_circles, moving=True)
    truth = AnalyticBarrierSource(world_size=E.WORLD_SIZE, r_robot=E.AGENT_RADIUS,
                                  static_rects=E.STATIC_RECTS)
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=ALPHA)
    src = DynamicLidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5,
                                    dh_dt_mode=dh_dt_mode, v_assumed=V_ASSUMED)
    diag = EpisodeDiagnostics()

    assoc_total = assoc_wrong = 0
    assoc_vel_err = []
    ctype, outcome, steps = None, "timeout", 0

    for steps in range(1, E.MAX_STEPS + 1):
        ranges = scan(p, circles)
        owners, _ = ray_owners(p, rects=E.STATIC_RECTS, circles=circles,
                               circle_radius=E.OBSTACLE_RADIUS, n_rays=24, lidar_range=5.0,
                               world_size=E.WORLD_SIZE, agent_radius=E.AGENT_RADIUS)
        src.push(p, ranges, owners)
        h, g, dd = src.samples(p, vels)

        # --- SECONDARY: what nearest-neighbour association would have done. Measured only.
        true_owner = src.critical_owner(p)
        j = int(np.argmin(h))
        q_crit = p - g[:, j] * (h[j] + E.AGENT_RADIUS)
        nn_owner = associate_nearest(q_crit, circles, E.OBSTACLE_RADIUS)
        # Association error must be counted in VELOCITY-RELEVANT terms. True ownership
        # distinguishes wall from each rectangle, but both carry v = 0, so disagreeing about
        # which static surface was hit changes nothing. What matters is (a) circle vs static
        # and (b) which circle -- so both ids are collapsed to that before comparison.
        assoc_total += 1
        collapse = lambda o: o if o >= 1000 else WALL
        if collapse(nn_owner) != collapse(true_owner):
            assoc_wrong += 1
        v_true = vels[true_owner - 1000] if true_owner >= 1000 else np.zeros(2)
        v_nn = vels[nn_owner - 1000] if nn_owner >= 1000 else np.zeros(2)
        assoc_vel_err.append(float(np.linalg.norm(v_true - v_nn)))

        xi = build_xi(h, g, dd)
        info = {}
        u = ctrl.generate_controller(p, goal, xi, record=info)
        diag.record(info=info, u=u, xi=xi, cbf_rate=ALPHA,
                    h_true=truth.true_clearance(p, circles, E.OBSTACLE_RADIUS))

        p_new = p + np.asarray(u) * E.DT
        ctype = collision_type(p_new, circles)
        p = np.clip(p_new, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)

        circles = circles + vels * E.DT
        for i in range(len(circles)):
            for ax in range(2):
                lo, hi = E.OBSTACLE_RADIUS, E.WORLD_SIZE - E.OBSTACLE_RADIUS
                if circles[i, ax] < lo:
                    circles[i, ax] = lo
                    vels[i, ax] *= -1
                elif circles[i, ax] > hi:
                    circles[i, ax] = hi
                    vels[i, ax] *= -1

        if ctype is not None:
            outcome = "collision"
            break
        if float(np.linalg.norm(goal - p)) < E.TARGET_RADIUS:
            outcome = "success"
            break

    return diag.summary(
        seed=seed, outcome=outcome, steps=steps, dh_dt_mode=dh_dt_mode,
        velocity_source="oracle" if dh_dt_mode == "oracle" else "none",
        success=int(outcome == "success"), collision=int(outcome == "collision"),
        timeout=int(outcome == "timeout"),
        collision_type=ctype if outcome == "collision" else None,
        assoc_error_rate=assoc_wrong / assoc_total if assoc_total else float("nan"),
        assoc_vel_err_mean=float(np.mean(assoc_vel_err)) if assoc_vel_err else float("nan"),
        assoc_vel_err_max=float(np.max(assoc_vel_err)) if assoc_vel_err else float("nan"),
    )


def run_arm(mode, *, episodes):
    recs = [run_episode(seed=i, dh_dt_mode=mode) for i in range(episodes)]
    s = aggregate(recs)
    types = Counter(r["collision_type"] for r in recs if r["collision_type"])
    s.update(arm=mode, episodes=episodes,
             velocity_source="oracle" if mode == "oracle" else "none",
             success_rate=float(np.mean([r["success"] for r in recs])),
             collision_rate=float(np.mean([r["collision"] for r in recs])),
             timeout_rate=float(np.mean([r["timeout"] for r in recs])),
             collisions_wall=types.get("wall", 0),
             collisions_static=types.get("static", 0),
             collisions_dynamic=types.get("dynamic", 0))
    return s, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/phase3/exp3_dynamic_oracle.json")
    args = ap.parse_args()

    report = {"config": vars(args), "alpha": ALPHA, "v_assumed": V_ASSUMED, "arms": {}}
    arms = ["oracle", "zero", "conservative"]
    for m in arms:
        s, recs = run_arm(m, episodes=args.episodes)
        report["arms"][m] = {"summary": s, "episodes": recs}

    keys = ["velocity_source", "success_rate", "collision_rate", "timeout_rate",
            "collisions_dynamic", "collisions_static", "collisions_wall",
            "min_h_true", "min_h_true_min", "min_cbc", "min_cbc_solved",
            "frac_cbc_negative", "frac_infeasible", "n_infeasible", "n_solver_fail",
            "n_dcp_error", "frac_h_negative", "min_h_est", "frac_delta_positive", "mean_u_dev",
            "assoc_error_rate", "assoc_vel_err_mean", "assoc_vel_err_max", "steps"]
    print(f"{'metric':24s}" + "".join(f"{a:>15s}" for a in arms))
    for k in keys:
        row = [report["arms"][a]["summary"].get(k) for a in arms]
        if all(v is None for v in row):
            continue
        cells = "".join(f"{v:>15.4g}" if isinstance(v, (int, float))
                        else f"{str(v):>15s}" for v in row)
        print(f"{k:24s}{cells}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
