"""Experiment 0 -- Phase 1 gate: analytic obstacles, analytic CBF, no sensor.

WHAT IS BEING TESTED
    The controller mathematics alone. There is no LiDAR, no occupancy map, no A*, no
    reference governor, and no velocity estimation anywhere in this file. Barrier values are
    exact, so any failure here is in the controller, not in perception.

INPUTS       exact (h, grad_h, dh_dt) from dr_control.cbf_sources
CONTROLLER   drccp | cbf_qp | clf_only
ASSUMPTIONS  perfect state, perfect barrier information, gamma = the goal (no governor)
METRICS      Part D, Phase 1 subset -- see dr_control/diagnostics.py

The simulation reproduces RobotNavEnv's own dynamics and collision test exactly
(p <- clip(p + u*dt), collision at h <= 0) but does NOT import the env: Phase 1 must not
depend on the gym wrapper, and robot_env/ is not touched.

Episode budget is the development/validation tier from the plan. The 200-episode
EVAL_SEED_BASE block is reserved for Phase 6 and is never used here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dr_control.baselines import ClfCbfQpController, ClfOnlyController
from dr_control.cbf_sources import AnalyticBarrierSource
from dr_control.diagnostics import EpisodeDiagnostics, aggregate
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi

# Environment constants, mirrored from config.json / RobotNavEnv so that Phase 1 results are
# comparable with later phases. Kept as literals rather than imported, to keep this phase
# independent of the gym env.
WORLD_SIZE = 10.0
AGENT_RADIUS = 0.3
OBSTACLE_RADIUS = 0.3
TARGET_RADIUS = 0.6
MAX_SPEED = 1.0
DT = 0.1
MAX_STEPS = 500
STATIC_RECTS = [
    (2.0, 2.0, 0.5, 1.5),
    (7.0, 3.0, 1.0, 0.5),
    (4.5, 7.0, 0.5, 1.0),
    (2.0, 7.5, 1.5, 0.5),
    (7.5, 7.0, 0.5, 1.5),
]

#: Development/validation seeds. Disjoint from evaluate_week4.EVAL_SEED_BASE (1_000_000),
#: which is reserved for the final Phase 6 comparison.
DEV_SEED_BASE = 20_000

CONTROLLERS = {
    "drccp": ClfCbfDrccpController,
    "cbf_qp": ClfCbfQpController,
    "clf_only": ClfOnlyController,
}


def make_controller(name, **kwargs):
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; expected one of {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](**kwargs)


def sample_scenario(rng, n_circles, *, moving):
    """Start, goal and obstacles, all outside the static rectangles and clear of each other."""
    src = AnalyticBarrierSource(world_size=WORLD_SIZE, r_robot=AGENT_RADIUS,
                                static_rects=STATIC_RECTS)

    def free_point(min_clear, exclude, min_sep):
        for _ in range(400):
            q = rng.uniform(0.6, WORLD_SIZE - 0.6, 2)
            if src.true_clearance(q) < min_clear:
                continue
            if any(np.linalg.norm(q - e) < min_sep for e in exclude):
                continue
            return q
        return np.array([WORLD_SIZE / 2, WORLD_SIZE / 2])

    start = free_point(0.4, [], 0.0)
    goal = free_point(0.4, [start], 4.0)          # matches the env's min_separation = 4.0
    circles, vels = [], []
    for _ in range(n_circles):
        c = free_point(0.5, [start, goal] + circles, 1.5)
        circles.append(c)
        if moving:
            th = rng.uniform(0, 2 * np.pi)
            vels.append(np.array([np.cos(th), np.sin(th)]) * rng.uniform(0.6, 0.75))
        else:
            vels.append(np.zeros(2))
    circles = np.array(circles).reshape(-1, 2)
    vels = np.array(vels).reshape(-1, 2)
    return start, goal, circles, vels


def run_episode(controller, *, seed, n_circles, moving, source, max_steps=MAX_STEPS):
    rng = np.random.default_rng(seed)
    p, goal, circles, vels = sample_scenario(rng, n_circles, moving=moving)
    controller.reset()
    diag = EpisodeDiagnostics()

    outcome, steps = "timeout", 0
    for steps in range(1, max_steps + 1):
        h, grad, dh_dt = source.samples(p, circles, OBSTACLE_RADIUS, vels)
        xi = build_xi(h, grad, dh_dt)
        info = {}
        u = controller.generate_controller(p, goal, xi, record=info)
        diag.record(info=info, u=u, xi=xi, cbf_rate=getattr(controller, "rateh", 0.0),
                    h_true=float(np.min(h)))

        # RobotNavEnv dynamics, reproduced exactly.
        p = np.clip(p + np.asarray(u) * DT, AGENT_RADIUS, WORLD_SIZE - AGENT_RADIUS)
        if moving and len(circles):
            circles = circles + vels * DT
            for i in range(len(circles)):          # elastic reflection, as DeterministicMotion
                for ax in range(2):
                    lo, hi = OBSTACLE_RADIUS, WORLD_SIZE - OBSTACLE_RADIUS
                    if circles[i, ax] < lo:
                        circles[i, ax] = lo; vels[i, ax] *= -1
                    elif circles[i, ax] > hi:
                        circles[i, ax] = hi; vels[i, ax] *= -1

        if source.true_clearance(p, circles, OBSTACLE_RADIUS) <= 0.0:
            outcome = "collision"
            break
        if float(np.linalg.norm(goal - p)) < TARGET_RADIUS:
            outcome = "success"
            break

    return diag.summary(seed=seed, outcome=outcome, steps=steps,
                        success=int(outcome == "success"),
                        collision=int(outcome == "collision"),
                        timeout=int(outcome == "timeout"))


def run_arm(name, *, episodes, n_circles, moving, seed_base=DEV_SEED_BASE, **ctrl_kwargs):
    source = AnalyticBarrierSource(world_size=WORLD_SIZE, r_robot=AGENT_RADIUS,
                                   static_rects=STATIC_RECTS)
    controller = make_controller(name, max_v=MAX_SPEED, **ctrl_kwargs)
    t0 = time.perf_counter()
    records = [run_episode(controller, seed=seed_base + i, n_circles=n_circles,
                           moving=moving, source=source) for i in range(episodes)]
    summary = aggregate(records)
    summary.update(
        arm=name, episodes=episodes, n_circles=n_circles, moving=bool(moving),
        success_rate=float(np.mean([r["success"] for r in records])),
        collision_rate=float(np.mean([r["collision"] for r in records])),
        timeout_rate=float(np.mean([r["timeout"] for r in records])),
        wall_clock_s=time.perf_counter() - t0,
        params=dict(ctrl_kwargs),
    )
    return summary, records


def r_w_sweep(values, *, episodes, n_circles, moving):
    """Diagnostic 4: the EFFECT of r_W, reported as a curve.

    No monotonicity is asserted. Raising r_W tightens the constraint and so shrinks the
    feasible set, but the minimizer moves with it, so the achieved min CBC is not provably
    monotone. The expected direction is more conservative behaviour (larger min h, larger
    min CBC); a reversal is a finding to flag, not a test failure.
    """
    out = []
    for r_w in values:
        summary, _ = run_arm("drccp", episodes=episodes, n_circles=n_circles,
                             moving=moving, wasserstein_r=r_w)
        out.append({"wasserstein_r": r_w,
                    "min_h_true": summary["min_h_true"],
                    "min_h_true_min": summary.get("min_h_true_min"),
                    "min_cbc": summary["min_cbc"],
                    "collision_rate": summary["collision_rate"],
                    "success_rate": summary["success_rate"],
                    "mean_u_dev": summary["mean_u_dev"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20,
                    help="development tier 5-20, validation tier 50")
    ap.add_argument("--arms", nargs="+", default=["clf_only", "cbf_qp", "drccp"],
                    choices=sorted(CONTROLLERS))
    ap.add_argument("--n-circles", type=int, default=6)
    ap.add_argument("--moving", action="store_true")
    ap.add_argument("--sweep-rw", action="store_true", help="run the r_W effect curve")
    ap.add_argument("--out", default="results/phase1/exp0.json")
    args = ap.parse_args()

    report = {"config": vars(args), "arms": {}}
    for arm in args.arms:
        summary, records = run_arm(arm, episodes=args.episodes, n_circles=args.n_circles,
                                   moving=args.moving)
        report["arms"][arm] = {"summary": summary, "episodes": records}
        print(f"\n=== {arm} ===")
        for key in ("success_rate", "collision_rate", "timeout_rate", "min_h_true",
                    "min_h_true_min", "min_cbc", "min_cbc_min", "frac_cbc_negative",
                    "frac_delta_positive", "frac_cbf_active", "frac_h_negative",
                    "n_solver_fail", "n_dcp_error", "mean_solver_time", "mean_total_time",
                    "mean_u_dev", "steps", "wall_clock_s"):
            if key in summary:
                print(f"  {key:22s} {summary[key]}")

    if args.sweep_rw:
        report["r_w_sweep"] = r_w_sweep([0.0, 0.001, 0.004, 0.02, 0.1],
                                        episodes=args.episodes,
                                        n_circles=args.n_circles, moving=args.moving)
        print("\n=== r_W effect (diagnostic, no monotonicity asserted) ===")
        for row in report["r_w_sweep"]:
            print("  " + json.dumps(row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
