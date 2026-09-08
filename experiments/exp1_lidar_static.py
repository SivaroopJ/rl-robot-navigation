"""Experiment 1 -- Phase 2 gate: 24-ray LiDAR replaces the analytic barrier source.

WHAT IS BEING TESTED
    The sensor-to-CBF conversion, and nothing else. The controller, alpha, r_W, epsilon and
    the u=0 infeasibility fallback are all unchanged from Phase 1. Scenarios, dynamics,
    seeds and the episode loop are identical to experiments/exp0_analytic.py, so an
    analytic-vs-LiDAR difference is attributable to the sensor alone.

INPUTS       24 ray ranges (robot_env.lidar_core.cast_rays -- the env's own sensor) + ego pose
CONTROLLER   drccp, alpha = 0.4, r_W = 0.004, eps = 0.1, u = 0 on infeasibility  (UNCHANGED)
ASSUMPTIONS  known ego pose; no obstacle ground truth; no sensor noise (Phase 4); dh_dt = 0
METRICS      Phase 1 metrics + sensing error, detection miss rate, frac_infeasible

Phase 2 is static-only. `--static-circles` adds NON-MOVING circles, which is how the 24-ray
detection limit is exercised: rectangles are large and easy to see, small discs are not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dr_control.cbf_sources import AnalyticBarrierSource, circle_barrier, rect_barrier, wall_barriers
from dr_control.diagnostics import EpisodeDiagnostics, aggregate
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
from dr_control.lidar_cbf import LidarBarrierSource
from robot_env.lidar_core import cast_rays, ray_angles
import experiments.exp0_analytic as E


def scan(p, circles):
    """The environment's own sensor model, called directly."""
    return cast_rays(p, n_rays=24, lidar_range=5.0, world_size=E.WORLD_SIZE,
                     agent_radius=E.AGENT_RADIUS, rects=E.STATIC_RECTS,
                     circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=E.OBSTACLE_RADIUS)


def expected_lidar_clearance(p, circles):
    """Clearance under the LiDAR CONVENTION, from exact geometry.

    The comparison target for sensing error. Walls carry the -r_robot conservatism the
    convention imposes; rectangles and circles are exact. Pooling them would mix a constant
    convention offset with genuine discretisation error and make both unreadable.
    """
    vals = [h - E.AGENT_RADIUS for h, _, _ in wall_barriers(p, E.WORLD_SIZE, E.AGENT_RADIUS)]
    vals += [rect_barrier(p, r, E.AGENT_RADIUS)[0] for r in E.STATIC_RECTS]
    vals += [circle_barrier(p, c, E.OBSTACLE_RADIUS, E.AGENT_RADIUS)[0]
             for c in np.asarray(circles, float).reshape(-1, 2)]
    return float(min(vals))


def run_episode(*, seed, source_kind, n_circles, alpha=0.4):
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    p, goal, circles, _ = E.sample_scenario(rng, n_circles, moving=False)
    truth = AnalyticBarrierSource(world_size=E.WORLD_SIZE, r_robot=E.AGENT_RADIUS,
                                  static_rects=E.STATIC_RECTS)
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=alpha)
    lidar = LidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5)
    diag = EpisodeDiagnostics()

    h_err, grad_err, missed, seen = [], [], 0, 0
    band_seen, band_missed = np.zeros(5, int), np.zeros(5, int)
    outcome, steps = "timeout", 0

    for steps in range(1, E.MAX_STEPS + 1):
        if source_kind == "analytic":
            h, g, dd = truth.samples(p, circles, E.OBSTACLE_RADIUS)
        else:
            ranges = scan(p, circles)
            lidar.push(p, ranges)
            h, g, dd = lidar.samples(p)

            # --- sensing error, measured against the convention-corrected ground truth
            h_exp = expected_lidar_clearance(p, circles)
            h_err.append(float(np.min(h)) - h_exp)

            # Gradient angular error, measured against the surface that is nearest UNDER THE
            # LIDAR CONVENTION. Comparing against the raw-analytic nearest would mix in the
            # deliberate -r_robot wall offset and report a large angle whenever the two
            # conventions disagree about which surface is closest -- an artefact, not error.
            gt_h, gt_g, _ = truth.samples(p, circles, E.OBSTACLE_RADIUS)
            conv = gt_h.copy()
            conv[:4] -= E.AGENT_RADIUS          # first 4 samples are walls, by construction
            j = int(np.argmin(conv))
            k = int(np.argmin(h))
            cosang = float(np.clip(gt_g[:, j] @ g[:, k], -1.0, 1.0))
            grad_err.append(np.degrees(np.arccos(cosang)))

            # --- detection: is each circle within lidar range actually seen by some ray?
            ang = ray_angles(24)
            dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            pts = p[None, :] + ranges[:, None] * dirs
            for c in np.asarray(circles, float).reshape(-1, 2):
                d = float(np.linalg.norm(c - p))
                if d - E.OBSTACLE_RADIUS >= 5.0:
                    continue
                seen += 1
                band = min(int(d), 4)          # 0-1, 1-2, 2-3, 3-4, 4+ metres
                band_seen[band] += 1
                if np.min(np.linalg.norm(pts - c[None, :], axis=1)) > E.OBSTACLE_RADIUS + 1e-3:
                    missed += 1
                    band_missed[band] += 1

        xi = build_xi(h, g, dd)
        info = {}
        u = ctrl.generate_controller(p, goal, xi, record=info)
        diag.record(info=info, u=u, xi=xi, cbf_rate=alpha,
                    h_true=truth.true_clearance(p, circles, E.OBSTACLE_RADIUS))

        p = np.clip(p + np.asarray(u) * E.DT, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)

        if truth.true_clearance(p, circles, E.OBSTACLE_RADIUS) <= 0.0:
            outcome = "collision"
            break
        if float(np.linalg.norm(goal - p)) < E.TARGET_RADIUS:
            outcome = "success"
            break

    return diag.summary(
        seed=seed, outcome=outcome, steps=steps,
        success=int(outcome == "success"), collision=int(outcome == "collision"),
        timeout=int(outcome == "timeout"),
        h_err_mean=float(np.mean(h_err)) if h_err else float("nan"),
        h_err_max=float(np.max(h_err)) if h_err else float("nan"),
        h_err_min=float(np.min(h_err)) if h_err else float("nan"),
        grad_err_mean=float(np.mean(grad_err)) if grad_err else float("nan"),
        grad_err_p95=float(np.percentile(grad_err, 95)) if grad_err else float("nan"),
        circles_in_range=seen, circles_missed=missed,
        band_seen=band_seen.tolist(), band_missed=band_missed.tolist(),
    )


def detection_curve(n_trials=4000, seed=0):
    """Empirical P(detect) for a lone disc vs range, against the angular derivation.

    Derivation: a disc of radius R at range d subtends asin(R/d); rays are spaced
    delta = 2*pi/24, and the bearing offset is uniform, so
        P(detect) ~= min(1, 2*asin(R/d) / delta)
    with guaranteed detection for d <= R / sin(delta/2) ~= 2.30 m.
    """
    rng = np.random.default_rng(seed)
    ang = ray_angles(24)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    delta = 2 * np.pi / 24
    rows = []
    for d in (1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0, 4.5):
        hits = valid = 0
        for _ in range(n_trials):
            bearing = rng.uniform(0, 2 * np.pi)
            c = np.array([5.0, 5.0]) + d * np.array([np.cos(bearing), np.sin(bearing)])
            if not (0.4 < c[0] < 9.6 and 0.4 < c[1] < 9.6):
                continue
            valid += 1
            r = cast_rays([5.0, 5.0], n_rays=24, lidar_range=5.0, world_size=E.WORLD_SIZE,
                          agent_radius=E.AGENT_RADIUS, rects=(), circles=c.reshape(1, 2),
                          circle_radius=E.OBSTACLE_RADIUS)
            pts = np.array([5.0, 5.0])[None, :] + r[:, None] * dirs
            hits += int(np.min(np.linalg.norm(pts - c[None, :], axis=1))
                        <= E.OBSTACLE_RADIUS + 1e-3)
        rows.append({"range_m": d, "trials": valid,
                     "p_detect_measured": hits / max(valid, 1),
                     "p_detect_predicted": float(min(
                         1.0, 2 * np.arcsin(min(1.0, E.OBSTACLE_RADIUS / d)) / delta))})
    return rows


def run_arm(source_kind, *, episodes, n_circles):
    records = [run_episode(seed=i, source_kind=source_kind, n_circles=n_circles)
               for i in range(episodes)]
    s = aggregate(records)
    s.update(arm=source_kind, episodes=episodes, n_circles=n_circles,
             success_rate=float(np.mean([r["success"] for r in records])),
             collision_rate=float(np.mean([r["collision"] for r in records])),
             timeout_rate=float(np.mean([r["timeout"] for r in records])))
    tot_seen = sum(r["circles_in_range"] for r in records)
    tot_missed = sum(r["circles_missed"] for r in records)
    s["detection_miss_rate"] = tot_missed / tot_seen if tot_seen else float("nan")
    bs = np.sum([r["band_seen"] for r in records], axis=0)
    bm = np.sum([r["band_missed"] for r in records], axis=0)
    s["detection_miss_by_range_band"] = [
        {"band_m": lbl, "seen": int(n), "missed": int(m),
         "miss_rate": float(m / n) if n else None}
        for lbl, n, m in zip(["0-1", "1-2", "2-3", "3-4", "4+"], bs, bm)]
    return s, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--n-circles", type=int, default=6,
                    help="STATIC circles; 0 for rectangles and walls only")
    ap.add_argument("--out", default="results/phase2/exp1_lidar_static.json")
    args = ap.parse_args()

    report = {"config": vars(args), "arms": {}}
    keys = ["success_rate", "collision_rate", "timeout_rate", "min_h_true", "min_h_true_min",
            "min_cbc", "min_cbc_solved", "frac_cbc_negative", "frac_infeasible",
            "n_infeasible", "n_solver_fail", "n_dcp_error", "frac_delta_positive",
            "mean_u_dev", "h_err_mean", "h_err_max", "h_err_min", "grad_err_mean",
            "grad_err_p95", "detection_miss_rate", "steps"]

    for kind in ("analytic", "lidar"):
        s, recs = run_arm(kind, episodes=args.episodes, n_circles=args.n_circles)
        report["arms"][kind] = {"summary": s, "episodes": recs}

    print(f"{'metric':24s}{'analytic':>14s}{'lidar':>14s}")
    for k in keys:
        a = report["arms"]["analytic"]["summary"].get(k)
        l = report["arms"]["lidar"]["summary"].get(k)
        if a is None and l is None:
            continue
        fmt = lambda v: f"{v:>14.4g}" if isinstance(v, (int, float)) else f"{'-':>14s}"
        print(f"{k:24s}{fmt(a)}{fmt(l)}")

    report["detection_curve"] = detection_curve()
    print("\n24-ray detection of a 0.3 m disc (guaranteed below "
          f"{E.OBSTACLE_RADIUS/np.sin(np.pi/24):.2f} m):")
    print(f"  {'range':>8s}{'measured':>12s}{'predicted':>12s}")
    for r in report["detection_curve"]:
        print(f"  {r['range_m']:>8.1f}{r['p_detect_measured']:>12.3f}"
              f"{r['p_detect_predicted']:>12.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
