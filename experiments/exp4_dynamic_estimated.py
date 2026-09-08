"""Experiment 4 -- Phase 4 gate: LiDAR-estimated obstacle velocity.

WHAT IS BEING TESTED
    Only the source of dh_dt. alpha = 0.4, r_W = 0.004, eps = 0.1, the u = 0 fallback, the CBF
    formulation, the solver configuration and LiDAR-derived positions are all unchanged from
    Phase 3. h and grad_h are computed by identical code in every arm.

ARMS
    oracle      dh_dt from ground-truth velocity + ground-truth identity   (upper bound)
    estimated   dh_dt from dr_control.velocity_tracker, LiDAR only         (the result)
    zero        dh_dt = 0                                                  (lower bound)

STRICT PAIRING, GUARANTEED BY CONSTRUCTION
    Obstacle trajectories are PRE-COMPUTED per episode before any arm runs, from a dedicated
    RNG that is advanced once per step and never sees the robot. Every arm then indexes the
    same table. This matters for OU motion in particular: if the noise were drawn inline, the
    draws would desynchronise as soon as the arms' robot paths diverged, and the "paired"
    comparison would silently stop being paired.

THE TRUTH CHANNEL IS METRICS-ONLY
    A second barrier source in Phase 3's `oracle` mode is fed the same scans purely to obtain
    the TRUE dh_dt for exactly the same critical points, so that e = dh_dt_est - dh_dt_true is
    an elementwise comparison on identical geometry. It never touches the control path of the
    `estimated` arm; `assert_same_geometry` checks that both sources really do produce
    identical h and grad_h, which is what makes e attributable to velocity alone.
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
from dr_control.dynamic_cbf import DynamicLidarBarrierSource, ray_owners
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import LidarVelocityTracker
from robot_env.dynamic_obstacles import SmoothStochasticMotion
from robot_env.lidar_core import cast_rays
import experiments.exp0_analytic as E

ALPHA, R_W, EPS = 0.4, 0.004, 0.1
TAU = R_W / EPS                      # 0.04, the DR floor min CBC must clear
V_ASSUMED = 0.75
BIND_TOL = 5e-3                      # |min CBC - tau| below which the barrier is "binding"

# OU parameters, taken from config.json's dynamic_obstacles block (the env's own settings).
OU_KW = dict(angular_noise_sigma=0.35, angular_velocity_decay=0.8, max_turn_rate=0.7,
             boundary_margin=2.2, boundary_steer_gain=4.0, steer_full_error=np.pi / 4,
             steer_full_depth=0.35, randomize_initial_heading=True)


def scan(p, circles):
    return cast_rays(p, n_rays=24, lidar_range=5.0, world_size=E.WORLD_SIZE,
                     agent_radius=E.AGENT_RADIUS, rects=E.STATIC_RECTS,
                     circles=np.asarray(circles, float).reshape(-1, 2),
                     circle_radius=E.OBSTACLE_RADIUS)


def precompute_trajectory(seed, motion, n_steps=E.MAX_STEPS, n_circles=6):
    """Robot-independent obstacle trajectory, identical for every arm.

    Returns (p0, goal, positions[T+1,N,2], velocities[T+1,N,2]).
    """
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    p0, goal, circles, vels = E.sample_scenario(rng, n_circles, moving=True)
    P = np.zeros((n_steps + 1, len(circles), 2))
    V = np.zeros_like(P)
    c, v = circles.copy(), vels.copy()

    if motion == "ou":
        mm = SmoothStochasticMotion(world_size=E.WORLD_SIZE,
                                    obstacle_radius=E.OBSTACLE_RADIUS, **OU_KW)
        # Dedicated stream, seeded from the episode seed and advanced once per step.
        ou_rng = np.random.default_rng(900_000 + seed)
        speed = float(np.mean(np.linalg.norm(v, axis=1))) if len(v) else 0.0
        v = mm.reset(ou_rng, c, speed)

    P[0], V[0] = c, v
    for t in range(1, n_steps + 1):
        if motion == "cv":
            c = c + v * E.DT
            for i in range(len(c)):
                for ax in range(2):
                    lo, hi = E.OBSTACLE_RADIUS, E.WORLD_SIZE - E.OBSTACLE_RADIUS
                    if c[i, ax] < lo:
                        c[i, ax] = lo
                        v[i, ax] *= -1
                    elif c[i, ax] > hi:
                        c[i, ax] = hi
                        v[i, ax] *= -1
        else:
            c, v = mm.step(ou_rng, c, v, E.DT)
            c, v = c.copy(), v.copy()
        P[t], V[t] = c, v
    return p0, goal, P, V


def bounce_steps(V):
    """Indices at which any obstacle's velocity direction changes sharply. Metrics only."""
    out = np.zeros(len(V), dtype=bool)
    for t in range(1, len(V)):
        a, b = V[t - 1], V[t]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 1e-9) & (nb > 1e-9)
        if np.any(ok):
            cos = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
            out[t] = bool(np.any(cos < np.cos(np.deg2rad(30.0))))
    return out


def make_source(arm, tracker_kw):
    if arm == "estimated":
        tr = LidarVelocityTracker(r_nominal=tracker_kw.pop("r_nominal", 0.3), dt=E.DT,
                                  **tracker_kw)
        return EstimatedLidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5, tracker=tr,
                                           v_assumed=V_ASSUMED)
    mode = "oracle" if arm == "oracle" else "zero"
    return DynamicLidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5, dh_dt_mode=mode,
                                     v_assumed=V_ASSUMED)


def run_episode(*, seed, arm, motion, tracker_kw=None):
    tracker_kw = dict(tracker_kw or {})
    p, goal, P, V = precompute_trajectory(seed, motion)
    bounces = bounce_steps(V)
    truth = AnalyticBarrierSource(world_size=E.WORLD_SIZE, r_robot=E.AGENT_RADIUS,
                                  static_rects=E.STATIC_RECTS)
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=ALPHA)
    src = make_source(arm, tracker_kw)
    # METRICS ONLY -- never consulted by the estimated arm's control path.
    truth_src = DynamicLidarBarrierSource(r_robot=E.AGENT_RADIUS, k_scans=5,
                                          dh_dt_mode="oracle", v_assumed=V_ASSUMED)
    diag = EpisodeDiagnostics()

    err_rows, track_rows = [], []
    geometry_mismatch = 0
    ctype, outcome, steps = None, "timeout", 0
    collided_unconfirmed = collided_no_track = False

    for t in range(E.MAX_STEPS):
        steps = t + 1
        circles, vels = P[t], V[t]
        ranges = scan(p, circles)

        # ---- metrics channel (ground truth) -------------------------------------------
        owners, _ = ray_owners(p, rects=E.STATIC_RECTS, circles=circles,
                               circle_radius=E.OBSTACLE_RADIUS, n_rays=24, lidar_range=5.0,
                               world_size=E.WORLD_SIZE, agent_radius=E.AGENT_RADIUS)
        truth_src.push(p, ranges, owners)
        h_t, g_t, dd_true = truth_src.samples(p, vels)

        # ---- control channel -----------------------------------------------------------
        if arm == "estimated":
            src.push(p, ranges)
            h, g, dd = src.samples(p)
        else:
            src.push(p, ranges, owners)
            h, g, dd = src.samples(p, vels)

        if h.shape != h_t.shape or not np.allclose(h, h_t, atol=1e-12):
            geometry_mismatch += 1

        xi = build_xi(h, g, dd)
        info = {}
        u = ctrl.generate_controller(p, goal, xi, record=info)
        h_true_now = truth.true_clearance(p, circles, E.OBSTACLE_RADIUS)
        diag.record(info=info, u=u, xi=xi, cbf_rate=ALPHA, h_true=h_true_now)

        # ---- projected error e = dh_dt_est - dh_dt_true, per sample --------------------
        binding = abs(float(np.min(xi @ np.array([1.0, ALPHA, u[0], u[1]]))) - TAU) < BIND_TOL
        j_crit = int(np.argmin(h * ALPHA + dd))
        n = min(len(dd), len(dd_true))
        for i in range(n):
            e = float(dd[i] - dd_true[i])
            ttc = h_true_now / max(-float(dd_true[i]), 1e-6) if dd_true[i] < 0 else np.inf
            err_rows.append({"e": e, "h_true": h_true_now, "binding": binding,
                             "critical": i == j_crit, "ttc": float(min(ttc, 1e3)),
                             "since_bounce": int(t - np.max(np.nonzero(bounces[:t + 1])[0]))
                             if np.any(bounces[:t + 1]) else -1})

        # ---- track-quality record (estimated arm only) ---------------------------------
        if arm == "estimated":
            tid, tk = src.critical_track(p)
            # TRUE owner of the same critical point (metrics only), so velocity error is
            # measured against the object that actually produced the return rather than
            # against "the fastest obstacle anywhere".
            true_owner = truth_src.critical_owner(p)
            v_true_crit = (vels[true_owner - 1000] if true_owner >= 1000 and len(vels)
                           else np.zeros(2))
            v_est_crit = tk.velocity if (tk is not None and tk.confirmed) else np.zeros(2)
            track_rows.append({
                "step": t, "track_id": tid,
                # has_track and confirmed are DIFFERENT conditions and must not be pooled:
                # a critical point on static structure legitimately has no track, and
                # dh_dt = 0 is correct there -- that is not confirmation latency.
                "has_track": bool(tid >= 0),
                "owner_is_dynamic": bool(true_owner >= 1000),
                "age": tk.age if tk else -1,
                "n_points": tk.n_points_last if tk else 0,
                "confirmed": bool(tk.confirmed) if tk else False,
                "misses": tk.misses if tk else -1,
                "frames_to_confirm": (tk.frames_to_confirm if tk else None),
                "est_speed": float(np.linalg.norm(v_est_crit)),
                "true_speed": float(np.linalg.norm(v_true_crit)),
                "vel_err": float(np.linalg.norm(v_est_crit - v_true_crit)),
                "e_crit": float(dd[j_crit] - dd_true[j_crit]) if n else 0.0,
                "binding": binding, "h_true": h_true_now,
                "n_tracks": len(src.tracker.tracks),
                "n_confirmed": sum(1 for x in src.tracker.tracks if x.confirmed),
            })

        # ---- step the world ------------------------------------------------------------
        p_new = p + np.asarray(u) * E.DT
        from experiments.exp3_dynamic_oracle import collision_type
        ctype = collision_type(p_new, P[t + 1])
        p = np.clip(p_new, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)

        if ctype is not None:
            outcome = "collision"
            if arm == "estimated" and track_rows:
                last = track_rows[-1]
                collided_unconfirmed = last["has_track"] and not last["confirmed"]
                collided_no_track = last["owner_is_dynamic"] and not last["has_track"]
            break
        if float(np.linalg.norm(goal - p)) < E.TARGET_RADIUS:
            outcome = "success"
            break

    rec = diag.summary(
        seed=seed, arm=arm, motion=motion, outcome=outcome, steps=steps,
        success=int(outcome == "success"), collision=int(outcome == "collision"),
        timeout=int(outcome == "timeout"),
        collision_type=ctype if outcome == "collision" else None,
        geometry_mismatch=geometry_mismatch,
        collided_while_unconfirmed=bool(collided_unconfirmed),
        collided_with_untracked_dynamic=bool(collided_no_track),
    )
    return rec, err_rows, track_rows


# ------------------------------------------------------------------ error analysis
def quantiles(x, qs=(50, 90, 95, 99)):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def error_report(rows):
    if not rows:
        return {}
    e = np.array([r["e"] for r in rows])
    h = np.array([r["h_true"] for r in rows])
    binding = np.array([r["binding"] for r in rows])
    crit = np.array([r["critical"] for r in rows])

    def cell(mask, label):
        sub = e[mask]
        if not len(sub):
            return {"cell": label, "n": 0}
        return {"cell": label, "n": int(mask.sum()), "mean": float(sub.mean()),
                "p_optimistic": float(np.mean(sub > 0)), **quantiles(sub),
                "max": float(sub.max()), "min": float(sub.min())}

    out = {
        "overall": cell(np.ones(len(e), bool), "all"),
        # THE HEADLINE SAFETY METRIC
        "headline_optimistic": cell((h < 0.6) & binding, "h<0.6 & binding"),
        "by_clearance": [
            cell((h >= lo) & (h < hi), f"h in [{lo},{hi})")
            for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 1e9))],
        "by_binding": [cell(binding, "binding"), cell(~binding, "not binding")],
        "by_critical": [cell(crit, "critical sample"), cell(~crit, "non-critical")],
        "by_ttc": [cell(np.array([r["ttc"] for r in rows]) < 1.0, "ttc < 1s"),
                   cell(np.array([r["ttc"] for r in rows]) >= 1.0, "ttc >= 1s")],
        "by_since_bounce": [
            cell(np.array([0 <= r["since_bounce"] <= 5 for r in rows]), "<=5 steps post-bounce"),
            cell(np.array([r["since_bounce"] > 5 for r in rows]), ">5 steps post-bounce")],
    }
    return out


def confirmation_report(track_rows, records):
    """Requirement 6: is 3-of-5 confirmation too slow for the hazard geometry?

    "No track" and "unconfirmed track" are reported SEPARATELY. A critical point lying on
    static structure legitimately has no track and dh_dt = 0 is correct for it; pooling the
    two would inflate the apparent latency problem with cases that are not latency at all.
    """
    if not track_rows:
        return {}
    n = len(track_rows)
    ftc = [r["frames_to_confirm"] for r in track_rows if r["frames_to_confirm"] is not None]
    dyn = [r for r in track_rows if r["owner_is_dynamic"]]
    dyn_no_track = [r for r in dyn if not r["has_track"]]
    dyn_unconf = [r for r in dyn if r["has_track"] and not r["confirmed"]]
    speeds = [r["true_speed"] for r in dyn if r["true_speed"] > 0]
    speed = float(np.mean(speeds)) if speeds else 0.0
    return {
        "frames_to_confirm_mean": float(np.mean(ftc)) if ftc else float("nan"),
        "frames_to_confirm_p90": float(np.percentile(ftc, 90)) if ftc else float("nan"),
        "obstacle_distance_during_confirmation_m":
            float(np.mean(ftc) * E.DT * speed) if ftc else float("nan"),
        "frac_steps_critical_owner_dynamic": len(dyn) / n,
        "frac_dynamic_critical_with_no_track": len(dyn_no_track) / max(len(dyn), 1),
        "frac_dynamic_critical_unconfirmed": len(dyn_unconf) / max(len(dyn), 1),
        "frac_dynamic_critical_unconfirmed_and_binding":
            sum(r["binding"] for r in dyn_unconf) / max(len(dyn), 1),
        "frac_dynamic_critical_unconfirmed_and_close":
            sum(r["h_true"] < 0.6 for r in dyn_unconf) / max(len(dyn), 1),
        "collisions_while_critical_track_unconfirmed":
            int(sum(r["collided_while_unconfirmed"] for r in records)),
        "collisions_with_untracked_dynamic_critical":
            int(sum(r["collided_with_untracked_dynamic"] for r in records)),
    }


def track_quality_report(track_rows):
    """Requirement 5: errors split by reconstruction point count and by range."""
    if not track_rows:
        return {}
    rows = [r for r in track_rows if r["owner_is_dynamic"] and r["has_track"]]
    if not rows:
        return {}

    def block(sel, label):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return {"cell": label, "n": 0}
        ve = np.array([r["vel_err"] for r in sub])
        ec = np.array([r["e_crit"] for r in sub])
        return {"cell": label, "n": len(sub),
                "vel_err_mean": float(ve.mean()), "vel_err_median": float(np.median(ve)),
                "vel_err_p90": float(np.percentile(ve, 90)),
                "vel_err_p99": float(np.percentile(ve, 99)), "vel_err_max": float(ve.max()),
                "e_mean": float(ec.mean()), "p_optimistic": float(np.mean(ec > 0))}

    return {
        "by_point_count": [block(lambda r: r["n_points"] == 1, "1 point"),
                           block(lambda r: r["n_points"] == 2, "2 points"),
                           block(lambda r: r["n_points"] >= 3, ">=3 points")],
        "by_range": [block(lambda r, a=a, b=b: a <= r["h_true"] + E.AGENT_RADIUS < b,
                           f"range [{a},{b})")
                     for a, b in ((0.3, 0.8), (0.8, 1.3), (1.3, 2.0), (2.0, 99.0))],
        "overall": block(lambda r: True, "all dynamic-critical steps"),
    }


def summarise(records, err_rows, track_rows, label):
    s = aggregate(records)
    types = Counter(r["collision_type"] for r in records if r["collision_type"])
    s.update(arm=label, episodes=len(records),
             success_rate=float(np.mean([r["success"] for r in records])),
             collision_rate=float(np.mean([r["collision"] for r in records])),
             timeout_rate=float(np.mean([r["timeout"] for r in records])),
             collisions_dynamic=types.get("dynamic", 0),
             collisions_static=types.get("static", 0),
             collisions_wall=types.get("wall", 0),
             geometry_mismatch=int(sum(r["geometry_mismatch"] for r in records)))
    s["error"] = error_report(err_rows)
    s["confirmation"] = confirmation_report(track_rows, records)
    s["track_quality"] = track_quality_report(track_rows)
    return s


def mcnemar(a, b):
    """Exact two-sided McNemar on paired binary outcomes (1 = collision)."""
    from math import comb
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return {"only_b": n01, "only_a": n10, "p": 1.0}
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return {"only_b": n01, "only_a": n10, "p": float(p)}


def paired_ci(a, b, n_boot=10000, seed=12345):
    """Bootstrap CI for the paired difference in collision rate (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--motion", choices=["cv", "ou"], default="cv")
    ap.add_argument("--arms", nargs="+", default=["oracle", "estimated", "zero"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--r-nominal", type=float, default=0.3)
    ap.add_argument("--radius-free", action="store_true")
    ap.add_argument("--no-static-rejection", action="store_true")
    ap.add_argument("--no-confirmation", action="store_true")
    ap.add_argument("--greedy-association", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tracker_kw = dict(r_nominal=args.r_nominal, radius_free=args.radius_free,
                      use_static_rejection=not args.no_static_rejection,
                      use_confirmation=not args.no_confirmation,
                      association="greedy" if args.greedy_association else "hungarian")

    report = {"config": vars(args), "alpha": ALPHA, "r_W": R_W, "eps": EPS,
              "tracker": {k: v for k, v in tracker_kw.items()}, "arms": {}}
    outcomes = {}

    for arm in args.arms:
        recs, errs, trks = [], [], []
        for i in range(args.episodes):
            r, e, tk = run_episode(seed=i, arm=arm, motion=args.motion,
                                   tracker_kw=dict(tracker_kw))
            recs.append(r)
            errs.extend(e)
            trks.extend(tk)
        report["arms"][arm] = {"summary": summarise(recs, errs, trks, arm),
                               "episodes": recs}
        outcomes[arm] = [r["collision"] for r in recs]

    if "estimated" in outcomes:
        report["paired"] = {}
        for ref in ("oracle", "zero"):
            if ref in outcomes:
                report["paired"][f"estimated_vs_{ref}"] = {
                    **mcnemar(outcomes["estimated"], outcomes[ref]),
                    **paired_ci(outcomes["estimated"], outcomes[ref])}

    keys = ["success_rate", "collision_rate", "timeout_rate", "collisions_dynamic",
            "collisions_static", "collisions_wall", "min_h_true", "min_h_true_min",
            "min_cbc_solved", "frac_infeasible", "n_infeasible", "mean_u_dev",
            "n_dcp_error", "frac_h_negative", "geometry_mismatch", "steps"]
    print(f"motion = {args.motion}   episodes = {args.episodes}   tracker = {tracker_kw}\n")
    print(f"{'metric':26s}" + "".join(f"{a:>14s}" for a in args.arms))
    for k in keys:
        row = [report["arms"][a]["summary"].get(k) for a in args.arms]
        if all(v is None for v in row):
            continue
        print(f"{k:26s}" + "".join(
            f"{v:>14.4g}" if isinstance(v, (int, float)) else f"{'-':>14s}" for v in row))

    for a in args.arms:
        err = report["arms"][a]["summary"].get("error", {})
        if not err:
            continue
        hl = err.get("headline_optimistic", {})
        ov = err.get("overall", {})
        print(f"\n[{a}] projected error e = dh_dt_est - dh_dt_true")
        print(f"   overall  n={ov.get('n')}  mean={ov.get('mean'):+.4f}  "
              f"P(e>0)={ov.get('p_optimistic'):.3f}  p50={ov.get('p50'):+.4f} "
              f"p90={ov.get('p90'):+.4f} p95={ov.get('p95'):+.4f} p99={ov.get('p99'):+.4f}")
        if hl.get("n"):
            print(f"   HEADLINE h<0.6 & binding  n={hl['n']}  P(e>0)={hl['p_optimistic']:.3f}"
                  f"  p90={hl['p90']:+.4f} p95={hl['p95']:+.4f} p99={hl['p99']:+.4f}"
                  f"  max={hl['max']:+.4f}")

    if "paired" in report:
        print("\npaired comparisons (collision as the outcome):")
        for k, v in report["paired"].items():
            print(f"   {k}: diff={v['diff']:+.4f} CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                  f"  McNemar only_a={v['only_a']} only_b={v['only_b']} p={v['p']:.5f}")

    out = Path(args.out or f"results/phase4/exp4_{args.motion}{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
