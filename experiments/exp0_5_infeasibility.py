"""Phase 1.5 -- DIAGNOSTIC ONLY: why the DRCCP goes infeasible with moving obstacles.

THIS FILE CHANGES NOTHING. It imports the Phase 1 controller and barrier source read-only
and analyses their behaviour. No remedy is selected or integrated; the Phase 1 gate stays
reproducible (`experiments/exp0_analytic.py` and results/phase1/exp0_*.json are untouched).

WHAT MAKES A STEP INFEASIBLE
----------------------------
The CLF constraint cannot cause infeasibility: delta is a free variable bounded only below,
so the CLF is always satisfiable. Infeasibility therefore comes from the barrier constraints
plus the action box alone.

With the reference's N = 5 kept samples and eps = 0.1, eps*N = 0.5 < 1. Each sample carries
probability mass 1/5 = 0.2 > 0.1, so the worst-epsilon tail lies entirely inside the single
worst sample and CVaR collapses to the minimum. The DR constraint is therefore exactly

    min_i CBC_i  >=  r_W * ||u_bar||_inf / eps  =:  tau

and with |u| <= max_v = 1 and alpha = 0.4, ||u_bar||_inf = max(1, alpha, |u_x|, |u_y|) = 1,
so tau = r_W / eps = 0.04. Feasibility of a step is then the linear program

    exists u :  |u_x|, |u_y| <= max_v ,  g_i . u  >=  rhs_i := tau - dh_dt_i - alpha * h_i

which this module solves directly with scipy.linprog rather than inferring it from CVXPY's
status string.

SAMPLE PROVENANCE
-----------------
AnalyticBarrierSource.samples emits, in order: 4 walls, then one per static rectangle, then
one per circle. Labels are reconstructed from those counts -- the source is not modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from dr_control.cbf_sources import AnalyticBarrierSource
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi
import experiments.exp0_analytic as E

ALPHA_REF = 0.4
R_W_REF = 0.004
EPS_REF = 0.1
TAU = R_W_REF / EPS_REF          # 0.04, derived above

N_WALLS = 4
N_RECTS = len(E.STATIC_RECTS)


def sample_labels(n_total):
    """'wall' / 'static' / 'dynamic' per row of the FULL sample set, by construction order."""
    labels = ["wall"] * N_WALLS + ["static"] * N_RECTS
    labels += ["dynamic"] * (n_total - len(labels))
    return np.array(labels)


def kind(label):
    """Collapse to the two families the question asks about."""
    return "dynamic" if label == "dynamic" else "static"


# --------------------------------------------------------------------- LP primitives
def _rows(xi, alpha, tau):
    """A_ub, b_ub for  g_i . u >= rhs_i  written as  -g_i . u <= -rhs_i."""
    g = xi[:, 2:4]
    rhs = tau - xi[:, 0] - alpha * xi[:, 1]
    return -g, -rhs


def is_feasible(xi, alpha=ALPHA_REF, tau=TAU, max_v=E.MAX_SPEED):
    A, b = _rows(xi, alpha, tau)
    r = linprog(c=np.zeros(2), A_ub=A, b_ub=b,
                bounds=[(-max_v, max_v)] * 2, method="highs")
    return bool(r.success)


def min_uniform_slack(xi, alpha=ALPHA_REF, tau=TAU, max_v=E.MAX_SPEED):
    """Smallest s >= 0 such that  g_i . u + s >= rhs_i  for all i is satisfiable.

    This is the minimum CBF slack, in the same units as the barrier constraint
    (distance/time, i.e. m/s), that would restore joint feasibility.
    """
    A, b = _rows(xi, alpha, tau)
    A3 = np.hstack([A, -np.ones((A.shape[0], 1))])
    r = linprog(c=[0, 0, 1], A_ub=A3, b_ub=b,
                bounds=[(-max_v, max_v)] * 2 + [(0, None)], method="highs")
    return float(r.x[2]) if r.success else float("nan")


def min_control_scale(xi, alpha=ALPHA_REF, tau=TAU):
    """Smallest rho with |u| <= rho making the constraints jointly satisfiable.

    rho / max_v is the factor by which the action box would have to grow. rho = inf (no
    finite bound helps) is reported as nan and means the conflict is not a speed problem.
    """
    A, b = _rows(xi, alpha, tau)
    n = A.shape[0]
    # variables [ux, uy, rho];  |ux| <= rho  ->  ux - rho <= 0, -ux - rho <= 0
    A3 = np.vstack([
        np.hstack([A, np.zeros((n, 1))]),
        [[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]],
    ])
    b3 = np.concatenate([b, np.zeros(4)])
    r = linprog(c=[0, 0, 1], A_ub=A3, b_ub=b3,
                bounds=[(None, None)] * 2 + [(0, None)], method="highs")
    return float(r.x[2]) if r.success else float("nan")


def min_alpha_feasible(xi, tau=TAU, cap=64.0, tol=1e-3):
    """Smallest alpha >= the reference value at which the step becomes feasible, else nan.

    DIRECTION MATTERS AND IS EASY TO GET BACKWARDS. The constraint is

        g_i . u  >=  tau - dh_dt_i - alpha * h_i

    and every h_i is positive here (frac_h_negative = 0 throughout Phase 1), so RAISING alpha
    shrinks the right-hand side and LOOSENS the constraint; lowering alpha tightens it. A
    smaller alpha is therefore more conservative, not less -- which is why the "reduced alpha"
    rollouts below collide far more, not less. Feasibility is monotone increasing in alpha, so
    bisection upward from the reference value is valid.

    nan means no alpha up to `cap` restores feasibility: the conflict is purely directional
    (constraints demanding opposed directions) and no barrier rate can fix it.
    """
    if is_feasible(xi, alpha=ALPHA_REF, tau=tau):
        return ALPHA_REF
    if not is_feasible(xi, alpha=cap, tau=tau):
        return float("nan")
    lo, hi = ALPHA_REF, cap
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_feasible(xi, alpha=mid, tau=tau):
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------- categorisation
def categorise(xi, labels, alpha=ALPHA_REF, tau=TAU, max_v=E.MAX_SPEED):
    """Locate a MINIMAL infeasible subset and name the conflict.

    Singletons first (a lone constraint no admissible u can meet is a control-bound
    limitation), then pairs, then anything higher-order.
    """
    n = len(xi)
    singles = [i for i in range(n) if not is_feasible(xi[[i]], alpha, tau, max_v)]
    if singles:
        return "control-bound limitation", [singles[0]]

    for i, j in itertools.combinations(range(n), 2):
        if not is_feasible(xi[[i, j]], alpha, tau, max_v):
            a, b = sorted((kind(labels[i]), kind(labels[j])))
            return f"{a}-{b} conflict", [i, j]

    for k in (3, 4, 5):
        if k > n:
            break
        for subset in itertools.combinations(range(n), k):
            if not is_feasible(xi[list(subset)], alpha, tau, max_v):
                return f"other ({k}-way conflict)", list(subset)
    return "other (unclassified)", []


# ----------------------------------------------------------------------- rollouts
def step_world(p, u, circles, vels):
    """RobotNavEnv dynamics, identical to experiments/exp0_analytic.run_episode."""
    p = np.clip(p + np.asarray(u) * E.DT, E.AGENT_RADIUS, E.WORLD_SIZE - E.AGENT_RADIUS)
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
    return p, circles, vels


def slack_fallback_u(xi, alpha=ALPHA_REF, tau=TAU, max_v=E.MAX_SPEED):
    """u from the minimum-uniform-slack LP: the least-violating admissible action."""
    A, b = _rows(xi, alpha, tau)
    A3 = np.hstack([A, -np.ones((A.shape[0], 1))])
    r = linprog(c=[0, 0, 1], A_ub=A3, b_ub=b,
                bounds=[(-max_v, max_v)] * 2 + [(0, None)], method="highs")
    return np.array(r.x[:2]) if r.success else np.zeros(2)


def rollout(seed, *, variant, alpha=ALPHA_REF, analyse=False):
    """One moving-obstacle episode. `variant` selects only the INFEASIBLE-step fallback."""
    rng = np.random.default_rng(E.DEV_SEED_BASE + seed)
    p, goal, circles, vels = E.sample_scenario(rng, 6, moving=True)
    src = AnalyticBarrierSource(world_size=E.WORLD_SIZE, r_robot=E.AGENT_RADIUS,
                                static_rects=E.STATIC_RECTS)
    ctrl = ClfCbfDrccpController(max_v=E.MAX_SPEED, cbf_rate=alpha)

    n_steps = n_infeas = 0
    infeasible_at = []
    records = []
    outcome = "timeout"
    last_infeasible_gap = None

    for step in range(E.MAX_STEPS):
        h, g, dd = src.samples(p, circles, E.OBSTACLE_RADIUS, vels)
        xi_all = build_xi(h, g, dd)
        labels_all = sample_labels(len(xi_all))

        order = np.argsort(xi_all[:, 1] * alpha + xi_all[:, 0])
        keep = order[: ctrl.n_keep]
        xi_kept, labels_kept = xi_all[keep], labels_all[keep]

        feasible = is_feasible(xi_kept, alpha)
        n_steps += 1
        if not feasible:
            n_infeas += 1
            infeasible_at.append(step)
            if analyse:
                cat, subset = categorise(xi_kept, labels_kept, alpha)
                records.append({
                    "seed": seed, "step": step, "category": cat,
                    "min_slack": min_uniform_slack(xi_kept, alpha),
                    "min_control_scale": min_control_scale(xi_kept, alpha),
                    "max_alpha_feasible": min_alpha_feasible(xi_kept),
                    "feasible_at_rw0": bool(is_feasible(xi_kept, alpha, tau=0.0)),
                    "feasible_at_rw0_alpha16": bool(is_feasible(xi_kept, 1.6, tau=0.0)),
                    "feasible_unbounded_control": bool(
                        is_feasible(xi_kept, alpha, max_v=1e6)),
                    "feasible_rw0_unbounded": bool(
                        is_feasible(xi_kept, alpha, tau=0.0, max_v=1e6)),
                    "conflict": [
                        {"label": labels_kept[i], "dh_dt": float(xi_kept[i, 0]),
                         "h": float(xi_kept[i, 1]),
                         "grad": [float(xi_kept[i, 2]), float(xi_kept[i, 3])]}
                        for i in subset],
                })

        if feasible or variant == "zero":
            u = ctrl.generate_controller(p, goal, xi_kept)
        elif variant == "slack":
            u = slack_fallback_u(xi_kept, alpha)
        else:
            raise ValueError(variant)

        p, circles, vels = step_world(p, u, circles, vels)

        if src.true_clearance(p, circles, E.OBSTACLE_RADIUS) <= 0.0:
            outcome = "collision"
            last_infeasible_gap = (step - infeasible_at[-1]) if infeasible_at else None
            break
        if float(np.linalg.norm(goal - p)) < E.TARGET_RADIUS:
            outcome = "success"
            break

    return {
        "seed": seed, "outcome": outcome, "steps": n_steps,
        "n_infeasible": n_infeas,
        "any_infeasible": n_infeas > 0,
        "steps_since_last_infeasible_at_collision": last_infeasible_gap,
        "records": records,
    }


def summarise(rolls, label):
    n = len(rolls)
    coll = [r for r in rolls if r["outcome"] == "collision"]
    steps = sum(r["steps"] for r in rolls)
    infeas = sum(r["n_infeasible"] for r in rolls)
    coll_with = [r for r in coll if r["any_infeasible"]]
    near = [r for r in coll_with
            if r["steps_since_last_infeasible_at_collision"] is not None
            and r["steps_since_last_infeasible_at_collision"] <= 10]
    return {
        "variant": label, "episodes": n,
        "success_rate": sum(r["outcome"] == "success" for r in rolls) / n,
        "collision_rate": len(coll) / n,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rolls) / n,
        "total_steps": steps,
        "infeasible_steps": infeas,
        "frac_infeasible_steps": infeas / steps if steps else 0.0,
        "episodes_with_any_infeasible": sum(r["any_infeasible"] for r in rolls) / n,
        "collisions": len(coll),
        "collisions_with_infeasibility_in_episode": len(coll_with),
        "collisions_within_10_steps_of_infeasibility": len(near),
        "collisions_with_no_infeasible_step": len(coll) - len(coll_with),
        "episodes_infeasible_but_no_collision": sum(
            r["any_infeasible"] and r["outcome"] != "collision" for r in rolls),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/phase1/exp0_5_infeasibility.json")
    args = ap.parse_args()

    print(f"tau = r_W/eps = {TAU}   alpha_ref = {ALPHA_REF}   max_v = {E.MAX_SPEED}\n")

    # --- reference behaviour, with full per-step analysis
    base = [rollout(i, variant="zero", analyse=True) for i in range(args.episodes)]
    base_sum = summarise(base, "u=0 (reference fallback)")

    recs = [r for roll in base for r in roll["records"]]
    cats = Counter(r["category"] for r in recs)
    slacks = np.array([r["min_slack"] for r in recs]) if recs else np.array([])
    scales = np.array([r["min_control_scale"] for r in recs]) if recs else np.array([])
    max_alpha = np.array([r["max_alpha_feasible"] for r in recs]) if recs else np.array([])

    # --- diagnostic variants (NOT integrated into the controller)
    variants = [base_sum,
                summarise([rollout(i, variant="slack") for i in range(args.episodes)],
                          "min-slack feasibility restoration")]
    for a in (0.8, 1.6):
        variants.append(summarise(
            [rollout(i, variant="zero", alpha=a) for i in range(args.episodes)],
            f"alpha = {a} (looser barrier; u=0 fallback)"))

    report = {
        "tau": TAU, "alpha_ref": ALPHA_REF, "episodes": args.episodes,
        "categories": dict(cats),
        "relaxation": {
            "n": int(len(recs)),
            "min_slack_mean": float(np.nanmean(slacks)) if len(slacks) else None,
            "min_slack_median": float(np.nanmedian(slacks)) if len(slacks) else None,
            "min_slack_max": float(np.nanmax(slacks)) if len(slacks) else None,
            "min_slack_p90": float(np.nanpercentile(slacks, 90)) if len(slacks) else None,
            "min_control_scale_median": float(np.nanmedian(scales)) if len(scales) else None,
            "min_control_scale_max": float(np.nanmax(scales)) if len(scales) else None,
            "n_control_scale_unbounded": int(np.sum(~np.isfinite(scales))) if len(scales) else 0,
            "min_alpha_feasible_median": float(np.nanmedian(max_alpha)) if len(max_alpha) else None,
            "n_no_alpha_restores_feasibility": int(sum(
                not np.isfinite(r["max_alpha_feasible"]) for r in recs)),
            "n_feasible_if_rw0": int(sum(r["feasible_at_rw0"] for r in recs)),
            "n_feasible_if_rw0_and_alpha16": int(sum(r["feasible_at_rw0_alpha16"] for r in recs)),
            "n_feasible_with_unbounded_control": int(sum(
                r["feasible_unbounded_control"] for r in recs)),
            "n_feasible_rw0_and_unbounded_control": int(sum(
                r["feasible_rw0_unbounded"] for r in recs)),
        },
        "variants": variants,
        "representative_conflicts": recs[:6],
    }

    print("=== infeasibility categories ===")
    tot = sum(cats.values()) or 1
    for k, v in cats.most_common():
        print(f"  {k:34s} {v:5d}  ({100*v/tot:5.1f}%)")

    print("\n=== required relaxation (over infeasible steps) ===")
    for k, v in report["relaxation"].items():
        print(f"  {k:34s} {v}")

    print("\n=== variant comparison (diagnostic only) ===")
    hdr = ["success_rate", "collision_rate", "timeout_rate", "frac_infeasible_steps",
           "collisions", "collisions_with_infeasibility_in_episode",
           "collisions_within_10_steps_of_infeasibility",
           "collisions_with_no_infeasible_step",
           "episodes_infeasible_but_no_collision"]
    for v in variants:
        print(f"\n  {v['variant']}")
        for k in hdr:
            print(f"    {k:46s} {v[k]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
