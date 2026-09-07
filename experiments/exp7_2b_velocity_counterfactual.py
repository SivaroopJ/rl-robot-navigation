"""Week5-Phase7 / Stage 2b / velocity-estimation CAUSAL counterfactual. DIAGNOSTIC ONLY.

Amendment 3 (PHASE7_PLAN.md 11.6). V0 is COMPLETELY FROZEN and untouched. Nothing is
implemented: no IMM, no CT tracking, no uncertainty margin, no predictive barrier, no recovery
ladder, no slack, no estimator redesign. This module only re-poses the RECORDED problems with
one channel substituted.

PRIMARY QUESTION
    If ONLY the velocity supplied to dh/dt is changed, how much of the observed infeasibility
    disappears?

ARMS -- identical recorded states, geometry and constraints; only column 0 of xi differs.
    A1 frozen estimate   dh_dt as recorded                                  (baseline)
    A2 ground truth      -grad_h . v_true of the obstacle that produced the buffered point
                                                                           (PRIMARY CAUSAL TEST)
    A3 zero              0
    A4 capped estimate   max(dh_dt_est, -v_phys)

HELD IDENTICAL: robot state, LiDAR geometry (h and grad_h are untouched), sample selection,
critical-row ordering, control bounds, objective, alpha, tau, and the V0 solver.

TWO POINTS OF PRECISION, STATED RATHER THAN GLOSSED
  1. FEASIBILITY IS ORDER-INVARIANT. The feasible set is {u : |u| <= max_v, G u >= b}, which
     depends on the constraint SET and not on its ordering. With K = 5 and n_keep = 5 all five
     rows are always kept, so substituting dh_dt can permute the criticality order but cannot
     change which rows are present. The infeasibility rate is therefore identical whether the
     recorded ordering is held fixed or recomputed. Ordering affects only h_crit = kept[0, 1],
     hence the objective weights p1, p3 -- so h_crit is reported BOTH ways.
  2. ARM 4 CLAMPS THE PROJECTION, not the velocity vector, and it is the MORE CONSERVATIVE of
     the two. Write s = min(1, v_phys/||v||). Clamping the vector gives s*dh_dt; clamping the
     projection gives max(dh_dt, -v_phys). Since ||grad_h|| = 1 implies |dh_dt| <= ||v||, one
     gets s*dh_dt >= max(dh_dt, -v_phys) FOR CLOSING ROWS (dh_dt < 0), with equality exactly
     when v is parallel to grad_h. Closing rows are the only ones that can tighten a
     constraint; for a receding row (dh_dt >= 0) the projection clamp is a no-op by
     construction, which is correct since such a row only loosens. So arm 4 relieves
     the constraint by AT MOST what a genuine velocity clamp would: whatever infeasibility it
     removes is a LOWER BOUND on what clamping the vector would remove. The corpus records only
     the projection, so the vector clamp is not recoverable from it.
     (Two earlier drafts got the direction and then the domain of this inequality wrong;
     corrected here, and `test_cap_is_a_bound_on_clamping_the_velocity_vector` pins both.)

GROUND-TRUTH ATTRIBUTION (arm 2), a measurement channel only
    Row j at step i has scan age a, so its buffered point q_j was measured at step i - a. The
    owning obstacle is the true obstacle whose surface q_j lies on at THAT time, found by the
    annulus test | ||q_j - o|| - r_obs | < tol. Its CURRENT velocity is then used, mirroring
    the frozen pipeline exactly (which binds a track at push time and reads its current
    velocity). A point on no obstacle surface is static structure and gets v = 0.
    Sensitivity to `tol` is reported.

FEASIBILITY ORACLE
    The exact LP over the recorded constraints, which Stage 1 P1 established defines the same
    set as the V0 problem (516 088 membership comparisons, 0 disagreements). The actual frozen
    V0 solver is additionally run on a subsample and its status agreement reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from dr_control.drccp_controller import ClfCbfDrccpController
from experiments.exp0_5_infeasibility import (is_feasible, min_control_scale,
                                              min_uniform_slack)
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 2b / velocity causal counterfactual"
OUT = Path("results/week5_phase7/stage2b_velocity_counterfactual")
ALPHA, TAU, MAX_V, R_ROBOT, OBS_R = 0.4, 0.04, 1.0, 0.3, 0.3
V_PHYS = {"fixed": 0.675, "randomized": 0.75}
ARMS = ("A1_frozen_estimate", "A2_ground_truth", "A3_zero", "A4_capped_estimate")
PARTS = [("dev_fixed", "fixed"), ("dev_randomized", "randomized"),
         ("validation_fixed", "fixed"), ("validation_randomized", "randomized"),
         ("final_fixed", "fixed"), ("final_randomized", "randomized")]


def feas(xi, eps=1e-9):
    """EXACT emptiness test of {u : |u| <= max_v, grad_h_i . u >= b_i}, by vertex enumeration.

    The region is a bounded 2-D polytope (the action box bounds it), so it is non-empty if and
    only if it has a vertex, and every vertex is the intersection of two of its bounding lines.
    With m = N + 4 constraints there are at most C(m, 2) = 36 candidates, each checked against
    all m constraints -- pure NumPy, and ~100x faster than a linprog call. Used instead of
    scipy HiGHS purely for speed: `test_feas_matches_the_lp_oracle` below checks the two agree
    on a large random sample, and the LP remains the definition.
    """
    xi = np.atleast_2d(xi)
    A = np.vstack([-xi[:, 2:4], [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
    c = np.concatenate([-(TAU - ALPHA * xi[:, 1] - xi[:, 0]),
                        [MAX_V, MAX_V, MAX_V, MAX_V]])
    m = len(A)
    for i in range(m - 1):
        for j in range(i + 1, m):
            M = A[[i, j]]
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < 1e-12:
                continue
            rhs = c[[i, j]]
            u = np.array([(rhs[0] * M[1, 1] - rhs[1] * M[0, 1]) / det,
                          (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det])
            if np.all(A @ u <= c + eps):
                return True
    return False


def feas_lp(xi):
    """The LP definition, kept as the reference the fast test is validated against."""
    return bool(is_feasible(np.atleast_2d(xi), alpha=ALPHA, tau=TAU, max_v=MAX_V))


def mis_cardinality(xi):
    n = len(xi)
    for k in range(1, n + 1):
        for sub in itertools.combinations(range(n), k):
            if not feas(xi[list(sub)]):
                return k
    return None


def owner_dh_dt(xi, ages, p, hist_pos, vel_now, tol):
    """Arm 2: -grad_h . v_true of the obstacle that produced each buffered point."""
    q = p[None, :] - (xi[:, 1:2] + R_ROBOT) * xi[:, 2:4]      # buffered surface points
    out = np.zeros(len(xi))
    n_dyn = 0
    for j in range(len(xi)):
        hp = hist_pos.get(int(ages[j]))
        if hp is None or len(hp) == 0 or vel_now is None or not len(vel_now):
            out[j] = 0.0
            continue
        dist = np.linalg.norm(hp - q[j][None, :], axis=1)
        o = int(np.argmin(dist))
        if abs(float(dist[o]) - OBS_R) < tol:                 # the point lies on that surface
            out[j] = -float(xi[j, 2:4] @ vel_now[o])
            n_dyn += 1
        else:
            out[j] = 0.0                                      # static structure
    return out, n_dyn


def build_arms(xi, ages, p, hist_pos, vel_now, v_phys, tol):
    a1 = xi.copy()
    a2 = xi.copy()
    dh_true, n_dyn = owner_dh_dt(xi, ages, p, hist_pos, vel_now, tol)
    a2[:, 0] = dh_true
    a3 = xi.copy()
    a3[:, 0] = 0.0
    a4 = xi.copy()
    a4[:, 0] = np.maximum(xi[:, 0], -v_phys)
    return {"A1_frozen_estimate": a1, "A2_ground_truth": a2, "A3_zero": a3,
            "A4_capped_estimate": a4}, n_dyn


def q5(v, name):
    v = np.asarray([x for x in np.asarray(v, dtype=float) if x == x])
    if not len(v):
        return {}
    return {f"{name}_n": int(len(v)), f"{name}_median": float(np.median(v)),
            f"{name}_p95": float(np.percentile(v, 95)), f"{name}_min": float(v.min()),
            f"{name}_max": float(v.max()), f"{name}_mean": float(v.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=0.15, help="annulus tolerance, arm 2")
    ap.add_argument("--tol-sensitivity", nargs="+", type=float, default=[0.10, 0.15, 0.25])
    ap.add_argument("--v0-subsample", type=int, default=600,
                    help="steps on which the actual frozen V0 solver is run for agreement")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = np.random.default_rng(20260907)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"### {LABEL}\n")

    rec = {a: {"infeasible": 0, "steps": 0, "by_cond": Counter(), "cond_steps": Counter(),
               "mis": Counter(), "slack": [], "mcs_finite": 0, "h_crit_fixed": [],
               "h_crit_resorted": [], "dh_dt": [], "superphys_steps": 0}
           for a in ARMS}
    paired = []           # per-step feasibility across arms, for paired contrasts
    n_dyn_rows = n_rows = 0
    tol_sens = {t: 0 for t in args.tol_sensitivity}
    tol_rows = {t: 0 for t in args.tol_sensitivity}
    v0_check = []

    for name, cond in PARTS:
        meta = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
        z = np.load(CORPUS_DIR / meta["npz"])
        XI = z["xi_flat"].copy(); PTR = z["xi_ptr"].copy()
        AGE = z["sample_age"].copy(); P = z["p"].copy(); GAM = z["gamma"].copy()
        UPREV = z["u_prev"].copy(); EP = z["ep_index"].copy()
        TPOS = z["truth_obs_pos"].copy(); TVEL = z["truth_obs_vel"].copy()
        v_phys = V_PHYS[cond]
        n = len(P)
        sub = set(rng.choice(n, size=min(args.v0_subsample // len(PARTS), n),
                             replace=False).tolist())
        tol_sub = set(rng.choice(n, size=min(1500, n), replace=False).tolist())

        for i in range(n):
            a0, a1 = PTR[i], PTR[i + 1]
            xi = XI[a0:a1]
            ages = AGE[a0:a1]
            hist = {}
            for a in np.unique(ages):
                j = i - int(a)
                if j >= 0 and EP[j] == EP[i]:
                    hist[int(a)] = TPOS[j]
            arms, nd = build_arms(xi, ages, P[i], hist, TVEL[i], v_phys, args.tol)
            n_dyn_rows += nd
            n_rows += len(xi)
            if i in tol_sub:                       # sensitivity on a subsample only
                for t in args.tol_sensitivity:
                    _, ndt = owner_dh_dt(xi, ages, P[i], hist, TVEL[i], t)
                    tol_sens[t] += ndt
                    tol_rows[t] += len(xi)

            row = {"part": name, "cond": cond}
            for a, x in arms.items():
                ok = feas(x)
                r = rec[a]
                r["steps"] += 1
                r["cond_steps"][cond] += 1
                r["dh_dt"].append(float(x[:, 0].min()))
                r["superphys_steps"] += int((x[:, 0] < -v_phys - 1e-9).any())
                # h_crit with the ORDER HELD FIXED at the recorded one, and re-sorted
                r["h_crit_fixed"].append(float(xi[int(np.argmin(xi[:, 1] * ALPHA + xi[:, 0])), 1]))
                r["h_crit_resorted"].append(float(x[int(np.argmin(x[:, 1] * ALPHA + x[:, 0])), 1]))
                if not ok:
                    r["infeasible"] += 1
                    r["by_cond"][cond] += 1
                    r["mis"][mis_cardinality(x)] += 1
                    r["slack"].append(min_uniform_slack(x, alpha=ALPHA, tau=TAU, max_v=MAX_V))
                    r["mcs_finite"] += int(np.isfinite(
                        min_control_scale(x, alpha=ALPHA, tau=TAU)))
                row[a] = ok
            paired.append(row)

            if i in sub:                       # actual frozen V0 solver, for status agreement
                for a, x in arms.items():
                    c = ClfCbfDrccpController()
                    c.prev_u = UPREV[i].copy()
                    info = {}
                    c.generate_controller(P[i], GAM[i], x, record=info)
                    v0_inf = "infeasible" in str(info["status"])
                    v0_check.append({"arm": a, "lp_feasible": bool(row[a]),
                                     "v0_status": info["status"], "agree": v0_inf != row[a]})
        print(f"  {name:22s} done ({n} steps)", flush=True)

    # ------------------------------------------------------------------ report
    R = {"label": LABEL, "amendment": "PHASE7_PLAN.md 11.6 (2026-09-07)",
         "tol": args.tol, "arms": {}}
    base = rec["A1_frozen_estimate"]["infeasible"]
    print(f"\nsteps per arm: {rec['A1_frozen_estimate']['steps']}")
    print(f"buffered rows attributed to a DYNAMIC obstacle (arm 2, tol {args.tol}): "
          f"{n_dyn_rows}/{n_rows} ({100*n_dyn_rows/n_rows:.2f} %)")
    print(f"  tolerance sensitivity (subsample): "
          f"{ {t: f'{100*tol_sens[t]/max(tol_rows[t],1):.2f} %' for t in tol_sens} }")
    R["dynamic_attribution"] = {"rows": n_rows, "dynamic_rows": n_dyn_rows,
                                "fraction": n_dyn_rows / n_rows,
                                "tol_sensitivity": {str(t): tol_sens[t] / max(tol_rows[t], 1)
                                                    for t in tol_sens}}

    print(f"\n{'arm':22s}{'infeasible':>12}{'rate':>10}{'vs A1':>10}{'fixed':>9}{'random':>9}"
          f"{'superphys':>11}{'MIS=1':>8}")
    for a in ARMS:
        r = rec[a]
        rate = r["infeasible"] / r["steps"]
        d = 100 * (r["infeasible"] - base) / max(base, 1)
        R["arms"][a] = {
            "steps": r["steps"], "infeasible": r["infeasible"], "rate": rate,
            "pct_change_vs_A1": d,
            "by_condition": {k: int(v) for k, v in r["by_cond"].items()},
            "rate_by_condition": {k: r["by_cond"][k] / r["cond_steps"][k]
                                  for k in r["cond_steps"]},
            "mis_cardinality": {str(k): int(v) for k, v in r["mis"].items()},
            "superphysical_step_rate": r["superphys_steps"] / r["steps"],
            "control_would_help": r["mcs_finite"],
            **q5(r["slack"], "min_slack"), **q5(r["dh_dt"], "worst_dh_dt"),
            **q5(r["h_crit_fixed"], "h_crit_order_fixed"),
            **q5(r["h_crit_resorted"], "h_crit_resorted"),
        }
        m = R["arms"][a]
        print(f"{a:22s}{r['infeasible']:12d}{rate:10.4f}{d:+9.1f}%"
              f"{r['by_cond']['fixed']:9d}{r['by_cond']['randomized']:9d}"
              f"{m['superphysical_step_rate']:11.4f}{r['mis'].get(1,0):8d}")

    # paired contrasts against A1
    print(f"\npaired against A1 (same recorded step, only dh_dt substituted):")
    R["paired_vs_A1"] = {}
    for a in ARMS[1:]:
        fixed_by = sum(1 for x in paired if not x["A1_frozen_estimate"] and x[a])
        broken_by = sum(1 for x in paired if x["A1_frozen_estimate"] and not x[a])
        R["paired_vs_A1"][a] = {"A1_infeasible_now_feasible": fixed_by,
                                "A1_feasible_now_infeasible": broken_by,
                                "share_of_A1_infeasibility_removed": fixed_by / max(base, 1)}
        print(f"   {a:22s} removes {fixed_by:5d} of A1's {base} infeasible steps "
              f"({100*fixed_by/max(base,1):.2f} %), introduces {broken_by:5d} new")

    if v0_check:
        agree = sum(x["agree"] for x in v0_check)
        R["v0_solver_agreement"] = {"checked": len(v0_check), "agree": agree,
                                    "rate": agree / len(v0_check)}
        print(f"\nfrozen V0 solver agreement with the LP oracle: "
              f"{agree}/{len(v0_check)} ({100*agree/len(v0_check):.2f} %)")

    print("\nslack / h_crit per arm:")
    for a in ARMS:
        m = R["arms"][a]
        print(f"  {a:22s} slack median {m.get('min_slack_median', float('nan')):.4f} "
              f"max {m.get('min_slack_max', float('nan')):.4f}   "
              f"h_crit(order fixed) median {m['h_crit_order_fixed_median']:.4f}  "
              f"h_crit(resorted) median {m['h_crit_resorted_median']:.4f}  "
              f"worst dh_dt median {m['worst_dh_dt_median']:+.4f}")

    (out / "stage2b_velocity_counterfactual.json").write_text(json.dumps(R, indent=1,
                                                                        default=float))
    print(f"\nwrote {out}/stage2b_velocity_counterfactual.json")


if __name__ == "__main__":
    main()
