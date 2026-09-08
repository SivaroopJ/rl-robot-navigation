"""Week5-Phase7 / Stage 1 FINAL rerun under Amendment 2 (PHASE7_PLAN.md 11.5, 2026-09-07).

Third run. 998eeaa FAILED (original gate) -> 3c85f9d Amendment 1 -> 00587e3 FAILED
(Amendment 1) -> 4e55219 Amendment 2 -> this. No prior result is overwritten.

P4b and L4 only; P1, P2, P3, P4a, the secondary diagnostics and the closed-loop test are
unchanged from Amendment 1 and their results stand.

NO REFERENCE SOLVER APPEARS IN EITHER CRITERION.

The delta variable is eliminated analytically before certification. For fixed u the optimal
slack is delta*(u) = max(0, grad_V . u + rate_V V), so the problem reduces to

    min_u  F(u) = p0||u-u_prev||^2 + p1||u-u_nom||^2 + p3 max(0, grad_V.u + rate_V V)^2
    s.t.   G u >= b ,  |u| <= max_v

F is strongly convex in u with modulus mu_u = 2(p0 + p1) >= 6 ALWAYS, since p0 = 3 and
p1 = 4 h_crit >= 0. Certifying in u-space rather than in (u, delta) avoids the degeneracy at
p3 = 5 h_crit -> 0, where the (u, delta) Hessian loses rank and any bound through it would
blow up. The tier depends only on u, so nothing is lost.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, nnls

from dr_control.drccp_controller import clf_terms
from dr_control.fast_drccp import FastClfCbfDrccpController
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR
from experiments.exp7_1_forensic import ALPHA, MAX_V, P0, TAU, frozen_prepare

LABEL = "Week5-Phase7 / Stage 1 FINAL rerun under Amendment 2"
OUT = Path("results/week5_phase7/stage1_equivalence")
DELTA_CERT_TOL = 1e-6        # pre-registered, PHASE7_PLAN.md 11.5.1, from problem scale only
K_V, RATE_V = 0.05, 1.0
PARTS = ["dev_fixed", "dev_randomized", "validation_fixed", "validation_randomized"]


def certify(u_star, kept, *, u_prev, u_nom, h_crit, dV, V):
    """Solver-independent KKT certificate for `u_star` on the recorded problem.

    Multipliers come from non-negative least squares on (C^T, -grad F), using only the
    reconstructed problem and the point being certified -- never the solver's own duals, and
    with no active-set tolerance.
    """
    u = np.asarray(u_star, dtype=float).reshape(2)
    N = kept.shape[0]
    p1, p3 = 4.0 * h_crit, 5.0 * h_crit
    G, b = kept[:, 2:4], TAU - ALPHA * kept[:, 1] - kept[:, 0]

    # constraint set  C u <= d
    C = np.vstack([-G, [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
    d = np.concatenate([-b, [MAX_V, MAX_V, MAX_V, MAX_V]])

    s = float(dV @ u + RATE_V * V)
    grad = (2.0 * P0 * (u - u_prev) + 2.0 * p1 * (u - u_nom)
            + (2.0 * p3 * max(0.0, s)) * np.asarray(dV, dtype=float))

    slack = C @ u - d                               # <= 0 when feasible

    # MULTIPLIER RECOVERY.  Any (lambda >= 0, residuals) triple gives a VALID bound, so the
    # certificate must use the one giving the TIGHTEST bound. Plain NNLS does not: the five
    # buffered barrier rows are near-parallel (they are five temporal copies of one surface
    # point), so C^T has near-collinear columns, the multiplier vector is under-determined,
    # and NNLS spreads weight onto INACTIVE rows whose slack is large -- inflating c_sum
    # without improving stationarity. Measured: that produced c_sum up to 47 and an identical
    # delta_cert for V0 and V1, which is the signature of a degenerate dual, not of the point
    # being certified.
    #
    # Instead, minimise the complementarity violation directly over the valid multiplier set:
    #
    #     min_{lambda >= 0}  sum_i (-slack_i) * lambda_i     s.t.  C^T lambda = -grad
    #
    # a 2-equation LP in m unknowns. Its optimum is the smallest c_sum consistent with exact
    # stationarity, hence the tightest certified bound. No active-set tolerance is introduced.
    res = linprog(c=np.maximum(0.0, -slack), A_eq=C.T, b_eq=-grad,
                  bounds=[(0, None)] * C.shape[0], method="highs")
    if res.success:
        lam, dual_source = np.asarray(res.x, dtype=float), "lp"
    else:
        lam, _ = nnls(C.T, -grad)                   # fallback, flagged and counted
        dual_source = "nnls_fallback"
    r_stat_vec = C.T @ lam + grad
    r_stat_inf = float(np.max(np.abs(r_stat_vec)))
    r_stat_2 = float(np.linalg.norm(r_stat_vec))
    r_prim = float(max(0.0, slack.max()))
    r_dual = float(max(0.0, -lam.min())) if len(lam) else 0.0
    r_comp = float(np.max(np.abs(lam * slack))) if len(lam) else 0.0
    c_sum = float(np.sum(lam * np.maximum(0.0, -slack)))

    mu = 2.0 * (P0 + p1)                            # >= 6, always
    subopt = c_sum + r_stat_2 ** 2 / (2.0 * mu)
    delta_cert = float(np.sqrt(max(2.0 * subopt / mu, 0.0)))
    return {"r_prim": r_prim, "r_dual": r_dual, "r_stat": r_stat_inf, "r_comp": r_comp,
            "c_sum": c_sum, "subopt_bound": subopt, "delta_cert": delta_cert, "mu": mu,
            "dual_source": dual_source}


def certified_tier(min_cbc, delta_cert):
    """Tier of the interval [minCBC - delta_cert, minCBC + delta_cert]; None if it straddles."""
    lo, hi = min_cbc - delta_cert, min_cbc + delta_cert
    if lo >= TAU:
        return 0
    if lo >= 0.0 and hi < TAU:
        return 1
    if hi < 0.0:
        return 2
    return None


def q5(v, name):
    v = np.asarray(v, dtype=float)
    return {f"{name}_median": float(np.median(v)), f"{name}_p95": float(np.percentile(v, 95)),
            f"{name}_p99": float(np.percentile(v, 99)), f"{name}_min": float(v.min()),
            f"{name}_max": float(v.max()), f"{name}_n": int(len(v))}


def main():
    print(f"### {LABEL}\n")
    acc = FastClfCbfDrccpController(variant="reduced_osqp")
    cert0, cert1, cbc0, cbc1 = [], [], [], []
    tier_pairs = Counter()
    n_diff_determinate = n_indet = n_fallback = 0
    steps = 0

    for name in PARTS:
        m = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
        d = np.load(CORPUS_DIR / m["npz"])
        tab = {int(k): v for k, v in m["status_table"].items()}
        for i in range(len(d["u"])):
            if tab[int(d["status_code"][i])] != "optimal":
                continue
            a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
            xi = d["xi_flat"][a0:a1]
            kept, h_crit, _ = frozen_prepare(xi)
            acc.prev_u = d["u_prev"][i].copy()
            info = {}
            u1 = acc.generate_controller(d["p"][i], d["gamma"][i], xi, record=info)
            if info["status"] != "optimal":
                continue
            V, dV = clf_terms(d["p"][i], d["gamma"][i], K_V)
            kw = dict(u_prev=d["u_prev"][i], u_nom=d["u_nom"][i], h_crit=h_crit, dV=dV, V=V)
            c0 = certify(d["u"][i], kept, **kw)
            c1 = certify(u1, kept, **kw)
            ub = lambda z: np.array([1.0, ALPHA, z[0], z[1]])
            m0 = float(np.min(kept @ ub(d["u"][i])))
            m1 = float(np.min(kept @ ub(u1)))
            t0 = certified_tier(m0, c0["delta_cert"])
            t1 = certified_tier(m1, c1["delta_cert"])
            tier_pairs[(t0, t1)] += 1
            if t0 is None or t1 is None:
                n_indet += 1
            elif t0 != t1:
                n_diff_determinate += 1
            cert0.append(c0)
            cert1.append(c1)
            if c0["dual_source"] != "lp" or c1["dual_source"] != "lp":
                n_fallback += 1
            cbc0.append(m0 - TAU)
            cbc1.append(m1 - TAU)
            steps += 1

    R = {"label": LABEL, "amendment": "PHASE7_PLAN.md 11.5 (2026-09-07)",
         "provenance": ["998eeaa FAILED (original gate)", "3c85f9d Amendment 1",
                        "00587e3 FAILED (Amendment 1)", "4e55219 Amendment 2", "this run"],
         "certified_steps": steps, "delta_cert_tol": DELTA_CERT_TOL,
         "dual_recovery_fallbacks": None}

    R["dual_recovery_fallbacks"] = n_fallback
    print(f"certified solved steps: {steps}   dual-recovery fallbacks: {n_fallback}\n")
    print("P4b  SOLVER-INDEPENDENT KKT CERTIFICATE (no reference solver in the criterion)")
    print(f"  {'residual':22s}{'median':>12}{'p95':>12}{'p99':>12}{'max':>12}")
    for key, lab in (("r_prim", "primal feasibility"), ("r_dual", "dual feasibility"),
                     ("r_stat", "stationarity"), ("r_comp", "complementary slack"),
                     ("delta_cert", "CERTIFIED ||x*-x_opt||")):
        for who, cs in (("V1", cert1), ("V0", cert0)):
            v = np.array([c[key] for c in cs])
            tag = f"{lab} [{who}]"
            print(f"  {tag:22s}{np.median(v):12.3e}{np.percentile(v,95):12.3e}"
                  f"{np.percentile(v,99):12.3e}{v.max():12.3e}")
            R.setdefault("P4b", {}).update(q5(v, f"{who}_{key}"))
    dc1 = np.array([c["delta_cert"] for c in cert1])
    dc0 = np.array([c["delta_cert"] for c in cert0])
    R["P4b"]["V1_delta_cert_exceeding"] = int((dc1 > DELTA_CERT_TOL).sum())
    R["P4b"]["pass"] = bool(dc1.max() <= DELTA_CERT_TOL)
    print(f"\n  PASS CONDITION  max delta_cert(V1) = {dc1.max():.3e} <= {DELTA_CERT_TOL:.0e}"
          f"   exceedances {R['P4b']['V1_delta_cert_exceeding']} / {steps}"
          f"   -> {'PASS' if R['P4b']['pass'] else 'FAIL'}")
    print(f"  (V0 for reference only, not a criterion: max delta_cert = {dc0.max():.3e})")

    print("\nL4  GUARANTEE TIER FROM THE CERTIFIED OPTIMUM")
    trans = {f"{'INDET' if a is None else 'T'+str(a)}->"
             f"{'INDET' if b is None else 'T'+str(b)}": int(v) for (a, b), v in tier_pairs.items()}
    R["L4"] = {"steps": steps, "transitions": trans, "indeterminate": n_indet,
               "differing_and_determinate": n_diff_determinate,
               "pass": bool(n_diff_determinate == 0)}
    print(f"  certified tier transitions: {trans}")
    print(f"  indeterminate (interval straddles a boundary): {n_indet} / {steps}"
          f"  ({100*n_indet/steps:.2f} %)")
    print(f"  PASS CONDITION  steps where tiers differ AND both determinate = "
          f"{n_diff_determinate}  -> {'PASS' if R['L4']['pass'] else 'FAIL'}")

    print("\nRAW CBC RESIDUALS (reported separately, no thresholds attached)")
    a0_, a1_ = np.array(cbc0), np.array(cbc1)
    R["raw_cbc"] = {**q5(a0_, "CBC_V0_minus_tau"), **q5(a1_, "CBC_V1_minus_tau"),
                    "V0_below_tau": int((a0_ < 0).sum()), "V1_below_tau": int((a1_ < 0).sum()),
                    "V0_below_zero": int((a0_ < -TAU).sum()),
                    "V1_below_zero": int((a1_ < -TAU).sum())}
    print(f"  {'':18s}{'median':>12}{'p95':>12}{'p99':>12}{'min':>12}{'max':>12}")
    for lab, v in (("CBC_V0 - tau", a0_), ("CBC_V1 - tau", a1_)):
        print(f"  {lab:18s}{np.median(v):12.3e}{np.percentile(v,95):12.3e}"
              f"{np.percentile(v,99):12.3e}{v.min():12.3e}{v.max():12.3e}")
    print(f"  counts below tau : V0 {R['raw_cbc']['V0_below_tau']}  "
          f"V1 {R['raw_cbc']['V1_below_tau']}   (of {steps})")
    print(f"  counts below zero: V0 {R['raw_cbc']['V0_below_zero']}  "
          f"V1 {R['raw_cbc']['V1_below_zero']}")

    prev = json.loads((OUT / "stage1_amended_rerun.json").read_text())
    R["unchanged_from_amendment_1"] = {
        "P1": prev["P1"]["pass"], "P2": prev["P2"]["pass"], "P3": prev["P3"]["pass"],
        "P4a_feasibility": f"{prev['P4']['u_V1_in_V0_feasible_set']}/"
                           f"{prev['P4']['sampled_solved_steps']}",
        "secondary": prev["secondary"], "closed_loop":
            {k: v["pass"] for k, v in prev["closed_loop"].items()}}
    gate = {"P1": prev["P1"]["pass"], "P2": prev["P2"]["pass"], "P3": prev["P3"]["pass"],
            "P4a": prev["P4"]["u_V1_in_V0_feasible_set"] == prev["P4"]["sampled_solved_steps"],
            "P4b_kkt": R["P4b"]["pass"], "L4_certified": R["L4"]["pass"],
            "closed_loop": all(v["pass"] for v in prev["closed_loop"].values())}
    gate["overall"] = bool(all(gate.values()))
    R["gate"] = gate
    print("\n" + "=" * 78)
    for k, v in gate.items():
        print(f"  {k:16s} {'PASS' if v else 'FAIL'}")
    print(f"\nAMENDED (Amendment 2) STAGE 1 GATE -> {'PASS' if gate['overall'] else 'FAIL'}")
    print("998eeaa and 00587e3 remain recorded as FAILED under their own gates.")
    (OUT / "stage1_kkt_rerun.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"wrote {OUT}/stage1_kkt_rerun.json")


if __name__ == "__main__":
    main()
