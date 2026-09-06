"""Week5-Phase7 / Stage 1 / FORENSIC DIAGNOSIS of the failed gate. Changes nothing.

Read-only. No tolerance, status mapping, controller, objective, sample selection, tau or alpha
is modified anywhere in this file. It only measures.

INDEPENDENCE RULE (brief section 3): the accelerated implementation is NEVER used as evidence
that it is equivalent. Every equivalence claim below is checked by re-deriving the FROZEN V0
problem from the recorded inputs and testing it with machinery that belongs to neither
implementation:
    * V0 membership oracle: for a FIXED u, solve the (t, s_i) feasibility LP with scipy HiGHS
    * reduced membership: direct arithmetic, min_i (u_bar . xi_i) >= tau
    * optimum: the frozen formulation solved by CLARABEL at tol 1e-14
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from dr_control.drccp_controller import ClfCbfDrccpController, clf_terms, nominal_action
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 1 / forensic diagnosis"
OUT = Path("results/week5_phase7/stage1_equivalence")
ALPHA, R_W, EPS, MAX_V, N_KEEP, P0, K_V, RATE_V = 0.4, 0.004, 0.1, 1.0, 5, 3.0, 0.05, 1.0
TAU = R_W / EPS


# ---------------------------------------------------------------- independent V0 machinery
def frozen_prepare(xi):
    """The frozen sort/truncate, transcribed from drccp_controller.generate_controller."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    order = np.argsort(xi[:, 1] * ALPHA + xi[:, 0])
    kept = xi[order][:N_KEEP]
    return kept, float(kept[0, 1]), order


def v0_membership(u, kept):
    """Is `u` in the V0 feasible set? Solve the (t, s_i) LP for FIXED u. Independent of both
    implementations: this is the literal constraint block, with u substituted in.

        r_W*|u_bar_j|/eps <= t - sum_i s_i/(N*eps)   j = 1..4
        s_i >= 0 ,  s_i >= t - u_bar . xi_i
    """
    u = np.asarray(u, dtype=float).reshape(2)
    if np.max(np.abs(u)) > MAX_V + 1e-12:
        return False
    N = kept.shape[0]
    u_bar = np.array([1.0, ALPHA, u[0], u[1]])
    c_i = kept @ u_bar
    lhs = R_W * np.max(np.abs(u_bar)) / EPS
    # variables z = [t, s_1..s_N]
    A, b = [], []
    row = np.zeros(1 + N)
    row[0] = -1.0
    row[1:] = 1.0 / (N * EPS)
    A.append(row)
    b.append(-lhs)                                   # lhs <= t - sum s/(N eps)
    for i in range(N):
        r = np.zeros(1 + N)
        r[0] = 1.0
        r[1 + i] = -1.0
        A.append(r)
        b.append(float(c_i[i]))                      # t - s_i <= c_i
    res = linprog(c=np.zeros(1 + N), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None)] + [(0, None)] * N, method="highs")
    return bool(res.success)


def reduced_membership(u, kept):
    """Is `u` in the reduced feasible set? Direct arithmetic, no solver at all."""
    u = np.asarray(u, dtype=float).reshape(2)
    if np.max(np.abs(u)) > MAX_V + 1e-12:
        return False
    u_bar = np.array([1.0, ALPHA, u[0], u[1]])
    return bool(np.min(kept @ u_bar) >= R_W * np.max(np.abs(u_bar)) / EPS - 1e-12)


def arbiter(p, gamma, kept, u_prev):
    """Frozen formulation, CLARABEL at tol 1e-14. Belongs to neither implementation."""
    h = float(kept[0, 1])
    if h < 0:
        return None, np.nan, "h_crit<0"
    N = kept.shape[0]
    u, si, t, d = cp.Variable(2), cp.Variable(N), cp.Variable(), cp.Variable()
    ub = cp.hstack([1.0, ALPHA, u[0], u[1]])
    cons = [R_W * cp.abs(ub) / EPS <= (t - cp.sum(si) / (N * EPS)) * np.ones(4),
            si >= 0, cp.abs(u) <= MAX_V]
    cons += [si[i] >= t - ub @ kept[i] for i in range(N)]
    V, dV = clf_terms(p, gamma, K_V)
    cons += [dV @ u + RATE_V * V <= d, d >= 0]
    un = nominal_action(p, gamma, MAX_V)
    prob = cp.Problem(cp.Minimize(P0 * cp.sum_squares(u - u_prev)
                                  + 4 * h * cp.sum_squares(u - un)
                                  + 5 * h * cp.square(d)), cons)
    try:
        prob.solve(solver="CLARABEL", tol_gap_abs=1e-14, tol_gap_rel=1e-14, tol_feas=1e-14)
    except cp.error.SolverError as e:
        return None, np.nan, f"SolverError:{e}"
    if prob.status != "optimal" or u.value is None:
        return None, np.nan, prob.status
    return np.asarray(u.value).reshape(2), float(prob.value), prob.status


def lp_exact_feasible(kept):
    """Exact emptiness test of {u: |u|<=max_v, G u >= b}. tau is CONSTANT here, which is
    itself verified in check_tau_constant()."""
    G, b = kept[:, 2:4], TAU - ALPHA * kept[:, 1] - kept[:, 0]
    r = linprog(c=np.zeros(2), A_ub=-G, b_ub=-b, bounds=[(-MAX_V, MAX_V)] * 2, method="highs")
    return bool(r.success)


# ---------------------------------------------------------------- section 3 checks
def check_tau_constant(rng, n=200000):
    """||u_bar||_inf = max(1, alpha, |u_x|, |u_y|) must equal 1 for EVERY admissible u."""
    u = rng.uniform(-MAX_V, MAX_V, (n, 2))
    v = np.maximum(np.maximum(1.0, ALPHA), np.max(np.abs(u), axis=1))
    return float(v.min()), float(v.max())


def check_feasible_set_equality(kept, rng, n_grid=41, n_rand=400):
    """Do the V0 and reduced feasible sets contain exactly the same u? Grid + random probes.
    Neither side uses the accelerated implementation."""
    pts = [np.array([a, b]) for a in np.linspace(-MAX_V, MAX_V, n_grid)
           for b in np.linspace(-MAX_V, MAX_V, n_grid)]
    pts += [rng.uniform(-MAX_V, MAX_V, 2) for _ in range(n_rand)]
    dis = []
    for u in pts:
        a, b = v0_membership(u, kept), reduced_membership(u, kept)
        if a != b:
            u_bar = np.array([1.0, ALPHA, u[0], u[1]])
            dis.append((u.tolist(), a, b, float(np.min(kept @ u_bar) - TAU)))
    return len(pts), dis


def check_coefficients(kept_from_trace, xi_full):
    """Constraint/objective coefficients, ordering, h/dh_dt, h_crit -- re-derived from the
    RECORDED xi and compared with what the corpus recorded the frozen controller using."""
    kept, h_crit, order = frozen_prepare(xi_full)
    return {
        "kept_matches_recorded": bool(np.array_equal(kept, kept_from_trace)),
        "h_crit": h_crit,
        "p1": 4.0 * h_crit, "p3": 5.0 * h_crit, "p0": P0,
        "sort_key_first": float(kept[0, 1] * ALPHA + kept[0, 0]),
        "sort_is_ascending": bool(np.all(np.diff(kept[:, 1] * ALPHA + kept[:, 0]) >= -1e-15)),
        "b_rows": (TAU - ALPHA * kept[:, 1] - kept[:, 0]).tolist(),
        "grad_norms": np.linalg.norm(kept[:, 2:4], axis=1).tolist(),
    }


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(0)
    print(f"### {LABEL}\n")
    report = {"label": LABEL}

    # ---- tau constancy -----------------------------------------------------------------
    lo, hi = check_tau_constant(rng)
    report["tau_check"] = {"min_norm_inf": lo, "max_norm_inf": hi, "tau": TAU,
                           "constant": lo == hi == 1.0}
    print(f"[3] tau: ||u_bar||_inf over 200k admissible u -> [{lo}, {hi}]; "
          f"tau = r_W/eps = {TAU}  constant={lo == hi == 1.0}")

    # ---- enriched infeasible corpus: full cross-tabulation ------------------------------
    meta = json.loads((CORPUS_DIR / "meta_infeasible_enriched.json").read_text())
    parts = json.loads((CORPUS_DIR / "INDEX.json").read_text())["parts"]
    table = {}
    for pt in parts:
        table.update({int(k): v for k, v in pt["status_table"].items()})
    d = np.load(CORPUS_DIR / meta["npz"])
    n = len(d["u"])
    dis = json.loads((OUT / "stage1_disagreements.json").read_text())

    lp_true, h_neg, kept_all = [], 0, []
    for i in range(n):
        a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
        kept, h_crit, _ = frozen_prepare(d["xi_flat"][a0:a1])
        kept_all.append(kept)
        h_neg += h_crit < 0
        lp_true.append(lp_exact_feasible(kept))
    lp_true = np.array(lp_true)
    print(f"\n[4] enriched infeasible corpus: {n} steps, all recorded V0 status in "
          f"{sorted(set(table[int(c)] for c in d['status_code']))}")
    print(f"    LP-exact says FEASIBLE on {int(lp_true.sum())} of {n} "
          f"({100*lp_true.mean():.3f} %)  -- i.e. V0 misclassification rate")
    print(f"    h_crit < 0 on {h_neg} steps")
    report["enriched"] = {
        "steps": n, "v0_status_counts":
            {k: int(v) for k, v in Counter(table[int(c)] for c in d["status_code"]).items()},
        "lp_exact_feasible": int(lp_true.sum()), "h_crit_negative": int(h_neg),
    }

    # cross-tab per variant, using the recorded disagreements
    for variant, r in dis["variants"].items():
        ct = Counter()
        for i in range(n):
            st_v0 = table[int(d["status_code"][i])]
            ct[(st_v0, "same", bool(lp_true[i]))] += 1
        for x in r["disagreements"]:
            i = x["i"]
            st_v0 = table[int(d["status_code"][i])]
            ct[(st_v0, "same", bool(lp_true[i]))] -= 1
            ct[(st_v0, x["status_var"], bool(lp_true[i]))] += 1
        report.setdefault("crosstab", {})[variant] = {
            f"V0={a} | ACC={b} | LP_feasible={c}": int(v) for (a, b, c), v in ct.items()}

    # ---- independent feasible-set equality on the failing cases -------------------------
    print("\n[3] INDEPENDENT feasible-set equality (V0 LP oracle vs reduced arithmetic);")
    print("    the accelerated implementation is not consulted.")
    fail_idx = sorted({x["i"] for r in dis["variants"].values() for x in r["disagreements"]})
    extra = list(rng.choice(n, size=40, replace=False))
    checked = []
    for i in fail_idx + extra:
        npts, bad = check_feasible_set_equality(kept_all[i], rng)
        checked.append({"i": int(i), "probes": npts, "disagreements": len(bad),
                        "is_gate_failure_case": i in fail_idx,
                        "worst": (max(abs(x[3]) for x in bad) if bad else 0.0)})
    tot_bad = sum(c["disagreements"] for c in checked)
    print(f"    {len(checked)} steps x {checked[0]['probes']} u-probes = "
          f"{len(checked)*checked[0]['probes']} membership comparisons")
    print(f"    disagreements between the V0 and reduced feasible sets: {tot_bad}")
    report["feasible_set_equality"] = {"steps": len(checked),
                                       "probes_per_step": checked[0]["probes"],
                                       "total_disagreements": int(tot_bad),
                                       "detail": checked}

    # ---- coefficient audit on the failing cases -----------------------------------------
    print("\n[3] coefficient audit on the gate-failure steps")
    audit = []
    for i in fail_idx:
        a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
        rec_kept = np.asarray(d["xi_kept_flat"][d["xi_kept_ptr"][i]:d["xi_kept_ptr"][i + 1]])
        c = check_coefficients(rec_kept, d["xi_flat"][a0:a1])
        c["i"] = int(i)
        c["lp_exact_feasible"] = bool(lp_true[i])
        c["min_b_minus_reach"] = float(np.min(
            np.abs(rec_kept[:, 2:4]) @ np.array([MAX_V, MAX_V])
            - (TAU - ALPHA * rec_kept[:, 1] - rec_kept[:, 0])))
        audit.append(c)
        print(f"    step {i:5d}: kept==recorded {c['kept_matches_recorded']}  "
              f"h_crit={c['h_crit']:+.5f}  p1={c['p1']:+.4f} p3={c['p3']:+.4f}  "
              f"sorted={c['sort_is_ascending']}  |grad|={min(c['grad_norms']):.6f}"
              f"..{max(c['grad_norms']):.6f}  LP_feas={c['lp_exact_feasible']}  "
              f"slack_margin={c['min_b_minus_reach']:+.2e}")
    report["coefficient_audit"] = audit

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_forensic.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {OUT}/stage1_forensic.json")


if __name__ == "__main__":
    main()
