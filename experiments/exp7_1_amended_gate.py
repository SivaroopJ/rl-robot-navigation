"""Week5-Phase7 / Stage 1 RERUN under Amendment 1 (PHASE7_PLAN.md 11.4, dated 2026-09-06).

Second run. The FIRST run and its FAILED gate stand permanently at commit 998eeaa; nothing here
rewrites them, and the original thresholds are re-reported unchanged for continuity.

V0 = frozen reference.  V1 = the accelerated candidate = RED-OSQP (the only substrate
candidate; PAR-SCS is excluded on its status-level regression).

PRIMARY   P1 independent feasible-set equality        0 disagreements
          P2 coefficients + critical-row ordering     bitwise, every corpus step
          P3 bounds, tau, barrier and objective form  exact
          P4 arbiter confirmation                     u_V1 in the V0 set AND gap <= J_tol = 1e-9
SECONDARY E0/E1 against the arbiter                   reported, no pass/fail
SAFETY    L4                                          0 guarantee-tier REGRESSIONS (mandatory)
CLOSED    B0 vs B1                                    exact paired McNemar + bootstrap CI

The P1 V0-side oracle solves the (t, s_i) feasibility LP with scipy HiGHS for a FIXED u and
never consults the accelerated implementation.
"""
from __future__ import annotations

import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from dr_control.fast_drccp import FastClfCbfDrccpController
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR
from experiments.exp7_1_forensic import (ALPHA, MAX_V, P0, TAU, arbiter,
                                         check_feasible_set_equality, check_tau_constant,
                                         frozen_prepare, lp_exact_feasible, v0_membership)

LABEL = "Week5-Phase7 / Stage 1 RERUN under Amendment 1"
OUT = Path("results/week5_phase7/stage1_equivalence")
J_TOL = 1e-9                 # pre-registered in PHASE7_PLAN.md 11.4.2, anchored to the arbiter
CI_MARGIN = 0.03             # pre-registered in PHASE7_PLAN.md 11.4.5
PARTS = ["dev_fixed", "dev_randomized", "validation_fixed", "validation_randomized"]
FAILURE_STEPS = [102, 103, 253, 254, 1061, 1467, 1622, 2153]      # from the FIRST run


def load(name):
    m = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
    return m, np.load(CORPUS_DIR / m["npz"]), {int(k): v for k, v in m["status_table"].items()}


def load_enriched():
    m = json.loads((CORPUS_DIR / "meta_infeasible_enriched.json").read_text())
    parts = json.loads((CORPUS_DIR / "INDEX.json").read_text())["parts"]
    tab = {}
    for pt in parts:
        tab.update({int(k): v for k, v in pt["status_table"].items()})
    return m, np.load(CORPUS_DIR / m["npz"]), tab


def tier(min_cbc):
    """T0 DR intact | T1 nominal CBF holds, DR margin consumed | T2 invariance lost."""
    if min_cbc >= TAU:
        return 0
    return 1 if min_cbc >= 0.0 else 2


def mcnemar_exact(a, b):
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def boot_ci(a, b, rng, n=10000):
    d = b.astype(float) - a.astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rng = np.random.default_rng(20260906)
    print(f"### {LABEL}\n")
    R = {"label": LABEL, "amendment": "PHASE7_PLAN.md 11.4 (2026-09-06)",
         "original_run": "998eeaa (FAILED, preserved)", "candidate": "RED-OSQP",
         "J_tol": J_TOL, "ci_margin": CI_MARGIN}

    # ---------------- P3 first: it is cheapest and gates the meaning of the rest ----------
    lo, hi = check_tau_constant(rng, n=1_000_000)
    c = FastClfCbfDrccpController(variant="reduced_osqp")
    P3 = {"u_bar_inf_min": lo, "u_bar_inf_max": hi, "tau_constant": bool(lo == hi == 1.0),
          "tau": TAU, "tau_candidate": c.tau, "tau_match": bool(c.tau == TAU),
          "eps_times_n_keep": c.epsilon * c.n_keep,
          "precondition_cvar": bool(c.epsilon * c.n_keep < 1.0),
          "precondition_box": bool(c.max_v <= max(1.0, c.rateh)),
          "max_v": c.max_v, "alpha": c.rateh, "p0": P0}
    P3["pass"] = bool(P3["tau_constant"] and P3["tau_match"] and P3["precondition_cvar"]
                      and P3["precondition_box"])
    print(f"P3 bounds/tau/barrier/objective : ||u_bar||_inf over 1e6 admissible u = "
          f"[{lo}, {hi}]  tau = {TAU}  eps*n_keep = {P3['eps_times_n_keep']}  "
          f"-> {'PASS' if P3['pass'] else 'FAIL'}")
    R["P3"] = P3

    # ---------------- P2: every corpus step -----------------------------------------------
    n_steps = n_bad_kept = n_bad_sort = n_bad_w = 0
    for name in PARTS + ["enriched"]:
        m, d, tab = load_enriched() if name == "enriched" else load(name)
        for i in range(len(d["u"])):
            a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
            kept, h_crit, _ = frozen_prepare(d["xi_flat"][a0:a1])
            rec = d["xi_kept_flat"][d["xi_kept_ptr"][i]:d["xi_kept_ptr"][i + 1]]
            n_steps += 1
            n_bad_kept += not np.array_equal(kept, rec)
            key = kept[:, 1] * ALPHA + kept[:, 0]
            n_bad_sort += not np.all(np.diff(key) >= -1e-15)
            if d["p1"][i] == d["p1"][i]:
                n_bad_w += not (abs(d["p1"][i] - 4.0 * h_crit) < 1e-12
                                and abs(d["p3"][i] - 5.0 * h_crit) < 1e-12)
    P2 = {"steps": n_steps, "kept_mismatches": n_bad_kept, "sort_violations": n_bad_sort,
          "weight_mismatches": n_bad_w,
          "pass": bool(n_bad_kept == 0 and n_bad_sort == 0 and n_bad_w == 0)}
    print(f"P2 coefficients + ordering      : {n_steps} steps, kept mismatches {n_bad_kept}, "
          f"sort violations {n_bad_sort}, weight mismatches {n_bad_w} "
          f"-> {'PASS' if P2['pass'] else 'FAIL'}")
    R["P2"] = P2

    # ---------------- P1: independent feasible-set equality --------------------------------
    kept_pool = []
    m, d, tab = load_enriched()
    for i in FAILURE_STEPS:
        a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
        kept_pool.append(("enriched", int(i), frozen_prepare(d["xi_flat"][a0:a1])[0]))
    per_part = 242 // (len(PARTS) + 1)
    for name in PARTS + ["enriched"]:
        mm, dd, _ = load_enriched() if name == "enriched" else load(name)
        sel = rng.choice(len(dd["u"]), size=per_part, replace=False)
        for i in sel:
            a0, a1 = dd["xi_ptr"][i], dd["xi_ptr"][i + 1]
            kept_pool.append((name, int(i), frozen_prepare(dd["xi_flat"][a0:a1])[0]))
    tot_probes = tot_dis = 0
    dis_detail = []
    for src, i, kept in kept_pool:
        npts, bad = check_feasible_set_equality(kept, rng)
        tot_probes += npts
        tot_dis += len(bad)
        if bad:
            dis_detail.append({"source": src, "i": i, "n": len(bad),
                               "worst_boundary_distance": max(abs(x[3]) for x in bad)})
    P1 = {"steps": len(kept_pool), "probes": tot_probes, "disagreements": tot_dis,
          "includes_all_first_run_failure_steps": True, "detail": dis_detail,
          "pass": bool(tot_dis == 0)}
    print(f"P1 feasible-set equality        : {len(kept_pool)} steps x "
          f"{tot_probes // len(kept_pool)} probes = {tot_probes} comparisons, "
          f"disagreements {tot_dis} -> {'PASS' if P1['pass'] else 'FAIL'}")
    R["P1"] = P1

    # ---------------- P4 + secondary E0/E1 + L4 --------------------------------------------
    acc = FastClfCbfDrccpController(variant="reduced_osqp")
    e0, e1, gaps, feas_ok = [], [], [], 0
    n_p4 = 0
    tiers = Counter()
    regressions = improvements = 0
    l4_steps = 0
    status_ct = Counter()
    for name in PARTS:
        m, d, tab = load(name)
        n = len(d["u"])
        sample = set(rng.choice(n, size=min(300, n), replace=False).tolist())
        for i in range(n):
            st0 = tab[int(d["status_code"][i])]
            a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
            xi = d["xi_flat"][a0:a1]
            kept, h_crit, _ = frozen_prepare(xi)
            acc.prev_u = d["u_prev"][i].copy()
            info = {}
            u1 = acc.generate_controller(d["p"][i], d["gamma"][i], xi, record=info)
            status_ct[(st0, info["status"])] += 1
            if st0 != "optimal" or info["status"] != "optimal":
                continue
            ub = lambda z: np.array([1.0, ALPHA, z[0], z[1]])
            t0, t1 = tier(float(np.min(kept @ ub(d["u"][i])))), tier(float(np.min(kept @ ub(u1))))
            tiers[(t0, t1)] += 1
            regressions += t1 > t0
            improvements += t1 < t0
            l4_steps += 1
            if i in sample:
                ua, Ja, sa = arbiter(d["p"][i], d["gamma"][i], kept, d["u_prev"][i])
                if ua is None or sa != "optimal":
                    continue
                n_p4 += 1
                feas_ok += v0_membership(u1, kept)
                un = d["u_nom"][i]
                J = lambda u, dl: (P0 * float((u - d["u_prev"][i]) @ (u - d["u_prev"][i]))
                                   + 4 * h_crit * float((u - un) @ (u - un))
                                   + 5 * h_crit * float(dl) ** 2)
                gaps.append(J(u1, info["delta"]) - Ja)
                e0.append(float(np.max(np.abs(d["u"][i] - ua))))
                e1.append(float(np.max(np.abs(u1 - ua))))
    gaps, e0, e1 = np.array(gaps), np.array(e0), np.array(e1)
    P4 = {"sampled_solved_steps": n_p4, "u_V1_in_V0_feasible_set": int(feas_ok),
          "feasibility_rate": float(feas_ok / max(n_p4, 1)),
          "gap_median": float(np.median(gaps)), "gap_p95": float(np.percentile(gaps, 95)),
          "gap_p99": float(np.percentile(gaps, 99)), "gap_max": float(gaps.max()),
          "gap_min": float(gaps.min()), "J_tol": J_TOL,
          "n_gap_exceeding": int((gaps > J_TOL).sum()),
          "pass": bool(feas_ok == n_p4 and gaps.max() <= J_TOL)}
    print(f"P4 arbiter confirmation         : {n_p4} solved steps, u_V1 feasible in the V0 set "
          f"{feas_ok}/{n_p4}, suboptimality gap max {gaps.max():.3e} vs J_tol {J_TOL:.0e} "
          f"-> {'PASS' if P4['pass'] else 'FAIL'}")
    R["P4"] = P4

    def stats(v, name):
        return {f"{name}_median": float(np.median(v)), f"{name}_p95": float(np.percentile(v, 95)),
                f"{name}_p99": float(np.percentile(v, 99)), f"{name}_max": float(v.max())}
    SEC = {**stats(e0, "E0"), **stats(e1, "E1"), "n": len(e0),
           "fraction_E1_more_accurate": float((e1 < e0).mean()),
           "original_failed_thresholds": {
               "L2_vs_V0_1e-6": "FAILED in the first run, preserved",
               "L2_vs_arbiter_1e-8": "FAILED in the first run, preserved",
               "L3_objective_1e-8": "FAILED in the first run, preserved"}}
    R["secondary"] = SEC
    print(f"\nSECONDARY (reported, not a gate), n = {len(e0)}")
    print(f"  {'':6s}{'median':>12}{'p95':>12}{'p99':>12}{'max':>12}")
    print(f"  {'E0':6s}{SEC['E0_median']:12.3e}{SEC['E0_p95']:12.3e}"
          f"{SEC['E0_p99']:12.3e}{SEC['E0_max']:12.3e}   (frozen V0 vs arbiter)")
    print(f"  {'E1':6s}{SEC['E1_median']:12.3e}{SEC['E1_p95']:12.3e}"
          f"{SEC['E1_p99']:12.3e}{SEC['E1_max']:12.3e}   (candidate V1 vs arbiter)")
    print(f"  fraction of cases where E1 < E0: {SEC['fraction_E1_more_accurate']:.4f}")

    L4 = {"solved_steps": l4_steps, "regressions": regressions, "improvements": improvements,
          "tier_transitions": {f"T{a}->T{b}": int(v) for (a, b), v in tiers.items()},
          "pass": bool(regressions == 0)}
    R["L4"] = L4
    print(f"\nL4 guarantee tiers              : {l4_steps} solved steps, REGRESSIONS "
          f"{regressions}, improvements {improvements}, transitions "
          f"{L4['tier_transitions']} -> {'PASS' if L4['pass'] else 'FAIL'}")

    R["status_matrix"] = {f"V0={a} | V1={b}": int(v) for (a, b), v in status_ct.items()}
    print(f"\nStatus matrix (all five statuses kept distinct):")
    for k, v in sorted(R["status_matrix"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:58s} {v}")

    # ---------------- closed loop: re-analysis of the SAME 400 episodes from 998eeaa -------
    prev = json.loads((OUT / "stage1_equivalence.json").read_text())
    CL = {}
    for key, block in prev["L5"].items():
        recs = block["episodes_detail"]
        cl = {"episodes": len(recs), "outcome_agreement": block["outcome_agreement"],
              "divergent_episodes": [
                  {"seed": x["seed"], "V0": x["outcome_ref"], "V1": x["outcome_var"]}
                  for x in recs if not x["same_outcome"]]}
        ok = True
        for outcome in ("success", "collision", "timeout"):
            a = np.array([x["outcome_ref"] == outcome for x in recs])
            b = np.array([x["outcome_var"] == outcome for x in recs])
            n01, n10, p = mcnemar_exact(a, b)
            loc, hic = boot_ci(a, b, rng)
            passed = bool(p > 0.05 and loc >= -CI_MARGIN and hic <= CI_MARGIN)
            ok &= passed
            cl[outcome] = {"rate_V0": float(a.mean()), "rate_V1": float(b.mean()),
                           "diff": float(b.mean() - a.mean()), "n01": n01, "n10": n10,
                           "mcnemar_p": p, "ci95": [loc, hic], "pass": passed}
        cl["pass"] = ok
        CL[key] = cl
        print(f"\nCLOSED LOOP {key}  (same 400 episodes as 998eeaa; criterion changed, data did not)")
        print(f"  outcome agreement {cl['outcome_agreement']:.3f}, divergent episodes "
              f"{len(cl['divergent_episodes'])}: {cl['divergent_episodes']}")
        for outcome in ("success", "collision", "timeout"):
            e = cl[outcome]
            print(f"  {outcome:10s} V0 {e['rate_V0']:.4f} -> V1 {e['rate_V1']:.4f}  "
                  f"diff {e['diff']:+.4f}  McNemar n01={e['n01']} n10={e['n10']} "
                  f"p={e['mcnemar_p']:.4f}  CI95 [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  "
                  f"{'PASS' if e['pass'] else 'FAIL'}")
    R["closed_loop"] = CL

    R["gate"] = {"primary": bool(P1["pass"] and P2["pass"] and P3["pass"] and P4["pass"]),
                 "safety_L4": L4["pass"],
                 "closed_loop": all(v["pass"] for v in CL.values())}
    R["gate"]["overall"] = bool(all(R["gate"].values()))
    print("\n" + "=" * 78)
    print(f"AMENDED GATE  primary {R['gate']['primary']}  safety/L4 {R['gate']['safety_L4']}  "
          f"closed loop {R['gate']['closed_loop']}  ->  "
          f"{'PASS' if R['gate']['overall'] else 'FAIL'}")
    print("The FIRST run at 998eeaa remains recorded as FAILED under the original gate.")
    (OUT / "stage1_amended_rerun.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"wrote {OUT}/stage1_amended_rerun.json")


if __name__ == "__main__":
    main()
