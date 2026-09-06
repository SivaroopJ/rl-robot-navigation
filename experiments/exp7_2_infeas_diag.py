"""Week5-Phase7 / Stage 2 / D1a: why the FROZEN V0 DR-CBF goes infeasible. DIAGNOSTIC ONLY.

Nothing is implemented, changed, tuned or remedied. No recovery policy, no slack variable, no
tau relaxation, no alpha modification, no predictive barrier, no uncertainty margin, no tracker
change, no controller change. V0 is untouched. This module reads the Stage 0 corpus and runs
LP feasibility analysis over the RECORDED problems.

Stage 1 is closed and FAILED (cc692ab); V0 is the substrate. The accelerated implementation is
not used anywhere in this file.

THE HYPOTHESIS UNDER TEST (plan Finding C) -- a LEADING HYPOTHESIS, not a finding:
    temporal staleness in the 5-scan LiDAR barrier buffer creates an inconsistency between
    historical surface geometry and current estimated obstacle velocity, and that inconsistency
    is what makes the problem infeasible.

WHY K=1 ALONE CANNOT TEST IT, AND WHAT DOES
    The plan's D1a-M3 counterfactual drops from 5 constraints to 1. That is CONFOUNDED: any
    5-constraint problem becomes easier when 4 constraints are deleted, whether or not the
    deleted ones were stale. K=1 measures "fewer constraints", not "less staleness".

    The discriminating test is LEAVE-ONE-AGE-OUT: remove exactly one row and re-test, for each
    age in turn. Every resulting subset has 4 rows, so constraint count is held FIXED and only
    the age of the removed row varies. If staleness drives infeasibility, removing the OLDEST
    row should restore feasibility far more often than removing the NEWEST. If all ages are
    equally implicated, the conflict is simply among five constraints and the staleness
    hypothesis gains no support from it. Both counterfactuals are reported.

GEOMETRY RECONSTRUCTION
    lidar_cbf.samples emits g = (p - q)/dist and h = dist - r_robot from the CURRENT p to a
    buffered point q, so the buffered surface point is recovered exactly as
        q_i = p - (h_i + r_robot) * g_i
    Row index IS scan age (the buffer is iterated newest-first), asserted in Stage 0.

    Ground truth (truth_obs_pos / truth_obs_vel) is used ONLY as a measurement channel, never
    to alter any reconstructed problem.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from experiments.exp0_5_infeasibility import (is_feasible, min_alpha_feasible,
                                              min_control_scale, min_uniform_slack)
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 2 / D1a infeasibility diagnosis"
OUT = Path("results/week5_phase7/stage2_infeasibility_diagnosis")
ALPHA, TAU, MAX_V, R_ROBOT, DT = 0.4, 0.04, 1.0, 0.3, 0.1
OBS_R = 0.3
PARTS = [("dev_fixed", "fixed"), ("dev_randomized", "randomized"),
         ("validation_fixed", "fixed"), ("validation_randomized", "randomized"),
         ("final_fixed", "fixed"), ("final_randomized", "randomized")]


def feas(kept):
    return bool(is_feasible(np.atleast_2d(kept), alpha=ALPHA, tau=TAU, max_v=MAX_V))


def minimal_infeasible_subset(xi):
    """Smallest row subset that is already infeasible. Returns (cardinality, rows)."""
    n = len(xi)
    for k in range(1, n + 1):
        for sub in itertools.combinations(range(n), k):
            if not feas(xi[list(sub)]):
                return k, list(sub)
    return None, []


def row_reach(row):
    """Largest attainable grad_h . u over the action box, minus what the row demands.

    Negative => this row ALONE is unsatisfiable, i.e. a control-bound limitation.
    """
    g, b = row[2:4], TAU - ALPHA * row[1] - row[0]
    return float(np.abs(g) @ np.array([MAX_V, MAX_V]) - b)


def true_closing(p, obs_pos, obs_vel):
    """(true clearance to the nearest dynamic obstacle, closing rate, TTC). Metrics only."""
    if len(obs_pos) == 0:
        return np.inf, 0.0, np.inf
    d = np.linalg.norm(obs_pos - p[None, :], axis=1)
    j = int(np.argmin(d))
    h = float(d[j]) - R_ROBOT - OBS_R
    u = (p - obs_pos[j]) / max(d[j], 1e-12)
    closing = -float(u @ obs_vel[j])          # >0 means the obstacle is closing on us
    ttc = h / closing if closing > 1e-6 else np.inf
    return h, closing, min(ttc, 1e3)


#: Physically attainable obstacle speed in each condition, from the environment configuration
#: (RobotNavEnv obstacle_speed = 0.675 fixed; the randomized regime additionally samples the
#: episode speed from [0.6, 0.75]). Used ONLY as a measurement yardstick.
V_PHYS = {"fixed": 0.675, "randomized": 0.75}


def analyse_step(xi, ages, tids, p, truth_now, truth_hist, condition):
    """Everything Stage 2 asks about ONE recorded infeasible step."""
    n = len(xi)
    r = {"n_rows": n, "ages": ages.tolist(), "track_ids": tids.tolist(),
         "h": xi[:, 1].tolist(), "dh_dt": xi[:, 0].tolist(),
         "h_crit": float(np.min(xi[:, 1] * ALPHA + xi[:, 0])),
         "h_min": float(xi[:, 1].min()), "dh_dt_min": float(xi[:, 0].min())}

    # ---- D1a-M1 minimal infeasible subset ------------------------------------------------
    k, rows = minimal_infeasible_subset(xi)
    r["mis_cardinality"] = k
    r["mis_rows"] = rows
    r["mis_ages"] = ages[rows].tolist() if rows else []
    r["mis_track_ids"] = tids[rows].tolist() if rows else []

    # ---- singleton / control-bound check --------------------------------------------------
    reach = [row_reach(row) for row in xi]
    r["row_reach"] = reach
    r["n_rows_unsatisfiable_alone"] = int(sum(x < 0 for x in reach))
    r["control_bound_limited"] = bool(r["n_rows_unsatisfiable_alone"] > 0)

    # ---- D1a-M3 K=1 counterfactual (CONFOUNDED, reported as such) --------------------------
    r["feasible_K1_newest"] = feas(xi[[0]])
    r["feasible_prefix"] = [feas(xi[: j + 1]) for j in range(n)]

    # ---- leave-one-age-out: the CONTROLLED counterfactual ---------------------------------
    loo = {}
    for j in range(n):
        keep = [q for q in range(n) if q != j]
        loo[int(ages[j])] = feas(xi[keep])
    r["feasible_leave_one_out"] = loo
    r["loo_any"] = bool(any(loo.values()))
    r["loo_oldest_fixes"] = bool(loo.get(int(ages.max()), False))
    r["loo_newest_fixes"] = bool(loo.get(int(ages.min()), False))

    # ---- D1a-M4/M5/M6 ----------------------------------------------------------------------
    s = min_uniform_slack(xi, alpha=ALPHA, tau=TAU, max_v=MAX_V)
    r["min_slack"] = s
    r["tier"] = 0 if s <= 0 else (1 if s <= TAU else 2)
    r["min_control_scale"] = min_control_scale(xi, alpha=ALPHA, tau=TAU)
    r["min_alpha_feasible"] = min_alpha_feasible(xi, tau=TAU)

    # ---- staleness geometry ----------------------------------------------------------------
    q = p[None, :] - (xi[:, 1:2] + R_ROBOT) * xi[:, 2:4]      # buffered surface points
    r["buffered_points"] = q.tolist()
    r["range_from_ego"] = np.linalg.norm(q - p[None, :], axis=1).tolist()
    disp = []
    for j in range(n):
        hp = truth_hist.get(int(ages[j]))
        if hp is None or len(hp) == 0:
            disp.append(float("nan"))
            continue
        # the true obstacle that most plausibly produced this buffered point, at that time
        dd = np.linalg.norm(hp - q[j][None, :], axis=1)
        o = int(np.argmin(dd))
        moved = float(np.linalg.norm(truth_now["obs_pos"][o] - hp[o])) \
            if o < len(truth_now["obs_pos"]) else float("nan")
        disp.append(moved)
    r["owner_displacement_since_scan"] = disp        # metres the owning obstacle has moved
    r["max_owner_displacement"] = float(np.nanmax(disp)) if len(disp) else float("nan")

    # ---- COMPETING EXPLANATION: is the DEMANDED closing-rate compensation physical? --------
    # Row i demands  grad_h_i . u >= tau - alpha h_i - dh_dt_i. The dh_dt_i term is
    # -grad_h_i . v_track, i.e. the ESTIMATED closing rate. If the estimator reports a speed
    # above what an obstacle can physically travel, the row can demand more than the action box
    # can deliver -- infeasibility caused by estimator overshoot, not by staleness.
    v_phys = V_PHYS[condition]
    r["dh_dt_exceeds_physical"] = [bool(x < -v_phys - 1e-9) for x in xi[:, 0]]
    r["n_rows_superphysical"] = int(sum(r["dh_dt_exceeds_physical"]))
    r["worst_dh_dt_over_physical"] = float(max(0.0, -xi[:, 0].min() - v_phys))

    # TRUE dh/dt for the same buffered point, from ground truth: the owning obstacle's true
    # velocity projected on the same gradient. Difference = the estimator's projected error.
    dh_true, e_proj = [], []
    for j in range(len(xi)):
        hp = truth_hist.get(int(ages[j]))
        if hp is None or len(hp) == 0 or not len(truth_now["obs_vel"]):
            dh_true.append(float("nan"))
            e_proj.append(float("nan"))
            continue
        dd = np.linalg.norm(hp - q[j][None, :], axis=1)
        o = int(np.argmin(dd))
        # a buffered point far from every dynamic obstacle came from static structure (v = 0)
        v_true = truth_now["obs_vel"][o] if dd[o] < OBS_R + 0.25 else np.zeros(2)
        t_ = -float(xi[j, 2:4] @ v_true)
        dh_true.append(t_)
        e_proj.append(float(xi[j, 0]) - t_)
    r["dh_dt_true"] = dh_true
    r["projected_error"] = e_proj                    # e > 0 optimistic, e < 0 pessimistic
    with np.errstate(invalid="ignore"):
        r["min_projected_error"] = float(np.nanmin(e_proj)) if len(e_proj) else float("nan")
        r["n_rows_pessimistic"] = int(np.nansum(np.asarray(e_proj) < -0.05))

    h_t, closing, ttc = true_closing(p, truth_now["obs_pos"], truth_now["obs_vel"])
    r.update(true_clearance_dyn=h_t, true_closing_rate=closing, true_ttc=ttc)
    return r


def classify(r):
    """Attribution required by the brief. Mutually exclusive, evaluated in this order."""
    if r["control_bound_limited"]:
        return "control-bound limitation"
    rows, ages, tids = r["mis_rows"], r["mis_ages"], r["mis_track_ids"]
    if not rows:
        return "unclassified"
    same_age = len(set(ages)) == 1
    same_track = len(set(tids)) == 1
    if same_age:
        return "instantaneous conflict (single scan)"
    if same_track:
        return "stale-vs-current, SAME track"          # the hypothesis's precise signature
    if -1 in tids:
        return "cross-age, involves an untracked (static) return"
    return "cross-age, DIFFERENT tracks"


def q5(v, name):
    v = np.asarray([x for x in v if x == x], dtype=float)
    if not len(v):
        return {}
    return {f"{name}_n": int(len(v)), f"{name}_median": float(np.median(v)),
            f"{name}_p95": float(np.percentile(v, 95)), f"{name}_min": float(v.min()),
            f"{name}_max": float(v.max()), f"{name}_mean": float(v.mean())}


def boot_ci(x, rng, n=10000):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", nargs="+", default=[p for p, _ in PARTS])
    ap.add_argument("--feasible-control", type=int, default=3000,
                    help="feasible steps sampled per part for the contrast")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = np.random.default_rng(20260907)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"### {LABEL}\n")

    infeas, control, base = [], [], {}
    for name, cond in PARTS:
        if name not in args.parts:
            continue
        m = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
        d = np.load(CORPUS_DIR / m["npz"])
        tab = {int(k): v for k, v in m["status_table"].items()}
        names = np.array([tab[int(c)] for c in d["status_code"]])
        is_inf = np.char.find(names.astype(str), "infeasible") >= 0
        base[name] = {"condition": cond, "steps": int(len(names)),
                      "infeasible": int(is_inf.sum()),
                      "rate": float(is_inf.mean()),
                      "status_counts": {k: int(v) for k, v in Counter(names).items()}}
        ep = d["ep_index"]
        st = d["step"]
        feas_idx = np.nonzero(~is_inf)[0]
        ctrl_idx = set(rng.choice(feas_idx, size=min(args.feasible_control, len(feas_idx)),
                                  replace=False).tolist())
        for i in range(len(names)):
            if not (is_inf[i] or i in ctrl_idx):
                continue
            a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
            xi = d["xi_flat"][a0:a1]
            ages = d["sample_age"][a0:a1]
            tids = d["sample_track_id"][a0:a1]
            p = d["p"][i]
            hist = {}
            for a in np.unique(ages):
                j = i - int(a)
                if j >= 0 and ep[j] == ep[i]:
                    hist[int(a)] = d["truth_obs_pos"][j]
            truth_now = {"obs_pos": d["truth_obs_pos"][i], "obs_vel": d["truth_obs_vel"][i]}
            if is_inf[i]:
                r = analyse_step(xi, ages, tids, p, truth_now, hist, cond)
                r.update(part=name, condition=cond, step_index=int(i),
                         ep_index=int(ep[i]), step_in_ep=int(st[i]), status=names[i],
                         truth_clearance=float(d["truth_clearance"][i]))
                r["class"] = classify(r)
                infeas.append(r)
            else:
                h_t, closing, ttc = true_closing(p, truth_now["obs_pos"],
                                                 truth_now["obs_vel"])
                control.append({"part": name, "condition": cond, "n_rows": len(xi),
                                "h_min": float(xi[:, 1].min()),
                                "dh_dt_min": float(xi[:, 0].min()),
                                "true_ttc": ttc, "true_closing_rate": closing,
                                "n_distinct_tracks": int(len(set(tids.tolist()))),
                                "min_slack_would_be": 0.0})
        print(f"  {name:22s} {base[name]['infeasible']:5d} / {base[name]['steps']:6d} "
              f"infeasible ({100*base[name]['rate']:.3f} %)", flush=True)

    R = {"label": LABEL, "base_rates": base, "n_infeasible_analysed": len(infeas),
         "n_feasible_control": len(control)}
    print(f"\nanalysed {len(infeas)} infeasible steps, {len(control)} feasible controls\n")

    # ---------------- 1. distribution ------------------------------------------------------
    by_cond = Counter(r["condition"] for r in infeas)
    R["distribution"] = {
        "by_condition": dict(by_cond),
        **q5([r["h_min"] for r in infeas], "h_min"),
        **q5([r["h_crit"] for r in infeas], "criticality"),
        **q5([r["true_ttc"] for r in infeas], "true_ttc"),
        **q5([r["true_closing_rate"] for r in infeas], "true_closing_rate"),
        **q5([r["truth_clearance"] for r in infeas], "true_clearance"),
        "n_distinct_tracks": dict(Counter(len(set(r["track_ids"])) for r in infeas)),
        "control_bound_limited": int(sum(r["control_bound_limited"] for r in infeas)),
    }
    ctrl_ttc = np.array([c["true_ttc"] for c in control])
    inf_ttc = np.array([r["true_ttc"] for r in infeas])
    R["distribution"]["control_true_ttc_median"] = float(np.median(ctrl_ttc))
    R["distribution"]["control_h_min_median"] = float(np.median([c["h_min"] for c in control]))
    print("1. DISTRIBUTION")
    print(f"   by condition: {dict(by_cond)}")
    print(f"   h_min        median {np.median([r['h_min'] for r in infeas]):.4f} "
          f"(feasible controls {R['distribution']['control_h_min_median']:.4f})")
    print(f"   true TTC     median {np.median(inf_ttc):.3f} s "
          f"(feasible controls {np.median(ctrl_ttc):.3f} s), "
          f"frac < 1 s {float((inf_ttc < 1).mean()):.3f} vs {float((ctrl_ttc < 1).mean()):.3f}")
    print(f"   distinct track ids per step: {R['distribution']['n_distinct_tracks']}")
    print(f"   rows unsatisfiable alone (control-bound): "
          f"{R['distribution']['control_bound_limited']} / {len(infeas)}")

    # ---------------- 2. minimal infeasible subsets ---------------------------------------
    card = Counter(r["mis_cardinality"] for r in infeas)
    R["minimal_infeasible_subsets"] = {"cardinality": {str(k): int(v) for k, v in card.items()}}
    print("\n2. MINIMAL INFEASIBLE SUBSETS")
    for k in sorted(x for x in card if x is not None):
        print(f"   cardinality {k}: {card[k]:5d}  ({100*card[k]/len(infeas):.2f} %)")

    # ---------------- 3. staleness -----------------------------------------------------------
    age_in_mis = Counter()
    for r in infeas:
        for a in r["mis_ages"]:
            age_in_mis[int(a)] += 1
    span = Counter(int(max(r["mis_ages"]) - min(r["mis_ages"])) if r["mis_ages"] else -1
                   for r in infeas)
    R["staleness"] = {"age_membership_in_mis": {str(k): int(v) for k, v in age_in_mis.items()},
                      "mis_age_span": {str(k): int(v) for k, v in span.items()},
                      **q5([r["max_owner_displacement"] for r in infeas], "owner_displacement")}
    print("\n3. TEMPORAL STALENESS")
    tot_mis_rows = sum(age_in_mis.values())
    print("   age membership in the minimal infeasible subset "
          "(uniform would be ~equal across ages):")
    for a in sorted(age_in_mis):
        print(f"     age {a}: {age_in_mis[a]:5d}  ({100*age_in_mis[a]/tot_mis_rows:.2f} %)")
    print(f"   MIS age span (max-min): "
          f"{ {k: v for k, v in sorted(span.items())} }")
    od = [r["max_owner_displacement"] for r in infeas]
    print(f"   owning-obstacle displacement since its scan: median "
          f"{np.nanmedian(od):.4f} m, p95 {np.nanpercentile(od, 95):.4f} m")

    # ---------------- 4 + leave-one-out ------------------------------------------------------
    k1 = np.array([r["feasible_K1_newest"] for r in infeas])
    loo_any = np.array([r["loo_any"] for r in infeas])
    loo_old = np.array([r["loo_oldest_fixes"] for r in infeas])
    loo_new = np.array([r["loo_newest_fixes"] for r in infeas])
    per_age_fix = Counter()
    per_age_n = Counter()
    for r in infeas:
        for a, ok in r["feasible_leave_one_out"].items():
            per_age_n[int(a)] += 1
            per_age_fix[int(a)] += bool(ok)
    ci_k1 = boot_ci(k1, rng)
    R["counterfactuals"] = {
        "K1_newest_feasible": float(k1.mean()), "K1_ci95": list(ci_k1),
        "prefix_feasible": {str(j): float(np.mean([r["feasible_prefix"][j]
                                                   for r in infeas if len(r["feasible_prefix"]) > j]))
                            for j in range(5)},
        "leave_one_out_any": float(loo_any.mean()),
        "leave_one_out_oldest": float(loo_old.mean()),
        "leave_one_out_newest": float(loo_new.mean()),
        "leave_one_out_by_age": {str(a): float(per_age_fix[a] / per_age_n[a])
                                 for a in sorted(per_age_n)},
    }
    print("\n4. COUNTERFACTUALS")
    print(f"   K=1 (newest scan only) feasible : {k1.mean():.4f}  "
          f"CI95 [{ci_k1[0]:.4f}, {ci_k1[1]:.4f}]   <-- CONFOUNDED: drops 4 constraints")
    print(f"   age-prefix feasibility {{0..j}}   : "
          f"{ {k: round(v, 4) for k, v in R['counterfactuals']['prefix_feasible'].items()} }")
    print(f"   LEAVE-ONE-OUT (4 rows kept, constraint count held FIXED):")
    for a in sorted(per_age_n):
        print(f"     remove age {a}: restores feasibility "
              f"{per_age_fix[a]/per_age_n[a]:.4f}")
    print(f"   any single removal helps: {loo_any.mean():.4f}   "
          f"oldest {loo_old.mean():.4f} vs newest {loo_new.mean():.4f}")

    # ---------------- 5. attribution ----------------------------------------------------------
    cls = Counter(r["class"] for r in infeas)
    R["attribution"] = {k: int(v) for k, v in cls.items()}
    print("\n5. ATTRIBUTION")
    for k, v in cls.most_common():
        print(f"   {k:46s} {v:5d}  ({100*v/len(infeas):.2f} %)")

    # ---------------- 6 + M4/M5/M6 ------------------------------------------------------------
    slack = np.array([r["min_slack"] for r in infeas])
    tiers = Counter(r["tier"] for r in infeas)
    mcs = np.array([r["min_control_scale"] for r in infeas])
    maf = np.array([r["min_alpha_feasible"] for r in infeas])
    R["remediability"] = {**q5(slack, "min_slack"),
                          "tier_counts": {str(k): int(v) for k, v in tiers.items()},
                          "min_control_scale_finite": int(np.isfinite(mcs).sum()),
                          "min_control_scale_nan": int(np.isnan(mcs).sum()),
                          **q5(mcs, "min_control_scale"),
                          "alpha_restores_feasibility": int(np.isfinite(maf).sum()),
                          **q5(maf, "min_alpha_feasible")}
    print("\n6. REMEDIABILITY (measured, nothing adopted)")
    print(f"   required uniform slack: median {np.median(slack):.4f}, p95 "
          f"{np.percentile(slack, 95):.4f}, max {slack.max():.4f}   (tau = {TAU})")
    print(f"   tier of that slack: T0 {tiers[0]}, T1 {tiers[1]}, T2 {tiers[2]}")
    print(f"   more control authority would help: {int(np.isfinite(mcs).sum())} / {len(infeas)} "
          f"({100*np.isfinite(mcs).mean():.2f} %)")
    print(f"   some alpha >= 0.4 restores feasibility: {int(np.isfinite(maf).sum())} / "
          f"{len(infeas)} ({100*np.isfinite(maf).mean():.2f} %)")

    # ---------------- 7. competing explanation: estimator overshoot -------------------------
    sp = np.array([r["n_rows_superphysical"] for r in infeas])
    ov = np.array([r["worst_dh_dt_over_physical"] for r in infeas])
    pe = np.array([r["min_projected_error"] for r in infeas], dtype=float)
    cb = np.array([r["control_bound_limited"] for r in infeas])
    R["estimator_overshoot"] = {
        "steps_with_superphysical_row": int((sp > 0).sum()),
        "fraction": float((sp > 0).mean()),
        **q5(ov[ov > 0], "overshoot_amount"),
        **q5(pe, "min_projected_error"),
        "superphysical_given_control_bound": float((sp[cb] > 0).mean()) if cb.any() else None,
        "superphysical_given_not_control_bound": float((sp[~cb] > 0).mean())
        if (~cb).any() else None,
        "pessimistic_rows_median": float(np.nanmedian(
            [r["n_rows_pessimistic"] for r in infeas])),
    }
    eo = R["estimator_overshoot"]
    print("\n7. COMPETING EXPLANATION -- ESTIMATED CLOSING RATE ABOVE WHAT IS PHYSICAL")
    print(f"   steps with >=1 row demanding a super-physical closing rate: "
          f"{eo['steps_with_superphysical_row']} / {len(infeas)} ({100*eo['fraction']:.2f} %)")
    if eo.get("overshoot_amount_median") is not None:
        print(f"   overshoot beyond the physical obstacle speed: median "
              f"{eo['overshoot_amount_median']:.4f} m/s, max {eo['overshoot_amount_max']:.4f} m/s")
    print(f"   given the step is control-bound limited:     "
          f"{eo['superphysical_given_control_bound']}")
    print(f"   given it is NOT control-bound limited:       "
          f"{eo['superphysical_given_not_control_bound']}")
    print(f"   most PESSIMISTIC projected error on the step: median "
          f"{eo.get('min_projected_error_median')}, min {eo.get('min_projected_error_min')}")

    (out / "stage2_infeasibility.json").write_text(json.dumps(R, indent=1, default=float))
    (out / "stage2_per_step.json").write_text(json.dumps(infeas, indent=1, default=float))
    print(f"\nwrote {out}/stage2_infeasibility.json")


if __name__ == "__main__":
    main()
