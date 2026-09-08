"""Week5-Phase7 / Stage 5 / the A4 admission gate (PHASE7_PLAN.md 11.9.13-A4). Read-only.

A4, verbatim: "On 50 held-out validation episodes per condition, E1 must show BOTH (i) a lower
P(e > 0 | TTC < 1 s) than A0 AND (ii) a non-increase in the super-physical row rate. If it
fails, Stage 5 stops at the diagnostic, the final tier is not run, and the null is reported as
the result."

A4 IS DIRECTIONAL, NOT A SIGNIFICANCE TEST. Significance is reserved for the final tier
(11.9.13-B2). Confidence intervals are printed for information; requiring them to exclude zero
here would be tightening the gate after the fact, which 11.9.15 forbids. Everything outside A4
is printed as DESCRIPTIVE validation-tier context only -- 11.9 forbids drawing any Stage-5
conclusion from validation alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp7_5_estimator import OUT, admission_a4, cluster_boot_rate, mcnemar, paired_boot

A0, A1 = "A0_T1_baseline", "A1_T1_E1"
STAGE3_T1 = {"fixed": 0.397, "randomized": 0.498}      # in-environment comparator, NOT 0.142


def load(cond, tag=""):
    p = OUT / f"stage5_validation_{cond}{tag}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-tag", default="", help="suffix of the fixed-condition result file")
    ap.add_argument("--randomized-tag", default="")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(20260908)
    data = {"fixed": load("fixed", args.fixed_tag),
            "randomized": load("randomized", args.randomized_tag)}
    ep = {c: data[c]["episodes_raw"] for c in data}

    print("=" * 78)
    print("Week5-Phase7 / Stage 5 / VALIDATION TIER -- A4 ADMISSION GATE")
    print("50 held-out episodes per condition, seeds DEV_SEED_BASE+1000..+1049.")
    print("EVAL_SEED_BASE is NOT touched. E1 has no free parameter and is not tuned here.")
    print("=" * 78)

    gate = admission_a4(ep["fixed"][A0], ep["fixed"][A1],
                        ep["randomized"][A0], ep["randomized"][A1], rng)

    for cond in ("fixed", "randomized"):
        g = gate[cond]
        print(f"\n--- {cond.upper()} ---")
        print(f"(i)  P(e>0 | TTC<1s) pooled   A0 {g['p_e_pos_ttc1_A0']:.4f}  "
              f"A1 {g['p_e_pos_ttc1_A1']:.4f}   delta {g['delta']:+.4f}  "
              f"95% CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]   "
              f"-> {'PASS' if g['criterion_i_lower_optimistic_tail'] else 'FAIL'}")
        em = {a: data[cond]["summary"][a]["p_e_pos_ttc1_episode_mean"] for a in (A0, A1)}
        print(f"     episode-mean (Stage-3 comparable)  A0 {em[A0]:.4f}  A1 {em[A1]:.4f}   "
              f"Stage-3 T1 reference {STAGE3_T1[cond]:.3f}")
        print(f"(ii) super-physical step rate  A0 {g['superphys_A0']:.4f}  "
              f"A1 {g['superphys_A1']:.4f}   delta {g['delta_superphys']:+.4f}  "
              f"95% CI [{g['ci_superphys'][0]:+.4f}, {g['ci_superphys'][1]:+.4f}]   "
              f"-> {'PASS' if g['criterion_ii_superphys_not_increased'] else 'FAIL'}")
        print(f"     condition verdict: {'PASS' if g['pass'] else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"A4 GATE: {'PASS' if gate['A4_PASS'] else 'FAIL'}   ->  the 200-episode final run is "
          f"{'AUTHORISED' if gate['A4_PASS'] else 'NOT authorised (11.9.13-D2: terminate, report the null)'}")
    print("=" * 78)

    print("\nDESCRIPTIVE validation-tier context. NOT a Stage-5 conclusion (11.9), NOT part of")
    print("the gate, and NOT usable to tune anything. n = 50 resolves far less than n = 200.")
    for cond in ("fixed", "randomized"):
        a, b = ep[cond][A0], ep[cond][A1]
        sa, sb = data[cond]["summary"][A0], data[cond]["summary"][A1]
        print(f"\n--- {cond.upper()} ---")
        for label, num, den in (("total infeasibility", "n_infeasible", "steps"),
                                ("M1 singleton      ", "n_singleton", "steps"),
                                ("M2 multi          ", "n_multi", "steps"),
                                ("T0 share          ", "n_T0", "steps"),
                                ("T2 share          ", "n_T2", "steps")):
            ra, rb, d, lo, hi = cluster_boot_rate(a, b, num, den, rng)
            print(f"  {label}  A0 {ra:.4f}  A1 {rb:.4f}  delta {d:+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]")
        n01, n10, p = mcnemar([r["collision"] for r in a], [r["collision"] for r in b])
        d, lo, hi = paired_boot([r["collision"] for r in a], [r["collision"] for r in b], rng)
        print(f"  collision            A0 {sa['collision']:.3f}  A1 {sb['collision']:.3f}  "
              f"delta {d:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  McNemar n01={n01} n10={n10} p={p:.4f}")
        print(f"       split A0 dyn/stat/wall {sa['dynamic']}/{sa['static']}/{sa['wall']}   "
              f"A1 {sb['dynamic']}/{sb['static']}/{sb['wall']}")
        d, lo, hi = paired_boot([r["min_clearance"] for r in a], [r["min_clearance"] for r in b], rng)
        print(f"  mean min clearance   A0 {sa['min_clearance']:.4f}  A1 {sb['min_clearance']:.4f}  "
              f"delta {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  worst-case clearance A0 {sa['min_clearance_worst']:+.4f}  "
              f"A1 {sb['min_clearance_worst']:+.4f}")
        print(f"  success              A0 {sa['success']:.3f}  A1 {sb['success']:.3f}    "
              f"timeout A0 {sa['timeout']:.3f}  A1 {sb['timeout']:.3f}")
        print(f"  e p95/p99/max        A0 {sa['e_p95']:.3f}/{sa['e_p99']:.3f}/{sa['e_max']:.3f}  "
              f"A1 {sb['e_p95']:.3f}/{sb['e_p99']:.3f}/{sb['e_max']:.3f}")
        print(f"  pessimistic tail     A0 {sa['p_e_pessimistic']:.4f}  A1 {sb['p_e_pessimistic']:.4f}")
        print(f"  clamp rate (T1)      A0 {sa['frac_clamped_steps']:.4f}  "
              f"A1 {sb['frac_clamped_steps']:.4f}")
        print(f"  solver failures      A0 {sa['n_solver_fail']}  A1 {sb['n_solver_fail']}")

    path = OUT / f"stage5_validation_a4_gate{args.out_tag}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; results are never overwritten (11.9.15)")
    path.write_text(json.dumps(gate, indent=2))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
