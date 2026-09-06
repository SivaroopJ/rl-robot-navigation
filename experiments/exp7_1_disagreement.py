"""Week5-Phase7 / Stage 1 / what the L1 status disagreements actually DO.

The pre-registered L1 gate demands exact status-string agreement on the enriched infeasible
corpus, and it FAILED: 3-5 steps in 2337 are labelled differently by each accelerated variant.
This module does not soften that. It asks the follow-up question the gate does not:

    for each disagreeing step, is the ACTION different?

The frozen controller returns u = 0 for EVERY non-optimal status -- `optimal_inaccurate`,
`infeasible`, `infeasible_inaccurate` and solver errors alike (drccp_controller._solve:
`if prob.status != "optimal" ... return np.zeros(2), False`). So relabelling one flavour of
failure as another is behaviourally inert, while relabelling a failure as `optimal` is not.
That distinction is what decides whether Stage 1 can proceed, and it is measured, not argued.

Three predicates are reported side by side, from strictest to the one the system acts on:

    P1  exact status string                 the pre-registered L1 criterion
    P2  coarse: "infeasible" in status      what dr_control/policy.py counts, and therefore
                                            what Direction 1's dependent variable is built on
    P3  action                              what the robot does
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from dr_control.fast_drccp import VARIANTS, FastClfCbfDrccpController
from experiments.exp0_5_infeasibility import is_feasible
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 1 / status-disagreement analysis"
OUT = Path("results/week5_phase7/stage1_equivalence")
ALPHA, TAU, MAX_V = 0.4, 0.04, 1.0


def coarse(status):
    """The predicate the frozen policy acts on."""
    if status == "optimal":
        return "use_u"
    return "infeasible" if "infeasible" in str(status) else "other_failure"


def main():
    meta = json.loads((CORPUS_DIR / "meta_infeasible_enriched.json").read_text())
    parts = json.loads((CORPUS_DIR / "INDEX.json").read_text())["parts"]
    table = {}
    for pt in parts:
        table.update({int(k): v for k, v in pt["status_table"].items()})
    d = np.load(CORPUS_DIR / meta["npz"])
    n = len(d["u"])
    print(f"### {LABEL}\nenriched infeasible corpus: {n} steps\n")

    report = {"label": LABEL, "steps": n, "variants": {}}
    for variant in VARIANTS:
        ctrl = FastClfCbfDrccpController(variant=variant)
        p1_same = p2_same = p3_same = 0
        detail, lp_says = [], Counter()
        for i in range(n):
            a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
            xi = d["xi_flat"][a0:a1]
            ctrl.prev_u = d["u_prev"][i].copy()
            info = {}
            u = ctrl.generate_controller(d["p"][i], d["gamma"][i], xi, record=info)
            st_ref = table[int(d["status_code"][i])]
            st_var = info["status"]
            same_action = np.array_equal(np.asarray(u, dtype=float), d["u"][i])
            p1_same += st_ref == st_var
            p2_same += coarse(st_ref) == coarse(st_var)
            p3_same += same_action
            if st_ref != st_var:
                kept = np.asarray(info["xi_kept"])
                lp = bool(is_feasible(kept, alpha=ALPHA, tau=TAU, max_v=MAX_V))
                lp_says[lp] += 1
                detail.append({
                    "i": int(i), "status_ref": st_ref, "status_var": st_var,
                    "coarse_ref": coarse(st_ref), "coarse_var": coarse(st_var),
                    "action_identical": bool(same_action),
                    "u_ref": d["u"][i].tolist(), "u_var": np.asarray(u).tolist(),
                    "lp_exact_feasible": lp,
                    "h_crit": float(info["h_crit"]),
                })
        r = {
            "P1_exact_status_agreement": p1_same / n,
            "P2_coarse_predicate_agreement": p2_same / n,
            "P3_action_agreement": p3_same / n,
            "n_disagreements": n - p1_same,
            "n_behaviourally_consequential": sum(1 for x in detail
                                                 if not x["action_identical"]),
            "lp_exact_says_feasible_on_disagreements": int(lp_says[True]),
            "lp_exact_says_infeasible_on_disagreements": int(lp_says[False]),
            "disagreements": detail,
        }
        report["variants"][variant] = r
        print(f"{variant:16s}  P1(exact)={r['P1_exact_status_agreement']:.5f}  "
              f"P2(coarse)={r['P2_coarse_predicate_agreement']:.5f}  "
              f"P3(action)={r['P3_action_agreement']:.5f}  "
              f"disagreements={r['n_disagreements']} of which consequential="
              f"{r['n_behaviourally_consequential']}")
        for x in detail:
            print(f"      step {x['i']:5d}: {x['status_ref']:22s} -> {x['status_var']:22s}"
                  f"  coarse {x['coarse_ref']}->{x['coarse_var']}"
                  f"  action identical={x['action_identical']}"
                  f"  LP-exact feasible={x['lp_exact_feasible']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_disagreements.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {OUT}/stage1_disagreements.json")


if __name__ == "__main__":
    main()
