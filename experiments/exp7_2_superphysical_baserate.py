"""Week5-Phase7 / Stage 2 / base rate of super-physical estimated closing rates. Read-only.

The Stage 2 report's central quantitative claim is a CONTRAST, and a contrast needs its
control committed alongside it. "68 % of infeasible steps carry a row demanding a
super-physical closing rate" means nothing without the same figure on feasible steps. This
module computes both over all 69 226 corpus steps and writes the result as an artefact, so the
risk ratios in STAGE2_REPORT.md section 7 are reproducible from the repository.

Diagnostic only: nothing is changed, tuned or remedied, and V0 is untouched.

A row demands a super-physical closing rate when dh_dt = -grad_h . v_track is more negative
than the fastest an obstacle can physically travel in that condition. V_PHYS is taken from the
environment configuration, and the randomized regime uses its upper bound 0.75, which makes the
test CONSERVATIVE (it under-counts super-physical rows).

NOTE ON npz ACCESS: the corpus is compressed, so indexing `z["xi_flat"]` inside a per-step loop
re-decompresses the whole array every time. The arrays are materialised once here; the earlier
in-loop version was abandoned for that reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 2 / super-physical closing-rate base rate"
OUT = Path("results/week5_phase7/stage2_infeasibility_diagnosis")
V_PHYS = {"fixed": 0.675, "randomized": 0.75}
PARTS = [("dev_fixed", "fixed"), ("dev_randomized", "randomized"),
         ("validation_fixed", "fixed"), ("validation_randomized", "randomized"),
         ("final_fixed", "fixed"), ("final_randomized", "randomized")]


def main():
    tot = inf = sp_inf = sp_feas = n_feas = 0
    wi, wf, per_part = [], [], {}
    for name, cond in PARTS:
        meta = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
        z = np.load(CORPUS_DIR / meta["npz"])
        dh = z["xi_flat"][:, 0].copy()          # materialise once; see the module docstring
        ptr = z["xi_ptr"].copy()
        code = z["status_code"].copy()
        tab = {int(k): v for k, v in meta["status_table"].items()}
        names = np.array([tab[int(c)] for c in code])
        is_inf = np.char.find(names.astype(str), "infeasible") >= 0
        v = V_PHYS[cond]

        mins = np.minimum.reduceat(dh, ptr[:-1])            # per-step worst dh_dt
        over = np.maximum(0.0, -mins - v)                   # m/s beyond the physical maximum
        sp = over > 0

        tot += len(mins)
        inf += int(is_inf.sum())
        n_feas += int((~is_inf).sum())
        sp_inf += int(sp[is_inf].sum())
        sp_feas += int(sp[~is_inf].sum())
        wi.append(over[is_inf])
        wf.append(over[~is_inf])
        per_part[name] = {"condition": cond, "v_phys": v, "steps": int(len(mins)),
                          "infeasible": int(is_inf.sum()),
                          "superphysical_infeasible": int(sp[is_inf].sum()),
                          "superphysical_feasible": int(sp[~is_inf].sum())}

    wi, wf = np.concatenate(wi), np.concatenate(wf)
    n_sp, n_nsp = sp_inf + sp_feas, tot - (sp_inf + sp_feas)
    R = {
        "label": LABEL, "steps": tot, "infeasible": inf, "feasible": n_feas,
        "p_superphysical_given_infeasible": sp_inf / inf,
        "p_superphysical_given_feasible": sp_feas / n_feas,
        "risk_ratio_exposure": (sp_inf / inf) / (sp_feas / n_feas),
        "p_infeasible_given_superphysical": sp_inf / n_sp,
        "p_infeasible_given_not_superphysical": (inf - sp_inf) / n_nsp,
        "risk_ratio_outcome": (sp_inf / n_sp) / ((inf - sp_inf) / n_nsp),
        "overshoot_infeasible": {"n": int((wi > 0).sum()),
                                 "median": float(np.median(wi[wi > 0])),
                                 "p95": float(np.percentile(wi[wi > 0], 95)),
                                 "max": float(wi.max())},
        "overshoot_feasible": {"n": int((wf > 0).sum()),
                               "median": float(np.median(wf[wf > 0])),
                               "p95": float(np.percentile(wf[wf > 0], 95)),
                               "max": float(wf.max())},
        "per_part": per_part,
    }
    print(f"### {LABEL}\n")
    print(f"all steps {tot}, infeasible {inf} ({100*inf/tot:.3f} %), feasible {n_feas}\n")
    print("P(>=1 super-physical estimated closing rate):")
    print(f"   infeasible steps : {sp_inf}/{inf} = {R['p_superphysical_given_infeasible']:.4f}")
    print(f"   feasible steps   : {sp_feas}/{n_feas} = {R['p_superphysical_given_feasible']:.4f}")
    print(f"   risk ratio       : {R['risk_ratio_exposure']:.2f}x")
    print(f"\nP(infeasible | >=1 super-physical row) = {R['p_infeasible_given_superphysical']:.4f}")
    print(f"P(infeasible | no  super-physical row) = "
          f"{R['p_infeasible_given_not_superphysical']:.5f}")
    print(f"   risk ratio = {R['risk_ratio_outcome']:.1f}x")
    print("\novershoot beyond physical (m/s), among steps with any:")
    for lab, k in (("infeasible", "overshoot_infeasible"), ("feasible", "overshoot_feasible")):
        s = R[k]
        print(f"   {lab:11s} n={s['n']:6d} median {s['median']:.4f} "
              f"p95 {s['p95']:.4f} max {s['max']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_superphysical_baserate.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"\nwrote {OUT}/stage2_superphysical_baserate.json")


if __name__ == "__main__":
    main()
