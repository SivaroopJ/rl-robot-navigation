"""Render the Phase 1 gate report from the exp0 JSON files."""
import json, sys
from pathlib import Path

KEYS = [("success_rate","success"),("collision_rate","collision"),("timeout_rate","timeout"),
        ("min_h_true","min h_true (mean)"),("min_h_true_min","min h_true (worst)"),
        ("min_cbc","min CBC constrained"),("min_cbc_min","  worst ep"),
        ("min_cbc_solved","min CBC | solved steps"),("min_cbc_all","min CBC all samples"),
        ("frac_cbc_negative","frac CBC<0"),("frac_delta_positive","frac delta>0"),
        ("frac_cbf_active","frac CBF active"),("frac_h_negative","frac h<0"),
        ("n_solver_fail","solver failures"),("n_infeasible","  of which infeasible"),("n_dcp_error","DCP errors"),
        ("mean_solver_time","mean solver s"),("mean_canon_time","mean canon s"),
        ("mean_total_time","mean total s"),("mean_u_dev","mean ||u-u_nom||"),
        ("steps","total steps")]

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n########## {Path(path).name}   {d['config']}")
    arms = list(d["arms"])
    print(f"{'metric':24s}" + "".join(f"{a:>14s}" for a in arms))
    for key, label in KEYS:
        row = [d["arms"][a]["summary"].get(key) for a in arms]
        if all(v is None for v in row): continue
        cells = "".join(f"{v:>14.4g}" if isinstance(v,(int,float)) else f"{'-':>14s}" for v in row)
        print(f"{label:24s}{cells}")
    if "r_w_sweep" in d:
        print("\n  r_W effect (diagnostic; no monotonicity asserted):")
        hdr = ["wasserstein_r","min_h_true","min_cbc","collision_rate","success_rate","mean_u_dev"]
        print("   " + "".join(f"{h:>16s}" for h in hdr))
        for r in d["r_w_sweep"]:
            print("   " + "".join(f"{r[h]:>16.5g}" for h in hdr))
