"""C1 — temporal-buffer / DR-sample-count diagnostic, N = k_scans in {1, 3, 5}.

Everything else is the reference configuration (no T1, no recovery). N = 5 is bit-identical to
C0-A. DIAGNOSTIC: N is not changed for H1 unless the SELECTION_RULES.md C1 stop condition fires,
in which case the script says STOP and the user decides.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.params import C1_K_SCANS, REFERENCE
from continuation.seeds import seed_block
from continuation.stats import compare_all, gates
from experiments.week6_continuation.common import run_stage, table

ARMS = [ArmSpec(f"C1-N{n}", "tuned", params=replace(REFERENCE, k_scans=n)) for n in C1_K_SCANS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    res, summ, meta = run_stage("C1", ARMS, seed_block("C1"), workers=a.workers)
    ref = "C1-N5"
    out = {"meta": meta, "paired_vs_N5": {}, "stop_condition": {}}
    stop = False
    for n in ("C1-N1", "C1-N3"):
        cmp = compare_all(res[n], res[ref])
        out["paired_vs_N5"][n] = cmp
        g = gates(summ[n], summ[ref], n_per_condition=100)
        lower = cmp["pooled"]["collision"]["diff"] < 0 and cmp["pooled"]["collision"]["p"] < 0.05
        fire = bool(lower and not g)
        out["stop_condition"][n] = {"collision_significantly_lower": lower,
                                    "gate_failures": g, "fires": fire}
        stop |= fire
    out["decision"] = ("STOP_FOR_REVIEW" if stop else "KEEP_N5")
    write_new(RESULTS / "C1" / "C1_decision.json", out)
    print(table(summ))
    for n, c in out["paired_vs_N5"].items():
        for cond in ("fixed", "randomized", "pooled"):
            for k in ("collision", "success"):
                v = c[cond][k]
                print(f"  {n} vs N5 {cond:10s} {k:9s} diff {v['diff']:+.3f} "
                      f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p']:.4f}")
            v = c[cond]["infeasible_step_rate"]
            print(f"  {n} vs N5 {cond:10s} inf-rate  {v['x']:.4f} vs {v['y']:.4f} "
                  f"diff {v['diff']:+.4f} CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
    print("C1 DECISION:", out["decision"])


if __name__ == "__main__":
    main()
