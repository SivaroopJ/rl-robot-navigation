"""RANDOM_EXP_R1 — basic random recovery (K = 8, H = 1, v_R = 0.2) vs frozen Tuned-CLF-DR-CBF.

NOT the LADDER rung R1. Report only; nothing is selected here (SELECTION_RULES.md).
"""
from __future__ import annotations

import argparse

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.random_recovery import RandomSpec
from continuation.seeds import seed_block
from continuation.stats import compare_all
from experiments.week6_continuation.common import load_tuned, print_paired, run_stage, table

R1_SPEC = RandomSpec(K=8, H=1, speeds=(0.2,), name="RANDOM_EXP_R1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tuned, _ = load_tuned()
    arms = [ArmSpec("Tuned", "tuned", params=tuned),
            ArmSpec("RANDOM_EXP_R1", "random", params=tuned, random=R1_SPEC)]
    res, summ, meta = run_stage("RANDOM_EXP_R1", arms, seed_block("RANDOM_EXP_R1"),
                                workers=a.workers)
    cmp = compare_all(res["RANDOM_EXP_R1"], res["Tuned"])
    write_new(RESULTS / "RANDOM_EXP_R1" / "R1_paired.json",
              {"meta": meta, "R1_vs_Tuned (positive = R1 higher)": cmp})
    print(table(summ))
    print_paired("R1 vs Tuned", cmp)


if __name__ == "__main__":
    main()
