"""RANDOM_EXP_R2 — speed escalation {0.2..1.0}, accept at m >= 0, else best over all speeds.

Arms: Tuned, RANDOM_EXP_R1 (re-run so R2 vs R1 is paired), RANDOM_EXP_R2. Decides the R3 speed
axis per SELECTION_RULES.md: escalation-start axis iff rank_key(R2) <= rank_key(R1).
"""
from __future__ import annotations

import argparse

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.random_recovery import RandomSpec
from continuation.seeds import seed_block
from continuation.stats import compare_all, rank_key
from experiments.week6_continuation.common import load_tuned, print_paired, run_stage, table
from experiments.week6_continuation.cont_r1 import R1_SPEC

SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)
R2_SPEC = RandomSpec(K=8, H=1, speeds=SPEEDS, accept_m=0.0, name="RANDOM_EXP_R2")
FIFTH = "step_time_ms_mean_ep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tuned, _ = load_tuned()
    arms = [ArmSpec("Tuned", "tuned", params=tuned),
            ArmSpec("RANDOM_EXP_R1", "random", params=tuned, random=R1_SPEC),
            ArmSpec("RANDOM_EXP_R2", "random", params=tuned, random=R2_SPEC)]
    res, summ, meta = run_stage("RANDOM_EXP_R2", arms, seed_block("RANDOM_EXP_R2"),
                                workers=a.workers)
    c_t = compare_all(res["RANDOM_EXP_R2"], res["Tuned"])
    c_1 = compare_all(res["RANDOM_EXP_R2"], res["RANDOM_EXP_R1"])
    k2 = rank_key(summ["RANDOM_EXP_R2"]["pooled"], FIFTH)
    k1 = rank_key(summ["RANDOM_EXP_R1"]["pooled"], FIFTH)
    axis = "escalation_start" if k2 <= k1 else "fixed_speed"
    write_new(RESULTS / "RANDOM_EXP_R2" / "R2_paired.json",
              {"meta": meta, "R2_vs_Tuned": c_t, "R2_vs_R1": c_1})
    write_new(RESULTS / "RANDOM_EXP_R2" / "r3_axis.json",
              {"axis": axis, "rank_key_R2": k2, "rank_key_R1": k1,
               "rule": "escalation_start iff rank_key(R2) <= rank_key(R1)"})
    print(table(summ))
    print_paired("R2 vs Tuned", c_t)
    print_paired("R2 vs R1", c_1)
    print(f"R3 SPEED AXIS: {axis}  (R2 key {k2} vs R1 key {k1})")


if __name__ == "__main__":
    main()
