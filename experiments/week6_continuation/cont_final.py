"""F — final paired comparison on the SEALED FINAL block (10_900_000 + 0..199, both conditions).

F-A  Original CLF-DR-CBF (frozen DRCBFPolicy)
F-B  Tuned-CLF-DR-CBF          (H2/tuned_frozen.json)
F-C  Tuned + existing LADDER   (tau = tau_eff, no T1 cap)
F-D  Random-CLF-DR-CBF         (Tuned + R3/random_frozen.json)

Runs ONCE: refuses if any F output exists, if either frozen file's hash differs, or if the
selection rules changed. Comparisons: A->B (tuning), B->C (LADDER after tuning),
B->D (random after tuning), D vs C (H4). Positive diff = first-named arm higher.
"""
from __future__ import annotations

import argparse

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.seeds import seed_block
from continuation.stats import compare_all
from experiments.week6_continuation.common import (load_random, load_tuned, print_paired,
                                                   run_stage, table)

COMPARISONS = (("F-B_Tuned", "F-A_Original", "tuning: Tuned vs Original"),
               ("F-C_Tuned+LADDER", "F-B_Tuned", "LADDER after tuning"),
               ("F-D_Random", "F-B_Tuned", "random recovery after tuning"),
               ("F-D_Random", "F-C_Tuned+LADDER", "H4: Random vs LADDER"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    assert rf["tuned_params"] == tf["params"], "random config was tuned on a different base"
    arms = [ArmSpec("F-A_Original", "original"),
            ArmSpec("F-B_Tuned", "tuned", params=tuned),
            ArmSpec("F-C_Tuned+LADDER", "ladder", params=tuned),
            ArmSpec("F-D_Random", "random", params=tuned, random=spec)]
    seeds = seed_block("FINAL", allow_final=True)
    res, summ, meta = run_stage("F", arms, seeds, workers=a.workers, progress=200)
    paired = {lab: compare_all(res[x], res[y]) for x, y, lab in COMPARISONS}
    write_new(RESULTS / "F" / "F_paired.json",
              {"meta": meta, "tuned": tf, "random": rf,
               "comparisons": {lab: {"x": x, "y": y, "result": paired[lab]}
                               for x, y, lab in COMPARISONS},
               "n_comparisons": len(COMPARISONS)})
    print(table(summ))
    for x, y, lab in COMPARISONS:
        print(f"\n{lab}  ({x} vs {y})")
        print_paired(lab[:14], paired[lab])


if __name__ == "__main__":
    main()
