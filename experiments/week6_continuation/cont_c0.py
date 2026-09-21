"""C0 — frozen baselines on the continuation C0 block (10_000_000 + 0..99, both conditions).

C0-A = frozen DRCBFPolicy (original CLF-DR-CBF).
C0-B = frozen historical Phase7Policy(T1, v_cap 0.96) + RecoveryLadder("ladder"). Historical
       reference only: never tuned, feeds nothing forward.
"""
from __future__ import annotations

import argparse

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.seeds import seed_block
from continuation.stats import compare_all
from experiments.week6_continuation.common import run_stage, table

ARMS = [ArmSpec("C0-A_original", "original"), ArmSpec("C0-B_T1+LADDER", "c0b")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    seeds = seed_block("C0", a.episodes)
    tag = "C0" if a.episodes is None else f"C0_smoke{a.episodes}"
    res, summ, meta = run_stage("C0", ARMS, seeds, workers=a.workers, tag=tag)
    paired = compare_all(res["C0-B_T1+LADDER"], res["C0-A_original"])
    write_new(RESULTS / "C0" / f"{tag}_paired.json",
              {"meta": meta, "B_vs_A (positive = C0-B higher)": paired})
    print(table(summ))
    for c in ("fixed", "randomized", "pooled"):
        for k in ("success", "collision", "dynamic_collision"):
            v = paired[c][k]
            print(f"  C0-B vs C0-A {c:10s} {k:18s} diff {v['diff']:+.3f} "
                  f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p']:.4f}")


if __name__ == "__main__":
    main()
