"""Week 6 Continuation / reference-equivalence gate (audit A12). MANDATORY before H1.

Reproduces EXISTING published per-episode records with the continuation harness; touches no
1_000_000 seed and makes no decision. Exact equality is required on every compared field:
the pipeline is deterministic given the seed, so any mismatch means something moved.

  G-A  kind="original"  (frozen DRCBFPolicy)       vs Stage-3 C0_frozen   21_000+0..49 fixed+randomized
  G-B  kind="tuned" @ REFERENCE (TunableDRCBFPolicy) vs the same records             <- the gate proper
  G-C  kind="tuned" @ REFERENCE                     vs Stage-3 C0_frozen   20_000+0..19 fixed
  G-D  kind="c0b"  (Phase7Policy T1 + LADDER)       vs Stage-4 LADDER      20_000+0..19 fixed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from continuation.harness import RESULTS, ArmSpec, run_block, write_new

S3 = Path("results/week5_phase7/stage3_velocity_intervention")
S4 = Path("results/week5_phase7/stage4_recovery")
FIELDS = ("seed", "outcome", "steps", "success", "collision", "timeout", "collision_type",
          "min_clearance", "spl")


def diff(ours, theirs, extra):
    bad = []
    for a, b in zip(ours, theirs):
        for k in FIELDS + extra:
            va, vb = a[k[0]] if isinstance(k, tuple) else a[k], b[k[1]] if isinstance(k, tuple) else b[k]
            if va != vb:
                bad.append({"seed": b["seed"], "field": k, "ours": va, "published": vb})
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    s3v = json.loads((S3 / "stage3_validation.json").read_text())
    s3d = json.loads((S3 / "stage3_dev.json").read_text())
    s4d = json.loads((S4 / "stage4_dev.json").read_text())

    out = {"label": "Week6-Continuation / reference-equivalence gate", "checks": {}}
    # G-A / G-B
    seeds = [r["seed"] for r in s3v["fixed"]["episodes"]["C0_frozen"]]
    assert seeds == list(range(21_000, 21_050))
    res = run_block([ArmSpec("original", "original"), ArmSpec("tuned_ref", "tuned")], seeds,
                    workers=a.workers)
    for cond in ("fixed", "randomized"):
        pub = s3v[cond]["episodes"]["C0_frozen"]
        for tag, arm in (("G-A", "original"), ("G-B", "tuned_ref")):
            bad = diff(res[arm][cond], pub, (("n_solver_fail", "n_solver_fail"),))
            out["checks"][f"{tag} {arm} vs Stage3 C0_frozen validation/{cond}"] = {
                "episodes": len(pub), "mismatches": bad, "pass": not bad}
    # G-C / G-D
    seeds = list(range(20_000, 20_020))
    assert [r["seed"] for r in s3d["fixed"]["episodes"]["C0_frozen"]] == seeds
    assert [r["seed"] for r in s4d["fixed"]["episodes"]["LADDER"]] == seeds
    res = run_block([ArmSpec("tuned_ref", "tuned"), ArmSpec("C0-B", "c0b")], seeds,
                    conditions=("fixed",), workers=a.workers)
    bad = diff(res["tuned_ref"]["fixed"], s3d["fixed"]["episodes"]["C0_frozen"],
               (("n_solver_fail", "n_solver_fail"),))
    out["checks"]["G-C tuned_ref vs Stage3 C0_frozen dev/fixed"] = {
        "episodes": 20, "mismatches": bad, "pass": not bad}
    bad = diff(res["C0-B"]["fixed"], s4d["fixed"]["episodes"]["LADDER"],
               (("n_infeasible", "n_infeasible"), ("rungs", "rungs")))
    out["checks"]["G-D C0-B vs Stage4 LADDER dev/fixed"] = {
        "episodes": 20, "mismatches": bad, "pass": not bad}

    out["GATE_PASS"] = all(c["pass"] for c in out["checks"].values())
    out["wall_s"] = time.time() - t0
    p = write_new(RESULTS / "gate" / "gate.json", out)
    for k, v in out["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  ({v['episodes']} eps, "
              f"{len(v['mismatches'])} mismatches)")
        for m in v["mismatches"][:5]:
            print("     ", m)
    print("GATE_PASS =", out["GATE_PASS"], "->", p)
    sys.exit(0 if out["GATE_PASS"] else 1)


if __name__ == "__main__":
    main()
