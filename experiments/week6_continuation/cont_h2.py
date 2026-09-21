"""H2 — fine DR-CBF/CLF search on the H1 shortlist: lambda_V x k_v, 50 episodes/condition.

Selection per SELECTION_RULES.md "H2": G1 + G2 (Delta 10 pp) vs a C0-A anchor on the same
block; Tuned-CLF-DR-CBF = rank-1 eligible candidate, frozen to H2/tuned_frozen.json (+ sha256).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.params import H2_CLF_RATE, H2_K_V, DRCBFParams
from continuation.stats import rank_key, select
from continuation.seeds import seed_block
from experiments.week6_continuation.common import load_json, run_stage, sha256, table

ANCHOR = ArmSpec("H2_anchor_C0-A", "original")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    h1 = load_json(RESULTS / "H1" / "H1_selection.json")
    short = [DRCBFParams.from_dict(s["params"]) for s in h1["shortlist"]]
    assert short and all(p.regime == "collapsed" for p in short)
    cands = []
    for p in short:
        for lv in H2_CLF_RATE:
            for kv in H2_K_V:
                q = replace(p, clf_rate=lv, k_v=kv)
                cands.append(ArmSpec(f"H2_{q.tag()}", "tuned", params=q))
    assert len({c.name for c in cands}) == len(cands)
    print(f"H2: {len(short)} classes x {len(H2_CLF_RATE)} lambda_V x {len(H2_K_V)} k_v = "
          f"{len(cands)} candidates + anchor")
    res, summ, meta = run_stage("H2", [ANCHOR] + cands, seed_block("H2"), workers=a.workers,
                                progress=500)
    elig, rej = select({c.name: summ[c.name] for c in cands}, summ[ANCHOR.name],
                       n_per_condition=50)
    by = {c.name: c for c in cands}
    out = {"meta": meta, "anchor": ANCHOR.name,
           "ranking": [{"name": n, "params": by[n].params.as_dict(),
                        "key": rank_key(summ[n]["pooled"]),
                        "fixed": {k: summ[n]["fixed"][k] for k in ("success_rate", "collision_rate")},
                        "randomized": {k: summ[n]["randomized"][k]
                                       for k in ("success_rate", "collision_rate")}}
                       for n in elig],
           "rejected": rej}
    write_new(RESULTS / "H2" / "H2_selection.json", out)
    print(table(summ, [ANCHOR.name] + elig[:10]))
    print(f"eligible {len(elig)} / {len(cands)}; rejected {len(rej)}")
    if not elig:
        raise SystemExit("STOP: no eligible H2 candidate")
    win = by[elig[0]]
    frozen = {"name": "Tuned-CLF-DR-CBF", "source_arm": win.name,
              "params": win.params.as_dict(), "selected_by": "SELECTION_RULES.md H2, rank 1",
              "rules_sha256": meta["rules_sha256"],
              "h2_pooled": {k: summ[win.name]["pooled"][k] for k in
                            ("success_rate", "collision_rate", "timeout_rate",
                             "min_clearance_mean", "infeasible_step_rate")}}
    p = write_new(RESULTS / "H2" / "tuned_frozen.json", frozen)
    (RESULTS / "H2" / "tuned_frozen.sha256").write_text(f"{sha256(p)}  {p}\n")
    print("TUNED-CLF-DR-CBF FROZEN:", json.dumps(frozen["params"]))


if __name__ == "__main__":
    main()
