"""H1 — coarse DR-CBF search: 5 alpha x 6 r_W x 4 eps = 120 cells, 20 episodes/condition.

Selection per SELECTION_RULES.md "H1": collapsed regime only, classes = effective_class(),
representative = lowest r_W then lowest eps, G1 + G2 (Delta 15 pp) vs the reference cell,
top-10 eligible classes -> H2. eps = 0.30 is ranked and reported separately, never forwarded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.params import REFERENCE, h1_grid
from continuation.seeds import seed_block
from continuation.stats import rank_key, select
from experiments.week6_continuation.common import run_stage, table

SHORTLIST = 10


def arm_for(p):
    return ArmSpec(f"H1_{p.tag()}", "tuned", params=p)


def degeneracy(res, members):
    """Fraction of (condition, seed) episodes on which all duplicate cells agree exactly."""
    names = [arm_for(p).name for p in members]
    agree = total = 0
    maxdiff = 0.0
    for c in ("fixed", "randomized"):
        for i in range(len(res[names[0]][c])):
            rows = [res[n][c][i] for n in names]
            same = all((r["outcome"], r["steps"]) == (rows[0]["outcome"], rows[0]["steps"])
                       for r in rows)
            agree += int(same)
            total += 1
            maxdiff = max(maxdiff, max(abs(r["min_clearance"] - rows[0]["min_clearance"])
                                       for r in rows))
    return {"cells": names, "episode_agreement": agree / total, "max_clearance_diff": maxdiff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    grid = h1_grid()
    arms = [arm_for(p) for p in grid]
    ref_name = arm_for(REFERENCE).name
    assert ref_name in [x.name for x in arms]
    res, summ, meta = run_stage("H1", arms, seed_block("H1"), workers=a.workers, progress=400)

    by_class = defaultdict(list)
    for p in grid:
        by_class[p.effective_class()].append(p)
    reps, degen = {}, {}
    for k, members in by_class.items():
        members.sort(key=lambda p: (p.wasserstein_r, p.epsilon))
        reps[k] = members[0]
        if len(members) > 1:
            degen[str(k)] = degeneracy(res, members)

    def pick(regime):
        cands = {arm_for(p).name: summ[arm_for(p).name]
                 for k, p in reps.items() if p.regime == regime}
        return select(cands, summ[ref_name], n_per_condition=20)

    elig, rej = pick("collapsed")
    elig_nc, rej_nc = pick("non_collapsed")
    short = elig[:SHORTLIST]
    by_name = {arm_for(p).name: p for p in grid}
    out = {
        "meta": meta, "reference": ref_name,
        "n_cells": len(grid), "n_classes_collapsed": sum(p.regime == "collapsed"
                                                        for p in reps.values()),
        "n_classes_non_collapsed": sum(p.regime != "collapsed" for p in reps.values()),
        "collapsed_ranking": [{"name": n, "params": by_name[n].as_dict(),
                               "key": rank_key(summ[n]["pooled"])} for n in elig],
        "collapsed_rejected": rej,
        "shortlist": [{"name": n, "params": by_name[n].as_dict()} for n in short],
        "non_collapsed_ranking_REPORT_ONLY": [{"name": n, "params": by_name[n].as_dict(),
                                               "key": rank_key(summ[n]["pooled"])}
                                              for n in elig_nc],
        "non_collapsed_rejected": rej_nc,
        "degeneracy_check": degen,
    }
    if not short:
        out["decision"] = "STOP: no eligible collapsed-regime class"
    write_new(RESULTS / "H1" / "H1_selection.json", out)

    print(table(summ, [ref_name] + [n for n in short if n != ref_name], conds=("pooled",)))
    print(f"\neligible collapsed classes: {len(elig)} / {out['n_classes_collapsed']}; "
          f"rejected {len(rej)}")
    ag = [d["episode_agreement"] for d in degen.values()]
    print(f"degeneracy: {len(degen)} duplicate classes, episode agreement "
          f"min {min(ag):.3f} mean {sum(ag) / len(ag):.3f}")
    print("SHORTLIST:", *short, sep="\n  ")
    if not short:
        raise SystemExit("STOP: no eligible class")


if __name__ == "__main__":
    main()
