"""RANDOM_EXP_R3 — tune the random recovery only (Tuned DR-CBF frozen).

--phase screen   : 45 configs = K {4,8,16} x H {1,3,5} x 5 speed schedules, 20 eps/condition,
                   G1 timeout gate only, top 4 by rank key (5th = mean step time) -> shortlist.
--phase validate : shortlist (4) + Tuned, 100 eps/condition, G1 + G2 (Delta 10 pp) vs Tuned,
                   rank-1 eligible -> R3/random_frozen.json (+ sha256).
Budget: 45*20*2 + 5*100*2 = 1 800 + 1 000 = 2 800 episodes (AUDIT.md section 7).
"""
from __future__ import annotations

import argparse
import json

from continuation.harness import ArmSpec, RESULTS, write_new
from continuation.random_recovery import RandomSpec
from continuation.seeds import seed_block
from continuation.stats import compare_all, rank_key, select
from experiments.week6_continuation.common import (load_json, load_tuned, print_paired,
                                                   run_stage, sha256, table)

K_SET, H_SET, SPEEDS = (4, 8, 16), (1, 3, 5), (0.2, 0.4, 0.6, 0.8, 1.0)
SHORTLIST = 4
FIFTH = "step_time_ms_mean_ep"


def schedules(axis):
    if axis == "escalation_start":
        return [(f"esc{v0:g}", tuple(v for v in SPEEDS if v >= v0 - 1e-12)) for v0 in SPEEDS]
    if axis == "fixed_speed":
        return [(f"fix{v:g}", (v,)) for v in SPEEDS]
    raise ValueError(axis)


def configs(axis, tuned):
    out = []
    for K in K_SET:
        for H in H_SET:
            for tag, sp in schedules(axis):
                name = f"R3_K{K}_H{H}_{tag}"
                out.append(ArmSpec(name, "random", params=tuned,
                                   random=RandomSpec(K=K, H=H, speeds=sp, name=name)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["screen", "validate"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tuned, _ = load_tuned()
    axis = load_json(RESULTS / "RANDOM_EXP_R2" / "r3_axis.json")["axis"]
    cfgs = configs(axis, tuned)
    assert len(cfgs) == 45
    by = {c.name: c for c in cfgs}

    if a.phase == "screen":
        res, summ, meta = run_stage("R3", cfgs, seed_block("RANDOM_EXP_R3_SCREEN"),
                                    workers=a.workers, tag="R3_screen", progress=300)
        elig, rej = select(summ, None, n_per_condition=20, fifth=FIFTH)
        short = elig[:SHORTLIST]
        write_new(RESULTS / "R3" / "R3_screen_selection.json", {
            "meta": meta, "axis": axis,
            "ranking": [{"name": n, "spec": by[n].random.as_dict(),
                         "key": rank_key(summ[n]["pooled"], FIFTH)} for n in elig],
            "rejected": rej, "shortlist": short})
        print(table(summ, elig[:12], conds=("pooled",)))
        print("R3 SHORTLIST:", short)
        if not short:
            raise SystemExit("STOP: no R3 configuration passed the screen")
        return

    sel = load_json(RESULTS / "R3" / "R3_screen_selection.json")
    short = sel["shortlist"]
    tuned_arm = ArmSpec("Tuned", "tuned", params=tuned)
    arms = [tuned_arm] + [by[n] for n in short]
    res, summ, meta = run_stage("R3", arms, seed_block("RANDOM_EXP_R3_VALID"),
                                workers=a.workers, tag="R3_validation")
    elig, rej = select({n: summ[n] for n in short}, summ["Tuned"], n_per_condition=100,
                       fifth=FIFTH)
    paired = {n: compare_all(res[n], res["Tuned"]) for n in short}
    write_new(RESULTS / "R3" / "R3_validation_selection.json", {
        "meta": meta, "ranking": [{"name": n, "key": rank_key(summ[n]["pooled"], FIFTH)}
                                  for n in elig],
        "rejected": rej, "paired_vs_Tuned": paired})
    print(table(summ))
    for n in short:
        print_paired(f"{n} vs Tuned", paired[n], keys=("success", "collision"))
    print(f"eligible {elig}; rejected {rej}")
    if not elig:
        raise SystemExit("STOP: no R3 configuration passed validation")
    win = by[elig[0]]
    frozen = {"name": "Random-CLF-DR-CBF", "source_arm": win.name,
              "spec": win.random.as_dict(), "tuned_params": tuned.as_dict(), "axis": axis,
              "selected_by": "SELECTION_RULES.md R3 validation, rank 1",
              "rules_sha256": meta["rules_sha256"]}
    p = write_new(RESULTS / "R3" / "random_frozen.json", frozen)
    (RESULTS / "R3" / "random_frozen.sha256").write_text(f"{sha256(p)}  {p}\n")
    print("RANDOM-CLF-DR-CBF FROZEN:", json.dumps(frozen["spec"]))


if __name__ == "__main__":
    main()
