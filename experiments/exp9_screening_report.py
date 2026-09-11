"""Week 6 / screening report. READ-ONLY over results/week6_generalization/screen/.

Descriptive by construction. Screening characterises maps; it selects nothing, and no number
here may be used to tune, reorder or reject anything (spec 4.3, and the standing rule since
Phase 7). X1-X7 are feasibility characterisation and are EXCLUDED from every benchmark
aggregate; they get their own section and no success rate is reported for them.

Paired McNemar is computed per cell for information ONLY. At n = 50 it resolves roughly 0.15
absolute, so a cell-level p-value is nearly always uninformative. Every claim in the report is
labelled descriptive unless it survives that.
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from generalization import suite

SCREEN = Path("results/week6_generalization/screen")
OUT = Path("results/week6_generalization")
ALPHA = 0.05


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load():
    rows = []
    for p in sorted(SCREEN.glob("*.json")):
        d = json.loads(p.read_text())
        A, B = d["episodes_A"], d["episodes_B"]
        n01s, n10s, ps = mcnemar([r["success"] for r in A], [r["success"] for r in B])
        n01c, n10c, pc = mcnemar([r["collision"] for r in A], [r["collision"] for r in B])
        rows.append({**{k: d[k] for k in ("id", "family", "level", "classification",
                                          "difficulty", "stress", "motion_model",
                                          "episodes", "meta", "seconds")},
                     "A": d["A"], "B": d["B"],
                     "p_success": ps, "p_collision": pc,
                     "n01_success": n01s, "n10_success": n10s,
                     "n01_collision": n01c, "n10_collision": n10c,
                     "sec_per_episode": d["seconds"] / max(2 * d["episodes"], 1)})
    return rows


def main():
    rows = load()
    man = suite.build_manifest()
    expected = len(suite.screening_cells(man))
    bench = [r for r in rows if not r["stress"]]
    stress = [r for r in rows if r["stress"]]

    L = []
    w = L.append
    w("# Week 6 / Generalization Suite — SCREENING REPORT")
    w("")
    w(f"**{len(rows)} of {expected} screening cells complete.** 50 paired episodes per cell, "
      "both arms, seeds `SCREEN_SEED_BASE = 8_000_000 … +49`.")
    w("")
    w("**Screening is descriptive.** It characterises maps; it selects nothing. No number here "
      "may be used to tune, reorder or reject any part of the preregistered design. At n = 50 a "
      "paired test resolves only ~0.15 absolute, so cell-level p-values are reported for "
      "information and are almost never conclusive.")
    w("")
    w("**X1–X7 are feasibility characterisation** and are excluded from every benchmark "
      "aggregate below. No success rate is reported for them.")
    w("")

    # ---------------------------------------------------------------- per family
    w("## 1. Results by family (benchmark families only)")
    w("")
    w("Episode-weighted means over the cells of each family. Δ = B − A.")
    w("")
    w("| family | cells | A succ | B succ | Δ succ | A coll | B coll | Δ coll | A inf | B inf | B recov |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    fam = defaultdict(list)
    for r in bench:
        fam[r["family"]].append(r)
    order = [f for f in ["F1'", "F2a", "F2b", "F3", "F4", "F5", "F6", "F8", "F9",
                         "F10", "F12", "F7"] if f in fam]
    for f in order:
        R = fam[f]
        m = lambda arm, k: float(np.mean([x[arm][k] for x in R]))
        w(f"| {f} | {len(R)} | {m('A','success'):.3f} | {m('B','success'):.3f} | "
          f"{m('B','success')-m('A','success'):+.3f} | {m('A','collision'):.3f} | "
          f"{m('B','collision'):.3f} | {m('B','collision')-m('A','collision'):+.3f} | "
          f"{m('A','frac_infeasible'):.4f} | {m('B','frac_infeasible'):.4f} | "
          f"{sum(x['B']['n_recovered'] for x in R)} |")
    w("")

    # ---------------------------------------------------------------- per config
    w("## 2. Results by configuration")
    w("")
    w("| id | model | A succ | B succ | Δ | A coll | B coll | Δ | p(coll) | class |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(bench, key=lambda x: (order.index(x["family"]) if x["family"] in order else 99,
                                          x["id"], x["motion_model"])):
        A, B = r["A"], r["B"]
        sig = "**" if r["p_collision"] < ALPHA else ""
        w(f"| {r['id']} | {r['motion_model'][:5]} | {A['success']:.2f} | {B['success']:.2f} | "
          f"{B['success']-A['success']:+.2f} | {A['collision']:.2f} | {B['collision']:.2f} | "
          f"{B['collision']-A['collision']:+.2f} | {sig}{r['p_collision']:.3f}{sig} | "
          f"{r['classification']} |")
    w("")

    # ---------------------------------------------------------------- collision split
    w("## 3. Collision split, timeout and infeasibility")
    w("")
    w("| family | A dyn/stat/wall | B dyn/stat/wall | A timeout | B timeout | A clr | B clr | A worst | B worst |")
    w("|---|---|---|---|---|---|---|---|---|")
    for f in order:
        R = fam[f]
        s = lambda arm, k: int(sum(x[arm][k] for x in R))
        m = lambda arm, k: float(np.mean([x[arm][k] for x in R]))
        mn = lambda arm: float(np.min([x[arm]["min_clearance_worst"] for x in R]))
        w(f"| {f} | {s('A','dynamic')}/{s('A','static')}/{s('A','wall')} | "
          f"{s('B','dynamic')}/{s('B','static')}/{s('B','wall')} | {m('A','timeout'):.3f} | "
          f"{m('B','timeout'):.3f} | {m('A','min_clearance'):.3f} | {m('B','min_clearance'):.3f} | "
          f"{mn('A'):+.4f} | {mn('B'):+.4f} |")
    w("")

    # ---------------------------------------------------------------- recovery
    w("## 4. Recovery behaviour (arm B) and guarantee tiers")
    w("")
    w("A recovered action is never described as safe: the tier and margin accompany it. "
      "T0 = DR guarantee intact, T1 = nominal CBF only, T2 = `m < 0`, forward invariance lost.")
    w("")
    w("| family | infeasible steps | recovered | T0 | T1 | T2 | rungs |")
    w("|---|---|---|---|---|---|---|")
    for f in order:
        R = fam[f]
        t = defaultdict(int)
        rung = defaultdict(int)
        for x in R:
            for k, v in x["B"]["tiers"].items():
                t[str(k)] += v
            for k, v in x["B"]["rungs"].items():
                rung[k] += v
        w(f"| {f} | {sum(x['B']['n_infeasible'] for x in R)} | "
          f"{sum(x['B']['n_recovered'] for x in R)} | {t['0']} | {t['1']} | {t['2']} | "
          f"{dict(rung)} |")
    w("")

    # ---------------------------------------------------------------- F10
    f10 = [r for r in rows if r["family"] == "F10"]
    if f10:
        w("## 5. F10 — reactive walls")
        w("")
        w("No replanning for either arm. The wall is visible only through LiDAR.")
        w("")
        w("| id | model | trigger | rect | A succ | B succ | A coll | B coll |")
        w("|---|---|---|---|---|---|---|---|")
        for r in sorted(f10, key=lambda x: (x["id"], x["motion_model"])):
            w(f"| {r['id']} | {r['motion_model'][:5]} | {r['meta']['trigger']} | "
              f"{r['meta']['rect']} | {r['A']['success']:.2f} | {r['B']['success']:.2f} | "
              f"{r['A']['collision']:.2f} | {r['B']['collision']:.2f} |")
        w("")

    # ---------------------------------------------------------------- F1'
    f1 = [r for r in rows if r["family"] == "F1'"]
    if f1:
        w("## 6. F1′ — staircase fidelity")
        w("")
        w("Interface/generalization test, **not** literal rotated-rectangle support. Every static "
          "obstacle remains an axis-aligned 4-tuple.")
        w("")
        w("| id | degrees | area ratio | Hausdorff | cells | rects |")
        w("|---|---|---|---|---|---|")
        seen = set()
        for r in sorted(f1, key=lambda x: x["id"]):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            m = r["meta"]
            w(f"| {r['id']} | {m['degrees']:.0f} | {m['area_ratio_overall']:.4f} | "
              f"{m['hausdorff_max']:.4f} | {m['cells_total']} | {m['n_rects_total']} |")
        w("")
        w(f"Bound: `cell x sqrt(2)` = {0.1*np.sqrt(2):.4f} m. "
          f"Max observed Hausdorff {max(r['meta']['hausdorff_max'] for r in f1):.4f} m.")
        w("")

    # ---------------------------------------------------------------- stress
    if stress:
        w("## 7. X1–X7 — feasibility characterisation (NOT part of any benchmark aggregate)")
        w("")
        w("**No success rate is reported.** The endpoints are what happens when safe action is "
          "scarce: infeasibility, recovery, achieved margin, tier, and collision.")
        w("")
        w("| id | model | A coll | B coll | A timeout | B timeout | A inf | B inf | B recov | B tiers | A worst clr | B worst clr |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(stress, key=lambda x: (x["id"], x["motion_model"])):
            A, B = r["A"], r["B"]
            w(f"| {r['id']} | {r['motion_model'][:5]} | {A['collision']:.2f} | "
              f"{B['collision']:.2f} | {A['timeout']:.2f} | {B['timeout']:.2f} | "
              f"{A['n_infeasible']} | {B['n_infeasible']} | {B['n_recovered']} | "
              f"{B['tiers']} | {A['min_clearance_worst']:+.4f} | {B['min_clearance_worst']:+.4f} |")
        w("")

    # ---------------------------------------------------------------- stats
    w("## 8. What is statistically meaningful, and what is not")
    w("")
    sig = [r for r in bench if r["p_collision"] < ALPHA]
    sigs = [r for r in bench if r["p_success"] < ALPHA]
    w(f"At n = 50 and alpha = {ALPHA}, **{len(sig)} of {len(bench)}** benchmark cells reach "
      f"significance on collision and **{len(sigs)}** on success under exact McNemar.")
    w("")
    if sig or sigs:
        w("| id | model | endpoint | A | B | n10 | n01 | p |")
        w("|---|---|---|---|---|---|---|---|")
        for r in sorted(set([x["id"] + "|" + x["motion_model"] for x in sig + sigs])):
            i, mm = r.split("|")
            x = next(y for y in bench if y["id"] == i and y["motion_model"] == mm)
            if x["p_collision"] < ALPHA:
                w(f"| {i} | {mm[:5]} | collision | {x['A']['collision']:.2f} | "
                  f"{x['B']['collision']:.2f} | {x['n10_collision']} | {x['n01_collision']} | "
                  f"{x['p_collision']:.4f} |")
            if x["p_success"] < ALPHA:
                w(f"| {i} | {mm[:5]} | success | {x['A']['success']:.2f} | "
                  f"{x['B']['success']:.2f} | {x['n10_success']} | {x['n01_success']} | "
                  f"{x['p_success']:.4f} |")
        w("")
    w("**Everything else in this report is descriptive.** Family-level rows average over cells "
      "and carry no test; they are for characterising the suite, not for concluding anything "
      "about the controllers. The final tier at 200 episodes is what the design reserves for "
      "inference.")
    w("")

    # ---------------------------------------------------------------- runtime
    w("## 9. Runtime")
    w("")
    rate = float(np.mean([r["sec_per_episode"] for r in rows]))
    tot = sum(r["seconds"] for r in rows)
    b = suite.budget(man)
    w(f"Measured **{rate:.2f} s/episode** over {len(rows)} completed cells "
      f"({tot/3600:.1f} h of cumulative compute).")
    w("")
    w("| tier | episodes | at the measured rate | at 8-way |")
    w("|---|---|---|---|")
    w(f"| screening (this run) | {b['screening_episodes']} | "
      f"{b['screening_episodes']*rate/3600:.1f} h | {b['screening_episodes']*rate/3600/8:.1f} h |")
    w(f"| final (not started) | {b['final_episodes']} | "
      f"{b['final_episodes']*rate/3600:.1f} h | {b['final_episodes']*rate/3600/8:.1f} h |")
    w("")
    slow = sorted(rows, key=lambda r: -r["sec_per_episode"])[:5]
    w("Slowest cells: " + ", ".join(f"`{r['id']}` {r['sec_per_episode']:.1f} s/ep" for r in slow))
    w("")

    (OUT / "screening_report.md").write_text("\n".join(L))
    (OUT / "screening_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"cells {len(rows)}/{expected}  rate {rate:.2f} s/ep  "
          f"significant(coll) {len(sig)}  significant(succ) {len(sigs)}")
    print("wrote", OUT / "screening_report.md")


if __name__ == "__main__":
    main()
