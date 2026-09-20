"""The pre-registered decision rules of phase 5 (RS_DESIGN.md sections 7.2, 7.3 and 8).

Pure functions over counts: nothing here runs, loads or writes an episode. A cell's counts are
`{"success": k, "collision": c, "episodes": n}` pooled over both motion conditions, and an arm is
the 20 generated cells (the M0 anchor is outside the pooled rules, section 8). Pooled rates weight
all 20 cells equally, so they are the mean of the cells' rates, kept in exact rational arithmetic
(`fractions.Fraction`) as `highdim.gates` does: a rule never flips on floating-point rounding at
its threshold. Step times are floats by nature and are compared as floats.

Two rules live here:

    choose_config   section 7.3, one candidate's parameter search on the first 20 RS_DIAG seeds:
                    the highest pooled success whose pooled collision rate is at most the
                    baseline's + 0.01 on the same episodes, else the default.
    select          section 8 on RS_TUNE: qualify at >= +2 pp success and <= +1 pp collision
                    against the baseline; the winner is the highest-success qualifier, ties to
                    fewer changed components then to the earlier-listed candidate; a candidate
                    whose p95 step time exceeds 100 ms is disqualified before the rule applies.

Readings of wording section 7.3 leaves open, recorded with the phase 5 run: the search compares
configurations to each other by pooled success alone (the collision limit is a filter, not a
score), and a tie goes to the earlier configuration of the pre-registered grid, whose first
member is the default. So a tie with the default keeps the default.
"""
from __future__ import annotations

from fractions import Fraction

from robustsuite import seeds as RS

#: Section 8: the qualifying margins, pooled over the 20 cells.
SUCCESS_MARGIN = Fraction(2, 100)
COLLISION_MARGIN = Fraction(1, 100)
#: Section 7.2: a candidate whose p95 control-step time exceeds this is disqualified, ms.
P95_LIMIT_MS = 100.0
#: Section 7.3: the collision limit of the parameter search, against the baseline's rate.
SEARCH_COLLISION_MARGIN = Fraction(1, 100)
#: Section 7.3: configurations a candidate's search may evaluate, the default included.
SEARCH_BUDGET = 8

#: The cells the pooled rates weight equally (the anchor is not one of them).
POOLED_CELLS = tuple(f"{c.family}_{c.obstacles}" for c in RS.CELLS)


def pooled(counts, *, cells=POOLED_CELLS):
    """The equal-weight pooled rates of one arm over the 20 cells, as exact Fractions.

    `cells` narrows the grid, which only a smoke run of the entry point does: the rules of
    sections 7.3 and 8 are defined on all 20 cells, so the default refuses anything else."""
    if set(counts) != set(cells):
        missing = sorted(set(cells) - set(counts))
        extra = sorted(set(counts) - set(cells))
        raise ValueError(f"the pooled rates need the cells {sorted(cells)}: "
                         f"missing {missing}, extra {extra}")
    for name, c in counts.items():
        if c["episodes"] <= 0:
            raise ValueError(f"cell {name} has no episode")
    n_cells = len(cells)
    def rate(key):
        return sum(Fraction(counts[c][key], counts[c]["episodes"]) for c in cells) / n_cells
    return {"success": rate("success"), "collision": rate("collision"), "cells": n_cells,
            "episodes": sum(c["episodes"] for c in counts.values()),
            "per_cell": {c: {"success": Fraction(counts[c]["success"], counts[c]["episodes"]),
                             "collision": Fraction(counts[c]["collision"],
                                                   counts[c]["episodes"]),
                             "episodes": counts[c]["episodes"]} for c in cells}}


def qualify(base, cand):
    """Section 8's two conditions for one candidate's pooled rates against the baseline's."""
    d_s = cand["success"] - base["success"]
    d_c = cand["collision"] - base["collision"]
    pass_s, pass_c = d_s >= SUCCESS_MARGIN, d_c <= COLLISION_MARGIN
    return {"d_success": d_s, "d_collision": d_c, "pass_success": pass_s,
            "pass_collision": pass_c, "qualifies": pass_s and pass_c}


def real_time(p95_ms):
    """Section 7.2: a p95 control-step time at most the limit."""
    return float(p95_ms) <= P95_LIMIT_MS


def select(baseline_counts, candidates, *, cells=POOLED_CELLS):
    """Section 8 on RS_TUNE.

    `candidates` are dicts {"name", "counts", "p95_ms", "components"} in the pre-registered list
    order (section 7.4), which is the last tie-break. Returns the baseline's and every
    candidate's pooled rates, each candidate's disqualification or qualification, and the winner
    (None: "no improvement found").
    """
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"the candidates are not distinct: {names}")
    base = pooled(baseline_counts, cells=cells)
    rows = []
    for order, c in enumerate(candidates):
        p = pooled(c["counts"], cells=cells)
        slow = not real_time(c["p95_ms"])
        row = {"name": c["name"], "order": order, "components": c["components"],
               "p95_ms": float(c["p95_ms"]), "pooled": p,
               "disqualified": (f"p95 {float(c['p95_ms']):.1f} ms exceeds "
                                f"{P95_LIMIT_MS:.0f} ms" if slow else None),
               "qualify": None if slow else qualify(base, p)}
        row["qualifies"] = bool(row["qualify"] and row["qualify"]["qualifies"])
        rows.append(row)
    winners = [r for r in rows if r["qualifies"]]
    winner = min(winners, key=lambda r: (-r["pooled"]["success"], r["components"], r["order"]),
                 default=None)
    return {"baseline": base, "candidates": rows,
            "winner": None if winner is None else winner["name"],
            "verdict": "no improvement found" if winner is None
                       else f"winner: {winner['name']}"}


def choose_config(baseline_counts, configurations, *, default="default", cells=POOLED_CELLS):
    """Section 7.3's parameter search for one candidate, on the first 20 RS_DIAG seeds.

    `configurations` are dicts {"name", "counts"}, the candidate's `default` configuration first,
    at most SEARCH_BUDGET of them. `baseline_counts` are the baseline's on the same episodes.
    """
    if not configurations:
        raise ValueError("the search needs at least the default configuration")
    if configurations[0]["name"] != default:
        raise ValueError(f"the search grid must start with the default {default!r}, "
                         f"got {configurations[0]['name']!r}")
    if len(configurations) > SEARCH_BUDGET:
        raise ValueError(f"the search budget is {SEARCH_BUDGET} configurations, "
                         f"got {len(configurations)}")
    base = pooled(baseline_counts, cells=cells)
    limit = base["collision"] + SEARCH_COLLISION_MARGIN
    rows = []
    for order, c in enumerate(configurations):
        p = pooled(c["counts"], cells=cells)
        rows.append({"name": c["name"], "order": order, "pooled": p,
                     "d_success": p["success"] - base["success"],
                     "d_collision": p["collision"] - base["collision"],
                     "within_collision_limit": p["collision"] <= limit})
    ok = [r for r in rows if r["within_collision_limit"]]
    best = min(ok, key=lambda r: (-r["pooled"]["success"], r["order"]), default=None)
    return {"baseline": base, "collision_limit": limit, "configurations": rows,
            "default": default, "chosen": default if best is None else best["name"],
            "fell_back": best is None}
