"""RS Experiment 1, phase 5: the selection rule and the 7.3 configuration search.

Design: MD_files/robustsuite/RS_DESIGN.md sections 7.3, 7.2 (the 100 ms rule) and 8. The seam is
`robustsuite.selection`: counts per cell in, a verdict out. Nothing here runs an episode.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from robustsuite import selection as SEL
from robustsuite import seeds as RS

CELLS = [f"{c.family}_{c.obstacles}" for c in RS.CELLS]


def counts(rates, n=100, collisions=0.0):
    """One cell-count table: `rates` is a success rate per cell (a scalar applies to all 20)."""
    if not isinstance(rates, (list, tuple)):
        rates = [rates] * len(CELLS)
    coll = collisions if isinstance(collisions, (list, tuple)) else [collisions] * len(CELLS)
    return {name: {"success": round(s * n), "collision": round(c * n), "episodes": n}
            for name, s, c in zip(CELLS, rates, coll, strict=True)}


def arm(name, rates, *, n=100, collisions=0.0, p95_ms=50.0, components=1):
    return {"name": name, "counts": counts(rates, n, collisions), "p95_ms": p95_ms,
            "components": components}


# --------------------------------------------------------------------------- pooling
def test_pooled_weights_every_cell_equally_and_is_exact():
    rates = [0.9] * 10 + [0.5] * 10
    p = SEL.pooled(counts(rates, n=50))
    assert p["success"] == Fraction(7, 10)
    assert isinstance(p["success"], Fraction)
    assert p["episodes"] == 20 * 50
    assert p["cells"] == 20


def test_pooled_is_not_the_episode_mean_when_cells_differ_in_size():
    c = counts(0.0, n=100)
    c[CELLS[0]] = {"success": 10, "collision": 0, "episodes": 10}     # a small, perfect cell
    assert SEL.pooled(c)["success"] == Fraction(1, 20)                # not 10 / 1910


def test_pooled_refuses_a_grid_that_is_not_the_twenty_cells():
    c = counts(0.5)
    c.pop(CELLS[0])
    with pytest.raises(ValueError):
        SEL.pooled(c)
    with pytest.raises(ValueError):
        SEL.pooled({**c, "rooms_elsewhere": {"success": 1, "collision": 0, "episodes": 2}})


def test_pooled_refuses_an_empty_cell():
    c = counts(0.5)
    c[CELLS[0]] = {"success": 0, "collision": 0, "episodes": 0}
    with pytest.raises(ValueError):
        SEL.pooled(c)


# --------------------------------------------------------------------------- qualifying
def test_qualifies_exactly_at_both_thresholds():
    base = SEL.pooled(counts(0.50, collisions=0.10))
    cand = SEL.pooled(counts(0.52, collisions=0.11))           # +2 pp success, +1 pp collision
    q = SEL.qualify(base, cand)
    assert q["d_success"] == Fraction(2, 100) and q["d_collision"] == Fraction(1, 100)
    assert q["pass_success"] and q["pass_collision"] and q["qualifies"]


def test_one_episode_below_the_success_threshold_does_not_qualify():
    base = SEL.pooled(counts(0.50, n=50))
    rates = [0.52] * 19 + [0.50]                               # one cell short of +2 pp pooled
    q = SEL.qualify(base, SEL.pooled(counts(rates, n=50)))
    assert q["d_success"] < Fraction(2, 100)
    assert not q["pass_success"] and not q["qualifies"]


def test_one_episode_above_the_collision_threshold_does_not_qualify():
    base = SEL.pooled(counts(0.50, n=50, collisions=0.10))
    coll = [0.11] * 19 + [0.12]                                # just over +1 pp pooled
    q = SEL.qualify(base, SEL.pooled(counts(0.60, n=50, collisions=coll)))
    assert q["d_collision"] > Fraction(1, 100)
    assert q["pass_success"] and not q["pass_collision"] and not q["qualifies"]


def test_a_collision_rate_below_the_baseline_passes_the_safety_side():
    base = SEL.pooled(counts(0.50, collisions=0.10))
    q = SEL.qualify(base, SEL.pooled(counts(0.60, collisions=0.02)))
    assert q["pass_collision"] and q["qualifies"]


# --------------------------------------------------------------------------- the 100 ms rule
def test_p95_at_the_limit_passes_and_above_it_disqualifies():
    assert SEL.passes_real_time(100.0) and not SEL.passes_real_time(100.0001)


def test_a_slow_candidate_is_disqualified_before_the_rule_applies():
    base = counts(0.50)
    out = SEL.select(base, [arm("slow", 0.90, p95_ms=101.0)])
    c = out["candidates"][0]
    assert c["disqualified"] == "p95 101.0 ms exceeds 100 ms"
    assert c["qualify"] is None and out["winner"] is None
    assert out["verdict"] == "no improvement found"


# --------------------------------------------------------------------------- selecting
def test_the_highest_success_qualifier_wins():
    out = SEL.select(counts(0.50), [arm("a", 0.55), arm("b", 0.60), arm("c", 0.51)])
    assert out["winner"] == "b"
    assert [c["qualifies"] for c in out["candidates"]] == [True, True, False]
    assert out["verdict"] == "winner: b"


def test_a_tie_on_success_goes_to_the_candidate_with_fewer_components():
    out = SEL.select(counts(0.50), [arm("a", 0.60, components=2), arm("b", 0.60, components=1)])
    assert out["winner"] == "b"


def test_a_tie_on_success_and_components_goes_to_the_earlier_candidate():
    out = SEL.select(counts(0.50), [arm("a", 0.60), arm("b", 0.60)])
    assert out["winner"] == "a"


def test_no_qualifier_means_no_winner():
    out = SEL.select(counts(0.50), [arm("a", 0.51), arm("b", 0.40)])
    assert out["winner"] is None and out["verdict"] == "no improvement found"


def test_a_slow_candidate_does_not_win_over_a_qualifying_one():
    out = SEL.select(counts(0.50), [arm("fast", 0.55), arm("slow", 0.99, p95_ms=150.0)])
    assert out["winner"] == "fast"


def test_select_refuses_two_candidates_with_the_same_name():
    with pytest.raises(ValueError):
        SEL.select(counts(0.50), [arm("a", 0.55), arm("a", 0.60)])


def test_select_carries_the_baseline_and_the_deltas_through():
    out = SEL.select(counts(0.50, collisions=0.10), [arm("a", 0.60, collisions=0.08)])
    assert out["baseline"]["success"] == Fraction(1, 2)
    c = out["candidates"][0]
    assert c["pooled"]["success"] == Fraction(3, 5)
    assert c["qualify"]["d_success"] == Fraction(1, 10)
    assert c["qualify"]["d_collision"] == Fraction(-2, 100)


def test_a_narrowed_grid_pools_only_the_cells_it_is_given():
    """Only a smoke run narrows the grid; the rules of 7.3 and 8 need all 20 cells."""
    two = CELLS[:2]
    c = {n: {"success": s, "collision": 0, "episodes": 10}
         for n, s in zip(two, (10, 0), strict=True)}
    assert SEL.pooled(c, cells=two)["success"] == Fraction(1, 2)
    out = SEL.select(c, [{"name": "a", "counts": c, "p95_ms": 50.0, "components": 1}],
                     cells=two)
    assert out["baseline"]["cells"] == 2 and not out["candidates"][0]["qualifies"]
    with pytest.raises(ValueError):                     # the default is still the 20 cells
        SEL.pooled(c)


# --------------------------------------------------------------------------- the 7.3 search
def test_the_search_takes_the_highest_success_inside_the_collision_limit():
    base = counts(0.50, collisions=0.10, n=1000)
    cfgs = [{"name": "default", "counts": counts(0.55, collisions=0.10, n=1000)},
            {"name": "best", "counts": counts(0.70, collisions=0.11, n=1000)},
            {"name": "greedy", "counts": counts(0.80, collisions=0.111, n=1000)}]  # over it
    out = SEL.choose_config(base, cfgs)
    assert out["chosen"] == "best" and not out["fell_back"]
    assert [c["within_collision_limit"] for c in out["configurations"]] == [True, True, False]


def test_the_search_falls_back_to_the_default_when_none_meets_the_collision_limit():
    base = counts(0.50, collisions=0.10)
    cfgs = [{"name": "default", "counts": counts(0.55, collisions=0.20)},
            {"name": "other", "counts": counts(0.90, collisions=0.30)}]
    out = SEL.choose_config(base, cfgs)
    assert out["chosen"] == "default" and out["fell_back"]


def test_the_search_breaks_a_tie_towards_the_earlier_configuration():
    base = counts(0.50, collisions=0.10)
    cfgs = [{"name": "default", "counts": counts(0.70, collisions=0.10)},
            {"name": "other", "counts": counts(0.70, collisions=0.09)}]
    assert SEL.choose_config(base, cfgs)["chosen"] == "default"


def test_the_search_refuses_a_grid_whose_first_configuration_is_not_the_default():
    base = counts(0.50)
    with pytest.raises(ValueError):
        SEL.choose_config(base, [{"name": "other", "counts": counts(0.70)}])


def test_the_search_limits_the_grid_to_the_pre_registered_budget():
    base = counts(0.50)
    cfgs = [{"name": "default", "counts": counts(0.5)}]
    cfgs += [{"name": f"c{i}", "counts": counts(0.5)} for i in range(SEL.SEARCH_BUDGET)]
    with pytest.raises(ValueError):
        SEL.choose_config(base, cfgs)
