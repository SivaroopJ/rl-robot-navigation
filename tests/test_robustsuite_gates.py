"""RS Experiment 1, phase 6: the GO gate (MD_files/robustsuite/RS_DESIGN.md section 9).

The seam is `robustsuite.gates`: paired per-episode records in, a verdict out. Nothing here runs
an episode. Records are constructed so the exact two-sided McNemar p sits on a known side of 0.05:
with d discordant pairs all one way, p = 2 / 2**d, so d = 5 gives 0.0625 (not significant) and
d = 6 gives 0.03125 (significant).
"""
from __future__ import annotations

import pytest

from robustsuite import gates as GT
from robustsuite import seeds as RS

CELLS = [f"{c.family}_{c.obstacles}" for c in RS.CELLS]
N = 20                                     # episodes per cell, 10 per motion condition


def recs(successes, collisions=(), n=N, cell=None):
    """One cell's records: `successes` and `collisions` are the indices that are 1."""
    out = []
    for i in range(n):
        motion = RS.MOTIONS[i % 2]
        out.append({"cell": cell, "motion": motion, "seed": 15_200_000 + i // 2,
                    "success": int(i in successes), "collision": int(i in collisions),
                    "timeout": int(i not in successes and i not in collisions)})
    return out


def arm(per_cell_successes, per_cell_collisions=None):
    """{cell: [records]} for the 20 cells, from a per-cell set of successful indices."""
    coll = per_cell_collisions or [()] * len(CELLS)
    return {c: recs(s, k, cell=c) for c, s, k in zip(CELLS, per_cell_successes, coll,
                                                     strict=True)}


def spread(n_success, n_collision=0):
    """The same outcome pattern in every cell."""
    return ([set(range(n_success))] * len(CELLS),
            [set(range(N - n_collision, N))] * len(CELLS))


def anchor(k_base, k_win, n=40):
    """A paired anchor: the winner's successes are a superset (k_win >= k_base) or a subset."""
    base = [{"cell": "m0_anchor", "motion": RS.MOTIONS[i % 2], "seed": i // 2,
             "success": int(i < k_base), "collision": 0, "timeout": int(i >= k_base)}
            for i in range(n)]
    win = [{**r, "success": int(i < k_win), "timeout": int(i >= k_win)}
           for i, r in enumerate(base)]
    return base, win


# --------------------------------------------------------------------------- pairing
def test_the_gate_refuses_records_that_are_not_paired():
    s, c = spread(10)
    base, win = arm(s, c), arm(s, c)
    win[CELLS[0]] = win[CELLS[0]][1:]                    # one episode short
    with pytest.raises(ValueError):
        GT.go_gate(base, win)


def test_the_gate_refuses_a_grid_that_is_not_the_twenty_cells():
    s, c = spread(10)
    base, win = arm(s, c), arm(s, c)
    base.pop(CELLS[0]), win.pop(CELLS[0])
    with pytest.raises(ValueError):
        GT.go_gate(base, win)


# --------------------------------------------------------------------------- condition 1
def test_a_significant_success_gain_with_no_safety_cost_is_GO():
    s, c = spread(10)
    win_s = [set(range(10)) | set(range(10, 16))] * len(CELLS)       # +6 discordant, one way
    out = GT.go_gate(arm(s, c), arm(win_s, c))
    assert out["success"]["higher"] and out["success"]["p"] < 0.05
    assert out["success"]["pass"] and not out["safety"]["fail"]
    assert out["verdict"] == "GO"


def test_a_success_gain_that_misses_significance_is_NO_GO():
    s, c = spread(10)
    win_s = [set(range(15))] + [set(range(10))] * (len(CELLS) - 1)   # 5 discordant in all:
    out = GT.go_gate(arm(s, c), arm(win_s, c))                       # p = 0.0625
    assert out["success"]["higher"] and out["success"]["p"] > 0.05
    assert not out["success"]["pass"] and out["verdict"] == "NO-GO"


def test_no_success_difference_is_NO_GO():
    s, c = spread(10)
    out = GT.go_gate(arm(s, c), arm(s, c))
    assert not out["success"]["higher"] and out["verdict"] == "NO-GO"


def test_a_significant_success_loss_is_NO_GO():
    s, c = spread(16)
    win_s = [set(range(10))] * len(CELLS)
    out = GT.go_gate(arm(s, c), arm(win_s, c))
    assert not out["success"]["higher"] and out["verdict"] == "NO-GO"


# --------------------------------------------------------------------------- condition 2
def test_a_significantly_higher_collision_rate_fails_the_safety_condition():
    base_s = [set(range(10))] * len(CELLS)
    base_c = [set()] * len(CELLS)
    win_s = [set(range(15))] * len(CELLS)                 # success up, significantly
    win_c = [set(range(15, 20))] * 2 + [set()] * (len(CELLS) - 2)    # 10 discordant collisions
    out = GT.go_gate(arm(base_s, base_c), arm(win_s, win_c))
    assert out["success"]["pass"]
    assert out["safety"]["higher"] and out["safety"]["p"] < 0.05 and out["safety"]["fail"]
    assert out["verdict"] == "NO-GO"


def test_a_collision_rise_that_misses_significance_does_not_fail_safety():
    base_s = [set(range(10))] * len(CELLS)
    base_c = [set()] * len(CELLS)
    win_s = [set(range(15))] * len(CELLS)
    win_c = [set(range(15, 20))] + [set()] * (len(CELLS) - 1)        # 5 discordant: p = 0.0625
    out = GT.go_gate(arm(base_s, base_c), arm(win_s, win_c))
    assert out["safety"]["higher"] and out["safety"]["p"] > 0.05
    assert not out["safety"]["fail"] and out["verdict"] == "GO"


def test_a_lower_collision_rate_never_fails_safety():
    base_s = [set(range(10))] * len(CELLS)
    base_c = [set(range(14, 20))] * len(CELLS)
    win_s = [set(range(14))] * len(CELLS)
    out = GT.go_gate(arm(base_s, base_c), arm(win_s, [set()] * len(CELLS)))
    assert not out["safety"]["higher"] and not out["safety"]["fail"]


# --------------------------------------------------------------------------- condition 3
def test_a_significantly_worse_anchor_fails_the_gate():
    s, c = spread(10)
    win_s = [set(range(16))] * len(CELLS)
    base_a, win_a = anchor(k_base=30, k_win=24)           # 6 discordant the wrong way
    out = GT.go_gate(arm(s, c), arm(win_s, c), anchor=(base_a, win_a))
    assert out["anchor"]["lower"] and out["anchor"]["p"] < 0.05 and out["anchor"]["fail"]
    assert out["verdict"] == "NO-GO"


def test_an_anchor_loss_that_misses_significance_does_not_fail_the_gate():
    s, c = spread(10)
    win_s = [set(range(16))] * len(CELLS)
    base_a, win_a = anchor(k_base=30, k_win=25)           # 5 discordant: p = 0.0625
    out = GT.go_gate(arm(s, c), arm(win_s, c), anchor=(base_a, win_a))
    assert out["anchor"]["lower"] and not out["anchor"]["fail"]
    assert out["verdict"] == "GO"


def test_a_better_anchor_never_fails_the_gate():
    s, c = spread(10)
    win_s = [set(range(16))] * len(CELLS)
    base_a, win_a = anchor(k_base=24, k_win=30)
    out = GT.go_gate(arm(s, c), arm(win_s, c), anchor=(base_a, win_a))
    assert not out["anchor"]["lower"] and not out["anchor"]["fail"]
    assert out["verdict"] == "GO"


def test_the_anchor_is_optional_and_then_cannot_fail():
    s, c = spread(10)
    win_s = [set(range(16))] * len(CELLS)
    out = GT.go_gate(arm(s, c), arm(win_s, c), anchor=None)
    assert out["anchor"] is None and out["verdict"] == "GO"


# --------------------------------------------------------------------------- pooling, per cell
def test_pooled_rates_weight_every_cell_equally():
    per_cell = [set(range(20))] * 10 + [set()] * 10       # half the cells perfect
    out = GT.go_gate(arm(per_cell), arm(per_cell))
    assert float(out["success"]["base"]) == 0.5 and float(out["success"]["winner"]) == 0.5


def test_every_cell_is_reported_with_its_own_paired_test_but_does_not_gate():
    s, c = spread(10)
    win_s = [set(range(16))] * 19 + [set(range(4))]       # one cell much worse
    out = GT.go_gate(arm(s, c), arm(win_s, c))
    assert set(out["per_cell"]) == set(CELLS)
    worst = out["per_cell"][CELLS[-1]]
    assert worst["d_success"] < 0 and worst["p"] < 0.05
    assert out["verdict"] == "GO"                          # per-cell results are not gated


def test_the_verdict_needs_condition_one_even_when_nothing_fails():
    s, c = spread(10)
    out = GT.go_gate(arm(s, c), arm(s, c))
    assert not out["safety"]["fail"] and out["verdict"] == "NO-GO"
