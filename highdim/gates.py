"""Pre-registered decision rules of HD Experiment 1 (HD_DESIGN.md sections 7-9), as pure functions.

Each takes per-episode records (dicts with at least `seed` and `success`) and returns a verdict dict.
Rates are compared in exact rational arithmetic so a rule never flips on floating-point rounding
at its threshold.
"""
from __future__ import annotations

from fractions import Fraction

STOP_THRESHOLD = Fraction(5, 100)       # section 7: S_ceiling - S_floor < 0.05 -> STOP


def _paired_successes(floor, ceiling):
    if [r["seed"] for r in floor] != [r["seed"] for r in ceiling]:
        raise ValueError("floor and ceiling records are not paired by seed")
    return sum(int(r["success"]) for r in floor), sum(int(r["success"]) for r in ceiling)


def stop_check(floor, ceiling):
    """Section 7 on HD_FINAL, pooled over conditions: STOP when S_ceiling - S_floor < 0.05."""
    n = len(floor)
    if n == 0:
        raise ValueError("no episodes")
    k_floor, k_ceiling = _paired_successes(floor, ceiling)
    gap = Fraction(k_ceiling - k_floor, n)
    return {"S_floor": k_floor / n, "S_ceiling": k_ceiling / n, "gap": float(gap), "n": n,
            "threshold": float(STOP_THRESHOLD), "stop": gap < STOP_THRESHOLD}
