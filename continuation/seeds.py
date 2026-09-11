"""Continuation seed blocks (audit section 5). Every block is disjoint from every earlier one.

Earlier reservations, NEVER used for a continuation decision:
    20_000 / 21_000          Phase 1-7 dev / validation   (read-only equivalence gate only)
    1_000_000 .. +199        Phase 6/7 final block         (never touched)
    7_000_000 / 8_000_000 / 9_000_000   Week 5.5 generalization  (never touched)

The FINAL block is sealed: `seed_block("FINAL")` raises unless `allow_final=True`, and
tests/test_week6_continuation.py asserts that only experiments/week6_continuation/cont_final.py
passes that flag.
"""
from __future__ import annotations

import numpy as np

BLOCKS = {
    "C0": (10_000_000, 100),
    "C1": (10_100_000, 100),
    "H1": (10_200_000, 20),
    "H2": (10_300_000, 50),
    "RANDOM_EXP_R1": (10_400_000, 100),
    "RANDOM_EXP_R2": (10_500_000, 100),
    "RANDOM_EXP_R3_SCREEN": (10_600_000, 20),
    "RANDOM_EXP_R3_VALID": (10_700_000, 100),
    "RESERVED_ABLATION": (10_800_000, 0),
    "FINAL": (10_900_000, 200),
}
BLOCK_STRIDE = 100_000

#: Tag mixed into the controller-side RNG so it can never coincide with env.np_random's stream.
CONTROLLER_RNG_TAG = 0x52434246          # "RCBF"

FORBIDDEN_RANGES = [(20_000, 22_000), (1_000_000, 1_000_200),
                    (7_000_000, 7_100_000), (8_000_000, 8_100_000), (9_000_000, 9_100_000)]


def seed_block(name, n=None, *, allow_final=False):
    if name not in BLOCKS:
        raise KeyError(name)
    if name == "FINAL" and not allow_final:
        raise PermissionError("the FINAL seed block is sealed; only cont_final.py may open it")
    base, size = BLOCKS[name]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"block {name} holds {size} episodes, requested {n}")
    return [base + i for i in range(n)]


def controller_rng(episode_seed):
    """Per-episode controller RNG, reproducible from the episode seed, independent of the env."""
    return np.random.default_rng([CONTROLLER_RNG_TAG, int(episode_seed)])


def _check_disjoint():
    spans = sorted((b, b + BLOCK_STRIDE) for b, _ in BLOCKS.values())
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        assert a1 <= b0, "continuation blocks overlap"
    for b, size in BLOCKS.values():
        assert size <= BLOCK_STRIDE
        for lo, hi in FORBIDDEN_RANGES:
            assert b + BLOCK_STRIDE <= lo or b >= hi, f"block {b} hits a reserved range"


_check_disjoint()
