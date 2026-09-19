"""RS seed blocks and the cell grid (RS_DESIGN.md sections 4-5). Disjoint from every earlier
reservation.

RS_FINAL is sealed: `seed_block("RS_FINAL")` raises unless `allow_final=True`, and
tests/test_robustsuite.py asserts that only the final-evaluation entry point passes that flag.

Every cell uses the same block of seed indices. The generator's randomness is derived from
(seed, family) only: families draw independent layouts, while all four obstacle conditions and
both motion conditions of a family share ONE family draw (RS_DESIGN 4.3, 5). A condition only
switches parts of that draw on (pedestrians, the block, the spawn), and a motion condition only
sets the pedestrian noise, so triggered cells pair with the +dynamic cell seed by seed. The env's
reset seed is the episode seed, and the per-episode Random RNG is
`continuation.seeds.controller_rng(episode_seed)`, reused unchanged.
"""
from __future__ import annotations

from typing import NamedTuple

from continuation import seeds as CS
from continuation.pursuit.config import PURSUIT_BLOCKS
from highdim import seeds as HDS

BLOCKS = {
    "RS_DIAG": (15_000_000, 50),        # baseline diagnostic (phase 2)
    "RS_TUNE": (15_100_000, 50),        # candidate selection (phase 5)
    "RS_FINAL": (15_200_000, 200),      # sealed final (phase 6)
}
#: The only file allowed to open RS_FINAL.
FINAL_ENTRY_POINT = "rs6_final.py"

FAMILIES = ("open_clutter", "rooms", "aisles", "corridors", "dense_clutter")
OBSTACLE_CONDITIONS = ("static", "dynamic", "trigger_block", "trigger_spawn")
MOTIONS = ("fixed", "randomized")


class Cell(NamedTuple):
    family: str
    obstacles: str


CELLS = tuple(Cell(f, o) for f in FAMILIES for o in OBSTACLE_CONDITIONS)
#: M0 with its original map and motion: a reference cell outside the pooled gate.
ANCHOR = Cell("m0", "anchor")

#: Earlier reservations the RS blocks must avoid, beyond continuation.seeds and pursuit.
OTHER_RESERVED = [(12_000_000, 13_000_000),     # Track B dev + eval
                  (13_000_000, 14_000_000)]     # Track A dev + eval


def seed_block(name, n=None, *, allow_final=False):
    if name not in BLOCKS:
        raise KeyError(name)
    if name == "RS_FINAL" and not allow_final:
        raise PermissionError(f"RS_FINAL is sealed; only {FINAL_ENTRY_POINT} opens it")
    base, size = BLOCKS[name]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"block {name} holds {size} seeds, requested {n}")
    return [base + i for i in range(n)]


def generator_entropy(cell, seed):
    """SeedSequence entropy for the scenario generator: per (seed, family), never per obstacle or
    motion condition. The M0 anchor is not generated (it uses the canonical env's own sampler)."""
    if cell.family not in FAMILIES:
        raise ValueError(f"{cell} is not a generated cell; the M0 anchor uses the canonical env")
    return (int(seed), FAMILIES.index(cell.family))


def _reserved():
    out = [(b, b + CS.BLOCK_STRIDE) for b, _ in CS.BLOCKS.values()]
    out += list(CS.FORBIDDEN_RANGES) + OTHER_RESERVED
    out += [(b, b + s) for b, s in PURSUIT_BLOCKS.values()]
    out += [(b, b + s) for b, s in HDS.BLOCKS.values()]
    out.append((HDS.TRAIN_BASE, HDS.TRAIN_BASE + 1000 * HDS.TRAIN_RUNS))
    return out


def check_disjoint():
    """RuntimeError if an RS block overlaps another RS block or any earlier reservation."""
    spans = sorted((b, b + s) for b, s in BLOCKS.values())
    for (_, a1), (b0, _) in zip(spans, spans[1:]):
        if a1 > b0:
            raise RuntimeError("RS blocks overlap each other")
    for lo, hi in spans:
        for r0, r1 in _reserved():
            if not (hi <= r0 or lo >= r1):
                raise RuntimeError(f"RS span [{lo}, {hi}) hits reservation [{r0}, {r1})")


check_disjoint()
