"""Seam 2: the scenario generator. (cell, motion condition, seed) -> a validated EpisodeSpec.

One entry point, `generate`. It never runs an episode. Every rule it enforces is fixed in
MD_files/robustsuite/RS_DESIGN.md section 4; the section numbers below point there.

DETERMINISM AND PAIRING. The only randomness is a SeedSequence with entropy
`robustsuite.seeds.generator_entropy(cell, seed)` = (seed, family index). The motion condition is
validated and then ignored: both motions get the identical spec (4.3, 5). The spec carries no
motion field for that reason; the environment is built for its motion condition instead.

FEASIBILITY (4.7). Resampling is at the level of the family draw. Each layout draw, and each
distance-bin redraw on a layout, counts one draw towards RESAMPLE_CAP; hitting the cap raises
FeasibilityError, so an infeasible spec is never returned. Every routing and clearance check
uses the frozen ShortestPathOracle (0.05 m grid) at the clearance radius R_C (4.1): free-space
connectivity and the start-goal path. Start and goal must also have Euclidean clearance >= R_C;
the oracle's endpoint test (its box-inflated grid cell must be free) is then applied on top, so
the combined check is at least as strict as either.

READING OF THE CAP (4.5, 4.7), recorded in ticket 02: each layout draw and each failed bin
attempt is one draw of the 200; after BIN_REDRAWS_PER_LAYOUT failed bins the layout is redrawn.

LATER TICKETS. Tickets 05-07 add the trigger, events and pedestrian routes to the family draw
(4.7: layout, start/goal, events with up to 50 attempts, then pedestrians; accepted only if every
condition's rules hold). That changes the accepted draws of the static cells too, so it must bump
GENERATOR_VERSION; no result made under an earlier version is paired with a later one.

SCOPE (ticket 02). Only the open_clutter family and the static condition are built. The other
families and conditions raise NotImplementedError until their tickets (03-07) add them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from evaluation.shortest_path import ShortestPathOracle
from robustsuite import seeds as RS

#: Bumped whenever a change alters any generated spec. Recorded in every result file.
GENERATOR_VERSION = "rs-gen-0.1"

WORLD = 10.0
#: Clearance radius: robot radius 0.3 + margin 0.15 (4.1).
R_C = 0.45
GRID = 200
#: Layout draws plus bin redraws, per family draw (4.7).
RESAMPLE_CAP = 200
#: M0's Euclidean start-goal distance bins (4.5).
DISTANCE_BINS = ((4.0, 5.5), (5.5, 7.0), (7.0, 8.5), (8.5, 10.5))

# Sampler budgets. Implementation details of "uniform within the stated ranges", not rules:
# a draw that exhausts one is a failed draw and counts towards RESAMPLE_CAP like any other.
#: Placement tries per rectangle before the layout draw fails.
RECT_TRIES = 200
#: Uniform candidate points per bin attempt; consecutive free points are paired (start, goal).
POINT_BATCH = 4000
#: Bin redraws on one layout before the layout itself is redrawn.
BIN_REDRAWS_PER_LAYOUT = 4

#: open_clutter (4.2): count, half-extent range, pairwise gap, distance from the outer wall.
OPEN_CLUTTER = {"count": (5, 7), "half": (0.25, 1.0), "gap": 1.2, "wall": 1.0}


class FeasibilityError(RuntimeError):
    """The resample cap was reached without a feasible family draw."""


@dataclass(frozen=True)
class EpisodeSpec:
    """One family draw. Rectangles are (centre x, centre y, half-width, half-height)."""
    generator_version: str
    family: str
    obstacles: str
    seed: int
    layout: tuple
    start: tuple
    goal: tuple
    distance_bin: int
    layout_draws: int          # layout draws made, the accepted one included
    bin_redraws: int           # failed bin attempts across all layouts
    feasible: bool = True      # never False: an infeasible draw raises instead

    def as_dict(self):
        d = asdict(self)
        d["layout"] = [list(r) for r in self.layout]
        d["start"], d["goal"] = list(self.start), list(self.goal)
        return d


def rect_gap(a, b):
    """Euclidean distance between two axis-aligned rectangles (0 if they touch or overlap)."""
    dx = max(abs(a[0] - b[0]) - (a[2] + b[2]), 0.0)
    dy = max(abs(a[1] - b[1]) - (a[3] + b[3]), 0.0)
    return float(np.hypot(dx, dy))


def clearance(p, layout):
    """Distance from point p to the nearest rectangle or outer wall."""
    x, y = float(p[0]), float(p[1])
    vals = [x, WORLD - x, y, WORLD - y]
    for cx, cy, hw, hh in layout:
        vals.append(rect_gap((x, y, 0.0, 0.0), (cx, cy, hw, hh)))
    return min(vals)


def oracle(layout, inflation=R_C):
    """The frozen shortest-path oracle on `layout`, obstacles inflated by `inflation`."""
    return ShortestPathOracle(WORLD, inflation, layout, resolution=GRID)


def free_space_connected(orc):
    """True if the oracle's free cells form one 8-connected component (the oracle's moves)."""
    _, n = ndimage.label(~orc.occupancy, structure=np.ones((3, 3), dtype=int))
    return n == 1


# --------------------------------------------------------------------------- families
def _draw_open_clutter(rng):
    """5-7 rectangles, half-extents in [0.25, 1.0], pairwise gap >= 1.2, >= 1.0 from the wall.
    None if a rectangle cannot be placed within RECT_TRIES."""
    p = OPEN_CLUTTER
    lo_n, hi_n = p["count"]
    n = int(rng.integers(lo_n, hi_n + 1))
    rects = []
    for _ in range(n):
        for _ in range(RECT_TRIES):
            hw, hh = (float(v) for v in rng.uniform(*p["half"], size=2))
            cx = float(rng.uniform(p["wall"] + hw, WORLD - p["wall"] - hw))
            cy = float(rng.uniform(p["wall"] + hh, WORLD - p["wall"] - hh))
            r = (cx, cy, hw, hh)
            if all(rect_gap(r, q) >= p["gap"] for q in rects):
                rects.append(r)
                break
        else:
            return None
    return tuple(rects)


FAMILY_DRAWS = {"open_clutter": _draw_open_clutter}
#: Conditions built so far. The draw is shared by all four; the others arrive with 05-07.
BUILT_CONDITIONS = ("static",)


# --------------------------------------------------------------------------- start / goal
def _draw_start_goal(rng, layout, orc, lo, hi):
    """Start and goal uniform in free space (clearance >= R_C) with |goal - start| in [lo, hi],
    joined by a clear path. None if this batch holds no such pair."""
    pts = rng.uniform(R_C, WORLD - R_C, size=(POINT_BATCH, 2))
    free = [q for q in pts if clearance(q, layout) >= R_C]
    for s, g in zip(free[0::2], free[1::2]):
        # the (0.2 s) path query runs only for in-bin pairs, and fails only when an endpoint's
        # grid cell is blocked, since free space is connected
        if lo <= float(np.linalg.norm(g - s)) <= hi and orc.path_length(s, g) is not None:
            return tuple(float(v) for v in s), tuple(float(v) for v in g)
    return None


# --------------------------------------------------------------------------- entry point
def generate(cell, motion, seed):
    """The validated EpisodeSpec for (cell, motion, seed). Raises FeasibilityError at the cap."""
    RS.check_motion(motion)
    entropy = RS.generator_entropy(cell, seed)        # ValueError for the M0 anchor
    if cell.family not in FAMILY_DRAWS or cell.obstacles not in BUILT_CONDITIONS:
        raise NotImplementedError(f"{cell} is not built yet")
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    draw = FAMILY_DRAWS[cell.family]

    draws = layout_draws = bin_redraws = 0
    while draws < RESAMPLE_CAP:
        draws += 1
        layout_draws += 1
        layout = draw(rng)
        if layout is None:
            continue
        orc = oracle(layout)
        if not free_space_connected(orc):
            continue
        for _ in range(BIN_REDRAWS_PER_LAYOUT):
            b = int(rng.integers(len(DISTANCE_BINS)))
            sg = _draw_start_goal(rng, layout, orc, *DISTANCE_BINS[b])
            if sg is not None:
                return EpisodeSpec(GENERATOR_VERSION, cell.family, cell.obstacles, int(seed),
                                   layout, sg[0], sg[1], b, layout_draws, bin_redraws)
            bin_redraws += 1
            draws += 1
            if draws >= RESAMPLE_CAP:
                break
    raise FeasibilityError(f"{cell} seed {seed}: no feasible draw in {RESAMPLE_CAP} draws")
