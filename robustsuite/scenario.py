"""Seam 2: the scenario generator. (cell, motion condition, seed) -> a validated EpisodeSpec.

One entry point, `generate`. It never runs an episode. Every rule it enforces is fixed in
MD_files/robustsuite/RS_DESIGN.md section 4; the section numbers below point there. The families
are in robustsuite.families, the pedestrian routes in robustsuite.pedestrians, and the paths on
the frozen oracle's grid in robustsuite.geometry.

DETERMINISM AND PAIRING. The only randomness is a SeedSequence with entropy
`robustsuite.seeds.generator_entropy(cell, seed)` = (seed, family index). ONE family draw serves
every obstacle condition and both motions (4.3, 5): the spec always carries the pedestrians, and
the condition, recorded in `obstacles`, only says which parts the environment switches on. The
motion condition is validated and then ignored, so both motions get the identical spec; the
environment is built for its motion condition instead.

FEASIBILITY (4.7). Resampling is at the level of the family draw, in the order layout, start and
goal, pedestrians. Each layout draw, each failed distance-bin attempt and each failed pedestrian
draw counts one draw towards RESAMPLE_CAP; hitting the cap raises FeasibilityError, so an
infeasible spec is never returned. Free-space connectivity and every route use the frozen
ShortestPathOracle's grid at the clearance radius R_C (4.1). Waypoints and pedestrian starts must
also have Euclidean clearance >= R_C, and the route and pedestrian legs are line segments that
keep it.

START AND GOAL (4.5). Candidate pairs are uniform in free space at clearance START_GOAL_CLEARANCE
(0.6 m); a pair is accepted when its distance lies in the drawn bin, it is joined by a route, and
the family's structural rule holds (the corridors rule reads the route). So start and goal are
uniform conditional on bin and rule. 0.6 m, not R_C, is the researcher's pre-run decision of
2026-09-19 (RS_DESIGN 14.1): it is M0's wall margin, and closer than that the frozen controller's
barrier value h = range - 0.3 is negative, CVXPY raises DCPError on every step and the robot
never leaves its start.

READING OF THE CAP (4.5, 4.7), recorded in ticket 02: each layout draw and each failed bin
attempt is one draw of the 200; after BIN_REDRAWS_PER_LAYOUT failed bins the layout is redrawn.
A failed pedestrian draw also counts one and redraws the layout (ticket 05).

LATER TICKETS. Tickets 06-07 add the trigger and events to the family draw (4.7: after start and
goal, up to 50 attempts, before the pedestrians, with the block zone reserved for pedestrians).
That changes the accepted draws of every cell, so it must bump GENERATOR_VERSION; no result made
under an earlier version is paired with a later one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from evaluation.shortest_path import ShortestPathOracle
from robustsuite import seeds as RS
from robustsuite.families import FAMILIES
from robustsuite.geometry import WORLD, Grid, clearances
from robustsuite.pedestrians import draw_route

#: Bumped whenever a change alters any generated spec. Recorded in every result file.
GENERATOR_VERSION = "rs-gen-0.2"

#: Clearance radius: robot radius 0.3 + margin 0.15 (4.1).
R_C = 0.45
#: Start and goal clearance: M0's wall margin (RS_DESIGN 14.1).
START_GOAL_CLEARANCE = 0.6
GRID = 200
#: Layout draws, bin redraws and pedestrian redraws, per family draw (4.7).
RESAMPLE_CAP = 200
#: M0's Euclidean start-goal distance bins (4.5).
DISTANCE_BINS = ((4.0, 5.5), (5.5, 7.0), (7.0, 8.5), (8.5, 10.5))

# Sampler budgets. Implementation details of "uniform within the stated ranges", not rules:
# a draw that exhausts one is a failed draw and counts towards RESAMPLE_CAP like any other.
#: Uniform candidate points per bin attempt; consecutive free points are paired (start, goal).
POINT_BATCH = 4000
#: Bin redraws on one layout before the layout itself is redrawn.
BIN_REDRAWS_PER_LAYOUT = 4
#: Candidate pairs per bin attempt that may be routed (each route is a grid search).
ROUTED_PAIRS = 40

#: Conditions built so far. The draw is shared by all four; the triggers arrive with 06-07.
BUILT_CONDITIONS = ("static", "dynamic")


class FeasibilityError(RuntimeError):
    """The resample cap was reached without a feasible family draw."""


@dataclass(frozen=True)
class EpisodeSpec:
    """One family draw. Rectangles are (centre x, centre y, half-width, half-height)."""
    generator_version: str
    family: str
    obstacles: str             # the condition: which parts of the draw the env switches on
    seed: int
    layout: tuple
    structure: dict            # family metadata: rooms and doorways, rows, corridor strips
    start: tuple
    goal: tuple
    distance_bin: int
    route: tuple               # start -> goal at R_C: the simplified oracle-grid path
    route_length: float        # the oracle's grid length of that path
    pedestrians: tuple         # robustsuite.pedestrians.Route, carried in every condition
    layout_draws: int          # layout draws made, the accepted one included
    bin_redraws: int           # failed bin attempts across all layouts
    pedestrian_redraws: int    # failed pedestrian draws across all layouts
    feasible: bool = True      # never False: an infeasible draw raises instead

    @property
    def active_pedestrians(self):
        """The pedestrians the environment runs in this spec's condition."""
        return () if self.obstacles == "static" else self.pedestrians

    def as_dict(self):
        """JSON-shaped: every tuple becomes a list."""
        def lists(v):
            if isinstance(v, (tuple, list)):
                return [lists(x) for x in v]
            if isinstance(v, dict):
                return {k: lists(x) for k, x in v.items()}
            return v
        return lists(asdict(self))


def oracle(layout, inflation=R_C):
    """The frozen shortest-path oracle on `layout`, obstacles inflated by `inflation`."""
    return ShortestPathOracle(WORLD, inflation, layout, resolution=GRID)


def free_space_connected(orc):
    """True if the oracle's free cells form one 8-connected component (the oracle's moves)."""
    _, n = ndimage.label(~orc.occupancy, structure=np.ones((3, 3), dtype=int))
    return n == 1


def _draw_start_goal(rng, layout, grid, lo, hi, fam, structure):
    """Start and goal uniform in free space (clearance >= START_GOAL_CLEARANCE), |goal - start|
    in [lo, hi], joined by a route at R_C, satisfying the family rule. None if this batch holds
    no such pair."""
    c = START_GOAL_CLEARANCE
    pts = rng.uniform(c, WORLD - c, size=(POINT_BATCH, 2))
    free = pts[clearances(pts, layout) >= c]
    routed = 0
    for s, g in zip(free[0::2], free[1::2]):
        if not lo <= float(np.linalg.norm(g - s)) <= hi:
            continue
        s, g = tuple(float(v) for v in s), tuple(float(v) for v in g)
        if not fam.rule_needs_route and not fam.rule(s, g, None, structure):
            continue
        routed += 1
        if routed > ROUTED_PAIRS:
            return None
        r = grid.route(s, g, layout, R_C)
        if r is not None and (not fam.rule_needs_route or fam.rule(s, g, r[0], structure)):
            return s, g, r
    return None


def generate(cell, motion, seed):
    """The validated EpisodeSpec for (cell, motion, seed). Raises FeasibilityError at the cap."""
    RS.check_motion(motion)
    entropy = RS.generator_entropy(cell, seed)        # ValueError for the M0 anchor
    if cell.obstacles not in BUILT_CONDITIONS:
        raise NotImplementedError(f"{cell} is not built yet")
    fam = FAMILIES[cell.family]
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))

    draws = layout_draws = bin_redraws = ped_redraws = 0
    while draws < RESAMPLE_CAP:
        draws += 1
        layout_draws += 1
        got = fam.draw(rng)
        if got is None:
            continue
        layout, structure = got
        orc = oracle(layout)
        if not free_space_connected(orc):
            continue
        grid = Grid(orc)
        sg = None
        for _ in range(BIN_REDRAWS_PER_LAYOUT):
            b = int(rng.integers(len(DISTANCE_BINS)))
            sg = _draw_start_goal(rng, layout, grid, *DISTANCE_BINS[b], fam, structure)
            if sg is not None:
                break
            bin_redraws += 1
            draws += 1
            if draws >= RESAMPLE_CAP:
                break
        if sg is None:
            continue
        start, goal, (rt, rt_len) = sg
        peds = []
        for _ in range(fam.pedestrians):
            r = draw_route(rng, layout, grid, R_C, start, goal)
            if r is None:
                break
            peds.append(r)
        if len(peds) < fam.pedestrians:
            ped_redraws += 1
            draws += 1
            continue
        return EpisodeSpec(GENERATOR_VERSION, cell.family, cell.obstacles, int(seed),
                           tuple(tuple(float(v) for v in r) for r in layout), structure,
                           start, goal, b, rt, float(rt_len), tuple(peds),
                           layout_draws, bin_redraws, ped_redraws)
    raise FeasibilityError(f"{cell} seed {seed}: no feasible draw in {RESAMPLE_CAP} draws")
