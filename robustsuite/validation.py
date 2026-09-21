"""The map validation report's measurements (RS_DESIGN.md 4.8): per-draw metrics, pedestrian-only
rollouts and renders. Nothing here runs a robot episode.

Every obstacle condition and both motions of a family share one family draw (4.3, 5), so the
metrics are computed once per (family, seed); a cell only switches parts of the draw on.

DETOUR. How much a fired block lengthens the way on: the frozen oracle's path length at the
clearance radius from the trigger centre to the goal with the block added, minus the same
without it. A block that spans from a box to the wall in open space can leave a short way round
the box, and this is the number that shows it (ticket 06 note).

BLOCK ENDS. What bounds the passage the block spans at each end: the outer wall or an obstacle.
"""
from __future__ import annotations

import numpy as np

from robustsuite import pedestrians as RP
from robustsuite.events import BLOCK_OVERLAP
from robustsuite.geometry import WORLD, Grid, arc_lengths, clearance, clearances, rect_gap, \
    segment_clearance, with_rect
from robustsuite.scenario import R_C, oracle
from robustsuite.scenario_env import RSScenarioEnv

#: Tolerance for "this passage end is the outer wall".
WALL_TOL = 1e-9
#: The environment's step, in seconds (config.json dt).
STEP_DT = 0.1
#: The largest step a pedestrian may take: 0.675 m/s over one step (RS_DESIGN 4.4).
MAX_STEP = RP.SPEED * STEP_DT
#: The clearance pedestrians keep from what they avoid, per motion (4.4): fixed ones walk their
#: legs at R_C; a randomized step is replaced when it would come within STEP_CLEARANCE.
FLOOR = {"fixed": R_C, "randomized": RP.STEP_CLEARANCE}


def summary(values):
    """n, mean and order statistics (linear interpolation) of a list of numbers."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    q = np.percentile(v, [0, 5, 50, 95, 100])
    return {"n": int(len(v)), "min": float(q[0]), "p5": float(q[1]), "median": float(q[2]),
            "p95": float(q[3]), "max": float(q[4]), "mean": float(v.mean())}


def block_ends(rect, axis, layout, world=WORLD):
    """("wall" | "obstacle", "wall" | "obstacle") for the low and high end of the passage the
    block spans along its long `axis`."""
    half = rect[2 + axis] - BLOCK_OVERLAP
    lo, hi = rect[axis] - half, rect[axis] + half
    return ("wall" if abs(lo) <= WALL_TOL else "obstacle",
            "wall" if abs(hi - world) <= WALL_TOL else "obstacle")


def block_kind(ends):
    """"wall-wall", "wall-obstacle" or "obstacle-obstacle", regardless of order."""
    return "-".join(sorted(ends, key=lambda e: e != "wall"))


def _polyline_clearance(poly, layout):
    return min(segment_clearance(a, b, layout) for a, b in zip(poly[:-1], poly[1:]))


def structural_passages(spec):
    """The family's own passage widths: doorways, aisles and cross-aisles, corridor strips, or,
    for clutter, the narrowest gap between two rectangles."""
    st = spec.structure
    if spec.family == "rooms":
        return [d["width"] for d in st["doorways"]]
    if spec.family == "aisles":
        return st["aisle_widths"] + st["cross_widths"] + (
            [st["mid_cross"][1]] if st["mid_cross"] else [])
    if spec.family == "corridors":
        return list(st["widths"].values())
    rects = spec.layout
    return [min(rect_gap(a, b) for i, a in enumerate(rects) for b in rects[i + 1:])]


def draw_metrics(spec):
    """Every per-draw number the report tabulates, JSON-shaped."""
    layout = spec.layout
    sw, blk, trig = spec.spawn, spec.block, spec.trigger
    free = oracle(layout)
    grid_free, grid_blocked = Grid(free), Grid(with_rect(free, blk.rect))
    before = grid_free.path(trig.centre, spec.goal)[0]
    after = grid_blocked.path(trig.centre, spec.goal)[0]
    ends = block_ends(blk.rect, blk.axis, layout)
    i = [tuple(q) for q in sw.scripted].index(tuple(sw.target))
    walk = float(arc_lengths(sw.scripted[i:])[-1]) if len(sw.scripted) - i > 1 else 0.0
    crossing = sw.variant == "crossing"
    return {
        "family": spec.family, "seed": spec.seed,
        "layout_draws": spec.layout_draws, "bin_redraws": spec.bin_redraws,
        "event_redraws": spec.event_redraws, "pedestrian_redraws": spec.pedestrian_redraws,
        "n_rects": len(layout),
        "start_goal_clearance": float(clearances([spec.start, spec.goal], layout).min()),
        "route_clearance": _polyline_clearance(spec.route, layout),
        "pedestrian_clearance": min(
            min(_polyline_clearance(r.entry, spec.pedestrian_layout),
                _polyline_clearance(r.loop, spec.pedestrian_layout)) for r in spec.pedestrians),
        "pedestrian_loop_length": [float(arc_lengths(r.loop)[-1]) for r in spec.pedestrians],
        "passages": [float(w) for w in structural_passages(spec)],
        "route_length": spec.route_length,
        "straight": float(np.linalg.norm(np.subtract(spec.goal, spec.start))),
        "distance_bin": spec.distance_bin,
        "n_pedestrians": len(spec.pedestrians),
        "trigger_fraction": trig.fraction,
        "block_offset": blk.offset,
        "block_passage": blk.passage,
        "block_length": 2.0 * blk.rect[2 + blk.axis],
        "block_ends": list(ends), "block_kind": block_kind(ends),
        "detour": float(after - before), "detour_ratio": float(after / before),
        "spawn_distance": float(np.linalg.norm(np.subtract(sw.start, trig.centre))),
        "spawn_clearance": float(clearance(sw.start, layout)),
        "spawn_entry_clearance": _polyline_clearance(sw.scripted, layout),
        "variant": sw.variant,
        "crossing_walk": walk if crossing else None,
        "crossing_zero": (walk == 0.0) if crossing else None,
    }


def pedestrian_trails(spec, motion, steps):
    """Pedestrian-only rollout (no robot), walked from step 0: positions (steps + 1, n, 2) of the
    regular pedestrians, on the reserved-zone layout, and (steps + 1, 2) of the spawned one, on
    the static layout."""
    rng = np.random.default_rng(spec.seed)
    trails = []
    for routes, layout in ((spec.pedestrians, spec.pedestrian_layout),
                           ([spec.spawn.route], spec.layout)):
        model = RP.PedestrianMotion(routes, layout, randomized=motion == "randomized")
        pos = np.zeros((len(routes), 2), dtype=np.float32)
        vel = model.reset(rng, pos, RP.SPEED)
        track = [pos.copy()]
        for _ in range(steps):
            pos, vel = model.step(rng, pos, vel, STEP_DT)
            track.append(pos.copy())
        trails.append(np.array(track))
    return trails[0], trails[1][:, 0]


def rollout_metrics(spec, motion, steps=500):
    """The pedestrians' smallest clearance to what they must avoid over a rollout, and the
    largest single step of any of them."""
    regular, spawned = pedestrian_trails(spec, motion, steps)
    moves = np.concatenate([np.linalg.norm(np.diff(regular, axis=0), axis=2).ravel(),
                            np.linalg.norm(np.diff(spawned, axis=0), axis=1)])
    return {"regular_min_clearance": float(clearances(regular.reshape(-1, 2),
                                                      spec.pedestrian_layout).min()),
            "spawn_min_clearance": float(clearances(spawned, spec.layout).min()),
            "max_step": float(moves.max())}


def anchor_trails(seed, motion, steps):
    """The M0 anchor through the scenario env, the robot held at its start: (M0 layout, start,
    goal, obstacle positions (steps + 1, n, 2)). M0's own motion models move the obstacles."""
    env = RSScenarioEnv(motion)
    try:
        env.install(None)
        env.reset(seed=seed)
        start, goal = env.agent_position.copy(), env.target_position.copy()
        track = [env.obstacle_positions.copy()]
        for _ in range(steps):
            env.step(np.zeros(2))
            track.append(env.obstacle_positions.copy())
        return tuple(env.static_obstacles), start, goal, np.array(track)
    finally:
        env.close()


# --------------------------------------------------------------------------- renders
def draw_rect(ax, r, **kw):
    """Rectangle r = (centre x, centre y, half-width, half-height) on a matplotlib axis."""
    from matplotlib.patches import Rectangle
    cx, cy, hw, hh = r
    ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, **kw))


def draw_frame(ax, layout):
    """The 10 x 10 m world with its static layout in grey."""
    ax.set_xlim(0, WORLD)
    ax.set_ylim(0, WORLD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in layout:
        draw_rect(ax, r, color="0.45", lw=0)


def draw_spec(ax, spec, obstacles):
    """One cell's view of a family draw: what `obstacles` switches on."""
    from matplotlib.patches import Circle
    draw_frame(ax, spec.layout)
    blk, trig, sw = spec.block, spec.trigger, spec.spawn
    if obstacles != "static":
        for r in spec.pedestrians:
            ax.plot(*np.asarray(r.loop).T, lw=0.6, alpha=0.6)
            ax.plot(*r.start, "o", ms=3, color="0.2")
    if obstacles == "dynamic":                      # the zone the pedestrians keep clear of
        draw_rect(ax, blk.rect, fill=False, ls="--", lw=0.8, color="tab:red")
    if obstacles == "trigger_block":
        draw_rect(ax, blk.rect, color="tab:red", alpha=0.85, lw=0)
    if obstacles in ("trigger_block", "trigger_spawn"):
        ax.add_patch(Circle(trig.centre, trig.radius, color="tab:orange", alpha=0.35, lw=0))
    if obstacles == "trigger_spawn":
        ax.plot(*np.asarray(sw.route.loop).T, lw=0.6, ls=":", color="tab:purple")
        ax.plot(*np.asarray(sw.scripted).T, lw=1.8, color="tab:purple")
        ax.plot(*sw.start, "x", ms=6, mew=2, color="tab:purple")
    ax.plot(*np.asarray(spec.route).T, lw=1.2, color="tab:blue")
    ax.plot(*spec.start, "o", ms=6, color="tab:green")
    ax.plot(*spec.goal, "*", ms=10, color="tab:green")


def draw_anchor(ax, layout, start, goal):
    """The M0 anchor: M0's map with one seed's start and goal."""
    draw_frame(ax, layout)
    ax.plot(*start, "o", ms=6, color="tab:green")
    ax.plot(*goal, "*", ms=10, color="tab:green")
