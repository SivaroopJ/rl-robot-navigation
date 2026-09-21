"""Triggered events for the scenario generator (RS_DESIGN.md 4.6): the trigger disc, the block
and the spawn.

The generator draws them after start and goal and before the pedestrians (4.7), in EVERY
obstacle condition, so all four conditions of a family share them; only `trigger_block` fires
the block and only `trigger_spawn` the spawn. In the other conditions the block is still the
regular pedestrians' reserved zone (4.4).

TRIGGER. A disc of radius TRIGGER_RADIUS centred on the start -> goal route (the spec's route, the
simplified oracle path at R_C, arm-independent) at arc-length fraction f ~ U(0.3, 0.7) of the
route polyline. It fires on the first step on which the robot's centre is inside it.

BLOCK. Centred on the route a further arc length d ~ U(1.5, 2.5) m beyond the trigger centre.
Axis-aligned: its long axis is the world axis closest to the perpendicular of the route direction,
the direction taken as the chord over +-DIRECTION_ARC of arc. Along the long axis the free extent
through the centre runs to the nearest static obstacle or outer wall on each side; that passage
must be <= MAX_PASSAGE wide. The block is BLOCK_THICKNESS thick, centred on the route point, and
spans the passage plus BLOCK_OVERLAP into the wall on each side.

FEASIBILITY (4.6), each checked on the frozen oracle's grid at R_C with the block added:
    - start, trigger centre and goal lie in one free component (a clear path from the trigger to
      the goal, and from the start to the trigger);
    - the block keeps >= R_C from the start and from the goal;
    - it keeps >= TRIGGER_RADIUS + AGENT_RADIUS from the trigger centre, so the robot, whose
      centre is inside the disc when the block appears, is never touching it (RS_DESIGN 14.2);
    - the start is outside the disc, so the robot has to enter it (14.2).
SPAWN (4.6). A pedestrian that appears when the trigger fires. Its start is uniform over the
annulus SPAWN_DISTANCE (2-3 m) around the trigger centre, so >= 1.5 m from every point of the
disc, in free space at R_C, >= 1.5 m from the robot start and >= 1.0 m from the goal, and
occluded: the segment from the trigger centre to it crosses a static rectangle. Its entry:
    crossing  an oracle path to the route point CROSSING_OFFSET (1-2 m) beyond the trigger
              centre, then on along the world axis nearest the route's perpendicular, in its
              arrival direction, for up to CROSSING_WALK (2 m), stopping R_C before any static
              obstacle or the outer wall;
    head_on   an oracle path to the route point HEAD_ON_OFFSET (2 m) beyond the trigger centre,
              then back along the route towards the start for HEAD_ON_WALK (3 m).
Every entry path keeps >= R_C from the static layout. Afterwards it walks an oracle path into its
own 8-waypoint cycle, like any pedestrian. The variant is the generator's first draw for the
seed, with equal probability, and is kept through every redraw, so feasibility cannot bias it.

THE SPAWN IGNORES THE BLOCK ZONE (researcher decision, RS_DESIGN 14.3): the block sits where the
entries go, and the spawned pedestrian exists only in trigger_spawn, where the block never
appears. It is routed on the static layout alone.

A failed d is redrawn up to OFFSET_TRIES times and a failed spawn start up to SPAWN_TRIES times,
then f is redrawn; the block and the spawn must both succeed for the same f. After EVENT_ATTEMPTS
draws of f the family draw fails and the layout is redrawn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsuite.geometry import WORLD, arc_lengths, clearance, points_at, rect_distances, \
    route_slice, same_component, segment_clear, segment_crosses, with_rect
from robustsuite.pedestrians import START_FROM_GOAL, START_FROM_ROBOT, Route, draw_cycle

#: The trigger disc's radius, in metres (4.6).
TRIGGER_RADIUS = 0.5
#: The trigger centre's arc-length fraction of the route, drawn uniformly (4.6).
TRIGGER_FRACTION = (0.3, 0.7)
#: The block centre's arc length beyond the trigger centre, in metres, drawn uniformly (4.6).
BLOCK_OFFSET = (1.5, 2.5)
#: Block thickness, and its overlap into the wall at each end, in metres (4.6).
BLOCK_THICKNESS = 0.4
BLOCK_OVERLAP = 0.2
#: The widest passage a block may span, in metres (4.6).
MAX_PASSAGE = 3.0
#: The local route direction is the chord from DIRECTION_ARC before to DIRECTION_ARC after.
DIRECTION_ARC = 0.5
#: The robot's radius (RobotNavEnv.AGENT_RADIUS, 4.1): the block keeps TRIGGER_RADIUS +
#: AGENT_RADIUS from the trigger centre (RS_DESIGN 14.2).
AGENT_RADIUS = 0.3
#: Draws of f per family draw (4.7: "up to 50 attempts").
EVENT_ATTEMPTS = 50
#: Draws of d per f before f is redrawn.
OFFSET_TRIES = 5

#: The spawn start's distance from the trigger centre, in metres (4.6).
SPAWN_DISTANCE = (2.0, 3.0)
#: The two entry variants, equally likely (4.6).
VARIANTS = ("crossing", "head_on")
#: Crossing: the route point it heads for, in metres beyond the trigger centre (drawn
#: uniformly), and how far it walks on across the route (4.6).
CROSSING_OFFSET = (1.0, 2.0)
CROSSING_WALK = 2.0
#: Head-on: the route point it heads for, and how far it then walks back along the route (4.6).
HEAD_ON_OFFSET = 2.0
HEAD_ON_WALK = 3.0
#: Spawn starts drawn per f before f is redrawn.
SPAWN_TRIES = 20


@dataclass(frozen=True)
class Trigger:
    centre: tuple
    radius: float
    fraction: float             # f: arc-length fraction of the route
    arc: float                  # f times the route's polyline length, in metres


@dataclass(frozen=True)
class Block:
    rect: tuple                 # (centre x, centre y, half-width, half-height)
    offset: float               # d: arc length beyond the trigger centre, in metres
    axis: int                   # the long axis: 0 = x, 1 = y
    passage: float              # the free width it spans, in metres


@dataclass(frozen=True)
class Spawn:
    start: tuple
    variant: str                # "crossing" or "head_on"
    target: tuple               # the route point the entry heads for
    scripted: tuple             # the scripted entry: start -> target -> end of its walk
    route: Route                # the whole walk: the scripted entry, then its waypoint cycle


def free_extent(p, axis, layout, world=WORLD):
    """(lo, hi): the free interval along `axis` of the line through p, bounded by the nearest
    rectangle or outer wall on each side. None if p is inside a rectangle."""
    o = 1 - axis
    lo, hi = 0.0, float(world)
    for r in layout:
        if abs(p[o] - r[o]) >= r[2 + o]:
            continue                                   # the line misses (or grazes) r
        near, far = r[axis] - r[2 + axis], r[axis] + r[2 + axis]
        if far <= p[axis]:
            lo = max(lo, far)
        elif near >= p[axis]:
            hi = min(hi, near)
        else:
            return None
    return lo, hi


def place_block(route, s, layout):
    """The block centred on the route at arc length s, or None if the passage there is wider
    than MAX_PASSAGE. Returns (rect, axis, passage width)."""
    cum = arc_lengths(route)
    p = points_at(route, [s], cum)[0]
    t = _local_direction(route, s, cum)
    axis = 0 if abs(t[1]) >= abs(t[0]) else 1          # perpendicular to the route direction
    ext = free_extent(p, axis, layout)
    if ext is None or ext[1] - ext[0] > MAX_PASSAGE:
        return None
    lo, hi = ext
    half = [0.0, 0.0]
    centre = [float(p[0]), float(p[1])]
    centre[axis] = (lo + hi) / 2
    half[axis] = (hi - lo) / 2 + BLOCK_OVERLAP
    half[1 - axis] = BLOCK_THICKNESS / 2
    return (centre[0], centre[1], half[0], half[1]), axis, float(hi - lo)


def _local_direction(route, s, cum):
    """The route's direction at arc s: the chord over +-DIRECTION_ARC."""
    before, after = points_at(route, [s - DIRECTION_ARC, s + DIRECTION_ARC], cum)
    return after - before


def _crossing_walk(target, arrival, direction, layout, radius):
    """The end of the crossing walk from `target`: along the world axis nearest the route's
    perpendicular, in the arrival direction, up to CROSSING_WALK and stopping `radius` before a
    static obstacle or the outer wall."""
    axis = 0 if abs(direction[1]) >= abs(direction[0]) else 1
    sign = 1.0 if arrival[axis] >= 0 else -1.0
    lo, hi = free_extent(target, axis, layout)
    room = (hi - target[axis]) if sign > 0 else (target[axis] - lo)
    end = np.array(target, dtype=float)
    end[axis] += sign * min(CROSSING_WALK, max(room - radius, 0.0))
    return tuple(float(v) for v in end)


def _draw_spawn(rng, variant, route, cum, s_t, start, goal, grid, layout, radius):
    """A Spawn for the trigger at arc s_t, or None when SPAWN_TRIES starts all fail."""
    length = float(cum[-1])
    c = points_at(route, [s_t], cum)[0]
    for _ in range(SPAWN_TRIES):
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = float(np.sqrt(rng.uniform(SPAWN_DISTANCE[0] ** 2, SPAWN_DISTANCE[1] ** 2)))
        s_goal = s_t + (float(rng.uniform(*CROSSING_OFFSET)) if variant == "crossing"
                        else HEAD_ON_OFFSET)
        p = tuple(float(v) for v in c + rad * np.array([np.cos(ang), np.sin(ang)]))
        if (clearance(p, layout) < radius
                or np.linalg.norm(np.subtract(p, start)) < START_FROM_ROBOT
                or np.linalg.norm(np.subtract(p, goal)) < START_FROM_GOAL
                or not segment_crosses(c, p, layout) or s_goal >= length):
            continue
        target = tuple(float(v) for v in points_at(route, [s_goal], cum)[0])
        got = grid.route(p, target, layout, radius)
        if got is None:
            continue
        approach = list(got[0])
        if variant == "crossing":
            arrival = np.subtract(approach[-1], approach[-2]) if len(approach) > 1 \
                else np.subtract(target, c)
            end = _crossing_walk(target, arrival, _local_direction(route, s_goal, cum), layout,
                                 radius)
            if not segment_clear(target, end, layout, radius):
                continue
            walk = [end] if end != target else []
        else:
            walk = route_slice(route, s_goal, max(s_goal - HEAD_ON_WALK, 0.0), cum)[1:]
        scripted = tuple(approach + walk)
        cycle = draw_cycle(rng, layout, grid, radius, scripted[-1])
        if cycle is None:
            continue
        whole = Route(p, cycle.waypoints, scripted + tuple(cycle.entry[1:]), cycle.loop)
        return Spawn(p, variant, target, scripted, whole)
    return None


def draw_variant(rng):
    """The seed's spawn variant: the generator's first draw, kept through every redraw (14.3)."""
    return VARIANTS[int(rng.integers(len(VARIANTS)))]


def draw_events(rng, variant, route, start, goal, orc, grid):
    """(Trigger, Block, Spawn, the oracle with the block added), or None when EVENT_ATTEMPTS
    draws of f all fail. `variant` is the seed's spawn variant; `orc` is the frozen oracle on
    the static layout at the clearance radius, and `grid` the Grid on it."""
    radius = orc.agent_radius
    layout = orc.static_obstacles
    cum = arc_lengths(route)
    length = float(cum[-1])
    for _ in range(EVENT_ATTEMPTS):
        f = float(rng.uniform(*TRIGGER_FRACTION))
        s_t = f * length
        c = points_at(route, [s_t], cum)[0]
        if np.linalg.norm(c - np.asarray(start)) <= TRIGGER_RADIUS:
            continue
        block = None
        for _ in range(OFFSET_TRIES):
            d = float(rng.uniform(*BLOCK_OFFSET))
            if s_t + d >= length:
                continue
            got = place_block(route, s_t + d, layout)
            if got is None:
                continue
            rect, axis, width = got
            if rect_distances([c], [rect])[0] < TRIGGER_RADIUS + AGENT_RADIUS:
                continue
            if rect_distances([start, goal], [rect]).min() < radius:
                continue
            blocked = with_rect(orc, rect)
            if not same_component(blocked, [start, c, goal]):
                continue
            block = Block(rect, d, axis, width)
            break
        if block is None:
            continue
        spawn = _draw_spawn(rng, variant, route, cum, s_t, start, goal, grid, layout, radius)
        if spawn is None:
            continue
        trigger = Trigger(tuple(float(v) for v in c), TRIGGER_RADIUS, f, s_t)
        return trigger, block, spawn, blocked
    return None
