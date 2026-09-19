"""Triggered events for the scenario generator (RS_DESIGN.md 4.6): the trigger disc and the block.

The generator draws them after start and goal and before the pedestrians (4.7), in EVERY
obstacle condition, so all four conditions of a family share them; only `trigger_block` fires
the block. In the other conditions the block is still the pedestrians' reserved zone (4.4).

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
A failed d is redrawn up to OFFSET_TRIES times, then f is redrawn; after EVENT_ATTEMPTS draws of
f the family draw fails and the layout is redrawn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsuite.geometry import WORLD, arc_lengths, points_at, rect_distances, same_component, \
    with_rect

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
    p, before, after = points_at(route, [s, s - DIRECTION_ARC, s + DIRECTION_ARC], cum)
    t = after - before
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


def draw_trigger_block(rng, route, start, goal, orc):
    """(Trigger, Block, the oracle with the block added), or None when EVENT_ATTEMPTS draws of f
    all fail. `orc` is the frozen oracle on the static layout at the clearance radius."""
    radius = orc.agent_radius
    layout = orc.static_obstacles
    length = float(arc_lengths(route)[-1])
    for _ in range(EVENT_ATTEMPTS):
        f = float(rng.uniform(*TRIGGER_FRACTION))
        s_t = f * length
        c = points_at(route, [s_t])[0]
        if np.linalg.norm(c - np.asarray(start)) <= TRIGGER_RADIUS:
            continue
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
            trigger = Trigger(tuple(float(v) for v in c), TRIGGER_RADIUS, f, s_t)
            return trigger, Block(rect, d, axis, width), blocked
    return None
