"""The five layout families (RS_DESIGN.md 4.2) and their structural start/goal rules (4.5).

Each family is a small grammar. `draw(rng)` returns (layout, structure) or None when the draw
fails (a rectangle that cannot be placed); the generator counts a None as one failed draw.
`structure` is JSON-able metadata the rules, the tests and the map validation report read:
doorways, rooms, rows, corridor strips. `rule(start, goal, route, structure)` is the family's
structural start/goal rule; the generator applies it to candidate pairs that already sit in their
distance bin, so start and goal stay uniform in free space conditional on bin and rule.

All widths, counts and positions are uniform within the ranges of RS_DESIGN 4.2. Choices the
design leaves open are named constants here (placement-try budgets, the mid cross-aisle's
position, the corridor network's placement and stub lengths) and are listed in ticket 03/04.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from robustsuite.geometry import WORLD, arc_lengths, free_rectangles_complement, points_at, \
    rect_gap

#: Placement tries per rectangle before the layout draw fails.
RECT_TRIES = 200


class Family(NamedTuple):
    draw: Callable
    rule: Callable
    pedestrians: int            # RS_DESIGN 4.2
    rule_needs_route: bool = False


def _scatter(rng, count, half, gap, wall):
    """Non-overlapping rectangles: count drawn from `count` (inclusive), half-extents from
    `half`, pairwise gap >= gap, >= wall from the outer wall. None if one cannot be placed."""
    n = int(rng.integers(count[0], count[1] + 1))
    rects = []
    for _ in range(n):
        for _ in range(RECT_TRIES):
            hw, hh = (float(v) for v in rng.uniform(*half, size=2))
            cx = float(rng.uniform(wall + hw, WORLD - wall - hw))
            cy = float(rng.uniform(wall + hh, WORLD - wall - hh))
            r = (cx, cy, hw, hh)
            if all(rect_gap(r, q) >= gap for q in rects):
                rects.append(r)
                break
        else:
            return None
    return rects


def _bins_only(start, goal, route, structure):
    return True


# --------------------------------------------------------------------------- clutter
OPEN_CLUTTER = {"count": (5, 7), "half": (0.25, 1.0), "gap": 1.2, "wall": 1.0}
DENSE_CLUTTER = {"count": (12, 18), "half": (0.15, 0.4), "gap": 1.1, "wall": 0.8}


def draw_open_clutter(rng):
    rects = _scatter(rng, **OPEN_CLUTTER)
    return None if rects is None else (tuple(rects), {})


def draw_dense_clutter(rng):
    rects = _scatter(rng, **DENSE_CLUTTER)
    return None if rects is None else (tuple(rects), {})


# --------------------------------------------------------------------------- rooms
WALL_HALF = 0.1                 # walls 0.2 m thick
ROOM_WALL_POS = (4.0, 6.0)
DOOR_WIDTH = (1.2, 1.6)
DOOR_JUNCTION_MARGIN = 0.5
FURNITURE_COUNT = (1, 2)
FURNITURE_HALF = (0.2, 0.5)
FURNITURE_DOOR_CLEARANCE = 1.0


def _wall_rect(axis, line, a, b):
    """A wall piece. axis "v": the vertical line x = line from y=a to y=b; "h": y = line."""
    mid, half = (a + b) / 2, (b - a) / 2
    return (line, mid, WALL_HALF, half) if axis == "v" else (mid, line, half, WALL_HALF)


def draw_rooms(rng):
    """3 or 4 rooms from full walls at [4, 6] m, one doorway per wall segment between two
    rooms (width [1.2, 1.6], >= 0.5 m from any junction), 1-2 furniture rectangles per room
    (half-extents [0.2, 0.5], >= 1.0 m from any doorway centre)."""
    n_rooms = int(rng.integers(3, 5))
    t = WALL_HALF
    if n_rooms == 4:
        xv, yh = (float(v) for v in rng.uniform(*ROOM_WALL_POS, size=2))
        segments = [("v", xv, 0.0, yh - t), ("v", xv, yh + t, WORLD),
                    ("h", yh, 0.0, xv - t), ("h", yh, xv + t, WORLD)]
        junctions = [(xv, yh)]
        rooms = [(0.0, 0.0, xv - t, yh - t), (xv + t, 0.0, WORLD, yh - t),
                 (0.0, yh + t, xv - t, WORLD), (xv + t, yh + t, WORLD, WORLD)]
    else:
        full = "v" if rng.random() < 0.5 else "h"
        p, q = (float(v) for v in rng.uniform(*ROOM_WALL_POS, size=2))
        low_half = bool(rng.random() < 0.5)          # the half the second wall splits
        a, b = (0.0, p - t) if low_half else (p + t, WORLD)
        segments = [(full, p, 0.0, q - t), (full, p, q + t, WORLD)]
        other = "h" if full == "v" else "v"
        segments.append((other, q, a, b))
        junctions = [(p, q) if full == "v" else (q, p)]
        split = [(a, 0.0, b, q - t), (a, q + t, b, WORLD)]       # (along-p, along-q) boxes
        whole = (p + t, 0.0, WORLD, WORLD) if low_half else (0.0, 0.0, p - t, WORLD)
        boxes = split + [whole]
        rooms = boxes if full == "v" else [(y0, x0, y1, x1) for x0, y0, x1, y1 in boxes]

    walls, doors = [], []
    for axis, line, s0, s1 in segments:
        w = float(rng.uniform(*DOOR_WIDTH))
        c = float(rng.uniform(s0 + DOOR_JUNCTION_MARGIN + w / 2, s1 - DOOR_JUNCTION_MARGIN - w / 2))
        walls += [_wall_rect(axis, line, s0, c - w / 2), _wall_rect(axis, line, c + w / 2, s1)]
        centre = (line, c) if axis == "v" else (c, line)
        doors.append({"axis": axis, "line": line, "centre": list(centre), "width": w,
                      "segment": [s0, s1]})
    walls += [(x, y, t, t) for x, y in junctions]            # fill the wall crossings

    furniture = []
    for x0, y0, x1, y1 in rooms:
        k = int(rng.integers(FURNITURE_COUNT[0], FURNITURE_COUNT[1] + 1))
        for _ in range(k):
            for _ in range(RECT_TRIES):
                hw, hh = (float(v) for v in rng.uniform(*FURNITURE_HALF, size=2))
                r = (float(rng.uniform(x0 + hw, x1 - hw)), float(rng.uniform(y0 + hh, y1 - hh)),
                     hw, hh)
                far = all(rect_gap(r, (d["centre"][0], d["centre"][1], 0.0, 0.0))
                          >= FURNITURE_DOOR_CLEARANCE for d in doors)
                if far and all(rect_gap(r, f) > 0.0 for f in furniture):
                    furniture.append(r)
                    break
            else:
                return None
    structure = {"n_rooms": n_rooms, "rooms": [list(r) for r in rooms], "doorways": doors}
    return tuple(walls + furniture), structure


def room_of(p, structure):
    for i, (x0, y0, x1, y1) in enumerate(structure["rooms"]):
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return i
    return None


def rooms_rule(start, goal, route, structure):
    a, b = room_of(start, structure), room_of(goal, structure)
    return a is not None and b is not None and a != b


# --------------------------------------------------------------------------- aisles
AISLE_ROWS = (3, 4)
SHELF_DEPTH = (0.6, 0.8)
AISLE_WIDTH = (1.4, 2.0)                  # aisles and cross-aisles
MID_CROSS_PROB = 0.5
#: The mid cross-aisle leaves at least this much row on each side (not fixed by RS_DESIGN).
MID_CROSS_MIN_ROW = 1.5
SAME_AISLE_FRACTION = 0.6


def draw_aisles(rng):
    """3-4 parallel shelf rows (depth [0.6, 0.8]) centred in the world, aisles between them
    ([1.4, 2.0]), a cross-aisle at both ends of the rows and, with probability 0.5, one mid
    cross-aisle cutting every row. Built in a frame where rows run along x, then transposed
    for the vertical orientation."""
    orientation = "h" if rng.random() < 0.5 else "v"
    n = int(rng.integers(AISLE_ROWS[0], AISLE_ROWS[1] + 1))
    depths = [float(v) for v in rng.uniform(*SHELF_DEPTH, size=n)]
    widths = [float(v) for v in rng.uniform(*AISLE_WIDTH, size=n - 1)]
    ends = [float(v) for v in rng.uniform(*AISLE_WIDTH, size=2)]
    along = (ends[0], WORLD - ends[1])
    mid = None
    if rng.random() < MID_CROSS_PROB:
        m = float(rng.uniform(*AISLE_WIDTH))
        c = float(rng.uniform(along[0] + MID_CROSS_MIN_ROW + m / 2,
                              along[1] - MID_CROSS_MIN_ROW - m / 2))
        mid = (c, m)
    height = sum(depths) + sum(widths)
    y, rows = WORLD / 2 - height / 2, []
    for i, d in enumerate(depths):
        rows.append((y, y + d))
        y += d + (widths[i] if i < n - 1 else 0.0)
    pieces = [along] if mid is None else [(along[0], mid[0] - mid[1] / 2),
                                          (mid[0] + mid[1] / 2, along[1])]
    rects = [((a + b) / 2, (y0 + y1) / 2, (b - a) / 2, (y1 - y0) / 2)
             for y0, y1 in rows for a, b in pieces]
    if orientation == "v":
        rects = [(cy, cx, hh, hw) for cx, cy, hw, hh in rects]
    structure = {"orientation": orientation, "rows": [list(r) for r in rows],
                 "along": list(along), "aisle_widths": widths, "cross_widths": ends,
                 "mid_cross": None if mid is None else list(mid)}
    return tuple(rects), structure


def aisle_of(p, structure):
    """(aisle index, along-row coordinate) of p, or None if p is not in an aisle between rows."""
    u, v = (p[0], p[1]) if structure["orientation"] == "h" else (p[1], p[0])
    lo, hi = structure["along"]
    if not lo <= u <= hi:
        return None
    rows = structure["rows"]
    for i in range(len(rows) - 1):
        if rows[i][1] <= v <= rows[i + 1][0]:
            return i, u
    return None


def aisles_rule(start, goal, route, structure):
    a, b = aisle_of(start, structure), aisle_of(goal, structure)
    if a is None or b is None:
        return False
    lo, hi = structure["along"]
    return a[0] != b[0] or abs(a[1] - b[1]) >= SAME_AISLE_FRACTION * (hi - lo)


# --------------------------------------------------------------------------- corridors
CORRIDOR_WIDTH = (1.5, 2.0)
#: Placement of the network (RS_DESIGN fixes only its topology and widths).
SPINE_CENTRE = (1.4, 3.0)          # spine centre, measured from its side of the world
EDGE_MARGIN = 0.5                  # corridors keep this much wall from the outer boundary
CORRIDOR_SEPARATION = 1.0          # minimum wall between two corridors that do not join
CONNECTOR_TOP = (7.0, WORLD - EDGE_MARGIN)
BRANCH_MIN_LENGTH = 2.0
STUB_LENGTH = (1.5, 3.0)
TURN_ANGLE, TURN_ARC = np.deg2rad(60.0), 1.0


def _strip(x0, x1, y0, y1):
    return ((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2)


def draw_corridors(rng):
    """A spine along the world, 2-3 perpendicular branches (T-junctions), a connector parallel
    to the spine joining the far ends of two branches (the loop), and an L-bend stub at the far
    end of one branch. Widths [1.5, 2.0]. Everything outside the strips is solid wall. Built with
    the spine along x near y = 0, then flipped and/or transposed.

    Constructive, so every accepted count is reachable: the branch widths and the stub length are
    drawn first, then the spare width is split uniformly over the gaps (>= CORRIDOR_SEPARATION of
    wall between branches, the stub's length on its outward side). The stub branch is an outer
    one and the stub turns outward, away from the loop. With 3 branches the stub branch is the
    free one and the loop joins the other two; with 2 branches both are in the loop, so the stub
    turns off a loop branch's far end opposite the connector (ticket 04's reading)."""
    transpose, flip = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    ws = float(rng.uniform(*CORRIDOR_WIDTH))
    ys = float(rng.uniform(*SPINE_CENTRE))
    spine_top = ys + ws / 2
    nb = int(rng.integers(2, 4))
    stub_branch = 0 if rng.random() < 0.5 else nb - 1
    for _ in range(RECT_TRIES):            # widths and stub length that fit, for this count
        widths = [float(v) for v in rng.uniform(*CORRIDOR_WIDTH, size=nb)]
        length = float(rng.uniform(*STUB_LENGTH))
        spare = (WORLD - 2 * EDGE_MARGIN) - sum(widths) - (nb - 1) * CORRIDOR_SEPARATION - length
        if spare >= 0:
            break
    else:
        return None
    cuts = np.sort(rng.uniform(0.0, spare, size=nb))
    gaps = list(np.diff(np.concatenate([[0.0], cuts, [spare]])))       # nb + 1 gaps
    gaps = [g + (CORRIDOR_SEPARATION if 0 < k < nb else 0.0) for k, g in enumerate(gaps)]
    gaps[0 if stub_branch == 0 else nb] += length
    branches, x = [], EDGE_MARGIN
    for k, w in enumerate(widths):
        x += gaps[k]
        branches.append((x, x + w))
        x += w
    pair = tuple(k for k in range(nb) if k != stub_branch) if nb == 3 else (0, 1)
    wc = float(rng.uniform(*CORRIDOR_WIDTH))
    top = float(rng.uniform(*CONNECTOR_TOP))
    conn_y = (top - wc, top)
    tops = [top if i in pair else None for i in range(nb)]
    if nb == 3:
        wb = widths[stub_branch]
        # the free branch's stub (as wide as the branch, at its top) keeps
        # CORRIDOR_SEPARATION of wall from the spine
        lo = spine_top + max(BRANCH_MIN_LENGTH, wb + CORRIDOR_SEPARATION)
        tops[stub_branch] = float(rng.uniform(lo, WORLD - EDGE_MARGIN))
    strips = {"spine": _strip(0.0, WORLD, ys - ws / 2, spine_top)}
    for i, (x0, x1) in enumerate(branches):
        strips[f"branch{i}"] = _strip(x0, x1, ys, tops[i])
    strips["connector"] = _strip(branches[pair[0]][0], branches[pair[1]][1], *conn_y)

    x0, x1 = branches[stub_branch]
    wb, tb = x1 - x0, tops[stub_branch]
    sx0, sx1 = (x0 - length, x0) if stub_branch == 0 else (x1, x1 + length)
    stub = _strip(sx0, sx1, tb - wb, tb)
    # safety net: the stub touches only its own branch (and, with two branches, the connector
    # it continues); every other corridor keeps CORRIDOR_SEPARATION of wall
    own = {f"branch{stub_branch}"} | ({"connector"} if nb == 2 else set())
    if not all(rect_gap(stub, s) >= CORRIDOR_SEPARATION
               for k, s in strips.items() if k not in own):
        return None
    strips["stub"] = stub

    def place(r):
        cx, cy, hw, hh = r
        if flip:
            cy = WORLD - cy
        return (cy, cx, hh, hw) if transpose else (cx, cy, hw, hh)

    strips = {k: place(v) for k, v in strips.items()}
    walls = free_rectangles_complement(list(strips.values()))
    widths = {k: 2 * min(v[2], v[3]) for k, v in strips.items()}
    structure = {"strips": {k: list(v) for k, v in strips.items()}, "widths": widths,
                 "branches": nb, "loop": list(pair), "stub_branch": stub_branch}
    return tuple(walls), structure


def has_turn(route, angle=TURN_ANGLE, arc=TURN_ARC):
    """True if at some vertex of the route the chord over the `arc` metres before it and the
    chord over the `arc` metres after it differ in direction by >= `angle` (RS_DESIGN 4.5:
    "a change of direction >= 60 deg sustained over >= 1 m")."""
    pts = np.asarray(route, dtype=float)
    cum = arc_lengths(pts)
    for i in range(1, len(pts) - 1):
        s = cum[i]
        if s < arc or s > cum[-1] - arc:
            continue
        before, after = points_at(pts, [s - arc, s + arc], cum)
        a, b = pts[i] - before, after - pts[i]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if np.arccos(np.clip(cos, -1.0, 1.0)) >= angle:
            return True
    return False


def corridors_rule(start, goal, route, structure):
    return has_turn(route)


FAMILIES = {
    "open_clutter": Family(draw_open_clutter, _bins_only, 5),
    "rooms": Family(draw_rooms, rooms_rule, 3),
    "aisles": Family(draw_aisles, aisles_rule, 4),
    "corridors": Family(draw_corridors, corridors_rule, 3, rule_needs_route=True),
    "dense_clutter": Family(draw_dense_clutter, _bins_only, 4),
}
