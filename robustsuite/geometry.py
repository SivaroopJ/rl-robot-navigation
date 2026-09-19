"""Geometry for the scenario generator: rectangles, clearance, paths on the frozen oracle's grid.

Rectangles are (centre x, centre y, half-width, half-height), the canonical environment's format.

PATHS. RS_DESIGN.md 4.1 says every routing check uses the frozen shortest-path oracle (200 x 200
grid, obstacles inflated by 0.45 m). The frozen ShortestPathOracle returns only a path LENGTH, so
`Grid` searches the oracle's own occupancy array with the oracle's own moves (8-connected, step
cost 1 or sqrt 2 cells, a move allowed when the target cell is free) and returns the cells too.
The search is scipy's Dijkstra, which is exact like the oracle's A*; the tests check the lengths
agree with `ShortestPathOracle.path_length`.

A box-inflated free cell centre has Euclidean clearance >= the inflation, so a grid path's cell
centres keep the clearance radius. `simplify` then replaces it by line segments, each checked
with its EXACT Euclidean clearance >= the radius (4.4: "simplified to line segments that keep
that clearance"): a segment's distance to a rectangle it does not cross is attained at one of
the segment's endpoints or one of the rectangle's corners.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

WORLD = 10.0
#: Strip edges are rounded to this many decimals before the complement is built, so float
#: noise in the corridor arithmetic cannot leave zero-width slivers.
SNAP_DIGITS = 9


def rect_gap(a, b):
    """Euclidean distance between two axis-aligned rectangles (0 if they touch or overlap)."""
    dx = max(abs(a[0] - b[0]) - (a[2] + b[2]), 0.0)
    dy = max(abs(a[1] - b[1]) - (a[3] + b[3]), 0.0)
    return float(np.hypot(dx, dy))


def rect_distances(points, layout):
    """Distance from each point (N, 2) to the nearest rectangle of `layout` (inf if none)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if not len(layout):
        return np.full(len(p), np.inf)
    r = np.asarray(layout, dtype=float)
    dx = np.maximum(np.abs(p[:, None, 0] - r[None, :, 0]) - r[None, :, 2], 0.0)
    dy = np.maximum(np.abs(p[:, None, 1] - r[None, :, 1]) - r[None, :, 3], 0.0)
    return np.hypot(dx, dy).min(axis=1)


def clearances(points, layout, world=WORLD):
    """Distance from each point (N, 2) to the nearest rectangle or outer wall."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    walls = np.minimum.reduce([p[:, 0], world - p[:, 0], p[:, 1], world - p[:, 1]])
    return np.minimum(walls, rect_distances(p, layout))


def clearance(p, layout, world=WORLD):
    return float(clearances([p], layout, world)[0])


def _point_segment_distance(p, a, b):
    """Distance from each point p (N, 2) to segment ab."""
    d = b - a
    L2 = float(d @ d)
    t = np.zeros(len(p)) if L2 == 0 else np.clip((p - a) @ d / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def _segment_crosses(a, b, r):
    """True if segment ab meets the closed rectangle r (Liang-Barsky clipping)."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for k in range(2):
        lo, hi = r[k] - r[2 + k], r[k] + r[2 + k]
        if abs(d[k]) < 1e-15:
            if not lo <= a[k] <= hi:
                return False
            continue
        u, v = (lo - a[k]) / d[k], (hi - a[k]) / d[k]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        if t0 > t1:
            return False
    return True


def segment_clearance(a, b, layout, world=WORLD):
    """Exact smallest clearance along segment ab: to the outer walls (attained at an endpoint)
    and to every rectangle (0 if crossed, else attained at an endpoint or a corner)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = float(clearances([a, b], [], world).min())
    for r in layout:
        if _segment_crosses(a, b, r):
            return 0.0
        cx, cy, hw, hh = r
        corners = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                            [cx + hw, cy + hh], [cx - hw, cy + hh]])
        out = min(out, float(rect_distances([a, b], [r]).min()),
                  float(_point_segment_distance(corners, a, b).min()))
    return out


def segment_clear(a, b, layout, radius, world=WORLD):
    """True if every point of segment ab has clearance >= radius."""
    return segment_clearance(a, b, layout, world) >= radius


def arc_lengths(pts):
    """Cumulative arc length at each vertex of a polyline, starting at 0."""
    p = np.asarray(pts, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def points_at(pts, s, cum=None):
    """The points at arc lengths s (array) along a polyline, clamped to its ends."""
    p = np.asarray(pts, dtype=float)
    cum = arc_lengths(p) if cum is None else cum
    s = np.atleast_1d(np.asarray(s, dtype=float))
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


def simplify(pts, layout, radius, world=WORLD):
    """Line-of-sight simplification of a polyline, keeping clearance >= radius.

    From each kept vertex, gallops forward (1, 2, 4, ... vertices) while the segment stays
    clear, then binary-searches the last clear jump. Every kept segment is checked exactly;
    the jump found is a clear one, not necessarily the farthest. Returns None if even a single
    original step is not clear."""
    p = [tuple(float(v) for v in q) for q in pts]
    last = len(p) - 1

    def clear(i, j):
        return segment_clear(p[i], p[j], layout, radius, world)

    out, i = [p[0]], 0
    while i < last:
        if not clear(i, i + 1):
            return None
        good, step = i + 1, 1
        while good < last:
            nxt = min(i + 2 * step, last)
            if not clear(i, nxt):
                bad = nxt
                break
            good, step = nxt, 2 * step
        else:
            bad = None
        while bad is not None and bad - good > 1:
            mid = (good + bad) // 2
            good, bad = (mid, bad) if clear(i, mid) else (good, mid)
        out.append(p[good])
        i = good
    return out


def free_rectangles_complement(strips, world=WORLD):
    """The solid rectangles filling the world outside the union of `strips` (rectangles).

    Coordinate compression over every strip edge, then cells not covered by any strip are
    merged into maximal horizontal runs, and runs with identical x-extent in consecutive rows
    are merged vertically. Deterministic."""
    def snap(v):
        return min(max(round(v, SNAP_DIGITS), 0.0), world)

    edges = [(snap(s[0] - s[2]), snap(s[0] + s[2]), snap(s[1] - s[3]), snap(s[1] + s[3]))
             for s in strips]
    xs = sorted({0.0, world, *[e[0] for e in edges], *[e[1] for e in edges]})
    ys = sorted({0.0, world, *[e[2] for e in edges], *[e[3] for e in edges]})

    def covered(x0, x1, y0, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return any(e[0] < cx < e[1] and e[2] < cy < e[3] for e in edges)

    runs = []                                    # per row band: [(x0, x1)]
    for j in range(len(ys) - 1):
        row, start = [], None
        for i in range(len(xs) - 1):
            solid = not covered(xs[i], xs[i + 1], ys[j], ys[j + 1])
            if solid and start is None:
                start = xs[i]
            if not solid and start is not None:
                row.append((start, xs[i]))
                start = None
        if start is not None:
            row.append((start, xs[-1]))
        runs.append(row)
    rects, open_ = [], {}                        # (x0, x1) -> y0 of the run still growing
    for j, row in enumerate(runs + [[]]):
        keys = set(row)
        for k in list(open_):
            if k not in keys:
                y0 = open_.pop(k)
                rects.append((k[0], k[1], y0, ys[j]))
        for k in row:
            open_.setdefault(k, ys[j])
    return [((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2)
            for x0, x1, y0, y1 in sorted(rects, key=lambda r: (r[2], r[0]))]


class Grid:
    """Shortest paths on a ShortestPathOracle's occupancy grid, with the oracle's moves."""

    def __init__(self, oracle):
        self.cell = oracle.cell
        self.res = oracle.resolution
        self.free = ~oracle.occupancy                         # [iy, ix]
        n = self.res
        self.node = -np.ones((n, n), dtype=np.int64)
        self.node[self.free] = np.arange(int(self.free.sum()))
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):       # each undirected move once
            x0, x1 = max(0, -dx), n - max(0, dx)
            a = self.node[0:n - dy, x0:x1]                     # cell (y, x)
            b = self.node[dy:n, x0 + dx:x1 + dx]               # cell (y + dy, x + dx)
            ok = (a >= 0) & (b >= 0)
            cost = float(np.hypot(dx, dy))
            rows += [a[ok], b[ok]]
            cols += [b[ok], a[ok]]
            w += [np.full(int(ok.sum()) * 2, cost)]
        m = int(self.free.sum())
        self.graph = csr_matrix((np.concatenate(w), (np.concatenate(rows),
                                                     np.concatenate(cols))), shape=(m, m))
        iy, ix = np.nonzero(self.free)                          # node k is (iy[k], ix[k])
        self.centres = np.column_stack([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def index(self, p):
        """The oracle's cell index of point p: (iy, ix)."""
        return (min(int(p[1] / self.cell), self.res - 1), min(int(p[0] / self.cell), self.res - 1))

    def node_of(self, p):
        return int(self.node[self.index(p)])

    def path(self, a, b):
        """(length in metres, [a, cell centres..., b]) or None if b is unreachable from a.
        The length is the oracle's: the cell-to-cell path, without the end offsets."""
        na, nb = self.node_of(a), self.node_of(b)
        if na < 0 or nb < 0:
            return None
        dist, pred = dijkstra(self.graph, indices=na, return_predecessors=True)
        if not np.isfinite(dist[nb]):
            return None
        chain, k = [], nb
        while k != na:
            chain.append(k)
            k = pred[k]
        chain.append(na)
        pts = [tuple(float(v) for v in self.centres[k]) for k in reversed(chain)]
        return float(dist[nb]) * self.cell, [tuple(a)] + pts[1:-1] + [tuple(b)]

    def route(self, a, b, layout, radius):
        """(a -> b as line segments keeping clearance >= radius, the oracle's grid length), or
        None if b is unreachable or the path cannot be simplified at that clearance."""
        got = self.path(a, b)
        if got is None:
            return None
        pts = simplify(got[1], layout, radius)
        return None if pts is None else (tuple(pts), got[0])
