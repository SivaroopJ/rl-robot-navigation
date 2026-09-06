"""A* over the known static map, returning WAYPOINTS, plus carrot following.

WHY THIS SUBCLASSES RATHER THAN EDITS ShortestPathOracle
---------------------------------------------------------
`evaluation/shortest_path.py` already builds exactly the grid we need -- walls inset by the
agent radius and each static rectangle inflated by it -- and already runs 8-connected A* over
it. But it returns only a LENGTH, because it exists to supply the SPL denominator. Editing it
would put the SPL metric of every earlier result at risk, so this class inherits the grid and
the neighbour set and adds a path-returning search alongside. `evaluation/` is untouched.

WHAT IS PLANNED OVER, AND WHAT IS NOT
--------------------------------------
The STATIC map only: walls and the five rectangles. Dynamic obstacles are deliberately NOT
planned against -- they are the local DR-CBF's job, from LiDAR. That separation is the point
of Phase 5: a global plan that is unaware of moving obstacles, and a local safety filter that
never sees the global map.

NO REPLANNING IS NEEDED HERE, BY CONSTRUCTION
----------------------------------------------
The static map is fixed and fully known at episode start, and the planner ignores dynamic
obstacles by design, so the plan cannot go stale. The path is computed once. If the barrier
constraint pushes the robot off the path, pure pursuit re-converges to it without a replan.
Replanning only becomes meaningful with an online map, which is the deferred 5b extension.
"""
from __future__ import annotations

import heapq

import numpy as np

from evaluation.shortest_path import _NEIGHBOURS, ShortestPathOracle


class StaticMapPlanner(ShortestPathOracle):
    """8-connected A* on the inherited static-map grid, returning a waypoint polyline."""

    def _centre(self, node):
        """Grid node (iy, ix) -> world (x, y). Mirrors ShortestPathOracle._index."""
        iy, ix = node
        return np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def path(self, start, goal):
        """Waypoints from start to goal, or None when either endpoint is blocked/unreachable.

        Returns an (M, 2) array whose first point is `start` and last is `goal`. Returning
        None rather than a fallback keeps a planner failure visible instead of silently
        degrading into straight-line behaviour.
        """
        si, gi = self._index(start), self._index(goal)
        if self.occupancy[si] or self.occupancy[gi]:
            return None

        came, best = {}, {si: 0.0}
        queue = [(0.0, 0.0, si)]
        found = False
        while queue:
            _, cost, node = heapq.heappop(queue)
            if node == gi:
                found = True
                break
            if cost > best.get(node, np.inf) + 1e-9:
                continue
            for dx, dy, weight in _NEIGHBOURS:
                nxt = (node[0] + dy, node[1] + dx)
                if not (0 <= nxt[0] < self.resolution and 0 <= nxt[1] < self.resolution):
                    continue
                if self.occupancy[nxt]:
                    continue
                new_cost = cost + weight
                if new_cost < best.get(nxt, np.inf) - 1e-12:
                    best[nxt] = new_cost
                    came[nxt] = node
                    h = float(np.hypot(nxt[0] - gi[0], nxt[1] - gi[1]))
                    heapq.heappush(queue, (new_cost + h, new_cost, nxt))
        if not found:
            return None

        nodes = [gi]
        while nodes[-1] != si:
            nodes.append(came[nodes[-1]])
        nodes.reverse()

        pts = [np.asarray(start, dtype=float).reshape(2)]
        pts += [self._centre(n) for n in nodes[1:-1]]
        pts.append(np.asarray(goal, dtype=float).reshape(2))
        return self.simplify(np.array(pts))

    # ------------------------------------------------------------------ simplification
    def visible(self, a, b, step=None):
        """True when the straight segment a->b stays in free space on the inflated grid."""
        a = np.asarray(a, dtype=float).reshape(2)
        b = np.asarray(b, dtype=float).reshape(2)
        d = float(np.linalg.norm(b - a))
        if d < 1e-12:
            return True
        step = step or (0.5 * self.cell)
        n = max(2, int(np.ceil(d / step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            q = a + t * (b - a)
            iy = min(int(q[1] / self.cell), self.resolution - 1)
            ix = min(int(q[0] / self.cell), self.resolution - 1)
            if iy < 0 or ix < 0 or self.occupancy[(iy, ix)]:
                return False
        return True

    def simplify(self, pts):
        """String-pulling: keep only the corners a straight line cannot skip.

        The raw grid path has one node per 0.05 m cell, which is hundreds of waypoints and
        useless as a carrot. Greedy line-of-sight reduction leaves the actual corners, and
        every retained segment is verified free on the same inflated grid the search used.
        """
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.visible(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)


class CarrotFollower:
    """Pure-pursuit reference along a fixed polyline. No governor -- that is a later step.

    Emits the point `lookahead` metres ahead of the robot's projection onto the path, which
    is what the CLF tracks and what `u_nom` points at. Beyond the path's end it emits the
    goal, so terminal behaviour is exactly the Phase 1-4 direct-to-goal case.
    """

    def __init__(self, path, *, lookahead=1.0):
        self.path = np.asarray(path, dtype=float).reshape(-1, 2)
        self.lookahead = float(lookahead)
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.total = float(self.seg_len.sum())
        self.progress = 0.0            # arclength of the projection, monotone non-decreasing

    @property
    def goal(self):
        return self.path[-1].copy()

    def _point_at(self, s):
        """World point at arclength s along the polyline."""
        if s <= 0:
            return self.path[0].copy()
        acc = 0.0
        for k, L in enumerate(self.seg_len):
            if acc + L >= s:
                t = (s - acc) / L if L > 1e-12 else 0.0
                return self.path[k] + t * (self.path[k + 1] - self.path[k])
            acc += L
        return self.path[-1].copy()

    def _project(self, p):
        """Arclength of the closest point on the polyline to p."""
        p = np.asarray(p, dtype=float).reshape(2)
        best_d, best_s, acc = np.inf, 0.0, 0.0
        for k, L in enumerate(self.seg_len):
            a, b = self.path[k], self.path[k + 1]
            if L < 1e-12:
                continue
            t = float(np.clip(np.dot(p - a, b - a) / (L * L), 0.0, 1.0))
            q = a + t * (b - a)
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d, best_s = d, acc + t * L
            acc += L
        return best_s

    def reference(self, p):
        """The carrot for the current robot position.

        Progress is kept monotone: the barrier constraint can push the robot sideways, and
        allowing the projection to slide backwards would let it re-chase path it has already
        covered.
        """
        self.progress = max(self.progress, self._project(p))
        return self._point_at(self.progress + self.lookahead)
