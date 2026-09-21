"""RS Experiment 1, phase 4: the three candidate arms (RS_DESIGN.md 7.4, revised 2026-09-20).

    detour        a LiDAR map of new static obstacles and Bug2-style wall following that rejoins
                  the reset-time A* path (ticket 11a)
    yield         a side step away from a tracked pedestrian on a close approach (ticket 11b)
    detour_yield  yield's layer over detour's layer over the frozen follower (ticket 11c)

Every candidate is the frozen `astar_random` stack with a behaviour layer installed as the
policy's follower after reset. The frozen DRCBFPolicy.predict reads its CLF reference as
`gamma = follower.reference(p)`, so a layer changes only gamma: the QP, its parameters, Random
recovery and the dynamics are untouched, and no frozen file is edited. A layer always calls the
follower it wraps first, every step, so the frozen CarrotFollower's progress evolves exactly as in
the baseline; with nothing to act on, a layer returns that gamma unchanged. No planner is called
after reset.

INFORMATION (RS_DESIGN 7.2 and 7.4). A layer reads:
    obs[4:28]         the LiDAR ranges, handed over by `see(obs)` before each predict() and sliced
                      exactly as the frozen predict() slices them (nothing past index 28);
    p                 the ego pose predict() passes to reference();
    the known map     the rectangles the policy's planner was built from at reset
                      (`pol.planner`), before any trigger can fire;
    the tracker       the frozen LiDAR velocity tracker's tracks, read only (position, velocity,
                      confirmed);
and its own history. Never obs[28:], the episode spec or the scenario-event log.

The readings of open wording are recorded in RS_DESIGN.md, "Readings before any tuning run
(2026-09-20, tickets 11a-11c)"; each is marked here as (reading N).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from robot_env.lidar_core import ray_angles

ARMS = ("detour", "yield", "detour_yield")

# ------------------------------------------------------------------ fixed numbers (7.4)
#: The new-obstacle grid, m.
CELL = 0.1
#: A hit farther than this from the known map is unexplained, m.
EXPLAIN = 0.15
#: Steps over which a cell's unexplained hits are counted.
WINDOW = 20
#: Trigger (a): arc of the original path checked ahead of the follower's progress, m.
AHEAD = 2.0
#: Trigger (a) and the rejoin point: path to new-cell distance, m.
NEAR = 0.45
#: q lies this much arc past the last near point, m.
REJOIN_PAST = 0.5
#: Trigger (b): moved less than STALL_DIST m over this many steps.
STALL_STEPS = 30
STALL_DIST = 0.1
#: A stall with no new cell nearby rejoins this much arc past the progress, m.
STALL_Q = 1.5
#: The side test: rays this many degrees from the direction to q.
SIDE_RAYS = (30.0, 90.0)
#: Wall following keeps the robot's centre this far from the wall surface, m.
WALL_GAP = 0.6
#: The leave rule's line-of-sight clearance, m.
LOS_CLEAR = 0.35
#: gamma = q until the robot is this close to q, m.
ARRIVE = 1.0
#: The detour flips to the other side once after this many steps, and gives up after GIVE_UP.
FLIP_STEPS = 200
GIVE_UP_STEPS = 400
#: Sampling of the original path, m (the planner's own grid).
PATH_STEP = 0.05
#: yield: a threat moves at least this fast, m/s.
MIN_SPEED = 0.3
#: yield: rays within this many degrees of the side direction decide the side.
YIELD_RAYS = 45.0


# ------------------------------------------------------------------ configurations (7.3, 7.4)
@dataclass(frozen=True)
class DetourConfig:
    K: int = 14
    L: float = 1.0
    stall: bool = True

    @property
    def name(self):
        return f"K{self.K}_L{self.L:.1f}_stall_{'on' if self.stall else 'off'}"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class YieldConfig:
    H: float = 3.0
    D: float = 1.2
    L: float = 1.0

    @property
    def name(self):
        return f"H{self.H:g}_D{self.D:.1f}_L{self.L:.1f}"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DetourYieldConfig:
    detour: DetourConfig = DetourConfig()
    yield_: YieldConfig = YieldConfig()

    @property
    def name(self):
        return f"{self.detour.name}+{self.yield_.name}"

    def as_dict(self):
        return {"detour": self.detour.as_dict(), "yield": self.yield_.as_dict()}


#: The pre-registered search grids (7.4): 8 configurations each, the default among them.
DETOUR_GRID = tuple(DetourConfig(K, L, s) for K in (8, 14) for L in (0.6, 1.0)
                    for s in (True, False))
YIELD_GRID = tuple(YieldConfig(H, D, L) for H in (2.0, 3.0) for D in (0.9, 1.2)
                   for L in (0.6, 1.0))
DEFAULTS = {"detour": DetourConfig(), "yield": YieldConfig(),
            "detour_yield": DetourYieldConfig()}


#: arm -> its pre-registered grid (7.4). detour_yield's configurations are pairs, so its grid is
#: the two grids it draws from; `configure` resolves a pair by name rather than enumerating 64.
GRIDS = {"detour": DETOUR_GRID, "yield": YIELD_GRID,
         "detour_yield": (DETOUR_GRID, YIELD_GRID)}


def check_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown candidate {arm!r}; expected one of {ARMS}")


def configurations(arm):
    """{name: config} of every configuration the harness accepts for `arm`.

    detour_yield accepts any pair of a detour and a yield grid configuration, named
    "<detour name>+<yield name>"; its search (7.4) runs two of them, see `search_grid`."""
    check_arm(arm)
    if arm == "detour_yield":
        return {DetourYieldConfig(d, y).name: DetourYieldConfig(d, y)
                for d in DETOUR_GRID for y in YIELD_GRID}
    return {c.name: c for c in GRIDS[arm]}


def configure(arm, name=None):
    """The named configuration of `arm`; None or "default" is the pre-registered default."""
    check_arm(arm)
    if name is None or name == "default":
        return DEFAULTS[arm]
    if arm == "detour_yield":
        parts = name.split("+")
        if len(parts) != 2:
            raise ValueError(f"{arm} has no configuration {name!r}")
        return DetourYieldConfig(configure("detour", parts[0]), configure("yield", parts[1]))
    table = {c.name: c for c in GRIDS[arm]}
    if name not in table:
        raise ValueError(f"{arm} has no configuration {name!r}")
    return table[name]


def search_grid(arm, chosen=None):
    """The configuration names the 7.3 search runs, the default first.

    detour and yield: their 8-point grids. detour_yield: both defaults, plus, once the searches
    of detour and yield are done, `chosen` = (detour name, yield name) taken together."""
    check_arm(arm)
    default = DEFAULTS[arm].name
    if arm != "detour_yield":
        return (default,) + tuple(n for n in configurations(arm) if n != default)
    if chosen is None:
        return (default,)
    pair = DetourYieldConfig(configure("detour", chosen[0]),
                             configure("yield", chosen[1])).name
    return (default,) if pair == default else (default, pair)


# ------------------------------------------------------------------ geometry helpers
def _rect_distance(pts, rects):
    """(n,) distance from each point to the nearest rectangle (0 inside); inf with none."""
    pts = np.atleast_2d(np.asarray(pts, float))
    if not len(rects):
        return np.full(len(pts), np.inf)
    r = np.asarray(rects, float)
    dx = np.maximum(np.abs(pts[:, None, 0] - r[None, :, 0]) - r[None, :, 2], 0.0)
    dy = np.maximum(np.abs(pts[:, None, 1] - r[None, :, 1]) - r[None, :, 3], 0.0)
    return np.hypot(dx, dy).min(axis=1)


def _box_distance(pts, lo, hi):
    """(n,) distance from each point to the boundary of the box [lo, hi]^2, 0 outside it."""
    pts = np.atleast_2d(np.asarray(pts, float))
    d = np.minimum(np.minimum(pts[:, 0] - lo, hi - pts[:, 0]),
                   np.minimum(pts[:, 1] - lo, hi - pts[:, 1]))
    return np.maximum(d, 0.0)


def _angle_diff(a, b):
    """Signed a - b wrapped to (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _ray_window(ranges, centre, lo, hi, *, n_rays):
    """The smallest range among the rays whose bearing is `lo`-`hi` degrees from `centre`
    (a direction vector). Both layers pick a side this way."""
    rel = np.degrees(_angle_diff(ray_angles(n_rays), np.arctan2(centre[1], centre[0])))
    return ranges[(rel >= lo - 1e-6) & (rel <= hi + 1e-6)].min()


def _rot(v, side):
    """v turned 90 degrees clockwise for side +1, counter-clockwise for side -1."""
    return np.array([v[1], -v[0]]) if side > 0 else np.array([-v[1], v[0]])


class _Polyline:
    """Vectorised arc-length sampling of the follower's path (its own _point_at, per point)."""

    def __init__(self, path):
        self.path = np.asarray(path, float).reshape(-1, 2)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self.cum[-1])

    def at(self, s):
        s = np.clip(np.asarray(s, float), 0.0, self.total)
        return np.stack([np.interp(s, self.cum, self.path[:, 0]),
                         np.interp(s, self.cum, self.path[:, 1])], axis=-1)


# ------------------------------------------------------------------ the new-obstacle map
class NewObstacleMap:
    """A 0.1 m grid of new static obstacles, from LiDAR (7.4, candidate 1 revised).

    A hit is unexplained when it is farther than 0.15 m from the known map: its rectangles and
    the outer wall where the LiDAR sees it, `lidar_inset` inside the world boundary (reading 1).
    A cell becomes new once unexplained hits land in it on at least K of the last 20 steps, and
    stays new for the episode. It also keeps the layer's clearance grid for the leave rule: a
    cell is blocked when its centre is within 0.35 m of a known rectangle, the world boundary or
    a new cell (reading 6).
    """

    def __init__(self, known_rects, K, *, world_size=10.0, lidar_inset=0.3, n_rays=24,
                 lidar_range=5.0):
        self.known = [tuple(float(v) for v in r) for r in known_rects]
        self.K = int(K)
        self.world_size = float(world_size)
        self.inset = float(lidar_inset)
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.n = int(round(self.world_size / CELL))
        ang = ray_angles(self.n_rays)
        self.dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.new = np.zeros((self.n, self.n), bool)            # [ix, iy]
        self.counts = np.zeros((self.n, self.n), int)
        self.window = deque()
        c = (np.arange(self.n) + 0.5) * CELL
        cx, cy = np.meshgrid(c, c, indexing="ij")
        centres = np.stack([cx.ravel(), cy.ravel()], axis=1)
        known = np.minimum(_rect_distance(centres, self.known),
                           _box_distance(centres, 0.0, self.world_size))
        self.blocked = (known < LOS_CLEAR).reshape(self.n, self.n)
        # the cells whose centre lies within LOS_CLEAR of a cell's square, as index offsets
        r = int(np.ceil((LOS_CLEAR + CELL / 2) / CELL))
        off = np.arange(-r, r + 1)
        ox, oy = np.meshgrid(off, off, indexing="ij")
        gap = np.hypot(np.maximum(np.abs(ox) * CELL - CELL / 2, 0.0),
                       np.maximum(np.abs(oy) * CELL - CELL / 2, 0.0))
        self._stencil = np.stack([ox[gap < LOS_CLEAR], oy[gap < LOS_CLEAR]], axis=1)
        # the last scan: hit points, and which may serve as a wall (explained or in a new cell)
        self.hits = np.zeros((0, 2))
        self.wall = np.zeros(0, bool)

    def index(self, pts):
        """(n, 2) integer cell indices [ix, iy] of world points, clipped to the grid."""
        return np.clip(np.floor(np.atleast_2d(pts) / CELL).astype(int), 0, self.n - 1)

    def explained(self, pts):
        d = np.minimum(_rect_distance(pts, self.known),
                       _box_distance(pts, self.inset, self.world_size - self.inset))
        return d <= EXPLAIN

    def update(self, p, ranges):
        """Take one scan at pose p. Returns the cells that became new on this step."""
        p = np.asarray(p, float).reshape(2)
        ranges = np.asarray(ranges, float).reshape(-1)
        hit = ranges < self.lidar_range - 1e-6               # as lidar_cbf.surface_points
        pts = p[None, :] + ranges[hit, None] * self.dirs[hit]
        expl = self.explained(pts)
        idx = self.index(pts[~expl]) if (~expl).any() else np.zeros((0, 2), int)
        idx = np.unique(idx, axis=0)                         # one count per cell per step
        self.window.append(idx)
        self.counts[idx[:, 0], idx[:, 1]] += 1
        if len(self.window) > WINDOW:
            old = self.window.popleft()
            self.counts[old[:, 0], old[:, 1]] -= 1
        fresh = np.argwhere((self.counts >= self.K) & ~self.new)
        for ix, iy in fresh:
            self.new[ix, iy] = True
            nb = self._stencil + (ix, iy)
            nb = nb[(nb >= 0).all(axis=1) & (nb < self.n).all(axis=1)]
            self.blocked[nb[:, 0], nb[:, 1]] = True
        self.hits = pts
        if len(pts):
            hi = self.index(pts)
            self.wall = expl | self.new[hi[:, 0], hi[:, 1]]
        else:
            self.wall = np.zeros(0, bool)
        return fresh

    def cells(self):
        """(m, 2) centres of the new cells."""
        return (np.argwhere(self.new) + 0.5) * CELL

    def distance(self, pts):
        """(n,) distance from each point to the nearest new cell's square (inf with none)."""
        c = self.cells()
        pts = np.atleast_2d(np.asarray(pts, float))
        if not len(c):
            return np.full(len(pts), np.inf)
        dx = np.maximum(np.abs(pts[:, None, 0] - c[None, :, 0]) - CELL / 2, 0.0)
        dy = np.maximum(np.abs(pts[:, None, 1] - c[None, :, 1]) - CELL / 2, 0.0)
        return np.hypot(dx, dy).min(axis=1)

    def segment_clear(self, a, b):
        """The straight segment a->b crosses no blocked cell of the clearance grid."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (CELL / 2))) + 1)
        pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
        if (pts < 0).any() or (pts > self.world_size).any():
            return False
        idx = self.index(pts)
        return not self.blocked[idx[:, 0], idx[:, 1]].any()


class _Layer:
    """What both behaviour layers share: they wrap a follower and are handed the observation
    before each predict(), of which they keep the LiDAR ranges only (reading 15)."""

    def __init__(self, inner, *, n_rays=24, lidar_range=5.0):
        self.inner = inner
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.ranges = None
        self.n_steps = 0

    def see(self, obs):
        """The observation of this step, before predict(); only obs[4:28] is kept."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        self.ranges = obs[4:4 + self.n_rays] * self.lidar_range
        if hasattr(self.inner, "see"):
            self.inner.see(obs)          # the layer below sees the same observation

    @property
    def path(self):
        return self.inner.path

    @property
    def progress(self):
        return self.inner.progress

    @property
    def goal(self):
        return self.inner.goal


# ------------------------------------------------------------------ candidate 1: detour
class DetourLayer(_Layer):
    """Follow the walls round a new obstacle, then rejoin the reset-time plan (candidate 1).

    `inner` is the frozen CarrotFollower (or anything with its `reference`, `progress` and
    `path`). States: None (the follower's gamma), "wall" (wall following), "leave" (gamma = q).
    `detours` logs every detour: its start step, trigger, rejoin arc, side, end step and reason.
    """

    def __init__(self, inner, known_rects, cfg, *, world_size=10.0, lidar_inset=0.3,
                 n_rays=24, lidar_range=5.0):
        super().__init__(inner, n_rays=n_rays, lidar_range=lidar_range)
        self.cfg = cfg
        self.map = NewObstacleMap(known_rects, cfg.K, world_size=world_size,
                                  lidar_inset=lidar_inset, n_rays=n_rays,
                                  lidar_range=lidar_range)
        self.line = _Polyline(inner.path)
        self.hist = deque(maxlen=STALL_STEPS + 1)   # positions since the last detour ended
        #: Trigger (a) looks at the path from here on: the q of the last detour (reading 4).
        self.trigger_floor = 0.0
        self.state = None
        self.q = None           # the rejoin point of the detour running now
        self.s_q = None         # its arc length along the original path
        self.d0 = None          # |p - q| when the detour started, the Bug2 leave bound
        self.side = None        # +1 to pass left of the direction to q, -1 to pass right
        self.t_last = None      # the last wall tangent, kept when no wall is in view
        self.steps = 0
        self.flipped = False
        self.detour_steps = 0
        self.detours = []

    # -- interface
    def reference(self, p):
        p = np.asarray(p, dtype=float).reshape(2)
        gamma = self.inner.reference(p)              # always: the follower's state as baseline
        self.n_steps += 1
        self.map.update(p, self.ranges)
        if self.state is None:
            self.hist.append(p.copy())
            trig = self._trigger()
            if trig is not None:
                self._start(p, *trig)
        if self.state is not None:
            g = self._detour(p)
            if g is not None:
                self.detour_steps += 1
                return g
        return gamma

    @property
    def active(self):
        return self.state is not None

    def new_cells(self):
        return self.map.cells()

    def summary(self):
        return {"attached": True, "n_detours": len(self.detours),
                "detour_steps": self.detour_steps,
                "n_new_cells": int(self.map.new.sum()), "detours": list(self.detours)}

    # -- triggers
    def blocked_stretch(self, progress, floor=0.0):
        """(first, last) arc of the run of path points within NEAR of a new cell whose first
        point is within AHEAD of the follower's `progress`, or None. Only arc at or past `floor`
        is considered, so a finished detour's stretch is not found again (readings 3 and 4)."""
        s0 = max(progress, floor)
        if not self.map.new.any() or s0 >= self.line.total:
            return None
        s = np.arange(s0, self.line.total + PATH_STEP / 2, PATH_STEP)
        close = self.map.distance(self.line.at(s)) <= NEAR
        first = np.flatnonzero(close & (s <= progress + AHEAD))
        if not len(first):
            return None
        i = j = int(first[0])
        while j + 1 < len(s) and close[j + 1]:
            j += 1
        return float(s[i]), float(s[j])

    def _trigger(self):
        stretch = self.blocked_stretch(self.inner.progress, self.trigger_floor)
        if stretch is not None:
            return "new_obstacle", min(stretch[1] + REJOIN_PAST, self.line.total)
        if (self.cfg.stall and len(self.hist) == self.hist.maxlen
                and np.linalg.norm(self.hist[-1] - self.hist[0]) < STALL_DIST):
            return "stall", min(self.inner.progress + STALL_Q, self.line.total)
        return None

    def _start(self, p, trigger, s_q):
        self.s_q = float(s_q)
        self.q = self.line.at(self.s_q)
        self.d0 = float(np.linalg.norm(p - self.q))
        self.side = self.pick_side(p, self.q)
        self.state = "wall"
        self.steps = 0
        self.flipped = False
        self.t_last = None
        self.detours.append({"start": self.n_steps, "trigger": trigger, "s_q": self.s_q,
                             "q": [float(v) for v in self.q], "d0": self.d0,
                             "side": "left" if self.side > 0 else "right",
                             "end": None, "reason": None})

    def pick_side(self, p, q):
        """+1 (pass on the left of the direction to q) or -1: the side whose rays 30-90 degrees
        from that direction have the larger minimum range; a tie goes left (reading 5)."""
        to_q, (lo, hi) = q - p, SIDE_RAYS
        left = _ray_window(self.ranges, to_q, lo, hi, n_rays=self.n_rays)
        right = _ray_window(self.ranges, to_q, -hi, -lo, n_rays=self.n_rays)
        return 1 if left >= right else -1

    # -- the detour itself
    def _end(self, reason):
        """End the detour. Either way its rejoin point is where later triggers look from, so a
        detour that gave up is not started again on the same stretch (reading 17)."""
        self.detours[-1].update(end=self.n_steps, reason=reason)
        self.trigger_floor = max(self.trigger_floor, self.s_q)
        self.state = None
        self.hist.clear()

    def _detour(self, p):
        """This step's gamma during a detour, or None once it has ended (reading 8)."""
        self.steps += 1
        if self.steps > GIVE_UP_STEPS:
            self._end("gave_up")
            return None
        if self.state == "wall":
            if self.steps > FLIP_STEPS and not self.flipped:
                self.side, self.flipped = -self.side, True
                self.detours[-1]["flipped_at"] = self.n_steps
            if self.can_leave(p):
                self.state = "leave"
        if self.state == "leave":
            if np.linalg.norm(p - self.q) < ARRIVE:
                self._end("rejoined")
                return None
            return self.q.copy()
        return self.wall_gamma(p)

    def can_leave(self, p):
        """The Bug2 rule: a clear line of sight to q, and closer to q than at the start."""
        return (np.linalg.norm(p - self.q) < self.d0
                and self.map.segment_clear(p, self.q))

    def wall_gamma(self, p):
        """gamma = p + L t + (d - 0.6)(-n) for the nearest hit on the known map or a new cell.
        With no such hit, the last tangent is kept, or q before there is one (reading 7)."""
        hits = self.map.hits[self.map.wall]
        if not len(hits):
            return p + self.cfg.L * self.t_last if self.t_last is not None else self.q.copy()
        d = np.linalg.norm(hits - p[None, :], axis=1)
        k = int(np.argmin(d))
        n = (p - hits[k]) / d[k]
        t = _rot(n, self.side)
        self.t_last = t
        return p + self.cfg.L * t - (d[k] - WALL_GAP) * n


# ------------------------------------------------------------------ candidate 2: yield
class YieldLayer(_Layer):
    """Step aside for an approaching pedestrian (candidate 2).

    `tracker` is the frozen LiDAR velocity tracker of the policy's barrier source, read only:
    the layer takes a track's `position`, `velocity` and `confirmed` and calls nothing that
    advances it. `planner` is the policy's frozen A* planner, used only as the known static map
    at 0.3 m clearance (its occupancy grid is the map inflated by the robot radius).
    """

    def __init__(self, inner, tracker, planner, cfg, *, n_rays=24, lidar_range=5.0):
        super().__init__(inner, n_rays=n_rays, lidar_range=lidar_range)
        self.tracker = tracker
        self.planner = planner
        self.cfg = cfg
        self.n_yield_steps = 0
        self.n_no_side = 0

    # -- interface
    def reference(self, p):
        p = np.asarray(p, dtype=float).reshape(2)
        gamma = self.inner.reference(p)          # always: the layer below keeps its own state
        self.n_steps += 1
        threat = self.threat(p)
        if threat is None:
            return gamma
        side = self.side_point(p, np.asarray(threat.velocity, float).reshape(2))
        if side is None:
            self.n_no_side += 1
            return gamma
        self.n_yield_steps += 1
        return side

    def summary(self):
        out = {"attached": True, "n_yield_steps": self.n_yield_steps,
               "n_no_side": self.n_no_side}
        if hasattr(self.inner, "summary"):
            out.update(self.inner.summary())
        return out

    # -- the rule
    def threat(self, p):
        """The confirmed track whose constant-velocity closest approach to p within H seconds is
        the earliest one under D, or None. Pure: it reads the tracker and changes nothing."""
        best = None
        for tr in self.tracker.tracks:
            if not tr.confirmed:
                continue
            v = np.asarray(tr.velocity, float).reshape(2)
            speed = float(np.linalg.norm(v))
            if speed < MIN_SPEED:
                continue
            r = np.asarray(tr.position, float).reshape(2) - p
            t = float(np.clip(-(r @ v) / speed ** 2, 0.0, self.cfg.H))
            if np.linalg.norm(r + t * v) < self.cfg.D and (best is None or t < best[0]):
                best = (t, tr)
        return None if best is None else best[1]

    def free(self, q):
        """q is free on the known static map at 0.3 m clearance.

        The grid is the frozen planner's own occupancy, which is the known map inflated by the
        robot radius (reading 12). The cell index repeats ShortestPathOracle._index rather than
        calling it, so this reads only the planner's public grid."""
        if (q < 0).any() or (q >= self.planner.world_size).any():
            return False
        n = self.planner.resolution
        iy = min(int(q[1] / self.planner.cell), n - 1)
        ix = min(int(q[0] / self.planner.cell), n - 1)
        return not self.planner.occupancy[(iy, ix)]

    def side_point(self, p, v):
        """p + L n on the side with more LiDAR room, or the other side if that point is not
        free on the known map, or None if neither is (readings 5 and 12)."""
        n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        first = n if self._room(n) >= self._room(-n) else -n
        for d in (first, -first):
            q = p + self.cfg.L * d
            if self.free(q):
                return q
        return None

    def _room(self, d):
        """The smallest LiDAR range among the rays within 45 degrees of the direction d."""
        return _ray_window(self.ranges, d, -YIELD_RAYS, YIELD_RAYS, n_rays=self.n_rays)


# ------------------------------------------------------------------ attaching to the policy
#: The summary of an episode in which no layer could be attached (the planner found no path, so
#: there is no follower to wrap and gamma is the frozen goal; reading 9).
NO_LAYER = {"attached": False}


def attach(arm, pol, cfg):
    """Install `arm`'s layer(s) as the policy's follower, AFTER pol.reset(). Returns the
    outermost layer, or None when the planner failed and there is no follower to wrap."""
    check_arm(arm)
    if pol.follower is None:
        return None
    kw = dict(world_size=pol.world_size, lidar_inset=pol.agent_radius, n_rays=pol.n_rays,
              lidar_range=pol.lidar_range)
    layer = pol.follower
    if arm in ("detour", "detour_yield"):
        detour = cfg.detour if arm == "detour_yield" else cfg
        layer = DetourLayer(layer, pol.planner.static_obstacles, detour, **kw)
    if arm in ("yield", "detour_yield"):
        yld = cfg.yield_ if arm == "detour_yield" else cfg
        layer = YieldLayer(layer, pol.src.tracker, pol.planner, yld,
                           n_rays=pol.n_rays, lidar_range=pol.lidar_range)
    pol.follower = layer
    return layer
