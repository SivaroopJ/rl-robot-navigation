"""
Start/goal sampling for RobotNavEnv.

WHY THIS EXISTS
---------------
The original scheme sampled both points uniformly over the free arena with a 2.0-unit
minimum separation (`RobotNavEnv._random_free_position`). Measured over 20k draws on
the shipped 10x10 layout that gives mean separation 5.21, median 4.92, p10 2.81, and
**31.7% of pairs closer than 4.0 units**. Because the agent covers up to 1.41 units per
step at dt=1.0, that produced the 4-6 step episodes recorded in every results/e*/ file
(e1 N=6: min 1, max 17, mean 5.1, zero timeouts out of 200).

Two Week-4 requirements fail at that episode length. An Ornstein-Uhlenbeck steering
process never leaves its initial transient in five steps, so a "randomized" obstacle is
indistinguishable from the constant-heading one it replaced. And Sibling Rivalry's
siblings diverge too little for its acceptance rule to ever bind, so it degenerates to
plain PPO.

WHAT THIS DOES INSTEAD
----------------------
Stratifies the pair over two axes at once:

  REGION   The free span is cut into a region_grid x region_grid lattice. A region is
           usable when its free-area fraction exceeds region_free_threshold. The start
           region is drawn uniformly over usable regions, so every part of the map is
           trained on rather than left to the tails of a uniform draw. On the shipped
           layout at K=4, 15 of 16 regions qualify; the one dropped is 16% free (it is
           mostly filled by the lower-left bar). K=3 gives 9/9 and K=5 gives 20/25.

  DISTANCE A band is drawn uniformly from distance_bins, then the goal is rejection
           sampled to land inside it. This puts EQUAL MASS in each band instead of the
           uniform-square distribution's peak near its mean, which is what makes the
           start-goal distance an experimental variable: every metric can then be
           reported per band, and "does SR help on the long crossings specifically?"
           becomes answerable rather than averaged away.

The two axes are drawn independently and the goal region is then constrained by the
band, so region coverage and distance coverage hold simultaneously.

WHAT IS DELIBERATELY NOT CHANGED
--------------------------------
Validity. `_is_free` is the same predicate `RobotNavEnv._random_free_position` applies:
inside the wall margin, and outside every static rectangle inflated by that margin. A
point this sampler returns would have been acceptable to the original code; only the
DISTRIBUTION over acceptable points differs.

`sampling="uniform"` does not route through this class at all -- the env keeps its
original call into `_random_free_position`, so the legacy path stays bit-identical
rather than merely equivalent. See `RobotNavEnv.reset`.

FALLBACKS ARE COUNTED, NOT SILENT
---------------------------------
Rejection sampling can fail: a band may be unreachable from a particular start point in
a particular region. Rather than loop forever or silently return a bad pair, the sampler
widens in two documented stages and increments `n_fallback`. The tests assert that rate
stays under 1%, so a future layout change that makes some band routinely infeasible
fails loudly instead of quietly reshaping the task distribution.
"""
from __future__ import annotations

import numpy as np


class StartGoalSampler:
    """Draws (start, goal) pairs stratified over map region and separation distance."""

    def __init__(self, world_size, wall_margin, static_obstacles, *, region_grid=4,
                 region_free_threshold=0.25, distance_bins=None, min_separation=4.0,
                 max_attempts=200, free_probe_resolution=25):
        """
        Parameters
        world_size: float
            Side length of the square arena.
        wall_margin: float
            Keep-out distance from the world boundary and from every static rectangle.
            Must be the same value `RobotNavEnv._random_free_position` uses, or this
            sampler and the env would disagree about which points are legal.
        static_obstacles: list of (cx, cy, half_w, half_h)
        region_grid: int
            Lattice resolution K. The free span is cut into K x K regions.
        region_free_threshold: float
            Minimum free-area fraction for a region to be drawn from.
        distance_bins: list of (low, high) or None
            Separation bands, drawn uniformly. None selects a default spanning
            min_separation to the arena's reachable maximum in four equal bands.
        min_separation: float
            Hard floor on separation, enforced even by the last-resort fallback.
        max_attempts: int
            Rejection-sampling budget per stage.
        free_probe_resolution: int
            Grid resolution used once at construction to measure each region's free area.
        """
        self.world_size = float(world_size)
        self.wall_margin = float(wall_margin)
        self.static_obstacles = [tuple(float(v) for v in o) for o in static_obstacles]
        self.region_grid = int(region_grid)
        self.region_free_threshold = float(region_free_threshold)
        self.min_separation = float(min_separation)
        self.max_attempts = int(max_attempts)

        self.low = self.wall_margin
        self.high = self.world_size - self.wall_margin
        if self.high <= self.low:
            raise ValueError(
                f"wall_margin {self.wall_margin} leaves no free span in a "
                f"{self.world_size} arena"
            )
        #: Largest separation two legal points could have (corner to corner).
        self.max_separation = float((self.high - self.low) * np.sqrt(2.0))

        if distance_bins is None:
            edges = np.linspace(self.min_separation, self.max_separation, 5)
            distance_bins = [(edges[i], edges[i + 1]) for i in range(4)]
        self.distance_bins = [(float(a), float(b)) for a, b in distance_bins]
        for lo, hi in self.distance_bins:
            if lo >= hi:
                raise ValueError(f"distance bin ({lo}, {hi}) is empty or inverted")
            if lo < self.min_separation:
                raise ValueError(
                    f"distance bin ({lo}, {hi}) starts below min_separation "
                    f"{self.min_separation}"
                )
            if hi > self.max_separation + 1e-9:
                raise ValueError(
                    f"distance bin ({lo}, {hi}) exceeds the reachable maximum "
                    f"{self.max_separation:.2f} for a {self.world_size} arena with "
                    f"wall_margin {self.wall_margin}"
                )

        self._edges = np.linspace(self.low, self.high, self.region_grid + 1)
        self.region_boxes, self.region_free_fraction = self._measure_regions(
            free_probe_resolution
        )
        self.usable_regions = [
            i for i, f in enumerate(self.region_free_fraction)
            if f > self.region_free_threshold
        ]
        if not self.usable_regions:
            raise ValueError(
                f"no region of the {self.region_grid}x{self.region_grid} lattice is more "
                f"than {self.region_free_threshold:.0%} free; the layout and the grid "
                "resolution are incompatible"
            )

        # Which (start region, goal region) pairs can satisfy each band. Precomputed
        # because the alternative -- draw a start region uniformly, then hope the band is
        # reachable -- fails badly: a mid-map start cannot reach the far band at all, so
        # 25% of draws fell through to the fallback and the bands came out 42/31/20/8
        # instead of 25/25/25/25. Restricting the start region to those that CAN satisfy
        # the drawn band is what makes the two strata independent.
        self.band_starts = []
        self.band_goals = []
        for low, high in self.distance_bins:
            goals_by_start = {}
            for a in self.usable_regions:
                reachable = []
                for b in self.usable_regions:
                    if b == a:
                        continue
                    d_min, d_max = self._box_pair_distance_range(
                        self.region_boxes[a], self.region_boxes[b]
                    )
                    if d_min <= high and d_max >= low:
                        reachable.append(b)
                if reachable:
                    goals_by_start[a] = reachable
            if not goals_by_start:
                raise ValueError(
                    f"distance bin ({low}, {high}) is unreachable: no pair of usable "
                    f"regions in the {self.region_grid}x{self.region_grid} lattice can be "
                    "that far apart. Widen the bin or lower region_grid."
                )
            self.band_starts.append(sorted(goals_by_start))
            self.band_goals.append(goals_by_start)

        # Sampling the start region uniformly WITHIN each band leaves the marginal over
        # regions badly uneven: a corner region is a feasible start for every band, a
        # central one only for the near bands, so corners get drawn ~2.5x as often
        # (measured min 132 against an expectation of 333 over 5000 draws). Iterative
        # proportional fitting rebalances the within-band weights so the marginal over
        # regions comes out uniform while each band still sums to one -- i.e. the two
        # strata hold simultaneously instead of trading off.
        self.band_start_weights = self._balance_start_weights()

        # The same imbalance afflicts the GOAL marginal, and more severely: picking
        # uniformly among the candidate goal regions left the worst region at 0.28 of its
        # uniform share. A second fitting pass, conditioned on (band, start region), pulls
        # the goal marginal level too. Both passes are static -- computed once here, never
        # updated from sampling history -- so episodes stay independent and a given seed
        # still reproduces a given pair.
        self.band_goal_weights = self._balance_goal_weights()

        #: Incremented whenever the primary rejection sampler could not satisfy the band.
        self.n_sampled = 0
        self.n_fallback = 0

    def _balance_start_weights(self, iterations=200, tol=1e-9):
        """Within-band start-region weights whose marginal over regions is uniform.

        Iterative proportional fitting: alternately rescale the weights so every region's
        marginal matches the uniform target, then renormalize each band back to a
        probability distribution. Both constraints are satisfiable together here because
        every usable region is a feasible start for at least one band, so the fitting
        converges rather than oscillating.

        Returns
        list of np.ndarray
            One weight vector per band, aligned with `self.band_starts[band]`.
        """
        n_bands = len(self.band_starts)
        weights = [np.full(len(s), 1.0 / len(s)) for s in self.band_starts]
        target = 1.0 / len(self.usable_regions)

        for _ in range(iterations):
            marginal = {r: 0.0 for r in self.usable_regions}
            for band in range(n_bands):
                for k, region in enumerate(self.band_starts[band]):
                    marginal[region] += weights[band][k] / n_bands

            worst = max(abs(m - target) for m in marginal.values())
            if worst < tol:
                break

            for band in range(n_bands):
                for k, region in enumerate(self.band_starts[band]):
                    if marginal[region] > 0.0:
                        weights[band][k] *= target / marginal[region]
                total = weights[band].sum()
                if total > 0.0:
                    weights[band] /= total

        return weights

    def _balance_goal_weights(self, iterations=200, tol=1e-9):
        """Goal-region weights, per (band, start region), with a uniform goal marginal.

        Same iterative proportional fitting as `_balance_start_weights`, but over the
        conditional P(goal | band, start), holding the already-fitted P(band) and
        P(start | band) fixed. Returns a dict keyed (band, start_region) -> weight vector
        aligned with `self.band_goals[band][start_region]`.
        """
        target = 1.0 / len(self.usable_regions)
        n_bands = len(self.band_starts)
        weights = {}
        context = []          # (band, start region, P(band) * P(start | band))
        for band in range(n_bands):
            for k, a in enumerate(self.band_starts[band]):
                cands = self.band_goals[band][a]
                weights[(band, a)] = np.full(len(cands), 1.0 / len(cands))
                context.append((band, a, self.band_start_weights[band][k] / n_bands))

        for _ in range(iterations):
            marginal = {r: 0.0 for r in self.usable_regions}
            for band, a, mass in context:
                for k, g in enumerate(self.band_goals[band][a]):
                    marginal[g] += mass * weights[(band, a)][k]

            if max(abs(m - target) for m in marginal.values()) < tol:
                break

            for band, a, _mass in context:
                w = weights[(band, a)]
                for k, g in enumerate(self.band_goals[band][a]):
                    if marginal[g] > 0.0:
                        w[k] *= target / marginal[g]
                total = w.sum()
                if total > 0.0:
                    weights[(band, a)] = w / total

        return weights

    # ------------------------------------------------------------------ geometry
    def _is_free(self, point):
        """The same validity predicate `RobotNavEnv._random_free_position` applies."""
        x, y = float(point[0]), float(point[1])
        if not (self.low <= x <= self.high and self.low <= y <= self.high):
            return False
        for cx, cy, hw, hh in self.static_obstacles:
            if abs(x - cx) < hw + self.wall_margin and abs(y - cy) < hh + self.wall_margin:
                return False
        return True

    def _measure_regions(self, resolution):
        """Free-area fraction of every region, measured once on a probe grid."""
        boxes, fractions = [], []
        for row in range(self.region_grid):
            for col in range(self.region_grid):
                x0, x1 = self._edges[col], self._edges[col + 1]
                y0, y1 = self._edges[row], self._edges[row + 1]
                boxes.append((x0, x1, y0, y1))
                xs = np.linspace(x0, x1, resolution)
                ys = np.linspace(y0, y1, resolution)
                free = sum(
                    1 for y in ys for x in xs if self._is_free((x, y))
                )
                fractions.append(free / (resolution * resolution))
        return boxes, fractions

    def region_index_of(self, point):
        """Region containing a point, or None when it lies outside the free span.

        Used by the coverage tests and the diagnostic heatmaps.
        """
        x, y = float(point[0]), float(point[1])
        if not (self.low <= x <= self.high and self.low <= y <= self.high):
            return None
        col = min(int((x - self.low) / (self.high - self.low) * self.region_grid),
                  self.region_grid - 1)
        row = min(int((y - self.low) / (self.high - self.low) * self.region_grid),
                  self.region_grid - 1)
        return row * self.region_grid + col

    @staticmethod
    def _box_distance_range(point, box):
        """(min, max) Euclidean distance from a point to anywhere in an axis-aligned box.

        The minimum is the distance to the box's nearest point (zero when inside); the
        maximum is always attained at one of the four corners. Used to discard goal
        regions that cannot possibly satisfy the requested band, so rejection sampling
        is not spent on them.
        """
        x, y = float(point[0]), float(point[1])
        x0, x1, y0, y1 = box
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        d_min = float(np.hypot(dx, dy))
        d_max = float(max(
            np.hypot(x - cx, y - cy)
            for cx in (x0, x1) for cy in (y0, y1)
        ))
        return d_min, d_max

    @staticmethod
    def _box_pair_distance_range(box_a, box_b):
        """(min, max) Euclidean distance between any point of one box and any of another.

        The minimum separates the boxes on each axis independently; the maximum is always
        attained at opposite corners. Used once at construction to decide which region
        pairs can serve which distance band.
        """
        ax0, ax1, ay0, ay1 = box_a
        bx0, bx1, by0, by1 = box_b
        dx = max(0.0, bx0 - ax1, ax0 - bx1)
        dy = max(0.0, by0 - ay1, ay0 - by1)
        d_min = float(np.hypot(dx, dy))
        d_max = float(np.hypot(max(abs(ax1 - bx0), abs(bx1 - ax0)),
                               max(abs(ay1 - by0), abs(by1 - ay0))))
        return d_min, d_max

    # ------------------------------------------------------------------ sampling
    def _sample_in_box(self, np_random, box):
        """A free point inside one region, or None if the budget is exhausted."""
        x0, x1, y0, y1 = box
        for _ in range(self.max_attempts):
            candidate = np.array([np_random.uniform(x0, x1), np_random.uniform(y0, y1)],
                                 dtype=np.float32)
            if self._is_free(candidate):
                return candidate
        return None

    def _sample_anywhere(self, np_random):
        """A free point anywhere in the arena, or None if the budget is exhausted."""
        for _ in range(self.max_attempts):
            candidate = np_random.uniform(self.low, self.high, size=2).astype(np.float32)
            if self._is_free(candidate):
                return candidate
        return None

    def sample(self, np_random):
        """Draw one (start, goal) pair, stratified over region and separation band.

        Parameters
        np_random: np.random.Generator
            The env's own generator, so the seeding contract is unchanged: the same seed
            reproduces the same pair.

        Returns
        start, goal: np.ndarray (2,) float32
        """
        self.n_sampled += 1
        band = int(np_random.integers(len(self.distance_bins)))
        low, high = self.distance_bins[band]
        starts = self.band_starts[band]
        weights = self.band_start_weights[band]
        goals_by_start = self.band_goals[band]

        # Each outer attempt commits to a start point and tries to place a goal in the
        # band from it. A start near its region's edge can fail even though its region is
        # feasible, so a failure re-draws the START too rather than burning the whole
        # budget on one unlucky point.
        for _ in range(self.max_attempts):
            start_region = starts[int(np_random.choice(len(starts), p=weights))]
            start = self._sample_in_box(np_random, self.region_boxes[start_region])
            if start is None:
                continue

            # Narrow the precomputed region list using the actual point: a region feasible
            # for the region-pair may still be out of band for this particular start.
            region_list = goals_by_start[start_region]
            goal_weights = self.band_goal_weights[(band, start_region)]
            keep = [
                k for k, r in enumerate(region_list)
                if (lambda rng_: rng_[0] <= high and rng_[1] >= low)(
                    self._box_distance_range(start, self.region_boxes[r])
                )
            ]
            if not keep:
                continue
            # Renormalize the fitted weights over the regions that survived narrowing by
            # the actual start point. A degenerate all-zero slice falls back to uniform.
            w = np.array([goal_weights[k] for k in keep], dtype=float)
            w = np.full(len(keep), 1.0 / len(keep)) if w.sum() <= 0.0 else w / w.sum()
            order = np_random.choice(len(keep), size=len(keep), replace=False, p=w)
            for idx in order:
                goal = self._sample_in_box(
                    np_random, self.region_boxes[region_list[keep[int(idx)]]]
                )
                if goal is None:
                    continue
                if low <= float(np.linalg.norm(goal - start)) <= high:
                    return start, goal

        # Fallback: band enforced, region strata abandoned. Counted, never silent.
        self.n_fallback += 1
        start = self._sample_anywhere(np_random)
        if start is None:
            raise RuntimeError(
                "could not place a start point: no free position found in "
                f"{self.max_attempts} attempts anywhere in the arena"
            )
        for _ in range(self.max_attempts):
            goal = self._sample_anywhere(np_random)
            if goal is not None and low <= float(np.linalg.norm(goal - start)) <= high:
                return start, goal

        # Last resort: band abandoned too, min_separation still enforced.
        for _ in range(self.max_attempts):
            goal = self._sample_anywhere(np_random)
            if goal is not None and float(np.linalg.norm(goal - start)) >= self.min_separation:
                return start, goal

        raise RuntimeError(
            f"could not place a goal at least {self.min_separation} from {start} in "
            f"{3 * self.max_attempts} attempts; the layout cannot support the configured "
            "min_separation"
        )

    @property
    def fallback_rate(self):
        """Share of pairs that needed stage 2 or later. Asserted low by the tests."""
        return 0.0 if self.n_sampled == 0 else self.n_fallback / self.n_sampled
