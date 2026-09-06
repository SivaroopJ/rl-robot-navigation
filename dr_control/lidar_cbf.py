"""Phase 2 -- LiDAR to CBF samples. Replaces the analytic barrier source of Phase 1.

INFORMATION LEDGER, ENFORCED BY CONSTRUCTION
--------------------------------------------
This module accepts a ranges vector and an ego pose. It never receives the environment, an
obstacle list, or an obstacle velocity, so ground truth about obstacles CANNOT leak into the
controller. Against the 52-d observation
`[dx, dy, vx, vy, l1..l24, (odx,ody,ovx,ovy) x 6]` that means:

    obs[0:2]    goal offset          USED  (via goal_from_observation)
    obs[2:4]    own velocity         available, unused in Phase 2
    obs[4:28]   24 LiDAR ranges      USED  (via ranges_from_observation)
    obs[28:52]  exact obstacle positions and velocities   NEVER READ
    ego pose    assumed known, as in the paper (Hector SLAM / P3D odometry)

`tests/test_dr_control_phase2.py::test_paired_obstacle_block_is_never_read` asserts that
perturbing obs[28:52] leaves the action bit-identical.

WHAT cast_rays ACTUALLY MEASURES -- VERIFIED, NOT ASSUMED
---------------------------------------------------------
Measured against analytic geometry (see the Phase 2 tests), the three surface families are
NOT treated alike by robot_env/lidar_core.py:

    surface      returned distance                  reading at env-collision threshold
    wall         robot centre -> INSET boundary     0.0     (boundary pre-inset by r_robot)
    rectangle    robot centre -> obstacle surface   r_robot
    circle       robot centre -> obstacle surface   r_robot

A LiDAR-only controller cannot tell which family a ray hit, so a single uniform subtraction
of `r_robot` is applied:

    h = range - r_robot

which is EXACT for rectangles and circles, and CONSERVATIVE by exactly r_robot for walls
(the robot is held r_robot further from a wall than strictly necessary). This is a deliberate
[adaptation]; the alternative -- guessing the family -- would need obstacle ground truth.

KNOWN SENSOR DEGENERACY  [measured, not repaired]
--------------------------------------------------
`cast_rays` requires t > 0, so when the robot sits exactly on the inset boundary (x = 9.7,
which RobotNavEnv's own position clamp makes exactly reachable) the wall-facing ray returns
lidar_range instead of 0 -- the sensor goes blind to the wall precisely at contact. Readings
approach 0 correctly for any x < 9.7 (1e-4 at x = 9.6999). We do not modify robot_env; this
is reported as a sensing limitation and counted by `n_boundary_degenerate`.

SAMPLE CONSTRUCTION, AND WHAT THE N SAMPLES MEAN HERE
------------------------------------------------------
Following the reference (clf_cbf_controller_node.py:289-312 with pre_process.py:170-180), a
rolling buffer of the last K scans is kept as world-frame surface points, and each scan
contributes ONE sample: its point closest to the CURRENT robot position. The reference
measures from the current position to buffered point clouds, not from the position at which
each scan was taken, and that is reproduced here.

This is scheme S-T of the plan (temporal samples). Note what it implies and what it does not:
with a noiseless sensor and static geometry the K samples still differ, because each buffered
scan saw a different set of surface points from a different vantage -- they are NOT identical.
But they are a lag window over a non-stationary process, not i.i.d. draws, so the Wasserstein
radius formula's i.i.d. premise does not hold for them. Scheme S-N (i.i.d. conditional
sensor-noise realizations) is a Phase 4 comparator.

A CONSEQUENCE WORTH STATING: each scan yields only its single nearest point, so the K samples
describe the ONE nearest obstacle at K instants, whereas Phase 1's analytic source produced a
distinct constraint per obstacle. Multi-obstacle coverage is therefore structurally weaker
here. That is the reference's design, not a porting artefact, and Phase 2 measures its cost.
"""
from __future__ import annotations

import numpy as np

from robot_env.lidar_core import ray_angles

N_RAYS = 24
LIDAR_RANGE = 5.0
WORLD_SIZE = 10.0


def ranges_from_observation(obs, *, n_rays=N_RAYS, lidar_range=LIDAR_RANGE):
    """Un-normalise obs[4:4+n_rays] back to metres. Reads no other slice."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    return obs[4:4 + n_rays] * lidar_range


def goal_from_observation(obs, p, *, world_size=WORLD_SIZE):
    """World-frame goal from the normalised offset obs[0:2] and the known ego pose."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    return np.asarray(p, dtype=float).reshape(2) + obs[0:2] * world_size


def surface_points(p, ranges, *, n_rays=N_RAYS, lidar_range=LIDAR_RANGE, miss_tol=1e-6):
    """World-frame endpoints of the rays that actually HIT something.

    A ray that hits nothing returns exactly `lidar_range`; those are dropped rather than
    turned into phantom surfaces at 5 m, which would fabricate obstacles in open space.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    ang = ray_angles(n_rays)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    hit = ranges < lidar_range - miss_tol
    return p[None, :] + ranges[hit, None] * dirs[hit], hit


class LidarBarrierSource:
    """Rolling buffer of K scans -> K barrier samples (scheme S-T).

    Phase 2 is static-only, so dh_dt is exactly 0: with a static world, h changes solely
    through the robot's own motion, which is the grad_h . u term, not the explicit partial
    derivative. Obstacle-velocity estimation is Phase 3/4 and is deliberately absent here.
    """

    def __init__(self, *, r_robot=0.3, k_scans=5, n_rays=N_RAYS, lidar_range=LIDAR_RANGE):
        self.r_robot = float(r_robot)
        self.k_scans = int(k_scans)
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.buffer = []                  # newest first, as in the reference
        self.n_boundary_degenerate = 0    # scans where every ray missed (see module docstring)
        self.n_empty_scans = 0

    def reset(self):
        self.buffer = []
        self.n_boundary_degenerate = 0
        self.n_empty_scans = 0

    def push(self, p, ranges):
        """Insert one scan as world-frame surface points; keep the newest k_scans."""
        pts, hit = surface_points(p, ranges, n_rays=self.n_rays, lidar_range=self.lidar_range)
        if len(pts) == 0:
            self.n_empty_scans += 1
            return
        self.buffer.insert(0, pts)
        if len(self.buffer) > self.k_scans:
            self.buffer.pop()

    def ready(self):
        return len(self.buffer) >= self.k_scans

    def samples(self, p):
        """(h, grad_h (2,N), dh_dt) -- one sample per buffered scan.

        Distance is measured from the CURRENT position p to each buffered point cloud, which
        is what the reference does; h is the clearance, radius already folded in.
        """
        p = np.asarray(p, dtype=float).reshape(2)
        h, grads = [], []
        for pts in self.buffer:
            d = np.linalg.norm(pts - p[None, :], axis=1)
            j = int(np.argmin(d))
            dist = float(d[j])
            if dist < 1e-12:
                # Robot coincident with a surface point: gradient undefined. Report maximal
                # violation with an arbitrary but unit direction rather than dividing by ~0.
                h.append(-self.r_robot)
                grads.append(np.array([1.0, 0.0]))
                continue
            h.append(dist - self.r_robot)
            grads.append((p - pts[j]) / dist)
        if not h:
            raise ValueError("no buffered scans; call push() before samples()")
        return np.array(h), np.column_stack(grads), np.zeros(len(h))

    def min_clearance_estimate(self, p):
        h, _, _ = self.samples(p)
        return float(np.min(h))
