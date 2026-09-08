"""Week5-Phase7 / Stage 3 (+ Stage 5 arm): DRCBFPolicy with a swappable barrier source.
NEW `[extension]`.

`dr_control/policy.py` is FROZEN and is not edited. This subclass changes exactly one thing --
which barrier source `reset()` builds -- and inherits `predict()` unchanged, so the control
logic, the observation slicing (nothing past index 28), the planner, the carrot follower and
the u = 0 fallback are all the frozen ones.
"""
from __future__ import annotations

import numpy as np

from dr_control.capped_velocity import (OracleVelocitySource, ProjectionCappedSource,
                                        V_CAP_DERIVED, VectorCappedSource)
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.policy import DRCBFPolicy
from dr_control.tracking2 import GeometricLidarVelocityTracker
from dr_control.velocity_tracker import LidarVelocityTracker

#: Stage 5 (11.9.9) adds ONE arm: E1_geometric_R = T1 + the geometry-derived measurement
#: covariance. It is T1 in every other respect -- same v_cap, same source class, same frozen
#: u = 0 fallback -- so A0 ("T1_projection_cap") and A1 ("E1_geometric_R") differ by exactly
#: the tracker's R. Existing arms are untouched and keep the frozen tracker.
ARMS = ("C0_frozen", "T1_projection_cap", "T2_vector_cap", "D_oracle", "E1_geometric_R")


class Phase7Policy(DRCBFPolicy):
    """C0 / T1 / T2 / D-oracle, selected by `arm`. C0 is byte-for-byte the frozen behaviour."""

    def __init__(self, *, arm="C0_frozen", v_cap=V_CAP_DERIVED, **kwargs):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        super().__init__(**kwargs)
        self.arm = arm
        self.v_cap = float(v_cap)

    def reset(self, obs, ego_pose):
        out = super().reset(obs, ego_pose)            # frozen: plans, builds the frozen source
        if self.arm == "C0_frozen":
            return out
        tk = (GeometricLidarVelocityTracker if self.arm == "E1_geometric_R"
              else LidarVelocityTracker)
        tracker = tk(r_nominal=self.r_nominal, dt=self.dt,
                     n_rays=self.n_rays, lidar_range=self.lidar_range)
        common = dict(r_robot=self.agent_radius, k_scans=5, n_rays=self.n_rays,
                      lidar_range=self.lidar_range, tracker=tracker)
        if self.arm in ("T1_projection_cap", "E1_geometric_R"):
            self.src = ProjectionCappedSource(v_cap=self.v_cap, **common)
        elif self.arm == "T2_vector_cap":
            self.src = VectorCappedSource(v_cap=self.v_cap, **common)
        else:
            self.src = OracleVelocitySource(**common)
        return out

    @property
    def n_clamped_steps(self):
        return int(getattr(self.src, "n_clamped_steps", 0))

    @property
    def n_clamped_rows(self):
        return int(getattr(self.src, "n_clamped_rows", 0))
