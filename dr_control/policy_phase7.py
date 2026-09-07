"""Week5-Phase7 / Stage 3: DRCBFPolicy with a swappable barrier source. NEW `[extension]`.

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
from dr_control.velocity_tracker import LidarVelocityTracker

ARMS = ("C0_frozen", "T1_projection_cap", "T2_vector_cap", "D_oracle")


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
        tracker = LidarVelocityTracker(r_nominal=self.r_nominal, dt=self.dt,
                                       n_rays=self.n_rays, lidar_range=self.lidar_range)
        common = dict(r_robot=self.agent_radius, k_scans=5, n_rays=self.n_rays,
                      lidar_range=self.lidar_range, tracker=tracker)
        if self.arm == "T1_projection_cap":
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
