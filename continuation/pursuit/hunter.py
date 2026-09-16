"""Hunter: pursuit nominal action + the same CLF-DR-CBF safety filter, no recovery.

NOMINAL ACTION (plan section 2)
    u_nom = max_speed_h * unit(p_protagonist - p_hunter), computed from the SHARED PRE-STEP state.
    This is the frozen `nominal_action(p, gamma, max_v)` with gamma = p_protagonist, so no new
    nominal law is written here.

SAFETY FILTER
    The frozen `ClfCbfDrccpController` with the tuned hyperparameters (audit section 7), CLF
    reference gamma = p_protagonist, and barrier rows from the hunter's OWN 24-ray LiDAR and its own
    velocity tracker / 5-scan buffer. The protagonist is the TARGET, never a barrier row: the hunter
    is not asked to avoid what it is chasing.

RECOVERY
    None. On a non-optimal solve the frozen fallback u = 0 applies, exactly as the frozen controller
    does. No LADDER, and no Random recovery: whether the protagonist's acceptance rule is meaningful
    for a pursuit objective is a separate, documented ablation (audit section 4.2).
"""
from __future__ import annotations

import numpy as np

from continuation.random_recovery import margins
from dr_control.drccp_controller import ClfCbfDrccpController, build_xi, nominal_action
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.velocity_tracker import LidarVelocityTracker


class HunterController:
    """One instance per episode, mirroring how the protagonist policy is built."""

    def __init__(self, *, params, max_speed=1.0, radius=0.3, n_rays=24, lidar_range=5.0,
                 dt=0.1, r_nominal=0.3, k_scans=5, filter_radius=None, protag_radius=0.3):
        """`filter_radius=None` is the audited hunter. A float selects `ProtagFilteredSource`
        (Baseline 1.5 / Predict 2.5): candidates within it of the reference passed to `act` are
        dropped from the barrier rows."""
        self.params = params
        self.max_speed = float(max_speed)
        self.radius = float(radius)
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.tau_eff = params.tau_eff()
        self.ctrl = ClfCbfDrccpController(max_v=self.max_speed, cbf_rate=params.alpha,
                                          clf_rate=params.clf_rate,
                                          wasserstein_r=params.wasserstein_r,
                                          epsilon=params.epsilon, k_v=params.k_v)
        tracker = LidarVelocityTracker(r_nominal=r_nominal, dt=dt, n_rays=self.n_rays,
                                       lidar_range=self.lidar_range)
        kw = dict(r_robot=self.radius, k_scans=int(k_scans), n_rays=self.n_rays,
                  lidar_range=self.lidar_range, tracker=tracker)
        if filter_radius is None:
            self.src = EstimatedLidarBarrierSource(**kw)
        else:
            from continuation.pursuit.hunter_policies import ProtagFilteredSource
            self.src = ProtagFilteredSource(filter_radius=filter_radius,
                                            protag_radius=protag_radius, **kw)
        self.filtering = filter_radius is not None
        self.reset()

    def reset(self):
        self.ctrl.reset()
        self.src.reset()
        self.n_steps = 0
        self.n_infeasible = 0
        self.n_solver_fail = 0
        self.n_fallback = 0
        #: A REAL intervention: the nominal pursuit action itself violates the DR constraint,
        #: i.e. min_i CBC_i(u_nom) < tau_eff, so the filter had to move off it for safety.
        #: (Counting any u != u_nom would be meaningless: the frozen objective also carries a
        #: smoothness term, so the two essentially never coincide exactly.)
        self.n_nominal_unsafe = 0
        self.n_materially_filtered = 0     # ||u - u_nom|| > 0.05 m/s
        self.u_dev = []
        self.solve_times = []
        self.last_step_infeasible = False
        self.n_no_barrier_rows = 0
        return self

    def act(self, p_hunter, ranges, p_target, *, filter_reference=None, nominal=None):
        """(u, record) in m/s. All inputs come from the shared pre-step state.

        `filter_reference`: Protag's CURRENT true position, used only by a filtering hunter to
        drop Protag's returns from the barrier rows. It is never the pursuit target.
        `nominal`: optional callable (p, src) -> (gamma, u_nom, info), called AFTER this step's
        scan is pushed (MPC pursuit, policies 3/3.5). None keeps the audited behaviour:
        gamma = p_target, u_nom = nominal_action(p, gamma, max_speed). The QP runs either way.
        """
        p = np.asarray(p_hunter, dtype=float).reshape(2)
        gamma = None if p_target is None else np.asarray(p_target, dtype=float).reshape(2)
        ranges = np.asarray(ranges, dtype=float).reshape(-1)
        if self.filtering:
            if filter_reference is None:
                raise ValueError("a filtering hunter needs filter_reference every step")
            self.src.push(p, ranges, exclude_center=filter_reference)
        else:
            if filter_reference is not None:
                raise ValueError("filter_reference given to a non-filtering hunter")
            self.src.push(p, ranges)
        if not self.src.buffer:
            # No surface point in any buffered scan (cannot happen on M0, where some wall is
            # always within 5 m, but a filter could in principle empty every scan). The frozen
            # source would raise; apply the frozen failure behaviour, u = 0, and count it.
            self.n_steps += 1
            self.n_no_barrier_rows += 1
            self.n_fallback += 1
            self.ctrl.prev_u = np.zeros(2)
            return np.zeros(2), {"status": "no_barrier_rows", "u_nom": np.zeros(2),
                                 "u_dev": 0.0}
        h, g, dd = self.src.samples(p)
        xi = build_xi(h, g, dd)
        nom_info = None
        if nominal is None:
            u_nom = nominal_action(p, gamma, self.max_speed)  # pursuit objective
        else:
            gamma, u_nom, nom_info = nominal(p, self.src)
            gamma = np.asarray(gamma, dtype=float).reshape(2)
            u_nom = np.asarray(u_nom, dtype=float).reshape(2)
        info = {}
        u = self.ctrl.generate_controller(p, gamma, xi, u_nom=u_nom, record=info)
        info["gamma"] = gamma
        if nom_info is not None:
            info["nominal_info"] = nom_info

        self.n_steps += 1
        status = str(info.get("status"))
        self.last_step_infeasible = "infeasible" in status
        if status != "optimal":
            self.n_solver_fail += 1
            self.n_fallback += 1                              # frozen u = 0 fallback
            if self.last_step_infeasible:
                self.n_infeasible += 1
        dev = float(np.linalg.norm(np.asarray(u, float) - u_nom))
        self.u_dev.append(dev)
        self.n_materially_filtered += int(dev > 0.05)
        kept = info.get("xi_kept")
        if kept is not None:
            m_nom = float(margins(kept, u_nom[None, :], self.ctrl.rateh)[0])
            info["m_nominal"] = m_nom
            self.n_nominal_unsafe += int(m_nom < self.tau_eff)
        t = info.get("total_time")
        if t is not None and t == t:
            self.solve_times.append(float(t))
        info["u_nom"] = u_nom
        info["u_dev"] = dev
        return np.asarray(u, dtype=float).reshape(2), info
