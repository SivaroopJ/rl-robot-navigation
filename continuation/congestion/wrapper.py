"""CongestionWrapper: CD-random-CLF-DR-CBF = this wrapper INSIDE the frozen Random recovery.

INSTALLATION ORDER. The wrapper replaces `ctrl.generate_controller`; RandomRecovery must be
attached AFTERWARDS, so Random's `_orig` is this wrapper and the call chain per step is
    RandomRecovery._wrapped -> CongestionWrapper._wrapped -> frozen generate_controller.
Random recovery therefore still sees the frozen QP's status and rows and is unchanged.

PER CALL
    1. u0 = the nominal the controller would use (u_nom if given, else the frozen
       nominal_action(p, gamma, max_v) -- exactly what generate_controller computes itself).
    2. monitor along p + t u0 (t = 0, 0.25, 0.5 s) -> margin = M_look - tau_eff and predicted
       clearance (own SensedObstacles); detector state; diagnostic signals.
    3. mode "shadow": call the frozen QP with the ORIGINAL (gamma, u_nom) arguments, unchanged.
       mode "active", state s >= 1: scale k = speed_scale[s]; for angle a in (0, +-15, +-30 deg)
       evaluate M_look along p + t k R(a) u0, keep the best (strictly better than the earlier
       candidates, so ties keep the original direction); then
            u_nom' = k R(a) u0,     gamma' = p + k R(a) (gamma - p).
       Only the nominal and the CLF reference change. Constraints, the control box and the
       recovery are untouched; an already-infeasible QP is still handled by Random recovery.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from dr_control.drccp_controller import nominal_action

from continuation.congestion.detector import CongestionDetector, DetectorConfig
from continuation.congestion.monitor import FeasibilityMonitor
from continuation.congestion.signals import free_gap, track_signals
from continuation.sensed_obstacles import SensedObstacles

MODES = ("shadow", "active")


@dataclass(frozen=True)
class CDConfig:
    mode: str = "shadow"
    lookahead_s: tuple = (0.0, 0.25, 0.5)
    grid_n: int = 9
    speed_scale: tuple = (1.0, 0.6, 0.3)          # indexed by state
    angle_offsets_deg: tuple = (0.0, 15.0, -15.0, 30.0, -30.0)
    gap_pass_distance: float = 1.0
    track_radius: float = 2.0
    recent_recovery_window: int = 10
    exclude_radius: float = 0.35
    #: Congestion fixes (defaults reproduce the pre-registered detector exactly).
    clearance_source: str = "sensed"      # sensed | rows      (fix 1)
    lookahead_velocity: str = "nominal"   # nominal | executed (fix 2)
    track_gate: str = "confirmed"         # confirmed | strict (fix 3)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.clearance_source not in ("sensed", "rows"):
            raise ValueError("clearance_source must be sensed or rows")
        if self.lookahead_velocity not in ("nominal", "executed"):
            raise ValueError("lookahead_velocity must be nominal or executed")
        if self.track_gate not in ("confirmed", "strict"):
            raise ValueError("track_gate must be confirmed or strict")
        if self.angle_offsets_deg[0] != 0.0:
            raise ValueError("the original direction must be the first candidate (tie rule)")
        if max(abs(a) for a in self.angle_offsets_deg) > 30.0 + 1e-9:
            raise ValueError("direction bias is bounded by 30 degrees")

    def as_dict(self):
        d = asdict(self)
        d["detector"] = self.detector.as_dict()
        return d


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class CongestionWrapper:
    def __init__(self, ctrl, *, source_fn, cfg: CDConfig, tau, n_rays, lidar_range,
                 r_robot=0.3, r_obstacle=0.3):
        self.ctrl = ctrl
        self.source_fn = source_fn
        self.cfg = cfg
        self.n_rays = int(n_rays)
        self.lidar_range = float(lidar_range)
        self.r_robot = float(r_robot)
        self.r_obstacle = float(r_obstacle)
        self.monitor = FeasibilityMonitor(alpha=ctrl.rateh, tau=tau, max_v=ctrl.max_v,
                                          n_keep=ctrl.n_keep, grid_n=cfg.grid_n,
                                          lookahead=cfg.lookahead_s,
                                          strict_tracks=(cfg.track_gate == "strict"),
                                          clearance_mode=cfg.clearance_source)
        self.detector = CongestionDetector(cfg.detector)
        self.recovery = None          # set to the RandomRecovery wrapper once it is attached
        self.steps = []
        self._ctx = {}
        self._orig = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._orig

    def begin_step(self, *, ranges=None, exclude_near=None, active=True):
        self._ctx = {"ranges": ranges, "exclude_near": exclude_near, "active": bool(active)}

    def _recent_recovery(self):
        w = self.recovery
        if w is None or not w.events:
            return False
        return (len(w.records) - w.events[-1]["t"]) <= self.cfg.recent_recovery_window

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        t0 = time.perf_counter()
        cfg, ctx = self.cfg, self._ctx
        p_arr = np.asarray(p, float).reshape(2)
        g_arr = np.asarray(gamma, float).reshape(2)
        u0 = (nominal_action(p_arr, g_arr, self.ctrl.max_v) if u_nom is None
              else np.asarray(u_nom, float).reshape(2))
        src = self.source_fn()
        obst = SensedObstacles.from_source(src, r_robot=self.r_robot, r_obstacle=self.r_obstacle,
                                           exclude_near=ctx.get("exclude_near"),
                                           exclude_radius=cfg.exclude_radius,
                                           strict_tracks=(cfg.track_gate == "strict"))
        # fix 2: extrapolate along the action actually executed last step (already CBF-filtered)
        u_look = (np.asarray(self.ctrl.prev_u, float).reshape(2)
                  if cfg.lookahead_velocity == "executed" else u0)
        ev = self.monitor.evaluate(src, p_arr, u_look, xi_now=xi, obstacles=obst)
        state = self.detector.update(ev["margin"], ev["clr_look"])
        ranges = ctx.get("ranges")
        gap = (free_gap(p_arr, ranges, u0, n_rays=self.n_rays, lidar_range=self.lidar_range,
                        pass_distance=cfg.gap_pass_distance)
               if ranges is not None else float("nan"))
        ts = track_signals(p_arr, self.ctrl.prev_u, obst, radius=cfg.track_radius)

        gamma_used, u_used, scale, angle = gamma, u_nom, 1.0, 0.0
        if cfg.mode == "active" and state > 0:
            k = float(cfg.speed_scale[state])
            best_m = None
            for a in cfg.angle_offsets_deg:
                m = self.monitor.evaluate(src, p_arr, k * (_rot(a) @ u0), xi_now=xi)["M_look"]
                if best_m is None or m > best_m + 1e-9:
                    best_m, angle = m, float(a)
            R = _rot(angle)
            u_used = k * (R @ u0)
            gamma_used = p_arr + k * (R @ (g_arr - p_arr))
            scale = k
        t_cd = (time.perf_counter() - t0) * 1e3

        u = self._orig(p, gamma_used, xi, u_nom=u_used, record=record)
        self.steps.append({
            "state": int(state), "margin": ev["margin"], "M0": ev["M"][0], "M_look": ev["M_look"],
            "clr_look": ev["clr_look"], "gap": gap, "n_near": ts["n_near"],
            "max_closing": ts["max_closing"], "min_ttc": ts["min_ttc"],
            "recent_recovery": bool(self._recent_recovery()), "scale": scale, "angle": angle,
            "excluded_tracks": obst.n_excluded_tracks, "cd_ms": t_cd,
            "active": bool(ctx.get("active", True))})
        return u
