"""Track B supervisor: risk-aware speed reduction around the FROZEN controller.

WHAT IT DOES AND DOES NOT DO
    It rescales the NOMINAL action's magnitude (direction preserved) and hands the scaled nominal to
    the unchanged frozen QP. It never writes the executed action, never touches the barrier rows,
    never bypasses the safety filter, and leaves the frozen u = 0 infeasibility fallback exactly as
    it is. Random recovery, when attached, is the frozen implementation and runs underneath this
    wrapper untouched.

WRAPPER ORDER
    ctrl.generate_controller  ->  [Supervisor]  ->  [RandomRecovery, if any]  ->  frozen QP
    The supervisor is outermost, so the scaled u_nom is what the frozen solve minimises against;
    the recovery still sees the frozen infeasibility status underneath.

STATE MACHINE (hysteresis + limits, so it cannot stall the robot)
    IDLE --risk >= RISK_ON--> ENGAGED (scale < 1)
    ENGAGED --risk < RISK_OFF--> IDLE
    ENGAGED --max_engaged steps--> COOLDOWN (scale = 1, no engagement) --cooldown steps--> IDLE
"""
from __future__ import annotations

import numpy as np

from dr_control.drccp_controller import nominal_action

from continuation.supervisor.risk import RISK_OFF, RISK_ON, risk_from_state

MIN_SCALE = 0.40            # the strongest slowdown allowed
MAX_ENGAGED = 50            # consecutive engaged steps before a forced stand-down
COOLDOWN = 20               # steps of forced non-engagement


class SpeedSupervisor:
    """Candidate 1: risk-aware speed reduction. Attach AFTER any recovery wrapper."""

    name = "speed_reduction"

    def __init__(self, ctrl, *, tracker, min_scale=MIN_SCALE, risk_on=RISK_ON, risk_off=RISK_OFF,
                 max_engaged=MAX_ENGAGED, cooldown=COOLDOWN, enabled=True):
        self.ctrl = ctrl
        self.tracker = tracker
        self.min_scale = float(min_scale)
        self.risk_on = float(risk_on)
        self.risk_off = float(risk_off)
        self.max_engaged = int(max_engaged)
        self.cooldown_len = int(cooldown)
        self.enabled = bool(enabled)
        self.records = []
        self.state = "IDLE"
        self._engaged_steps = 0
        self._cooldown = 0
        self.n_engaged = 0
        self.n_steps = 0
        self._inner = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._inner

    def _scale_for(self, risk):
        """Linear ramp from 1.0 at risk_on down to min_scale at risk = 1."""
        if risk <= self.risk_on:
            return 1.0
        span = max(1.0 - self.risk_on, 1e-9)
        return float(np.clip(1.0 - (risk - self.risk_on) / span * (1.0 - self.min_scale),
                             self.min_scale, 1.0))

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        self.n_steps += 1
        tracks = [t for t in self.tracker.tracks
                  if (t.confirmed or not self.tracker.use_confirmation)]
        risk, parts = risk_from_state(p, tracks, xi, self.ctrl.rateh)
        if u_nom is None:
            u_nom = nominal_action(p, gamma, self.ctrl.max_v)
        u_nom = np.asarray(u_nom, float).reshape(2)

        if self._cooldown > 0:
            self._cooldown -= 1
            self.state = "COOLDOWN"
        elif self.state == "ENGAGED":
            if risk < self.risk_off:
                self.state = "IDLE"
                self._engaged_steps = 0
            elif self._engaged_steps >= self.max_engaged:
                self.state = "COOLDOWN"
                self._cooldown = self.cooldown_len
                self._engaged_steps = 0
        elif risk >= self.risk_on:
            self.state = "ENGAGED"
            self._engaged_steps = 0

        scale = 1.0
        if self.enabled and self.state == "ENGAGED":
            self._engaged_steps += 1
            self.n_engaged += 1
            scale = self._scale_for(risk)
        u_nom_out = scale * u_nom

        u = self._inner(p, gamma, xi, u_nom=u_nom_out, record=record)
        rec = {} if record is None else record
        entry = {"risk": float(risk), "state": self.state, "scale": float(scale),
                 "u_nom_in": float(np.linalg.norm(u_nom)),
                 "u_nom_out": float(np.linalg.norm(u_nom_out)),
                 "infeasible": "infeasible" in str(rec.get("status")), **parts}
        self.records.append(entry)
        return u

    def summary(self):
        R = self.records
        eng = [r for r in R if r["state"] == "ENGAGED"]
        return {"steps": len(R), "engaged_steps": len(eng),
                "engaged_fraction": len(eng) / max(len(R), 1),
                "mean_risk": float(np.mean([r["risk"] for r in R])) if R else float("nan"),
                "mean_scale_when_engaged": (float(np.mean([r["scale"] for r in eng]))
                                            if eng else float("nan")),
                "cooldowns": sum(1 for a, b in zip(R, R[1:])
                                 if a["state"] != "COOLDOWN" and b["state"] == "COOLDOWN"),
                "infeasible_while_engaged": sum(1 for r in eng if r["infeasible"]),
                "infeasible_while_idle": sum(1 for r in R
                                             if r["state"] != "ENGAGED" and r["infeasible"])}
