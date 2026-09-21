"""Random-CLF-DR-CBF: stochastic velocity recovery on DR-CBF infeasibility.

NEW. The DR-CBF QP is the frozen one and runs EVERY step, unchanged. This wrapper acts ONLY on
steps the frozen problem declares infeasible (status contains "infeasible" -- the identical
trigger RecoveryLadder uses, audit A8); on feasible steps it returns the frozen action untouched.

THE RECOVERY (spec sections R1 / R2, decisions D5)
    On an infeasible step, draw phi ~ U(0, 2 pi) from the per-episode CONTROLLER rng and form K
    directions theta_j = phi + 2 pi j / K. For each speed v in the ascending schedule:
        u_j = v [cos theta_j, sin theta_j],   m_j = min_i CBC_i(u_j)  on the QP's own xi_kept,
        best_v = argmax_j m_j.
    Accept the FIRST speed whose best_v has m >= accept_m (= 0). If no speed is acceptable,
    execute the best candidate over ALL speeds tried. There is no LADDER fallback and no
    nominal blending: u = u_R during recovery.
    The chosen u_R is executed for H consecutive steps (strict persistence). During those steps
    the frozen QP still runs -- so the scan buffer and tracker keep updating and its feasibility
    is logged -- but its action is not used. ctrl.prev_u is set to the executed action, as the
    ladder does, so the frozen objective's smoothness term stays coherent.

WHY m >= 0 AND NOT m >= tau_eff (audit A6)
    The CLF row has an unbounded slack, so frozen-QP infeasibility means
    max_{|u|_inf <= max_v} min_i CBC_i < tau_eff. Every candidate has ||u||_2 <= v <= 1, so it
    lies in that box and cannot attain tau_eff. m >= 0 (nominal CBF on every kept sample, tier T1)
    is the only criterion that is attainable yet non-trivial.

INFORMATION LEDGER
    The wrapper sees exactly what the frozen controller sees: p, gamma and xi, the latter built
    from LiDAR ranges (obs[4:28]) and the LiDAR velocity tracker. It takes no env, no obstacle
    state and no observation; its only extra input is the controller rng, seeded from the
    episode seed and independent of env.np_random.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RandomSpec:
    K: int = 8
    H: int = 1
    speeds: tuple = (0.2,)
    accept_m: float = 0.0
    name: str = "RANDOM_EXP_R1"

    def __post_init__(self):
        if self.K < 1 or self.H < 1:
            raise ValueError("K and H must be >= 1")
        sp = tuple(float(s) for s in self.speeds)
        if not sp or any(s <= 0 or s > 1.0 + 1e-12 for s in sp):
            raise ValueError("speeds must lie in (0, 1.0]; the action bound is 1.0 m/s")
        if list(sp) != sorted(sp):
            raise ValueError("speed schedule must be ascending")
        object.__setattr__(self, "speeds", sp)

    def as_dict(self):
        d = asdict(self)
        d["speeds"] = list(self.speeds)
        return d


def candidate_directions(phi, K):
    th = phi + 2.0 * np.pi * np.arange(K) / K
    return th, np.column_stack([np.cos(th), np.sin(th)])


def margins(kept, U, alpha):
    """min_i CBC_i for each row of U (M, 2): kept (N, 4) = [dh_dt, h, gx, gy]."""
    kept = np.atleast_2d(np.asarray(kept, float))
    base = kept[:, 0] + alpha * kept[:, 1]                   # (N,)
    return np.min(base[None, :] + np.asarray(U, float) @ kept[:, 2:4].T, axis=1)


def select(kept, alpha, phi, spec: RandomSpec, max_v=1.0):
    """The deterministic part of one recovery event, given the random base angle phi."""
    th, dirs = candidate_directions(phi, spec.K)
    per_speed, best = [], None
    accepted = False
    for v in spec.speeds:
        U = np.clip(v * dirs, -max_v, max_v)
        m = margins(kept, U, alpha)
        j = int(np.argmax(m))
        row = {"v": float(v), "j": j, "theta": float(th[j] % (2 * np.pi)),
               "m": float(m[j]), "u": U[j].copy()}
        per_speed.append(row)
        if best is None or row["m"] > best["m"]:
            best = row
        if row["m"] >= spec.accept_m:
            best, accepted = row, True
            break
    return best, accepted, per_speed


class RandomRecovery:
    """Wraps a live frozen controller instance. The frozen module is never edited."""

    def __init__(self, ctrl, *, spec: RandomSpec, rng, tau, alpha=None, max_v=None):
        self.ctrl = ctrl
        self.spec = spec
        self.rng = rng
        self.tau = float(tau)
        self.alpha = ctrl.rateh if alpha is None else float(alpha)
        self.max_v = ctrl.max_v if max_v is None else float(max_v)
        self.records = []           # one per call
        self.events = []            # one per recovery event
        self._persist_left = 0
        self._persist_u = None
        self._t = 0
        self._orig = ctrl.generate_controller
        ctrl.generate_controller = self._wrapped

    def detach(self):
        self.ctrl.generate_controller = self._orig

    def _wrapped(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u = self._orig(p, gamma, xi, u_nom=u_nom, record=rec)       # FROZEN call, verbatim
        status = str(rec.get("status"))
        infeasible = "infeasible" in status
        kept = np.atleast_2d(np.asarray(rec.get("xi_kept", xi), dtype=float))
        t = self._t
        self._t += 1
        entry = {"t": t, "status": status, "infeasible": infeasible, "mode": "qp",
                 "t_recovery": 0.0}

        if self._persist_left > 0:                   # strict persistence, QP logged only
            self._persist_left -= 1
            u_out = self._persist_u
            entry.update(mode="persist", m=float(margins(kept, u_out[None, :], self.alpha)[0]))
            return self._emit(entry, rec, u_out)

        if not infeasible:
            entry["m"] = float(margins(kept, np.asarray(u, float)[None, :], self.alpha)[0])
            self.records.append(entry)
            rec.update(entry)
            return u                                  # INERT: frozen action untouched

        t0 = time.perf_counter()
        phi = float(self.rng.uniform(0.0, 2.0 * np.pi))
        best, accepted, per_speed = select(kept, self.alpha, phi, self.spec, self.max_v)
        dt_rec = time.perf_counter() - t0
        u_out = np.asarray(best["u"], float)
        m_u0 = float(margins(kept, np.zeros((1, 2)), self.alpha)[0])   # the frozen fallback's m
        ev = {"t": t, "phi": phi, "theta": best["theta"], "speed": best["v"], "m": best["m"],
              "accepted": bool(accepted), "m_u0": m_u0,
              "m_by_speed": [r["m"] for r in per_speed],
              "n_speeds_tried": len(per_speed), "n_candidates": len(per_speed) * self.spec.K,
              "t_recovery": dt_rec, "h_crit": float(kept[0, 1])}
        self.events.append(ev)
        self._persist_left = self.spec.H - 1
        self._persist_u = u_out.copy()
        entry.update(mode="event", m=best["m"], t_recovery=dt_rec)
        return self._emit(entry, rec, u_out)

    def _emit(self, entry, rec, u_out):
        self.records.append(entry)
        rec.update(entry)
        self.ctrl.prev_u = np.asarray(u_out, dtype=float).copy()   # keep frozen state coherent
        return np.asarray(u_out, dtype=float).copy()
