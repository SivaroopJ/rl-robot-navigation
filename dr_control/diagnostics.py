"""Per-step and per-episode controller telemetry -- Phase 1 subset of Part D.

The point of these numbers is attribution: when an episode fails we need to say whether it
was the barrier, the CLF, the solver, or the objective, without re-running anything. Phase 1
has no sensor and no planner, so the perception / velocity-estimation / planner fields of
Part D are absent by construction and are added in later phases.
"""
from __future__ import annotations

import numpy as np


class EpisodeDiagnostics:
    """Accumulates per-step controller telemetry and reduces it to one episode record."""

    def __init__(self):
        self.min_h_est = np.inf          # min barrier value the controller SAW
        self.min_h_true = np.inf         # min TRUE geometric clearance (ground truth)
        self.min_cbc = np.inf            # min CBC over the samples the controller CONSTRAINED
        self.min_cbc_all = np.inf        # min CBC over ALL samples, incl. unconstrained ones
        self.min_cbc_solved = np.inf     # min CBC restricted to steps that SOLVED. Separates
                                         # "the constraint was violated" (a real defect) from
                                         # "no admissible u existed, so u=0 was commanded"
                                         # (infeasibility -- a property of the scenario).
        self.n_steps = 0
        self.n_cbc_negative = 0          # CBC < 0 despite the constraint: a real violation
        self.n_delta_positive = 0        # CLF slack active: safety overrode progress
        self.n_cbf_active = 0            # barrier constraint tight
        self.n_solver_fail = 0
        self.n_dcp_error = 0
        self.n_h_negative = 0            # h_crit < 0 -- the reference's DCP trigger (B.1)
        self.n_infeasible = 0            # the admissible set was genuinely EMPTY: no u can
                                         # satisfy every kept barrier constraint at once.
                                         # Distinct from a numerical solver failure.
        self.solver_times = []
        self.total_times = []
        self.canon_times = []
        self.u_dev = []                  # ||u - u_nom||, magnitude of safety intervention
        self.statuses = []

    def record(self, *, info, u, xi, cbf_rate, h_true=None, cbc_tol=1e-6):
        from dr_control.drccp_controller import u_bar_value

        self.n_steps += 1
        status = info.get("status", "unknown")
        self.statuses.append(status)

        if status == "DCPError":
            self.n_dcp_error += 1
            self.n_solver_fail += 1
        elif status != "optimal":
            self.n_solver_fail += 1
            if "infeasible" in str(status):
                self.n_infeasible += 1

        h_crit = info.get("h_crit", np.nan)
        if h_crit == h_crit:
            self.min_h_est = min(self.min_h_est, float(h_crit))
            if h_crit < 0:
                self.n_h_negative += 1
        if h_true is not None:
            self.min_h_true = min(self.min_h_true, float(h_true))

        # `xi` is every sample that existed; `xi_kept` (when the controller reports it) is
        # the subset it actually constrained. Reporting both separates "the constraint was
        # violated" from "a sample outside the constrained subset was unsafe" -- for the
        # single-sample nominal QP the second is expected and is not a solver failure.
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        u_bar = u_bar_value(u, cbf_rate)
        cbc_all = float(np.min(xi @ u_bar))
        self.min_cbc_all = min(self.min_cbc_all, cbc_all)

        xi_kept = info.get("xi_kept")
        xi_kept = xi if xi_kept is None else np.atleast_2d(np.asarray(xi_kept, dtype=float))
        cbc = xi_kept @ u_bar
        cbc_min = float(np.min(cbc))
        self.min_cbc = min(self.min_cbc, cbc_min)
        if status == "optimal":
            self.min_cbc_solved = min(self.min_cbc_solved, cbc_min)
        if cbc_min < -cbc_tol:
            self.n_cbc_negative += 1
        if abs(cbc_min) <= 1e-4:
            self.n_cbf_active += 1

        delta = info.get("delta", np.nan)
        if delta == delta and delta > 1e-8:
            self.n_delta_positive += 1

        for key, sink in (("solver_time", self.solver_times),
                          ("total_time", self.total_times),
                          ("canon_time", self.canon_times)):
            v = info.get(key, np.nan)
            if v == v:
                sink.append(float(v))

        u_nom = info.get("u_nom")
        if u_nom is not None:
            self.u_dev.append(float(np.linalg.norm(np.asarray(u) - np.asarray(u_nom))))

    def summary(self, **extra):
        def frac(n):
            return n / self.n_steps if self.n_steps else 0.0

        def stat(xs, fn):
            return float(fn(xs)) if xs else float("nan")

        out = {
            "steps": self.n_steps,
            "min_h_est": _finite(self.min_h_est),
            "min_h_true": _finite(self.min_h_true),
            "min_cbc": _finite(self.min_cbc),
            "min_cbc_all": _finite(self.min_cbc_all),
            "min_cbc_solved": _finite(self.min_cbc_solved),
            "frac_cbc_negative": frac(self.n_cbc_negative),
            "frac_delta_positive": frac(self.n_delta_positive),
            "frac_cbf_active": frac(self.n_cbf_active),
            "frac_h_negative": frac(self.n_h_negative),
            "n_solver_fail": self.n_solver_fail,
            "n_infeasible": self.n_infeasible,
            "frac_infeasible": frac(self.n_infeasible),
            "n_dcp_error": self.n_dcp_error,
            "mean_solver_time": stat(self.solver_times, np.mean),
            "max_solver_time": stat(self.solver_times, np.max),
            "mean_total_time": stat(self.total_times, np.mean),
            "mean_canon_time": stat(self.canon_times, np.mean),
            "mean_u_dev": stat(self.u_dev, np.mean),
            "max_u_dev": stat(self.u_dev, np.max),
        }
        out.update(extra)
        return out


def _finite(v):
    return float(v) if np.isfinite(v) else float("nan")


def aggregate(records):
    """Mean over episodes for numeric fields; sums for count fields."""
    if not records:
        return {}
    sum_keys = {"n_solver_fail", "n_dcp_error", "n_infeasible", "steps"}
    out = {}
    for key in records[0]:
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        vals = [v for v in vals if v == v]      # drop NaN; a metric a controller never
        if not vals:                            # reports (e.g. canon_time for clf_only)
            continue                            # is absent, not zero
        vals = np.array(vals, dtype=float)
        if key in sum_keys:
            out[key] = float(np.nansum(vals))
        else:
            out[key] = float(np.nanmean(vals))
            out[key + "_min"] = float(np.nanmin(vals))
    return out
