"""Track B risk score: three runtime components, frozen on the B0 development block.

FROZEN ON DEV DATA (results/week6_trackb/B0/b0_steps_40.jsonl, seeds 12 000 000 + 0..39), never on
evaluation outcomes. Measured there on 10 286 feasible steps against "is the QP infeasible within
the next 3 steps" (base rate 0.0396):

    component                       AUC
    density  (confirmed tracks<2m)  0.731
    margin   (-m_u0)                0.739
    closing  (-worst closing speed) 0.691
    mean of the three               0.831

All three are RUNTIME quantities: the confirmed tracks come from the LiDAR velocity tracker the
frozen controller already runs, and m_u0 is read off the QP's own sample matrix. No ground truth.

    density = clip((n_tracks_within_2m - 2) / 3, 0, 1)
    margin  = clip((0.35 - m_u0) / 0.70, 0, 1)        m_u0 = min_i (dh_dt_i + alpha * h_i)
    closing = clip((-worst_closing_speed - 0.8) / 1.0, 0, 1)
    risk    = (density + margin + closing) / 3
"""
from __future__ import annotations

import numpy as np

RISK_ON = 0.50          # engage (dev: fires on 23.8 % of steps, recall 0.70, lift 2.96x)
RISK_OFF = 0.35         # release (hysteresis)
NEAR_RADIUS = 2.0


def components(*, n_tracks_near, m_u0, worst_closing):
    d = float(np.clip((n_tracks_near - 2.0) / 3.0, 0.0, 1.0))
    m = 0.0 if not np.isfinite(m_u0) else float(np.clip((0.35 - m_u0) / 0.70, 0.0, 1.0))
    c = float(np.clip((-worst_closing - 0.8) / 1.0, 0.0, 1.0))
    return d, m, c


def risk_from_state(p, tracks, xi, alpha):
    """The deployed computation: confirmed tracks + the QP's own rows. Returns (risk, parts)."""
    p = np.asarray(p, float).reshape(2)
    n_near, worst = 0, 0.0
    for t in tracks:
        q = np.asarray(t.position, float) - p
        d = float(np.linalg.norm(q))
        if d <= NEAR_RADIUS:
            n_near += 1
        if d > 1e-6:
            worst = min(worst, float(np.asarray(t.velocity, float) @ (q / d)))
    xi = np.atleast_2d(np.asarray(xi, float))
    m_u0 = float(np.min(xi[:, 0] + alpha * xi[:, 1])) if xi.size else float("nan")
    d_, m_, c_ = components(n_tracks_near=n_near, m_u0=m_u0, worst_closing=worst)
    return (d_ + m_ + c_) / 3.0, {"density": d_, "margin": m_, "closing": c_,
                                  "n_tracks_near": n_near, "m_u0": m_u0,
                                  "worst_closing": worst}
