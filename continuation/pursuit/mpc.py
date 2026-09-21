"""Short-horizon sampled MPC pursuit (policies 3 / 3.5): a better NOMINAL hunter action.

MPC never replaces the CLF-DR-CBF QP. It selects (gamma, u_nom); the frozen QP filters every
executed action and Random recovery handles QP infeasibility, exactly as for policies 1-2.5.

MODEL (verified against the code): single integrator p_{k+1} = p_k + dt u_k, dt = 0.1 s,
per-axis box |u_i| <= max_v. Candidates are built with Euclidean speed <= 1 m/s, so they lie
inside the box (the clip below is a no-op, kept as a guard).

CANDIDATES (approved decision 3). theta_0 = direction from the hunter to Protag's current true
position. Segment 0 (first `segment_steps` steps): heading theta_0 + h, h in heading_offsets.
Segment 1 (remaining steps): heading theta_0 + h + tr, tr in turn_offsets. Same speed v in both
segments, v in speeds. 7 x 3 x 2 = 42 sequences over N = 10 steps (1.0 s).

PROTAG PREDICTION. The policy-2 predictor `predict_position` evaluated at tau = k dt, k = 1..N
(prefix-consistent: each call integrates the same dt sub-steps). Protag is assumed not to react to
the candidate; documented simplification.

OBJECTIVE (approved decision 4), d_k = ||p^H_k - p_hat^P_k||, all distances in metres:
    J = w_T d_N + w_min min_k d_k - w_cap I[min_k d_k <= 0.65]
        + w_du sum_k ||u_{k+1} - u_k||^2 + w_clr sum_k max(0, 0.15 - clr_k)
clr_k from SensedObstacles (CBF convention). HARD REJECT if any clr_k < 0, k = 1..N.
argmin over non-rejected candidates; ties go to the earlier candidate (heading offset 0 first).
    gamma = best candidate's end-of-horizon position p^H_N   (decision 2)
    u_nom = best candidate's first control u_0
FALLBACK (all rejected): direct pursuit, gamma = p_P, u_nom = max_v unit(p_P - p_H), logged.
The rollout check is a planning aid, not a safety guarantee between or within steps.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from dr_control.drccp_controller import nominal_action

from continuation.pursuit.hunter_policies import MAX_PREDICTION_HORIZON, predict_position


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 10
    segment_steps: int = 5
    heading_offsets_deg: tuple = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)
    turn_offsets_deg: tuple = (0.0, -45.0, 45.0)
    speeds: tuple = (1.0, 0.5)
    w_T: float = 1.0
    w_min: float = 0.5
    w_cap: float = 2.0
    w_du: float = 0.1
    w_clr: float = 1.0
    clr_soft: float = 0.15          # = tau_eff / alpha, the controller's steady-state clearance
    capture_distance: float = 0.65  # PursuitConfig capture rule
    exclude_radius: float = 0.35    # policy 3.5 Protag-track exclusion (= filter radius)

    def __post_init__(self):
        if not 0 < self.segment_steps < self.horizon_steps:
            raise ValueError("need 0 < segment_steps < horizon_steps")
        if any(not 0 < s <= 1.0 for s in self.speeds):
            raise ValueError("candidate speeds must lie in (0, 1] m/s")

    @property
    def n_candidates(self):
        return len(self.heading_offsets_deg) * len(self.turn_offsets_deg) * len(self.speeds)

    def as_dict(self):
        d = asdict(self)
        d["n_candidates"] = self.n_candidates
        return d


def candidate_controls(p_hunter, p_protag, cfg: MPCConfig, max_v):
    """U (M, N, 2) m/s and the (heading, turn, speed) of each candidate."""
    d = np.asarray(p_protag, float) - np.asarray(p_hunter, float)
    theta0 = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0
    rows = [(h, tr, v) for h in cfg.heading_offsets_deg for tr in cfg.turn_offsets_deg
            for v in cfg.speeds]
    meta = np.asarray(rows, float)
    a0 = theta0 + np.deg2rad(meta[:, 0])
    a1 = a0 + np.deg2rad(meta[:, 1])
    v = meta[:, 2:3]
    U = np.empty((len(rows), cfg.horizon_steps, 2))
    U[:, :cfg.segment_steps] = (v * np.column_stack([np.cos(a0), np.sin(a0)]))[:, None, :]
    U[:, cfg.segment_steps:] = (v * np.column_stack([np.cos(a1), np.sin(a1)]))[:, None, :]
    return np.clip(U, -max_v, max_v), meta


def rollout(p_hunter, U, dt):
    """Positions after each step, k = 1..N: (M, N, 2)."""
    return np.asarray(p_hunter, float)[None, None, :] + dt * np.cumsum(U, axis=1)


def predict_protag_path(p_P, v_P, g_P, *, n_steps, dt, goal_accel, v_max, world_size,
                        body_radius):
    if n_steps * dt > MAX_PREDICTION_HORIZON + 1e-9:
        raise ValueError("MPC horizon exceeds the predictor's declared bound")
    out = []
    for k in range(1, n_steps + 1):
        q, _ = predict_position(p_P, v_P, g_P, horizon=k * dt, goal_accel=goal_accel, dt=dt,
                                v_max=v_max, world_size=world_size, body_radius=body_radius)
        if q is None:
            raise FloatingPointError("Protag state is non-finite; cannot predict")
        out.append(q)
    return np.asarray(out)


def score(P, U, pred, clr, cfg: MPCConfig):
    """Cost terms per candidate. Returns (J, reject, parts)."""
    d = np.linalg.norm(P - pred[None, :, :], axis=-1)                     # (M, N)
    d_N, d_min = d[:, -1], d.min(axis=1)
    cap = d_min <= cfg.capture_distance
    du = np.sum(np.sum(np.diff(U, axis=1) ** 2, axis=-1), axis=1)
    c_clr = np.maximum(0.0, cfg.clr_soft - clr).sum(axis=1)
    J = cfg.w_T * d_N + cfg.w_min * d_min - cfg.w_cap * cap + cfg.w_du * du + cfg.w_clr * c_clr
    reject = (clr < 0.0).any(axis=1)
    return J, reject, {"d_N": d_N, "d_min": d_min, "cap": cap, "du": du, "c_clr": c_clr}


def plan(p_hunter, p_protag, v_protag, goal_protag, obstacles, cfg: MPCConfig, *, dt, max_v,
         goal_accel, protag_v_max, world_size, body_radius):
    """(gamma, u_nom, info). Uses only the pre-step state and the hunter's sensed obstacles."""
    t0 = time.perf_counter()
    p_h = np.asarray(p_hunter, float).reshape(2)
    p_p = np.asarray(p_protag, float).reshape(2)
    U, meta = candidate_controls(p_h, p_p, cfg, max_v)
    P = rollout(p_h, U, dt)
    pred = predict_protag_path(p_p, v_protag, goal_protag, n_steps=cfg.horizon_steps, dt=dt,
                               goal_accel=goal_accel, v_max=protag_v_max,
                               world_size=world_size, body_radius=body_radius)
    times = dt * np.arange(1, cfg.horizon_steps + 1)
    clr = obstacles.clearance(P, times)
    J, reject, parts = score(P, U, pred, clr, cfg)
    info = {"n_candidates": int(len(U)), "n_rejected": int(reject.sum()),
            "pred_protag_end": pred[-1].copy()}
    if reject.all():
        u_nom = nominal_action(p_h, p_p, max_v)
        gamma = p_p.copy()
        info.update(fallback=True, best=None, pred_capture=False, cost=None,
                    heading=None, turn=None, speed=None)
    else:
        j = int(np.argmin(np.where(reject, np.inf, J)))
        u_nom = U[j, 0].copy()
        gamma = P[j, -1].copy()
        info.update(fallback=False, best=j, pred_capture=bool(parts["cap"][j]),
                    cost=float(J[j]), heading=float(meta[j, 0]), turn=float(meta[j, 1]),
                    speed=float(meta[j, 2]), min_clr=float(clr[j].min()))
    info["mpc_ms"] = (time.perf_counter() - t0) * 1e3
    return gamma, u_nom, info
