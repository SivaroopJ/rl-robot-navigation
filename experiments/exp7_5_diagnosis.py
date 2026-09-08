"""Week5-Phase7 / Stage 5 / DIAGNOSIS of the A4 failure. Read-only, changes nothing.

The A4 numbers compare two CLOSED-LOOP arms, so A0 and A1 visit different states and the
optimistic-tail difference confounds the estimator with the trajectory. This diagnostic removes
that confound: one trajectory is driven by the FROZEN A0 policy, and the identical (p_ego,
ranges) stream is fed to BOTH trackers in parallel. Any difference is then the estimator alone.

The pre-registered competing explanation for a rise in optimistic error alongside a fall in
super-physical rows is the SLUGGISH-FILTER failure mode of 11.9.8: a larger R makes the filter
trust the measurement less, so it reports SMALLER velocities -- which lowers the super-physical
rate mechanically while UNDERSTATING genuine closing. This script tests that directly.

Ground-truth obstacle state is used HERE ONLY as a metrics channel, exactly as in Stages 2-4.
It never touches either tracker: both receive (p_ego, ranges) and nothing else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dr_control.policy_phase7 import Phase7Policy
from dr_control.tracking2 import GeometricLidarVelocityTracker
from dr_control.velocity_tracker import LidarVelocityTracker
from experiments.exp0_analytic import DEV_SEED_BASE
from experiments.exp6_ppo_comparison import make_env
from experiments.exp7_5_estimator import OUT

LABEL = "Week5-Phase7 / Stage 5 / A4 failure diagnosis (open loop, identical input stream)"
MATCH_TOL = 0.35


def probe(tracker, p, ranges, truth_p, truth_v):
    """Per-track: matched true obstacle, estimated and true speed, and the velocity error."""
    rows = []
    for t in tracker.tracks:
        if not (t.confirmed or not tracker.use_confirmation):
            continue
        d = np.linalg.norm(np.asarray(truth_p, float) - t.position[None, :], axis=1)
        k = int(np.argmin(d))
        if float(d[k]) > MATCH_TOL:
            continue
        v_hat, v_true = t.velocity, np.asarray(truth_v, float)[k]
        g = t.position - p
        g = g / max(float(np.linalg.norm(g)), 1e-12)
        rows.append({
            "speed_hat": float(np.linalg.norm(v_hat)),
            "speed_true": float(np.linalg.norm(v_true)),
            "err": float(np.linalg.norm(v_hat - v_true)),
            # closing-rate error on this track's own gradient; > 0 is OPTIMISTIC
            "e_proj": float((-g @ v_hat) - (-g @ v_true)),
            # Track.n_points_last is the count from the detection that last updated this
            # track. Keying a dict on the post-update track position never matches the
            # detection centre, which silently zeroed every split in the first run.
            "n_points": int(t.n_points_last),
        })
    return rows


def run(cond, n_ep, seed_base):
    rnd = cond == "randomized"
    env = make_env(rnd)   # Stages 3-4 use the default obstacle_speed = 0.675 in BOTH conditions
    out = {"frozen": [], "E1": []}
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed_base + k)
        pol = Phase7Policy(arm="T1_projection_cap", world_size=env.WORLD_SIZE,
                           agent_radius=env.AGENT_RADIUS, static_obstacles=env.static_obstacles,
                           max_speed=env.MAX_SPEED, dt=env.dt, lidar_range=env.LIDAR_RANGE,
                           n_rays=env.N_LIDAR_RAYS)
        pol.reset(obs, env.agent_position)
        tks = {"frozen": LidarVelocityTracker(r_nominal=env.AGENT_RADIUS, dt=env.dt,
                                              n_rays=env.N_LIDAR_RAYS,
                                              lidar_range=env.LIDAR_RANGE),
               "E1": GeometricLidarVelocityTracker(r_nominal=env.AGENT_RADIUS, dt=env.dt,
                                                   n_rays=env.N_LIDAR_RAYS,
                                                   lidar_range=env.LIDAR_RANGE)}
        term = trunc = False
        while not (term or trunc):
            p = env.agent_position.copy()
            ranges = obs[4:28].copy() * env.LIDAR_RANGE
            tp = np.asarray(env.obstacle_positions, float).copy()
            tv = np.asarray(env.obstacle_velocities, float).copy()
            for name, tk in tks.items():
                tk.update(p, ranges)                      # IDENTICAL input to both
                out[name].extend(probe(tk, p, ranges, tp, tv))
            action, _ = pol.predict(obs, p)
            obs, _, term, trunc, _ = env.step(action)
    return out


def summarise(rows):
    if not rows:
        return {}
    sh = np.array([r["speed_hat"] for r in rows])
    st = np.array([r["speed_true"] for r in rows])
    er = np.array([r["err"] for r in rows])
    ep = np.array([r["e_proj"] for r in rows])
    np_ = np.array([r["n_points"] for r in rows])
    d = {"n_track_obs": int(len(rows)),
         "mean_speed_hat": float(sh.mean()), "mean_speed_true": float(st.mean()),
         "median_speed_hat": float(np.median(sh)),
         "speed_ratio_hat_over_true": float(sh.mean() / max(st.mean(), 1e-9)),
         "mean_abs_velocity_error": float(er.mean()),
         "median_abs_velocity_error": float(np.median(er)),
         "p_optimistic_proj": float((ep > 0).mean()),
         "mean_e_proj": float(ep.mean()),
         "p_superphysical_speed": float((sh > 0.75).mean())}
    for lab, sel in (("1pt", np_ == 1), ("2pt", np_ == 2), ("3+pt", np_ >= 3)):
        if sel.any():
            d[f"p_optimistic_proj_{lab}"] = float((ep[sel] > 0).mean())
            d[f"mean_abs_velocity_error_{lab}"] = float(er[sel].mean())
            d[f"mean_speed_hat_{lab}"] = float(sh[sel].mean())
            d[f"n_{lab}"] = int(sel.sum())
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=15)
    a = ap.parse_args()
    res = {"label": LABEL, "episodes_per_condition": a.episodes,
           "seed_base": DEV_SEED_BASE + 1_000, "note": "open loop; identical scans to both"}
    for cond in ("fixed", "randomized"):
        raw = run(cond, a.episodes, DEV_SEED_BASE + 1_000)
        res[cond] = {k: summarise(v) for k, v in raw.items()}
        print(f"\n=== {cond.upper()} (open loop, identical input stream) ===")
        f, e = res[cond]["frozen"], res[cond]["E1"]
        rows = [("mean estimated speed", "mean_speed_hat"),
                ("mean TRUE speed", "mean_speed_true"),
                ("estimated/true speed ratio", "speed_ratio_hat_over_true"),
                ("mean |v_hat - v_true|", "mean_abs_velocity_error"),
                ("P(optimistic projection)", "p_optimistic_proj"),
                ("mean projected error", "mean_e_proj"),
                ("P(speed > 0.75)", "p_superphysical_speed"),
                ("P(optimistic) 1-point", "p_optimistic_proj_1pt"),
                ("P(optimistic) 2-point", "p_optimistic_proj_2pt"),
                ("P(optimistic) 3+point", "p_optimistic_proj_3+pt"),
                ("mean |err| 1-point", "mean_abs_velocity_error_1pt"),
                ("mean speed_hat 1-point", "mean_speed_hat_1pt")]
        for lab, k in rows:
            if k in f and k in e:
                print(f"  {lab:30s} frozen {f[k]:+.4f}   E1 {e[k]:+.4f}   delta {e[k]-f[k]:+.4f}")
        print(f"  track observations: frozen {f['n_track_obs']}  E1 {e['n_track_obs']}")
    p = OUT / "stage5_validation_diagnosis.json"
    if p.exists():
        raise SystemExit(f"{p} exists; results are never overwritten")
    p.write_text(json.dumps(res, indent=2))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
