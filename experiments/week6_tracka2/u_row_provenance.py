"""Track A continuation - OFFLINE provenance of every track row built by the unlabelled arm.

Measurement only, after the action is computed. For each row: how far its estimated centre is from
the true hunter and from the nearest true dynamic obstacle, plus the track's own quality signals.
Answers whether the "unmatched" rows are phantom tracks or merely inaccurate centres.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuation.pursuit.hunter import HunterController
from continuation.unlabelled_threat.harness import arm_config, build_protagonist_arm, make_env
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/week6_tracka2/coverage")
DEV_BASE = 13_200_500          # post-gate validation block (pre-gate used 13_200_200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, _ = load_tuned()
    spec, _ = load_random()
    cfg = arm_config("U1_unlabelled")
    env = make_env(cfg)
    rows = []
    for k in range(a.episodes):
        seed = DEV_BASE + k
        obs, _ = env.reset(seed=seed)
        pol, wrap = build_protagonist_arm("U1_unlabelled", env, obs, cfg, tuned, spec, seed)
        hunter = HunterController(params=tuned, max_speed=cfg.hunter_max_speed,
                                  radius=cfg.hunter_radius, n_rays=env.N_LIDAR_RAYS,
                                  lidar_range=env.LIDAR_RANGE, dt=env.dt)
        term = trunc = False
        while not (term or trunc):
            pre = env.pre_step_state()
            a_prot, _ = pol.predict(obs, pre["p_prot"])
            for r in pol.src.last_rows:
                c = np.array([r["cx"], r["cy"]])
                d_h = float(np.linalg.norm(c - pre["p_hunter"]))
                d_o = min((float(np.linalg.norm(c - q)) for q in np.asarray(pre["obstacles"], float)),
                          default=float("inf"))
                # distance from the estimated centre to the nearest static rectangle surface
                d_rect = float("inf")
                for cx, cy, hw, hh in env.static_obstacles:
                    q = np.array([np.clip(c[0], cx - hw, cx + hw), np.clip(c[1], cy - hh, cy + hh)])
                    d_rect = min(d_rect, float(np.linalg.norm(c - q)))
                d_wall = float(min(c[0], c[1], env.WORLD_SIZE - c[0], env.WORLD_SIZE - c[1]))
                rows.append({"seed": seed, "d_hunter": d_h, "d_obstacle": d_o,
                             "d_rect": d_rect, "d_wall": d_wall, "speed": r["speed"],
                             "age": r["age"], "misses": r["misses"],
                             "dist_to_robot": r["dist"]})
            u_h, _ = hunter.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
            obs, _, term, trunc, _ = env.step_pursuit(a_prot, u_h)
        wrap.detach()
    env.close()

    TOL = 0.6
    real = [r for r in rows if min(r["d_hunter"], r["d_obstacle"]) <= TOL]
    phantom = [r for r in rows if min(r["d_hunter"], r["d_obstacle"]) > TOL]
    def q(v, k):
        x = [r[k] for r in v]
        return {"n": len(x), "mean": float(np.mean(x)) if x else float("nan"),
                "q10": float(np.quantile(x, .1)) if x else float("nan"),
                "q90": float(np.quantile(x, .9)) if x else float("nan")}
    summ = {"rows_total": len(rows), "rows_matching_a_real_body": len(real),
            "rows_matching_nothing": len(phantom),
            "phantom_fraction": len(phantom) / max(len(rows), 1),
            "phantom_near_static_geometry_frac": float(np.mean(
                [min(r["d_rect"], r["d_wall"]) <= 0.6 for r in phantom])) if phantom else float("nan"),
            "phantom_speed": q(phantom, "speed"), "real_speed": q(real, "speed"),
            "phantom_age": q(phantom, "age"), "real_age": q(real, "age"),
            "phantom_dist_to_robot": q(phantom, "dist_to_robot"),
            "real_dist_to_robot": q(real, "dist_to_robot"),
            "phantom_misses": q(phantom, "misses"), "real_misses": q(real, "misses")}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"row_provenance_{a.episodes}{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text(json.dumps({"tol": TOL, "summary": summ, "rows": rows}, indent=1, default=float))
    print(json.dumps(summ, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
