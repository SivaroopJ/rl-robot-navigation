"""Track A continuation - OFFLINE coverage diagnostic for the unlabelled-threat condition.

The controller has no labels, so the pilot cannot say whether the HUNTER was among the rows it
built. This script answers that question offline, using ground truth ONLY as a measurement channel
after the action has been computed. Nothing here is fed back to the controller: the episode is run
by the same `run_episode` code path, and the ground-truth comparison happens in this file.

Reported per step (U1_unlabelled only):
    hunter_in_range      true hunter distance <= LiDAR range               (ground truth)
    hunter_covered       some track row lay within `TOL` of the true hunter position
    decoy_rows           rows that matched a true dynamic obstacle instead
    unmatched_rows       rows matching neither (tracker artefacts)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from continuation.unlabelled_threat.harness import arm_config, build_protagonist_arm, make_env
from continuation.pursuit.hunter import HunterController
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/week6_tracka2/coverage")
DEV_BASE = 13_200_400          # post-gate validation block (pre-gate used 13_200_100)          # fresh, disjoint from the pilot seeds
TOL = 0.6                      # association tolerance, as in A-P1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, _ = load_tuned()
    spec, _ = load_random()
    cfg = arm_config("U1_unlabelled")
    env = make_env(cfg)
    rows_all, eps = [], []
    t0 = time.time()
    for k in range(a.episodes):
        seed = DEV_BASE + k
        obs, _ = env.reset(seed=seed)
        pol, wrap = build_protagonist_arm("U1_unlabelled", env, obs, cfg, tuned, spec, seed)
        hunter = HunterController(params=tuned, max_speed=cfg.hunter_max_speed,
                                  radius=cfg.hunter_radius, n_rays=env.N_LIDAR_RAYS,
                                  lidar_range=env.LIDAR_RANGE, dt=env.dt)
        term = trunc = False
        info = {}
        steps = 0
        per = []
        box = {}
        _orig = pol.ctrl.generate_controller

        def _wrapped(p_, gamma, xi, *, u_nom=None, record=None):
            rec = {} if record is None else record
            u = _orig(p_, gamma, xi, u_nom=u_nom, record=rec)
            box["xi_kept"] = rec.get("xi_kept")
            return u
        pol.ctrl.generate_controller = _wrapped
        while not (term or trunc):
            pre = env.pre_step_state()
            a_prot, _ = pol.predict(obs, pre["p_prot"])
            built = list(pol.src.last_rows)          # rows the controller actually used
            u_h, _ = hunter.act(pre["p_hunter"], env.hunter_lidar(), pre["p_prot"])
            # ---- measurement only, after the action is fixed ----
            p = pre["p_prot"]
            d_h = float(np.linalg.norm(pre["p_hunter"] - p))
            obs_pos = np.asarray(pre["obstacles"], float)
            covered, decoys, unmatched = 0, 0, 0
            for r in built:
                c = np.array([r["cx"], r["cy"]])              # the row's estimated centre
                dh = float(np.linalg.norm(c - pre["p_hunter"]))
                do = min((float(np.linalg.norm(c - q)) for q in obs_pos), default=np.inf)
                if dh <= TOL and dh <= do:
                    covered += 1
                elif do <= TOL:
                    decoys += 1
                else:
                    unmatched += 1
            kept = np.atleast_2d(np.asarray(box.get("xi_kept", np.zeros((0, 4))), float))
            vecs = {tuple(np.round(v, 12)) for v in pol.src.last_row_vectors}
            kept_track = sum(1 for row in kept if tuple(np.round(row, 12)) in vecs)
            per.append({"step": steps, "n_rows": len(built), "hunter_dist": d_h,
                        "kept_rows": int(len(kept)), "kept_track_rows": int(kept_track),
                        "hunter_in_range": bool(d_h <= env.LIDAR_RANGE),
                        "hunter_covered": bool(covered > 0), "decoy_rows": decoys,
                        "unmatched_rows": unmatched,
                        "infeasible": int(pol.last_step_infeasible)})
            obs, _, term, trunc, info = env.step_pursuit(a_prot, u_h)
            steps += 1
        pol.ctrl.generate_controller = _orig
        wrap.detach()
        rows_all += [{**r, "seed": seed} for r in per]
        eps.append({"seed": seed, "outcome": info.get("outcome"), "steps": steps})
    env.close()

    R = [r for r in rows_all if r["hunter_in_range"]]
    summ = {
        "episodes": a.episodes, "steps": len(rows_all),
        "steps_hunter_in_lidar_range": len(R),
        "coverage_given_in_range": float(np.mean([r["hunter_covered"] for r in R])) if R else float("nan"),
        "mean_rows_per_step": float(np.mean([r["n_rows"] for r in rows_all])),
        "mean_decoy_rows_per_step": float(np.mean([r["decoy_rows"] for r in rows_all])),
        "mean_unmatched_rows_per_step": float(np.mean([r["unmatched_rows"] for r in rows_all])),
        "steps_with_zero_rows": float(np.mean([r["n_rows"] == 0 for r in rows_all])),
        "mean_kept_rows": float(np.mean([r["kept_rows"] for r in rows_all])),
        "mean_kept_track_rows": float(np.mean([r["kept_track_rows"] for r in rows_all])),
        "mean_kept_scan_rows": float(np.mean([r["kept_rows"] - r["kept_track_rows"]
                                              for r in rows_all])),
        "frac_steps_all_kept_rows_are_track_rows": float(np.mean(
            [r["kept_track_rows"] >= r["kept_rows"] > 0 for r in rows_all])),
        "coverage_when_hunter_within_2m": (
            float(np.mean([r["hunter_covered"] for r in R if r["hunter_dist"] <= 2.0]))
            if any(r["hunter_dist"] <= 2.0 for r in R) else float("nan")),
        "wall_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"u_coverage_{a.episodes}{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text(json.dumps({"tol": TOL, "summary": summ, "episodes": eps,
                             "steps": rows_all}, indent=1, default=float))
    print(json.dumps(summ, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
