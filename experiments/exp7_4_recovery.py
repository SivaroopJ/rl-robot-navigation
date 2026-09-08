"""Week5-Phase7 / Stage 4: recovery of the residual M2 infeasibility. Pre-registered in
PHASE7_PLAN.md 11.8 (aa66cd7) before implementation.

BASELINE IS THE STAGE-3 T1 CONFIGURATION (v_cap = 0.96), NOT the frozen C0: Stage 3 changed the
states visited, so C0's infeasible steps are not the population Stage 4 acts on.

V0 frozen. alpha, tau, eps, v_cap, the objective, sample selection, the planner and the LiDAR
pipeline unchanged. No estimator change, IMM/CT, covariance, uncertainty or governor logic. The
u = 0 fallback is unchanged outside the treatment arms.

THE PRIMARY DELIVERABLE IS CHARACTERISATION, NOT OUTCOME: how much of the M2 population is
recovered, by which rung, and with what margins and tiers. P3 (no collision improvement) is
explicitly allowed to be null.

R1 "resolves" a step only in the sense that the tau = 0 optimisation becomes feasible. The
resulting action is NOT thereby safe -- the achieved margin m is recomputed from the RETURNED
action against the original DR constraint and reported with its tier. No recovered action is
described as safe anywhere in this module or its output.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from dr_control.capped_velocity import V_CAP_DERIVED
from dr_control.policy_phase7 import Phase7Policy
from dr_control.recovery import RecoveryLadder
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp0_analytic import DEV_SEED_BASE
from experiments.exp6_ppo_comparison import make_env, true_clearance
from experiments.exp7_2b_velocity_counterfactual import mis_cardinality

LABEL = "Week5-Phase7 / Stage 4 / M2 recovery ladder"
OUT = Path("results/week5_phase7/stage4_recovery")
ARMS = {"B4_baseline": "none", "R1_tau_zero": "R1", "R2_min_violation": "R2",
        "LADDER": "ladder", "R15_ablation": "R15"}
TIERS = {"dev": (DEV_SEED_BASE, 20), "validation": (DEV_SEED_BASE + 1_000, 50),
         "final": (EVAL_SEED_BASE, 200)}


def run_episode(env, oracle, seed, mode, v_cap):
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = Phase7Policy(arm="T1_projection_cap", v_cap=v_cap, world_size=env.WORLD_SIZE,
                       agent_radius=env.AGENT_RADIUS, static_obstacles=env.static_obstacles,
                       max_speed=env.MAX_SPEED, dt=env.dt, lidar_range=env.LIDAR_RANGE,
                       n_rays=env.N_LIDAR_RAYS)
    pol.reset(obs, env.agent_position)
    lad = RecoveryLadder(pol.ctrl, mode=mode)
    clear = [true_clearance(env)]
    traj = [start.copy()]
    mis = Counter()
    coll_after_inf = coll_feasible = 0
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        p = env.agent_position.copy()
        n0 = len(lad.records)
        action, _ = pol.predict(obs, p)
        steps += 1
        r = lad.records[-1] if len(lad.records) > n0 else None
        if r is not None and r["infeasible"]:
            hh, gg, dd = pol.src.samples(p)
            k = mis_cardinality(np.column_stack([dd, hh, gg[0], gg[1]]))
            mis[k] += 1
        obs, _, term, trunc, info = env.step(action)
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
        if info.get("collision"):
            was_inf = bool(r is not None and r["infeasible"])
            coll_after_inf += int(was_inf)
            coll_feasible += int(not was_inf)
    lad.detach()

    recs = lad.records
    inf = [r for r in recs if r["infeasible"]]
    rec_ok = [r for r in inf if r["recovered"]]
    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    sp = oracle.path_length(start, goal)
    rungs = Counter(r["rung"] for r in inf)
    tiers = Counter(r["tier"] for r in rec_ok)
    m_vals = [r["m"] for r in rec_ok]
    switches = sum(1 for a, b in zip(recs, recs[1:]) if a["rung"] != b["rung"])
    return {
        "seed": seed, "outcome": outcome, "steps": steps,
        "success": int(outcome == "success"), "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "min_clearance": float(np.min(clear)),
        "spl": (float(sp / max(plen, sp, 1e-9)) if (outcome == "success" and sp) else 0.0),
        "n_infeasible": len(inf), "frac_infeasible": len(inf) / max(steps, 1),
        "n_recovered": len(rec_ok),
        "rungs": dict(rungs), "tiers": {str(k): v for k, v in tiers.items()},
        "mis": {str(k): v for k, v in mis.items()},
        "m_min": float(np.min(m_vals)) if m_vals else float("nan"),
        "m_median": float(np.median(m_vals)) if m_vals else float("nan"),
        "decay_floor_min": float(min((r["decay_floor"] for r in rec_ok
                                      if r.get("decay_floor") is not None), default=0.0)),
        "coll_after_infeasible": coll_after_inf, "coll_while_feasible": coll_feasible,
        "rung_switch_rate": switches / max(steps, 1),
        "alpha_prime_max": float(max((r.get("alpha_prime", 0.0) for r in inf), default=0.0)),
        "r15_admission_pass": int(sum(1 for r in inf if r.get("r15_admission"))),
    }


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def agg(recs):
    def m(k):
        v = [r[k] for r in recs if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    rungs, tiers, mis = Counter(), Counter(), Counter()
    for r in recs:
        rungs.update(r["rungs"])
        tiers.update(r["tiers"])
        mis.update(r["mis"])
    ct = Counter(r["collision_type"] for r in recs if r["collision_type"])
    n_inf = int(sum(r["n_infeasible"] for r in recs))
    return {
        "episodes": len(recs), "success": m("success"), "collision": m("collision"),
        "timeout": m("timeout"), "spl": m("spl"), "min_clearance": m("min_clearance"),
        "min_clearance_worst": float(np.min([r["min_clearance"] for r in recs])),
        "dynamic": ct.get("dynamic", 0), "static": ct.get("static", 0),
        "wall": ct.get("wall", 0),
        "frac_infeasible": m("frac_infeasible"), "n_infeasible": n_inf,
        "n_recovered": int(sum(r["n_recovered"] for r in recs)),
        "recovery_rate": (sum(r["n_recovered"] for r in recs) / n_inf) if n_inf else 0.0,
        "rungs": dict(rungs), "tiers": dict(tiers), "mis_cardinality": dict(mis),
        "m_min": float(np.nanmin([r["m_min"] for r in recs])) if n_inf else float("nan"),
        "m_median": float(np.nanmedian([r["m_median"] for r in recs])) if n_inf else float("nan"),
        "decay_floor_min": float(np.min([r["decay_floor_min"] for r in recs])),
        "coll_after_infeasible": int(sum(r["coll_after_infeasible"] for r in recs)),
        "coll_while_feasible": int(sum(r["coll_while_feasible"] for r in recs)),
        "rung_switch_rate": m("rung_switch_rate"),
        "alpha_prime_max": float(np.max([r["alpha_prime_max"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["dev"], choices=list(TIERS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--conditions", nargs="+", default=["fixed", "randomized"])
    ap.add_argument("--v-cap", type=float, default=V_CAP_DERIVED)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"### {LABEL}   base = Stage-3 T1, v_cap = {args.v_cap}\n")

    for tier in args.tiers:
        base, n = TIERS[tier]
        block = {}
        for cond in args.conditions:
            env = make_env(cond == "randomized")
            oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
            per = {}
            for arm in args.arms:
                recs = [run_episode(env, oracle, base + i, ARMS[arm], args.v_cap)
                        for i in range(n)]
                per[arm] = recs
                s = agg(recs)
                print(f"{tier}/{cond:11s} {arm:16s} infeas {s['frac_infeasible']:.4f} "
                      f"recovered {s['n_recovered']}/{s['n_infeasible']} "
                      f"({s['recovery_rate']:.3f})  rungs {s['rungs']}  tiers {s['tiers']}  "
                      f"m_med {s['m_median']:+.4f} m_min {s['m_min']:+.4f}  "
                      f"coll {s['collision']:.3f} clear {s['min_clearance']:.3f}", flush=True)
            env.close()
            b = {a: agg(per[a]) for a in args.arms}
            b["paired_vs_B4"] = {}
            base_r = per["B4_baseline"]
            for arm in args.arms:
                if arm == "B4_baseline":
                    continue
                t = per[arm]
                d = {}
                for key in ("collision", "success", "timeout"):
                    n01, n10, p = mcnemar([r[key] for r in base_r], [r[key] for r in t])
                    d[key] = {"B4": float(np.mean([r[key] for r in base_r])),
                              "arm": float(np.mean([r[key] for r in t])),
                              "n01": n01, "n10": n10, "p": p}
                b["paired_vs_B4"][arm] = d
                c = d["collision"]
                print(f"   paired {arm:16s} collision {c['B4']:.3f}->{c['arm']:.3f} "
                      f"p={c['p']:.4f}", flush=True)
            # Keep only what the paired tests need. Storing every episode's full record for
            # 5 arms x 200 episodes is what exhausted memory on the first attempt.
            keep = ("seed", "success", "collision", "timeout", "min_clearance", "spl",
                    "n_infeasible", "n_recovered", "frac_infeasible", "collision_type",
                    "coll_after_infeasible", "coll_while_feasible")
            b["episodes"] = {a: [{k: r[k] for k in keep} for r in per[a]] for a in args.arms}
            block[cond] = b
            # written per CONDITION, so a kill cannot lose a completed condition
            (out / f"stage4_{tier}_{cond}.json").write_text(
                json.dumps(b, indent=1, default=float))
            del per
        (out / f"stage4_{tier}.json").write_text(json.dumps(block, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
