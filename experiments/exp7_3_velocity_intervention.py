"""Week5-Phase7 / Stage 3: the M1 intervention, evaluated on BOTH sides. Pre-registered in
PHASE7_PLAN.md 11.7 and clarified in 11.7.8 before implementation.

V0 FROZEN. alpha, tau, eps, the objective, sample selection and the u = 0 fallback are
untouched; the intervention is strictly upstream, in xi construction. Recovery is NOT used --
that is Stage 4, and using it here would mask estimator-induced infeasibility.

ARMS  C0 frozen | T1 projection cap (primary) | T2 vector cap | D-oracle (DIAGNOSTIC ONLY)

THE EFFICACY GATE IS MECHANISM-SPECIFIC (11.7.8). Three quantities are reported separately:
singleton/M1 (minimal infeasible subset of cardinality 1), multi-constraint/M2 (cardinality
>= 2), and total. The pre-registered prediction is that T1 reduces the SINGLETON class while
residual M2 remains; reducing total infeasibility without reducing singletons does NOT pass.

SAFETY IS A VETO. A significant paired collision increase rejects the arm regardless of any
infeasibility gain. At n = 200 the collision test resolves only ~0.08 absolute, so a
non-significant result is reported as "failed to detect harm", NEVER as proof of safety; the
higher-n endpoints (minimum true clearance, P(e>0 | TTC<1s)) are primary alongside it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from dr_control.policy_phase7 import ARMS, Phase7Policy
from dr_control.capped_velocity import V_CAP_DERIVED
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp0_analytic import DEV_SEED_BASE
from experiments.exp6_ppo_comparison import make_env, true_clearance
from experiments.exp7_2b_velocity_counterfactual import feas, mis_cardinality

LABEL = "Week5-Phase7 / Stage 3 / M1 velocity intervention"
OUT = Path("results/week5_phase7/stage3_velocity_intervention")
ALPHA, TAU, OBS_R, R_ROBOT = 0.4, 0.04, 0.3, 0.3
TIERS = {"dev": (DEV_SEED_BASE, 20), "validation": (DEV_SEED_BASE + 1_000, 50),
         "final": (EVAL_SEED_BASE, 200)}


def truth_dh_dt(h, g, p, truth_buf, vel_now, tol=0.15):
    """Ground-truth dh/dt per row -- METRICS CHANNEL ONLY, identical attribution to Stage 2b."""
    q = np.asarray(p, float)[None, :] - (np.asarray(h)[:, None] + R_ROBOT) * np.asarray(g).T
    out = np.zeros(len(h))
    for j in range(len(h)):
        if j >= len(truth_buf) or vel_now is None or not len(truth_buf[j]):
            continue
        d = np.linalg.norm(truth_buf[j] - q[j][None, :], axis=1)
        o = int(np.argmin(d))
        if abs(float(d[o]) - OBS_R) < tol and o < len(vel_now):
            out[j] = -float(np.asarray(g)[:, j] @ vel_now[o])
    return out


def run_episode(env, oracle, seed, arm, v_cap):
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = Phase7Policy(arm=arm, v_cap=v_cap, world_size=env.WORLD_SIZE,
                       agent_radius=env.AGENT_RADIUS, static_obstacles=env.static_obstacles,
                       max_speed=env.MAX_SPEED, dt=env.dt, lidar_range=env.LIDAR_RANGE,
                       n_rays=env.N_LIDAR_RAYS)
    pol.reset(obs, env.agent_position)
    tbuf = []
    traj, clear = [start.copy()], [true_clearance(env)]
    n_inf = n_single = n_multi = 0
    err_rows = []
    last_inf = False
    coll_after_inf = coll_feasible = 0
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        p = env.agent_position.copy()
        tbuf.insert(0, np.asarray(env.obstacle_positions, float).copy())
        if len(tbuf) > 5:
            tbuf.pop()
        if arm == "D_oracle":
            pol.src.push_truth(env.obstacle_positions, env.obstacle_velocities)
        vel_now = np.asarray(env.obstacle_velocities, float).copy()
        h_t = true_clearance(env)

        n0 = pol.n_steps
        action, _ = pol.predict(obs, p)
        steps += 1
        # the constraint set the controller just saw
        hh, gg, dd = pol.src.samples(p)
        xi = np.column_stack([dd, hh, gg[0], gg[1]])
        infeasible = not feas(xi)
        if infeasible:
            n_inf += 1
            k = mis_cardinality(xi)
            n_single += int(k == 1)
            n_multi += int(k is not None and k >= 2)
        # estimator error, metrics channel
        dt_true = truth_dh_dt(hh, gg, p, tbuf, vel_now)
        closing = max(-float(np.min(dt_true)), 0.0)
        ttc = h_t / closing if closing > 1e-6 else 1e3
        for j in range(len(dd)):
            err_rows.append((float(dd[j] - dt_true[j]), float(min(ttc, 1e3))))

        obs, _, term, trunc, info = env.step(action)
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
        if info.get("collision"):
            coll_after_inf += int(infeasible)
            coll_feasible += int(not infeasible)
        last_inf = infeasible

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    sp = oracle.path_length(start, goal)
    e = np.array([r[0] for r in err_rows])
    t = np.array([r[1] for r in err_rows])
    near = t < 1.0
    return {
        "seed": seed, "arm": arm, "outcome": outcome, "steps": steps,
        "success": int(outcome == "success"), "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "min_clearance": float(np.min(clear)),
        "spl": (float(sp / max(plen, sp, 1e-9)) if (outcome == "success" and sp) else 0.0),
        "n_infeasible": n_inf, "n_singleton": n_single, "n_multi": n_multi,
        "frac_infeasible": n_inf / max(steps, 1),
        "frac_singleton": n_single / max(steps, 1), "frac_multi": n_multi / max(steps, 1),
        "coll_after_infeasible": coll_after_inf, "coll_while_feasible": coll_feasible,
        "n_clamped_steps": pol.n_clamped_steps, "n_clamped_rows": pol.n_clamped_rows,
        "frac_clamped": pol.n_clamped_steps / max(steps, 1),
        "n_solver_fail": int(pol.n_solver_fail),
        "e_mean": float(e.mean()) if len(e) else float("nan"),
        "p_e_pos": float((e > 0).mean()) if len(e) else float("nan"),
        "p_e_pos_ttc1": float((e[near] > 0).mean()) if near.any() else float("nan"),
        "n_ttc1": int(near.sum()),
        "p_e_pessimistic": float((e < -0.05).mean()) if len(e) else float("nan"),
        "superphys_rows": int((np.array([]) if not len(e) else e * 0).sum()),
    }


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(x, rng, n=10000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg(recs):
    def m(k):
        v = [r[k] for r in recs if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    ct = Counter(r["collision_type"] for r in recs if r["collision_type"])
    return {
        "episodes": len(recs), "success": m("success"), "collision": m("collision"),
        "timeout": m("timeout"), "spl": m("spl"), "min_clearance": m("min_clearance"),
        "min_clearance_worst": float(np.min([r["min_clearance"] for r in recs])),
        "dynamic": ct.get("dynamic", 0), "static": ct.get("static", 0), "wall": ct.get("wall", 0),
        "frac_infeasible": m("frac_infeasible"), "frac_singleton": m("frac_singleton"),
        "frac_multi": m("frac_multi"),
        "n_infeasible": int(sum(r["n_infeasible"] for r in recs)),
        "n_singleton": int(sum(r["n_singleton"] for r in recs)),
        "n_multi": int(sum(r["n_multi"] for r in recs)),
        "coll_after_infeasible": int(sum(r["coll_after_infeasible"] for r in recs)),
        "coll_while_feasible": int(sum(r["coll_while_feasible"] for r in recs)),
        "frac_clamped": m("frac_clamped"), "p_e_pos": m("p_e_pos"),
        "p_e_pos_ttc1": m("p_e_pos_ttc1"), "p_e_pessimistic": m("p_e_pessimistic"),
        "steps": float(np.mean([r["steps"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["dev"], choices=list(TIERS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--conditions", nargs="+", default=["fixed", "randomized"])
    ap.add_argument("--v-cap", type=float, default=V_CAP_DERIVED)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = np.random.default_rng(20260907)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"### {LABEL}   v_cap = {args.v_cap}\n")
    R = {"label": LABEL, "v_cap": args.v_cap, "config": vars(args), "tiers": {}}

    for tier in args.tiers:
        base, n_eps = TIERS[tier]
        R["tiers"][tier] = {}
        for cond in args.conditions:
            env = make_env(cond == "randomized")
            oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
            per_arm = {}
            for arm in args.arms:
                recs = [run_episode(env, oracle, base + i, arm, args.v_cap)
                        for i in range(n_eps)]
                per_arm[arm] = recs
                s = agg(recs)
                print(f"{tier}/{cond:11s} {arm:20s} infeas {s['frac_infeasible']:.4f} "
                      f"(single {s['frac_singleton']:.4f} multi {s['frac_multi']:.4f})  "
                      f"succ {s['success']:.3f} coll {s['collision']:.3f} "
                      f"clear {s['min_clearance']:.3f} "
                      f"P(e>0|TTC<1s) {s['p_e_pos_ttc1']:.3f} clamp {s['frac_clamped']:.3f}",
                      flush=True)
            env.close()
            block = {a: agg(per_arm[a]) for a in args.arms}
            block["paired_vs_C0"] = {}
            c0 = per_arm["C0_frozen"]
            for arm in args.arms:
                if arm == "C0_frozen":
                    continue
                t = per_arm[arm]
                d = {}
                for key in ("collision", "success", "timeout"):
                    n01, n10, p = mcnemar([r[key] for r in c0], [r[key] for r in t])
                    lo, hi = boot(np.array([r[key] for r in t]) -
                                  np.array([r[key] for r in c0]), rng)
                    d[key] = {"C0": float(np.mean([r[key] for r in c0])),
                              "arm": float(np.mean([r[key] for r in t])),
                              "diff": float(np.mean([r[key] for r in t]) -
                                            np.mean([r[key] for r in c0])),
                              "n01": n01, "n10": n10, "p": p, "ci95": [lo, hi]}
                for key in ("frac_infeasible", "frac_singleton", "frac_multi",
                            "min_clearance", "p_e_pos_ttc1"):
                    lo, hi = boot(np.array([r[key] for r in t], float) -
                                  np.array([r[key] for r in c0], float), rng)
                    d[key] = {"C0": float(np.nanmean([r[key] for r in c0])),
                              "arm": float(np.nanmean([r[key] for r in t])),
                              "diff": float(np.nanmean([r[key] for r in t]) -
                                            np.nanmean([r[key] for r in c0])),
                              "ci95": [lo, hi]}
                block["paired_vs_C0"][arm] = d
                cl = d["collision"]
                print(f"   paired {arm:20s} collision {cl['C0']:.3f}->{cl['arm']:.3f} "
                      f"p={cl['p']:.4f} CI[{cl['ci95'][0]:+.4f},{cl['ci95'][1]:+.4f}]  "
                      f"singleton {d['frac_singleton']['C0']:.4f}->"
                      f"{d['frac_singleton']['arm']:.4f}  "
                      f"multi {d['frac_multi']['C0']:.4f}->{d['frac_multi']['arm']:.4f}",
                      flush=True)
            block["episodes"] = {a: per_arm[a] for a in args.arms}
            R["tiers"][tier][cond] = block
        (out / f"stage3_{tier}.json").write_text(json.dumps(R["tiers"][tier], indent=1,
                                                            default=float))
    (out / "stage3_index.json").write_text(json.dumps(
        {k: v for k, v in R.items() if k != "tiers"}, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
