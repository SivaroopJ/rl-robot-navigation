"""Week5-Phase7 / Stage 5: E1, the geometry-derived reconstruction covariance. Pre-registered in
PHASE7_PLAN.md 11.9 (9d38008), with the n = 1 derivation corrected in 11.9.18 (3b3a629) BEFORE
any code was written.

V0 FROZEN. alpha, tau, eps, v_cap, the objective, sample selection, the planner, the LiDAR
pipeline and the u = 0 fallback are all untouched. Recovery, where used, is Stage 4's LADDER
exactly as committed at 91bb969 -- never re-tuned. R1.5 is excluded. IMM, constant-turn,
covariance-into-the-barrier and uncertainty margins are NOT implemented and remain gated.

ARMS (11.9.9)
    A0  T1 + frozen u = 0          baseline, == Stage-4 B4
    A1  T1 + E1 + frozen u = 0     estimator main effect
    A2  T1 + LADDER                == Stage-4 LADDER
    A3  T1 + E1 + LADDER           interaction
    D-oracle                       DIAGNOSTIC ONLY, ledger-breaking, never a headline arm

THE PRIMARY CLAIM IS GUARANTEE PRESERVATION (T0), NOT COLLISIONS. At n = 200 the paired
collision test resolves only ~0.06-0.08 absolute, whereas the step-level endpoints rest on
10^4-10^5 steps. A null on collisions is expected and uninformative; it is never reported as
proof of safety.

VALIDATION TIER RUNS THE ADMISSION GATE A4 ONLY (11.9.13): A0 vs A1 on held-out seeds. It
decides whether the 200-episode final run is authorised. No Stage-5 conclusion is drawn from it,
and E1 is NOT tuned on it -- E1 has no free parameter to tune.

COMPARATOR FOR THE OPTIMISTIC TAIL. Stage-3 T1 in-environment: 0.397 (fixed) / 0.498
(randomized), episode-averaged. Phase 4's 0.142 is a DIFFERENT harness and subset and is not a
valid comparator. A0 measured in this same run is the primary comparator.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from dr_control.capped_velocity import V_CAP_DERIVED
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.policy_phase7 import Phase7Policy
from dr_control.recovery import RecoveryLadder, cbc, tier_of
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp0_analytic import DEV_SEED_BASE
from experiments.exp6_ppo_comparison import make_env, true_clearance
from experiments.exp7_2b_velocity_counterfactual import feas, mis_cardinality
from experiments.exp7_3_velocity_intervention import truth_dh_dt

LABEL = "Week5-Phase7 / Stage 5 / E1 geometry-derived reconstruction covariance"
OUT = Path("results/week5_phase7/stage5_estimator")
ALPHA, TAU = 0.4, 0.04
#: Metric threshold only (Stage 2). NEVER an input to any controller or estimator.
V_PHYS = {"fixed": 0.675, "randomized": 0.75}
#: arm -> (policy arm, recovery mode)
ARMS = {"A0_T1_baseline": ("T1_projection_cap", "none"),
        "A1_T1_E1": ("E1_geometric_R", "none"),
        "A2_T1_ladder": ("T1_projection_cap", "ladder"),
        "A3_T1_E1_ladder": ("E1_geometric_R", "ladder"),
        "D_oracle": ("D_oracle", "none")}
VALIDATION_ARMS = ("A0_T1_baseline", "A1_T1_E1")      # the A4 gate needs exactly these
TIERS = {"dev": (DEV_SEED_BASE, 20), "validation": (DEV_SEED_BASE + 1_000, 50),
         "final": (EVAL_SEED_BASE, 200)}


def run_episode(env, oracle, seed, arm, v_cap, v_phys):
    pol_arm, mode = ARMS[arm]
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = Phase7Policy(arm=pol_arm, v_cap=v_cap, world_size=env.WORLD_SIZE,
                       agent_radius=env.AGENT_RADIUS, static_obstacles=env.static_obstacles,
                       max_speed=env.MAX_SPEED, dt=env.dt, lidar_range=env.LIDAR_RANGE,
                       n_rays=env.N_LIDAR_RAYS)
    pol.reset(obs, env.agent_position)
    lad = RecoveryLadder(pol.ctrl, mode=mode) if mode != "none" else None

    tbuf = []
    traj, clear = [start.copy()], [true_clearance(env)]
    n_inf = n_single = n_multi = 0
    n_sp_steps = n_sp_rows = n_rows = 0
    tiers = Counter()
    rungs = Counter()
    err = []          # (e_used, e_raw, ttc)
    coll_after_inf = coll_feasible = 0
    term = trunc = False
    info = {}
    steps = 0
    while not (term or trunc):
        p = env.agent_position.copy()
        tbuf.insert(0, np.asarray(env.obstacle_positions, float).copy())
        if len(tbuf) > 5:
            tbuf.pop()
        if pol_arm == "D_oracle":
            pol.src.push_truth(env.obstacle_positions, env.obstacle_velocities)
        vel_now = np.asarray(env.obstacle_velocities, float).copy()
        h_t = true_clearance(env)

        action, _ = pol.predict(obs, p)
        steps += 1

        # The constraint set the controller just saw. RAW is the estimator's own output; the
        # arm's dh_dt is the T1 clamp of it, which is exactly ProjectionCappedSource.samples.
        hh, gg, raw = EstimatedLidarBarrierSource.samples(pol.src, p)
        used = np.maximum(raw, -v_cap) if pol_arm != "D_oracle" else raw
        xi = np.column_stack([used, hh, gg[0], gg[1]])
        infeasible = not feas(xi)
        if infeasible:
            n_inf += 1
            k = mis_cardinality(xi)
            n_single += int(k == 1)
            n_multi += int(k is not None and k >= 2)

        # Guarantee tier of the action actually returned, recomputed from the samples the
        # controller saw. NEVER a statement that the action is safe.
        m = cbc(xi, action, ALPHA)
        tiers[tier_of(m, TAU)] += 1
        if lad is not None:
            rungs[getattr(lad, "last_rung", None) or ("R0" if not infeasible else "unknown")] += 1

        # Super-physical rows: estimator output more negative than any obstacle can produce.
        sp = int(np.sum(raw < -v_phys - 1e-12))
        n_sp_rows += sp
        n_sp_steps += int(sp > 0)
        n_rows += len(raw)

        dt_true = truth_dh_dt(hh, gg, p, tbuf, vel_now)          # metrics channel only
        closing = max(-float(np.min(dt_true)), 0.0)
        ttc = h_t / closing if closing > 1e-6 else 1e3
        for j in range(len(used)):
            err.append((float(used[j] - dt_true[j]), float(raw[j] - dt_true[j]),
                        float(min(ttc, 1e3))))

        obs, _, term, trunc, info = env.step(action)
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
        if info.get("collision"):
            coll_after_inf += int(infeasible)
            coll_feasible += int(not infeasible)

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    sp_len = oracle.path_length(start, goal)
    e = np.array([r[0] for r in err]) if err else np.array([])
    e_raw = np.array([r[1] for r in err]) if err else np.array([])
    t = np.array([r[2] for r in err]) if err else np.array([])
    near = t < 1.0 if len(t) else np.array([], bool)
    tot = max(steps, 1)
    return {
        "seed": seed, "arm": arm, "outcome": outcome, "steps": steps,
        "success": int(outcome == "success"), "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "min_clearance": float(np.min(clear)),
        "spl": (float(sp_len / max(plen, sp_len, 1e-9))
                if (outcome == "success" and sp_len) else 0.0),
        # (3) M1, (4) M2 / total
        "n_infeasible": n_inf, "n_singleton": n_single, "n_multi": n_multi,
        "frac_infeasible": n_inf / tot, "frac_singleton": n_single / tot,
        "frac_multi": n_multi / tot,
        # (5) guarantee tiers
        "n_T0": tiers[0], "n_T1": tiers[1], "n_T2": tiers[2],
        "frac_T0": tiers[0] / tot, "frac_T1": tiers[1] / tot, "frac_T2": tiers[2] / tot,
        "rungs": dict(rungs),
        # (6) collisions
        "coll_after_infeasible": coll_after_inf, "coll_while_feasible": coll_feasible,
        # (1) accuracy, post-clamp (comparable to Stage-3 T1) and raw estimator output
        "p_e_pos_ttc1": float((e[near] > 0).mean()) if near.any() else float("nan"),
        "p_e_pos_ttc1_raw": float((e_raw[near] > 0).mean()) if near.any() else float("nan"),
        "n_ttc1": int(near.sum()),
        "n_e_pos_ttc1": int((e[near] > 0).sum()) if near.any() else 0,
        "p_e_pos": float((e > 0).mean()) if len(e) else float("nan"),
        "p_e_pessimistic": float((e < -0.05).mean()) if len(e) else float("nan"),
        "e_p95": float(np.percentile(e, 95)) if len(e) else float("nan"),
        "e_p99": float(np.percentile(e, 99)) if len(e) else float("nan"),
        "e_max": float(np.max(e)) if len(e) else float("nan"),
        "e_mean": float(np.mean(e)) if len(e) else float("nan"),
        # (2) super-physical
        "n_superphys_rows": n_sp_rows, "n_superphys_steps": n_sp_steps, "n_rows": n_rows,
        "frac_superphys_steps": n_sp_steps / tot,
        "frac_superphys_rows": n_sp_rows / max(n_rows, 1),
        "n_clamped_steps": pol.n_clamped_steps, "n_clamped_rows": pol.n_clamped_rows,
        "n_solver_fail": int(pol.n_solver_fail),
    }


# ------------------------------------------------------------------ statistics
def mcnemar(a, b):
    """Exact two-sided McNemar on the discordant pairs (as Stages 3 and 4)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_boot(a, b, rng, n=10000):
    """95 % CI on the paired difference mean(b) - mean(a), resampling EPISODE PAIRS."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (b[idx] - a[idx]).mean(axis=1)
    return float(np.mean(b - a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cluster_boot_rate(recs_a, recs_b, num, den, rng, n=10000):
    """95 % CI on a paired difference of POOLED step-level rates, resampling episodes.

    Steps within an episode are not independent, so a naive binomial interval would be
    anti-conservative. Pre-registered in 11.9.11.
    """
    na = np.array([r[num] for r in recs_a], float)
    da = np.array([r[den] for r in recs_a], float)
    nb = np.array([r[num] for r in recs_b], float)
    db = np.array([r[den] for r in recs_b], float)
    ra = float(na.sum() / max(da.sum(), 1))
    rb = float(nb.sum() / max(db.sum(), 1))
    idx = rng.integers(0, len(na), size=(n, len(na)))
    d = (nb[idx].sum(axis=1) / np.maximum(db[idx].sum(axis=1), 1)
         - na[idx].sum(axis=1) / np.maximum(da[idx].sum(axis=1), 1))
    return ra, rb, float(rb - ra), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def agg(recs):
    def m(k):
        v = [r[k] for r in recs if isinstance(r[k], (int, float)) and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    ct = Counter(r["collision_type"] for r in recs if r["collision_type"])
    steps = sum(r["steps"] for r in recs)
    return {
        "episodes": len(recs), "steps": int(steps),
        "success": m("success"), "collision": m("collision"), "timeout": m("timeout"),
        "dynamic": ct.get("dynamic", 0), "static": ct.get("static", 0),
        "wall": ct.get("wall", 0),
        "spl": m("spl"), "min_clearance": m("min_clearance"),
        "min_clearance_worst": float(np.min([r["min_clearance"] for r in recs])),
        "p_e_pos_ttc1_episode_mean": m("p_e_pos_ttc1"),
        "p_e_pos_ttc1_pooled": float(sum(r["n_e_pos_ttc1"] for r in recs)
                                     / max(sum(r["n_ttc1"] for r in recs), 1)),
        "n_ttc1": int(sum(r["n_ttc1"] for r in recs)),
        "p_e_pos_ttc1_raw_episode_mean": m("p_e_pos_ttc1_raw"),
        "p_e_pos": m("p_e_pos"), "p_e_pessimistic": m("p_e_pessimistic"),
        "e_p95": m("e_p95"), "e_p99": m("e_p99"), "e_max": m("e_max"), "e_mean": m("e_mean"),
        "frac_superphys_steps_pooled": float(sum(r["n_superphys_steps"] for r in recs)
                                             / max(steps, 1)),
        "frac_superphys_rows_pooled": float(sum(r["n_superphys_rows"] for r in recs)
                                            / max(sum(r["n_rows"] for r in recs), 1)),
        "frac_infeasible_pooled": float(sum(r["n_infeasible"] for r in recs) / max(steps, 1)),
        "frac_singleton_pooled": float(sum(r["n_singleton"] for r in recs) / max(steps, 1)),
        "frac_multi_pooled": float(sum(r["n_multi"] for r in recs) / max(steps, 1)),
        "n_infeasible": int(sum(r["n_infeasible"] for r in recs)),
        "n_singleton": int(sum(r["n_singleton"] for r in recs)),
        "n_multi": int(sum(r["n_multi"] for r in recs)),
        "frac_T0_pooled": float(sum(r["n_T0"] for r in recs) / max(steps, 1)),
        "frac_T1_pooled": float(sum(r["n_T1"] for r in recs) / max(steps, 1)),
        "frac_T2_pooled": float(sum(r["n_T2"] for r in recs) / max(steps, 1)),
        "coll_after_infeasible": int(sum(r["coll_after_infeasible"] for r in recs)),
        "coll_while_feasible": int(sum(r["coll_while_feasible"] for r in recs)),
        "frac_clamped_steps": float(sum(r["n_clamped_steps"] for r in recs) / max(steps, 1)),
        "n_solver_fail": int(sum(r["n_solver_fail"] for r in recs)),
    }


def admission_a4(fixed_a0, fixed_a1, rand_a0, rand_a1, rng):
    """A4 (11.9.13): E1 must show BOTH a lower optimistic tail AND a non-increase in the
    super-physical rate, in BOTH conditions. The conjunction is what makes the sluggish-filter
    failure mode non-passing."""
    out = {}
    # Explicit names: an earlier positional signature (A, B, ca, cb) paired fixed-A0 against
    # randomized-A0 instead of fixed-A0 against fixed-A1.
    for cond, a, b in (("fixed", fixed_a0, fixed_a1), ("randomized", rand_a0, rand_a1)):
        ra, rb, d, lo, hi = cluster_boot_rate(a, b, "n_e_pos_ttc1", "n_ttc1", rng)
        sa, sb, ds, slo, shi = cluster_boot_rate(a, b, "n_superphys_steps", "steps", rng)
        out[cond] = {
            "p_e_pos_ttc1_A0": ra, "p_e_pos_ttc1_A1": rb, "delta": d, "ci": [lo, hi],
            "criterion_i_lower_optimistic_tail": bool(d < 0),
            "superphys_A0": sa, "superphys_A1": sb, "delta_superphys": ds,
            "ci_superphys": [slo, shi],
            "criterion_ii_superphys_not_increased": bool(ds <= 0),
        }
        out[cond]["pass"] = bool(out[cond]["criterion_i_lower_optimistic_tail"]
                                 and out[cond]["criterion_ii_superphys_not_increased"])
    out["A4_PASS"] = bool(out["fixed"]["pass"] and out["randomized"]["pass"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=list(TIERS), default="validation")
    ap.add_argument("--arms", default=",".join(VALIDATION_ARMS))
    ap.add_argument("--condition", choices=["fixed", "randomized"], required=True)
    ap.add_argument("--v-cap", type=float, default=V_CAP_DERIVED)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.tier == "final" and set(a.arms.split(",")) == set(VALIDATION_ARMS):
        print("NOTE: the final tier is authorised only by the A4 gate; see 11.9.13.")
    base, n = TIERS[a.tier]
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}")
    rnd = a.condition == "randomized"
    env = make_env(rnd)   # Stages 3-4 use the default obstacle_speed = 0.675 in BOTH conditions
    oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    v_phys = V_PHYS[a.condition]

    recs = {}
    for arm in arms:
        rs = []
        for k in range(n):
            rs.append(run_episode(env, oracle, base + k, arm, a.v_cap, v_phys))
        recs[arm] = rs
        s = agg(rs)
        print(f"[{a.condition}/{a.tier}] {arm:16s} coll {s['collision']:.3f} "
              f"inf {s['frac_infeasible_pooled']:.4f} single {s['n_singleton']} "
              f"T0 {s['frac_T0_pooled']:.4f} "
              f"P(e>0|TTC<1) {s['p_e_pos_ttc1_pooled']:.4f} "
              f"sp {s['frac_superphys_steps_pooled']:.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"label": f"{LABEL} / {a.tier} / {a.condition}", "tier": a.tier,
           "condition": a.condition, "seed_base": base, "episodes": n,
           "v_cap": a.v_cap, "v_phys_metric_only": v_phys,
           "stage3_T1_reference_p_e_pos_ttc1": {"fixed": 0.397, "randomized": 0.498},
           "summary": {k: agg(v) for k, v in recs.items()},
           "episodes_raw": {k: v for k, v in recs.items()}}
    path = OUT / f"stage5_{a.tier}_{a.condition}{a.tag}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; results are never overwritten (11.9.15)")
    path.write_text(json.dumps(res, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
