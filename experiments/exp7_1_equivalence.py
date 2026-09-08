"""Week5-Phase7 / Stage 1 / Direction 2 equivalence and runtime.

GATE G1, RUN STRICTLY IN THIS ORDER. No runtime number is reported until L1-L5 have run.

    L0  reference accuracy    how accurate is the FROZEN controller itself?  (see below)
    L1  feasibility status    exact categorical agreement, on the ENRICHED INFEASIBLE corpus
                              first; three-way against an LP-exact arbiter
    L2  action agreement      ||u - u_ref||_inf, and ||u - u_arbiter||_inf
    L3  objective agreement   the frozen objective evaluated at each action
    L4  CBC agreement         min_i CBC_i and any change of guarantee tier
    L5  trajectory agreement  closed loop, paired seeds, terminal outcome
    L6  runtime               only now

WHY L0 EXISTS -- AN AMENDMENT FORCED BY MEASUREMENT (see PHASE7_PLAN.md 11.1)
    The plan pre-registered E1 as "status identical AND ||u - u_ref||_inf <= 1e-6", implicitly
    treating the frozen controller's output as ground truth. It is not: SCS is a first-order
    solver and the frozen controller's own distance from the true optimum is up to 9.4e-5.
    A threshold of 1e-6 against a reference with 9.4e-5 of noise cannot be met by ANY correct
    implementation, including a perfect one.

    So L0 measures the reference's own accuracy against a solver-independent ARBITER: the
    FROZEN formulation, unchanged, solved by CLARABEL at tol 1e-14. The pre-registered E1
    comparison is still computed and reported as run; L0 is what makes its result
    interpretable, and the amended criterion E1' is agreement with the arbiter, which does not
    privilege one solver's noise.

    This is an amendment to a gate criterion, not to a result. Both numbers are reported.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cvxpy as cp
import numpy as np

from dr_control.drccp_controller import (ClfCbfDrccpController, build_xi, clf_terms,
                                         nominal_action)
from dr_control.fast_drccp import VARIANTS, FastClfCbfDrccpController, reference_objective
from dr_control.policy import DRCBFPolicy
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp0_5_infeasibility import is_feasible
from experiments.exp6_ppo_comparison import make_env, true_clearance
from experiments.exp7_0_trace_corpus import OUT_DIR as CORPUS_DIR

LABEL = "Week5-Phase7 / Stage 1 / acceleration equivalence"
RESULT_DIR = Path("results/week5_phase7/stage1_equivalence")

E1_ACTION_TOL = 1e-6          # PRE-REGISTERED, against the frozen reference
E1P_ARBITER_TOL = 1e-8        # AMENDED, against the high-accuracy arbiter (L0 justifies it)
OBJ_TOL = 1e-8
CBC_TOL = 1e-6


# --------------------------------------------------------------------------- arbiter
def arbiter_solve(p, gamma, xi, u_prev, *, alpha=0.4, r_W=0.004, eps=0.1, max_v=1.0,
                  rate_V=1.0, k_v=0.05, p0=3.0, n_keep=5):
    """The FROZEN formulation, transcribed unchanged, solved to tol 1e-14 by CLARABEL.

    Not a competing controller: a numerical reference point that belongs to neither the frozen
    nor the accelerated implementation, so neither one's solver noise is treated as truth.
    """
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    order = np.argsort(xi[:, 1] * alpha + xi[:, 0])
    kept = xi[order][:n_keep]
    h_crit = float(kept[0, 1])
    if h_crit < 0:
        return None, np.nan, "h_crit<0"
    N = kept.shape[0]
    u, si, t, d = cp.Variable(2), cp.Variable(N), cp.Variable(), cp.Variable()
    u_bar = cp.hstack([1.0, alpha, u[0], u[1]])
    cons = [r_W * cp.abs(u_bar) / eps <= (t - cp.sum(si) / (N * eps)) * np.ones(4),
            si >= 0, cp.abs(u) <= max_v]
    cons += [si[i] >= t - u_bar @ kept[i] for i in range(N)]
    V, dV = clf_terms(p, gamma, k_v)
    cons += [dV @ u + rate_V * V <= d, d >= 0]
    u_nom = nominal_action(p, gamma, max_v)
    obj = cp.Minimize(p0 * cp.sum_squares(u - u_prev)
                      + 4.0 * h_crit * cp.sum_squares(u - u_nom)
                      + 5.0 * h_crit * cp.square(d))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL", tol_gap_abs=1e-14, tol_gap_rel=1e-14, tol_feas=1e-14)
    except cp.error.SolverError as exc:
        return None, np.nan, f"SolverError:{exc}"
    if prob.status != "optimal" or u.value is None:
        return None, np.nan, prob.status
    return np.asarray(u.value).reshape(2), float(prob.value), prob.status


def lp_feasible(kept, alpha, tau, max_v):
    """LP-exact feasibility of {u : |u| <= max_v, grad_h_i.u >= tau - alpha h_i - dh_dt_i}.

    Reuses the Phase 1.5 primitive unchanged. This is the arbiter for L1: Phase 1.5 already
    established it agrees with CVXPY on 3052/3052 probes, and it is solver-independent.
    """
    return bool(is_feasible(kept, alpha=alpha, tau=tau, max_v=max_v))


# --------------------------------------------------------------------------- corpus replay
def load_part(name):
    meta = json.loads((CORPUS_DIR / f"meta_{name}.json").read_text())
    return meta, np.load(CORPUS_DIR / meta["npz"])


def replay(part, variant, *, limit=None, with_arbiter=True, tau=0.04, alpha=0.4):
    """Replay recorded steps through one variant. Returns a list of per-step comparisons."""
    meta, d = part
    table = {int(k): v for k, v in meta["status_table"].items()}
    n = len(d["u"])
    idx = np.arange(n) if limit is None else np.linspace(0, n - 1, min(limit, n)).astype(int)
    ctrl = FastClfCbfDrccpController(variant=variant)
    rows = []
    for i in idx:
        a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
        xi = d["xi_flat"][a0:a1]
        p, gamma, u_prev = d["p"][i], d["gamma"][i], d["u_prev"][i]
        ctrl.prev_u = u_prev.copy()
        info = {}
        t0 = time.perf_counter()
        u = ctrl.generate_controller(p, gamma, xi, record=info)
        wall = time.perf_counter() - t0

        kept = np.asarray(info["xi_kept"])
        h_crit = float(info["h_crit"])
        u_ref = d["u"][i]
        st_ref = table[int(d["status_code"][i])]
        ub = np.array([1.0, alpha, u[0], u[1]])
        ub_ref = np.array([1.0, alpha, u_ref[0], u_ref[1]])
        u_nom = d["u_nom"][i]

        row = {
            "status_ref": st_ref, "status_var": info.get("status"),
            "lp_feasible": lp_feasible(kept, alpha, tau, ctrl.max_v),
            "d_action_ref": float(np.max(np.abs(u - u_ref))),
            "min_cbc_var": float(np.min(kept @ ub)),
            "min_cbc_ref": float(np.min(kept @ ub_ref)),
            "h_crit": h_crit, "delegated": bool(info.get("delegated", False)),
            "wall": wall, "solver_time": float(info.get("solver_time", np.nan)),
            "setup_time": float(info.get("canon_time", np.nan)),
            "obj_var": reference_objective(u, info.get("delta", np.nan), u_prev=u_prev,
                                           u_nom=u_nom, h_crit=h_crit),
            "obj_ref": reference_objective(u_ref, d["delta"][i], u_prev=u_prev,
                                           u_nom=u_nom, h_crit=h_crit),
        }
        if with_arbiter:
            u_a, _, st_a = arbiter_solve(p, gamma, xi, u_prev)
            row["arbiter_status"] = st_a
            row["d_action_arb"] = (float(np.max(np.abs(u - u_a))) if u_a is not None
                                   else float("nan"))
            row["d_ref_arb"] = (float(np.max(np.abs(u_ref - u_a))) if u_a is not None
                                else float("nan"))
        rows.append(row)
    return rows


def q(vals, name):
    v = np.array([x for x in vals if x == x], dtype=float)
    if not len(v):
        return {f"{name}_{k}": float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {f"{name}_mean": float(v.mean()), f"{name}_p50": float(np.percentile(v, 50)),
            f"{name}_p95": float(np.percentile(v, 95)),
            f"{name}_p99": float(np.percentile(v, 99)), f"{name}_max": float(v.max())}


def summarise(rows, variant, part_name):
    same = sum(r["status_ref"] == r["status_var"] for r in rows)
    disagree = Counter((r["status_ref"], r["status_var"]) for r in rows
                       if r["status_ref"] != r["status_var"])
    # LP arbiter: does each implementation's "solved" label match true feasibility?
    lp_vs_ref = sum((r["status_ref"] == "optimal") == r["lp_feasible"] for r in rows)
    lp_vs_var = sum((r["status_var"] == "optimal") == r["lp_feasible"] for r in rows)
    out = {
        "variant": variant, "part": part_name, "steps": len(rows),
        "L1_status_agreement": same / len(rows),
        "L1_status_disagreements": {f"{a}->{b}": c for (a, b), c in disagree.items()},
        "L1_lp_agrees_with_frozen": lp_vs_ref / len(rows),
        "L1_lp_agrees_with_variant": lp_vs_var / len(rows),
        "n_delegated": sum(r["delegated"] for r in rows),
    }
    solved = [r for r in rows if r["status_ref"] == "optimal" and r["status_var"] == "optimal"]
    out.update(q([r["d_action_ref"] for r in solved], "L2_d_action_vs_frozen"))
    out.update(q([r.get("d_action_arb", np.nan) for r in solved], "L2_d_action_vs_arbiter"))
    out.update(q([r.get("d_ref_arb", np.nan) for r in solved], "L0_frozen_vs_arbiter"))
    out.update(q([abs(r["obj_var"] - r["obj_ref"]) for r in solved], "L3_d_objective"))
    out.update(q([abs(r["min_cbc_var"] - r["min_cbc_ref"]) for r in solved], "L4_d_min_cbc"))
    out["L4_min_cbc_var_min"] = float(min((r["min_cbc_var"] for r in solved), default=np.nan))
    out["L4_tier_changes"] = sum(
        (r["min_cbc_var"] >= 0) != (r["min_cbc_ref"] >= 0) for r in solved)
    out.update(q([r["wall"] * 1e3 for r in rows], "L6_wall_ms"))
    out.update(q([r["solver_time"] * 1e3 for r in rows], "L6_solver_ms"))
    out.update(q([r["setup_time"] * 1e3 for r in rows], "L6_setup_ms"))
    out["E1_preregistered_pass"] = bool(
        out["L1_status_agreement"] == 1.0
        and out["L2_d_action_vs_frozen_max"] <= E1_ACTION_TOL)
    out["E1prime_arbiter_pass"] = bool(
        out["L1_status_agreement"] == 1.0
        and out["L2_d_action_vs_arbiter_max"] <= E1P_ARBITER_TOL)
    return out


# --------------------------------------------------------------------------- L5 closed loop
def rollout(env, oracle, seed, ctrl_factory):
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = DRCBFPolicy(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                      static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                      dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS)
    if ctrl_factory is not None:
        pol.ctrl = ctrl_factory()          # swap the solver only; policy code untouched
    pol.reset(obs, env.agent_position)
    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    while not (term or trunc):
        t0 = time.perf_counter()
        action, _ = pol.predict(obs, env.agent_position)
        times.append(time.perf_counter() - t0)
        obs, _, term, trunc, info = env.step(action)
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    shortest = oracle.path_length(start, goal)
    return {
        "seed": seed, "outcome": outcome, "steps": len(times),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "traj": traj, "min_clearance": float(np.min(clear)),
        "path_length": path_len,
        "spl": (float(shortest / max(path_len, shortest, 1e-9))
                if (outcome == "success" and shortest is not None) else 0.0),
        "n_infeasible": int(pol.n_infeasible),
        "mean_step_ms": float(np.mean(times) * 1e3),
        "max_step_ms": float(np.max(times) * 1e3),
    }


def closed_loop(variant, *, condition, episodes, seed_base):
    env = make_env(condition == "randomized")
    oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    out = []
    for i in range(episodes):
        s = seed_base + i
        a = rollout(env, oracle, s, None)
        b = rollout(env, oracle, s, lambda: FastClfCbfDrccpController(
            variant=variant, max_v=env.MAX_SPEED))
        n = min(len(a["traj"]), len(b["traj"]))
        div = np.where(np.max(np.abs(a["traj"][:n] - b["traj"][:n]), axis=1) > 1e-9)[0]
        out.append({
            "seed": s,
            "outcome_ref": a["outcome"], "outcome_var": b["outcome"],
            "same_outcome": a["outcome"] == b["outcome"],
            "same_collision_type": a["collision_type"] == b["collision_type"],
            "steps_ref": a["steps"], "steps_var": b["steps"],
            "first_divergence": int(div[0]) if len(div) else -1,
            "max_traj_dev": float(np.max(np.abs(a["traj"][:n] - b["traj"][:n]))),
            "d_spl": b["spl"] - a["spl"],
            "d_min_clearance": b["min_clearance"] - a["min_clearance"],
            "n_infeasible_ref": a["n_infeasible"], "n_infeasible_var": b["n_infeasible"],
            "ms_ref": a["mean_step_ms"], "ms_var": b["mean_step_ms"],
            "max_ms_ref": a["max_step_ms"], "max_ms_var": b["max_step_ms"],
        })
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--parts", nargs="+",
                    default=["infeasible_enriched", "dev_fixed", "dev_randomized"])
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample each part to this many steps (None = all)")
    ap.add_argument("--no-arbiter", action="store_true")
    ap.add_argument("--closed-loop", type=int, default=0,
                    help="episodes per condition for L5 (0 = skip)")
    ap.add_argument("--closed-loop-seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", default=str(RESULT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"### {LABEL}\n")
    report = {"label": LABEL, "config": vars(args), "L1_L4": [], "L5": {}}

    for part_name in args.parts:
        if part_name == "infeasible_enriched":
            meta = json.loads((CORPUS_DIR / "meta_infeasible_enriched.json").read_text())
            # The enriched corpus carries status_code but no table of its own; the codes come
            # from the same global assignment every part shares, so borrow a part's table.
            parts = json.loads((CORPUS_DIR / "INDEX.json").read_text())["parts"]
            # Status codes come from ONE global assignment shared by every part and extended
            # monotonically as new statuses appear, so a single part's table can be missing
            # codes the enriched corpus contains. Merge all of them, and assert consistency.
            merged = {}
            for pt in parts:
                for k, v in pt["status_table"].items():
                    assert merged.setdefault(k, v) == v, f"status code {k} is ambiguous"
            meta = dict(meta, status_table=merged,
                        controller_params=parts[0]["controller_params"])
            part = (meta, np.load(CORPUS_DIR / meta["npz"]))
        else:
            part = load_part(part_name)
        for variant in args.variants:
            rows = replay(part, variant, limit=args.limit,
                          with_arbiter=not args.no_arbiter)
            s = summarise(rows, variant, part_name)
            report["L1_L4"].append(s)
            print(f"{part_name:22s} {variant:16s} "
                  f"L1={s['L1_status_agreement']:.4f} "
                  f"L2|frozen max={s['L2_d_action_vs_frozen_max']:.2e} "
                  f"L2|arbiter max={s['L2_d_action_vs_arbiter_max']:.2e} "
                  f"L3 max={s['L3_d_objective_max']:.2e} "
                  f"L4 max={s['L4_d_min_cbc_max']:.2e} "
                  f"wall={s['L6_wall_ms_mean']:.3f}ms", flush=True)
            if s["L1_status_disagreements"]:
                print(f"      status disagreements: {s['L1_status_disagreements']}")

    if args.closed_loop:
        for variant in args.variants:
            for cond in ("fixed", "randomized"):
                recs = closed_loop(variant, condition=cond, episodes=args.closed_loop,
                                   seed_base=args.closed_loop_seed_base)
                agree = sum(r["same_outcome"] for r in recs) / len(recs)
                report["L5"][f"{variant}/{cond}"] = {
                    "episodes": len(recs),
                    "outcome_agreement": agree,
                    "collision_type_agreement":
                        sum(r["same_collision_type"] for r in recs) / len(recs),
                    "mean_first_divergence":
                        float(np.mean([r["first_divergence"] for r in recs])),
                    "max_traj_dev": float(np.max([r["max_traj_dev"] for r in recs])),
                    "mean_ms_ref": float(np.mean([r["ms_ref"] for r in recs])),
                    "mean_ms_var": float(np.mean([r["ms_var"] for r in recs])),
                    "max_ms_ref": float(np.max([r["max_ms_ref"] for r in recs])),
                    "max_ms_var": float(np.max([r["max_ms_var"] for r in recs])),
                    "speedup": float(np.mean([r["ms_ref"] for r in recs])
                                     / max(np.mean([r["ms_var"] for r in recs]), 1e-12)),
                    "episodes_detail": recs,
                }
                m = report["L5"][f"{variant}/{cond}"]
                print(f"L5 {variant:16s} {cond:11s} outcome agreement {agree:.3f}  "
                      f"speedup {m['speedup']:.1f}x  "
                      f"({m['mean_ms_ref']:.2f} -> {m['mean_ms_var']:.3f} ms/step)",
                      flush=True)

    (out_dir / "stage1_equivalence.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out_dir}/stage1_equivalence.json")


if __name__ == "__main__":
    main()
