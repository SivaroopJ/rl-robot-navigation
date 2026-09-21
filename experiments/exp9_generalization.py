"""Week 6 / Generalization Suite runner. Arm A (original) vs arm B (updated), paired.

Specified in MD_files/week6/generalization/MAP_SUITE_SPEC.md and approved before implementation.
Nothing under robot_env/, evaluation/, dr_control/ or optional_navigation/ is modified, and no
Phase 1-7 artefact is read for anything but the M0 regression gate.

ARMS, fixed, never tuned:
    A  DRCBFPolicy(use_planner=True)                       the frozen Phase-6 stack
    B  Phase7Policy(arm="T1_projection_cap", v_cap=0.96)   Stage-3 T1 cap
       + RecoveryLadder(mode="ladder")                     Stage-4 ladder, as committed 91bb969

v_cap = 0.96 EVERYWHERE, for every map, every speed, every radius. LiDAR range stays 5.0 m at
every world size. No per-map, per-speed or per-count tuning of anything.

FAIRNESS, asserted at runtime for every episode (spec 6): identical map, seed, start, goal,
initial obstacle states, horizon, motion model and RNG stream. Only the controller differs.
Neither arm reads obs[28:52] or any ground-truth obstacle state.

MODES
    --mode g0      the M0 regression gate. Must reproduce Phase 6 and the Phase-7 final
                   comparison EXACTLY before any map is generated.
    --mode smoke   one short episode per family, both arms, with the fairness assertions on.
    --mode screen  50 episodes per cell, both arms.
    --mode final   200 (benchmark) / 50 (X1-X7) episodes per cell, both arms.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dr_control.capped_velocity import V_CAP_DERIVED
from dr_control.policy import DRCBFPolicy
from dr_control.policy_phase7 import Phase7Policy
from dr_control.recovery import RecoveryLadder
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import true_clearance
from generalization import suite
from generalization.scenario_env import ScenarioEnv
from robot_env.robot_nav_env import RobotNavEnv

LABEL = "Week6 / Generalization Suite"
OUT = Path("results/week6_generalization")
ALPHA, TAU = 0.4, 0.04

#: The M0 gate targets. Phase 6 for arm A, the Phase-7 final comparison for arm B.
G0_TARGETS = {
    ("A", "deterministic"): (0.855, 0.140), ("A", "stochastic"): (0.725, 0.275),
    ("B", "deterministic"): (0.925, 0.065), ("B", "stochastic"): (0.815, 0.180),
}
EVAL_SEED_BASE_M0 = 1_000_000       # Phases 1-7 block; used ONLY by the gate


# --------------------------------------------------------------------------- construction
def make_env(entry, motion_model, *, config_path):
    """Build the environment for one (configuration, motion model) cell."""
    randomize = motion_model == "stochastic"
    kw = dict(config_path=config_path, render_mode=None, use_reward_shaping=True,
              randomize_dynamic_obstacles=randomize)
    sc = entry.get("scenario") or {}
    if entry.get("env") == "scenario":
        return ScenarioEnv(start_goal=sc.get("start_goal"),
                           reactive_walls=sc.get("reactive_walls"),
                           scripted=sc.get("scripted"), **kw)
    return RobotNavEnv(**kw)


def make_arm(arm, env):
    """The two controllers. Both take their geometry from the env, so they follow the map."""
    common = dict(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                  static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                  dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS,
                  use_planner=True)
    if arm == "A":
        return DRCBFPolicy(**common), None
    pol = Phase7Policy(arm="T1_projection_cap", v_cap=V_CAP_DERIVED, **common)
    return pol, "ladder"


# --------------------------------------------------------------------------- one episode
def run_episode(env, entry, arm, seed, oracle):
    obs, _ = env.reset(seed=seed)
    fair = {"start": env.agent_position.copy(), "goal": env.target_position.copy(),
            "obstacles": np.asarray(env.obstacle_positions, float).copy(),
            "max_steps": int(env.MAX_STEPS), "world": float(env.WORLD_SIZE),
            "n_static": len(env.static_obstacles)}
    pol, mode = make_arm(arm, env)
    pol.reset(obs, env.agent_position)
    lad = RecoveryLadder(pol.ctrl, mode=mode) if mode else None

    start, goal = env.agent_position.copy(), env.target_position.copy()
    traj, clear, times = [start.copy()], [true_clearance(env)], []
    term = trunc = False
    info = {}
    steps = 0
    infeasible_before = False
    coll_after_inf = coll_feasible = 0
    seen = 0
    rungs, tiers, ms = Counter(), Counter(), []
    n_recovered = 0
    inf_steps = []
    while not (term or trunc):
        t0 = time.perf_counter()
        action, _ = pol.predict(obs, env.agent_position)
        times.append(time.perf_counter() - t0)
        if lad is not None:
            for e in lad.records[seen:]:
                if e["infeasible"]:
                    rungs[e["rung"]] += 1
                    t = 0 if e["m"] >= lad.tau else (1 if e["m"] >= 0.0 else 2)
                    tiers[t] += 1
                    ms.append(float(e["m"]))
                    n_recovered += int(e["recovered"])
            seen = len(lad.records)
        infeasible_before = pol.last_step_infeasible
        if infeasible_before:
            inf_steps.append(steps)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        traj.append(env.agent_position.copy())
        clear.append(true_clearance(env))
        if info.get("collision"):
            coll_after_inf += int(infeasible_before)
            coll_feasible += int(not infeasible_before)
    if lad is not None:
        lad.detach()

    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    sp = oracle.path_length(start, goal)
    rec = {
        "seed": int(seed), "arm": arm, "outcome": outcome, "steps": steps,
        "success": int(outcome == "success"), "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "min_clearance": float(np.min(clear)),
        "path_length": plen,
        "spl": (float(sp / max(plen, sp, 1e-9)) if (outcome == "success" and sp) else 0.0),
        "n_infeasible": int(pol.n_infeasible), "n_solver_fail": int(pol.n_solver_fail),
        "frac_infeasible": pol.n_infeasible / max(steps, 1),
        "coll_after_infeasible": coll_after_inf, "coll_while_feasible": coll_feasible,
        "planner_failed": int(pol.planner_failed),
        "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
        "min_cbc_solved": (float(pol.min_cbc_solved)
                           if np.isfinite(pol.min_cbc_solved) else float("nan")),
        "mean_step_time": float(np.mean(times)) if times else float("nan"),
        "n_recovered": int(n_recovered),
        "rungs": dict(rungs), "tiers": {str(k): v for k, v in tiers.items()},
        "m_median": float(np.median(ms)) if ms else float("nan"),
        "m_min": float(np.min(ms)) if ms else float("nan"),
        "first_infeasible_step": (inf_steps[0] if inf_steps else -1),
        "wall_events": list(getattr(env, "wall_events", [])),
    }
    return rec, fair


def assert_paired(fa, fb, cid, seed):
    """Fairness invariants (spec 6). Raises rather than warns -- a violation invalidates the pair."""
    assert np.array_equal(fa["start"], fb["start"]), f"{cid} seed {seed}: start differs"
    assert np.array_equal(fa["goal"], fb["goal"]), f"{cid} seed {seed}: goal differs"
    assert np.array_equal(fa["obstacles"], fb["obstacles"]), f"{cid} seed {seed}: obstacles differ"
    assert fa["max_steps"] == fb["max_steps"], f"{cid} seed {seed}: horizon differs"
    assert fa["world"] == fb["world"], f"{cid} seed {seed}: world differs"
    assert fa["n_static"] == fb["n_static"], f"{cid} seed {seed}: static geometry differs"


# --------------------------------------------------------------------------- one cell
def run_cell(entry, motion_model, *, episodes, seed_base, cfg_dir, quiet=False):
    cfg_path = cfg_dir / f"{entry['id']}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(entry["config"], indent=1))

    envs = {a: make_env(entry, motion_model, config_path=str(cfg_path)) for a in ("A", "B")}
    oracle = ShortestPathOracle(envs["A"].WORLD_SIZE, envs["A"].AGENT_RADIUS,
                                envs["A"].static_obstacles)
    recs = {"A": [], "B": []}
    t0 = time.time()
    for k in range(episodes):
        s = seed_base + k
        ra, fa = run_episode(envs["A"], entry, "A", s, oracle)
        rb, fb = run_episode(envs["B"], entry, "B", s, oracle)
        assert_paired(fa, fb, entry["id"], s)
        recs["A"].append(ra)
        recs["B"].append(rb)
    for e in envs.values():
        e.close()
    dt = time.time() - t0
    if not quiet:
        sa, sb = agg(recs["A"]), agg(recs["B"])
        print(f"  {entry['id']:24s} {motion_model:13s} "
              f"A succ {sa['success']:.2f} coll {sa['collision']:.2f} | "
              f"B succ {sb['success']:.2f} coll {sb['collision']:.2f} | "
              f"{dt/max(2*episodes,1):.2f} s/ep", flush=True)
    return recs, dt


def agg(recs):
    def m(k):
        v = [r[k] for r in recs if isinstance(r[k], (int, float)) and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    ct = Counter(r["collision_type"] for r in recs if r["collision_type"])
    steps = sum(r["steps"] for r in recs)
    rungs, tiers = Counter(), Counter()
    for r in recs:
        rungs.update(r["rungs"]); tiers.update(r["tiers"])
    ms = [r["m_median"] for r in recs if r["m_median"] == r["m_median"]]
    return {
        "episodes": len(recs), "steps": int(steps),
        "success": m("success"), "collision": m("collision"), "timeout": m("timeout"),
        "dynamic": ct.get("dynamic", 0), "static": ct.get("static", 0), "wall": ct.get("wall", 0),
        "spl": m("spl"), "path_length": m("path_length"),
        "min_clearance": m("min_clearance"),
        "min_clearance_worst": float(np.min([r["min_clearance"] for r in recs])),
        "frac_infeasible": float(sum(r["n_infeasible"] for r in recs) / max(steps, 1)),
        "n_infeasible": int(sum(r["n_infeasible"] for r in recs)),
        "n_recovered": int(sum(r["n_recovered"] for r in recs)),
        "coll_after_infeasible": int(sum(r["coll_after_infeasible"] for r in recs)),
        "coll_while_feasible": int(sum(r["coll_while_feasible"] for r in recs)),
        "planner_failures": int(sum(r["planner_failed"] for r in recs)),
        "n_solver_fail": int(sum(r["n_solver_fail"] for r in recs)),
        "mean_u_dev": m("mean_u_dev"), "min_cbc_solved": m("min_cbc_solved"),
        "mean_step_time_ms": m("mean_step_time") * 1e3,
        "rungs": dict(rungs), "tiers": dict(tiers),
        "m_median_of_episode_medians": float(np.median(ms)) if ms else float("nan"),
    }


# --------------------------------------------------------------------------- modes
def mode_g0(args):
    """The M0 regression gate. EXACT reproduction required; otherwise stop."""
    print(f"{LABEL} / G0 -- M0 canonical regression gate")
    print(f"config.json unchanged, seeds EVAL_SEED_BASE {EVAL_SEED_BASE_M0}..+{args.episodes-1}\n")
    entry = {"id": "M0_canonical", "config": json.loads(Path("config.json").read_text()),
             "env": "base", "scenario": {}, "stress": False}
    entry["config"]["environment"]["n_dynamic_obstacles"] = 6
    entry["config"]["environment"]["obstacle_speed"] = 0.675
    out, ok = {}, True
    for mm in ("deterministic", "stochastic"):
        recs, _ = run_cell(entry, mm, episodes=args.episodes, seed_base=EVAL_SEED_BASE_M0,
                           cfg_dir=OUT / "configs" / "M0", quiet=True)
        for a in ("A", "B"):
            s = agg(recs[a])
            tsucc, tcoll = G0_TARGETS[(a, mm)]
            good = (abs(s["success"] - tsucc) < 1e-9) and (abs(s["collision"] - tcoll) < 1e-9)
            ok &= good
            out[f"{a}_{mm}"] = {"success": s["success"], "collision": s["collision"],
                                "target_success": tsucc, "target_collision": tcoll,
                                "pass": bool(good)}
            print(f"  arm {a} {mm:13s} success {s['success']:.3f} (want {tsucc:.3f})  "
                  f"collision {s['collision']:.3f} (want {tcoll:.3f})  "
                  f"{'PASS' if good else 'FAIL'}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "g0_regression.json"
    p.write_text(json.dumps({"label": f"{LABEL} / G0", "episodes": args.episodes,
                             "results": out, "G0_PASS": bool(ok)}, indent=2))
    print(f"\nG0: {'PASS' if ok else 'FAIL'}   -> {p}")
    if not ok:
        raise SystemExit("G0 FAILED -- stop. Do not generate or screen Week 6 maps.")


def mode_smoke(args):
    """One short episode per family, both arms, fairness assertions live."""
    man = suite.build_manifest()
    seen, rows = set(), []
    print(f"{LABEL} / smoke -- one configuration per family, {args.episodes} episode(s)\n")
    for e in man:
        if e["family"] in seen:
            continue
        seen.add(e["family"])
        mm = e["motion_models"][0]
        recs, dt = run_cell(e, mm, episodes=args.episodes,
                            seed_base=suite.SCREEN_SEED_BASE,
                            cfg_dir=OUT / "configs" / "smoke")
        rows.append({"id": e["id"], "family": e["family"], "motion_model": mm,
                     "sec_per_episode": dt / max(2 * args.episodes, 1),
                     "A": agg(recs["A"]), "B": agg(recs["B"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke.json").write_text(json.dumps(
        {"label": f"{LABEL} / smoke", "families": len(rows), "rows": rows}, indent=2))
    rate = float(np.mean([r["sec_per_episode"] for r in rows]))
    b = suite.budget(man)
    print(f"\nfamilies covered {len(rows)}   measured {rate:.2f} s/episode")
    print(f"projection at that rate: screening {b['screening_episodes']*rate/3600:.1f} h, "
          f"final {b['final_episodes']*rate/3600:.1f} h, "
          f"total {b['total_episodes']*rate/3600:.1f} h single-process "
          f"({b['total_episodes']*rate/3600/8:.1f} h at 8-way)")
    print("  (proposal assumed 5.65 s/episode; the measured rate above supersedes it)")


def mode_run(args, which):
    man = suite.build_manifest()
    cells = suite.screening_cells(man) if which == "screen" else suite.final_cells(man)
    if args.family:
        cells = [(e, mm) for e, mm in cells if e["family"] in args.family.split(",")]
    tier_dir = OUT / which
    tier_dir.mkdir(parents=True, exist_ok=True)
    print(f"{LABEL} / {which} -- {len(cells)} cells\n")
    rates = []
    for i, (e, mm) in enumerate(cells, 1):
        p = tier_dir / f"{e['id']}__{mm}.json"
        if p.exists():
            print(f"  [{i}/{len(cells)}] {p.name} exists, skipping (never overwritten)")
            continue
        n = (suite.SCREEN_EPISODES if which == "screen" else e["final_episodes"])
        base = (suite.SCREEN_SEED_BASE if which == "screen" else suite.GEN_EVAL_SEED_BASE)
        recs, dt = run_cell(e, mm, episodes=n, seed_base=base,
                            cfg_dir=OUT / "configs" / which)
        rates.append(dt / max(2 * n, 1))
        p.write_text(json.dumps({
            "label": f"{LABEL} / {which}", "id": e["id"], "family": e["family"],
            "level": e["level"], "classification": e["classification"],
            "difficulty": e["difficulty"], "stress": e["stress"],
            "motion_model": mm, "episodes": n, "seed_base": base,
            "meta": e["meta"], "seconds": dt,
            "A": agg(recs["A"]), "B": agg(recs["B"]),
            "episodes_A": recs["A"], "episodes_B": recs["B"]}, indent=1))
        if len(rates) in (5, 10, 20) or i == len(cells):
            r = float(np.mean(rates))
            b = suite.budget(man)
            print(f"    [rate] {r:.2f} s/ep over {len(rates)} cells -> "
                  f"{which} projection {(b['screening_episodes'] if which=='screen' else b['final_episodes'])*r/3600:.1f} h "
                  f"single-process, {(b['screening_episodes'] if which=='screen' else b['final_episodes'])*r/3600/8:.1f} h at 8-way",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["g0", "smoke", "screen", "final"], required=True)
    ap.add_argument("--episodes", type=int, default=200,
                    help="g0/smoke only; screen and final take their counts from the manifest")
    ap.add_argument("--family", default=None, help="comma-separated family filter")
    a = ap.parse_args()
    if a.mode == "g0":
        mode_g0(a)
    elif a.mode == "smoke":
        mode_smoke(a)
    else:
        mode_run(a, a.mode)


if __name__ == "__main__":
    main()
