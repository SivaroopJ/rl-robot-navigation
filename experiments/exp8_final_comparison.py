"""FINAL apples-to-apples evaluation of the UPDATED DR-CBF on the Phase-6 canonical harness.

EVALUATION / REPRODUCTION ONLY. No controller logic is written or modified here, nothing is
tuned, no seed or episode count changes, and no experimental branch is opened from the outcome.

CANONICAL BASIS
    experiments/exp6_ppo_comparison.py and results/phase6/exp6_{fixed,randomized}.json, which
    evaluated the initial DR-CBF and all twelve PPO checkpoints inside ONE harness on ONE seed
    block with ONE metric function, storing per-episode records for every arm. Those records are
    READ, never recomputed: PPO is not rerun.

    Environment, verbatim from exp6: make_env(randomize) -> RobotNavEnv(config.json,
    n_dynamic_obstacles=6, obstacle_speed=0.675, use_reward_shaping=True), max_steps = 500,
    dt = 0.1, 24-ray LiDAR at range 5.0, single integrator with max_speed = 1.0, the five fixed
    rectangles, seeds EVAL_SEED_BASE + 0..199, 200 paired episodes per condition.

    Metrics and statistics are exp6's OWN functions -- episode_record, summarise, mcnemar,
    paired_ci -- imported unchanged, so the arithmetic is identical rather than re-derived.

WEEK-7 PROVENANCE NOTE, contextual only
    results/week7/SUMMARY.txt evaluated the same _v4 checkpoints with obstacle_speed_range =
    [0.6, 0.75] (speed sampled per obstacle), not the scalar 0.675 exp6 uses. That is why its
    aggregate PPO numbers differ. It is NOT the comparator and is never used for inference.

CONTROLLER UNDER TEST
    Phase7Policy(arm="T1_projection_cap", v_cap=0.96, use_planner=True) + RecoveryLadder(
    mode="ladder"), i.e. the accepted Stage-3 T1 cap and the committed Stage-4 LADDER from
    91bb969, untouched. The policy and the ladder are constructed PER EPISODE, exactly as
    experiments/exp7_4_recovery.py does. No E1, IMM/CT, uncertainty, D-oracle, R1, R2-alone or
    R1.5, and no privileged ground truth: the controller reads obs[0:28] only.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np

from dr_control.capped_velocity import V_CAP_DERIVED, ProjectionCappedSource
from dr_control.policy_phase7 import Phase7Policy
from dr_control.recovery import RecoveryLadder
from dr_control.velocity_tracker import LidarVelocityTracker
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import (episode_record, make_env, mcnemar, paired_ci,
                                             summarise, true_clearance)

LABEL = "Week5-Phase7 / FINAL / updated DR-CBF on the Phase-6 canonical harness"
OUT = Path("results/week5_phase7/final_comparison")
PHASE6 = {c: Path(f"results/phase6/exp6_{c}.json") for c in ("fixed", "randomized")}
DR_KEY = "DR-CBF (LiDAR, A*)"


# --------------------------------------------------------------------------- integrity
def integrity_checks(env):
    """Pre-run checks, per the approved plan. Raises on any failure; returns the evidence."""
    out = {}

    out["EVAL_SEED_BASE"] = int(EVAL_SEED_BASE)
    assert EVAL_SEED_BASE == 1_000_000, "EVAL_SEED_BASE changed"

    obs, _ = env.reset(seed=EVAL_SEED_BASE)
    pol = build_policy(env)
    pol.reset(obs, env.agent_position)
    lad = RecoveryLadder(pol.ctrl, mode="ladder")

    out["barrier_source"] = type(pol.src).__name__
    out["tracker"] = type(pol.src.tracker).__name__
    out["v_cap"] = float(pol.src.v_cap)
    out["ladder_mode"] = lad.mode
    out["ladder_alpha"] = float(lad.alpha)
    out["ladder_tau"] = float(lad.tau)
    out["use_planner"] = bool(pol.use_planner)
    assert type(pol.src) is ProjectionCappedSource, out["barrier_source"]
    assert type(pol.src.tracker) is LidarVelocityTracker, out["tracker"]
    assert pol.src.v_cap == V_CAP_DERIVED == 0.96
    assert lad.mode == "ladder"
    assert pol.use_planner is True

    # No PPO checkpoint or SB3 policy can be reached from the controller's own package.
    banned_mods, banned_paths = set(), set()
    for f in sorted(Path("dr_control").glob("*.py")) + [Path("optional_navigation/planner.py")]:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):                       # strip docstrings before scanning
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
            if isinstance(node, ast.Import):
                for al in node.names:
                    if al.name.split(".")[0] in ("stable_baselines3", "torch", "sb3_contrib"):
                        banned_mods.add(f"{f}:{al.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in ("stable_baselines3", "torch", "sb3_contrib"):
                    banned_mods.add(f"{f}:{node.module}")
        if "models/" in ast.unparse(tree) or '"models"' in ast.unparse(tree):
            banned_paths.add(str(f))
    out["sb3_or_torch_imports_in_controller"] = sorted(banned_mods)
    out["models_path_references_in_controller"] = sorted(banned_paths)
    assert not banned_mods and not banned_paths

    # Observation-corruption invariance: obs[28:52] is the privileged ground-truth slice.
    acts = []
    for corrupt in (False, True):
        p = build_policy(env)
        o = obs.copy()
        if corrupt:
            o[28:52] = -7.5
        p.reset(o, env.agent_position)
        RecoveryLadder(p.ctrl, mode="ladder")
        a, _ = p.predict(o, env.agent_position, deterministic=True)
        acts.append(np.asarray(a, float))
    out["obs_corruption_invariant"] = bool(np.array_equal(acts[0], acts[1]))
    assert out["obs_corruption_invariant"]

    lad.detach()
    return out


# --------------------------------------------------------------------------- the run
def build_policy(env):
    return Phase7Policy(arm="T1_projection_cap", v_cap=V_CAP_DERIVED,
                        world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                        static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                        dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS,
                        use_planner=True)


def run_updated_drcbf(*, episodes, randomize, oracle, env):
    """exp6.run_drcbf's loop, with the policy replaced by T1 + LADDER. Metrics unchanged."""
    recs = []
    for i in range(episodes):
        s = EVAL_SEED_BASE + i
        obs, _ = env.reset(seed=s)
        start, goal = env.agent_position.copy(), env.target_position.copy()
        pol = build_policy(env)                       # per episode, as exp7_4_recovery does
        pol.reset(obs, env.agent_position)
        lad = RecoveryLadder(pol.ctrl, mode="ladder")
        traj, clear, times = [start.copy()], [true_clearance(env)], []
        term = trunc = False
        info = {}
        steps = 0
        infeasible_before_end = False
        while not (term or trunc):
            t0 = time.perf_counter()
            action, _ = pol.predict(obs, env.agent_position, deterministic=True)
            times.append(time.perf_counter() - t0)
            infeasible_before_end = pol.last_step_infeasible
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            traj.append(env.agent_position.copy())
            clear.append(true_clearance(env))
        outcome = ("success" if info.get("success") else
                   "collision" if info.get("collision") else "timeout")
        # SECONDARY diagnostics only; the headline metric set is exp6's.
        rungs, tiers, n_rec = {}, {}, 0
        for e in lad.records:
            rungs[e["rung"]] = rungs.get(e["rung"], 0) + 1
            if e["infeasible"]:
                t = 0 if e["m"] >= lad.tau else (1 if e["m"] >= 0.0 else 2)
                tiers[t] = tiers.get(t, 0) + 1
                n_rec += int(e["recovered"])
        recs.append(episode_record(
            env, s, oracle, outcome, steps, traj, start, goal, info,
            extra={"clearances": clear, "step_times": times,
                   "planner_failed": int(pol.planner_failed),
                   "n_infeasible": pol.n_infeasible,
                   "n_solver_fail": pol.n_solver_fail,
                   "collided_after_infeasible": int(outcome == "collision"
                                                    and infeasible_before_end),
                   "mean_u_dev": float(np.mean(pol.u_dev)) if pol.u_dev else float("nan"),
                   "min_cbc_solved": (float(pol.min_cbc_solved)
                                      if np.isfinite(pol.min_cbc_solved) else float("nan")),
                   "n_recovered": int(n_rec), "rungs": rungs, "tiers": tiers}))
        lad.detach()
    return recs


# --------------------------------------------------------------------------- comparison
def dyn_flag(recs):
    return [int(r["collision_type"] == "dynamic") for r in recs]


def compare_full(x, y, label):
    """exp6's own mcnemar + paired_ci, on success, collision and dynamic collision.

    Sign convention is exp6's: positive diff = x (the updated DR-CBF) higher.
    """
    out = {}
    for key, a, b in (("success", [r["success"] for r in x], [r["success"] for r in y]),
                      ("collision", [r["collision"] for r in x], [r["collision"] for r in y]),
                      ("dynamic_collision", dyn_flag(x), dyn_flag(y))):
        out[f"{label} [{key}]"] = {**mcnemar(a, b), **paired_ci(a, b)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=["fixed", "randomized"], required=True)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    randomize = a.condition == "randomized"
    env = make_env(randomize)
    oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)

    checks = integrity_checks(env)
    print("INTEGRITY:", json.dumps(checks), flush=True)

    t0 = time.time()
    upd = run_updated_drcbf(episodes=a.episodes, randomize=randomize, oracle=oracle, env=env)
    env.close()
    print(f"updated DR-CBF: {a.episodes} episodes in {time.time() - t0:.0f}s", flush=True)

    ref = json.loads(PHASE6[a.condition].read_text())["conditions"][a.condition]["arms"]
    init = ref[DR_KEY]["episodes"]
    assert [r["seed"] for r in init] == [r["seed"] for r in upd], "seed pairing mismatch"

    summ = {"updated_DR-CBF (T1+LADDER)": summarise(upd, "updated DR-CBF")}
    for k, v in ref.items():
        summ[k] = v["summary"]

    paired = {}
    paired.update(compare_full(upd, init, "updated vs initial DR-CBF"))
    for k, v in ref.items():
        if k == DR_KEY:
            continue
        assert [r["seed"] for r in v["episodes"]] == [r["seed"] for r in upd]
        paired.update(compare_full(upd, v["episodes"], f"updated vs {k}"))

    res = {"label": f"{LABEL} / {a.condition}", "condition": a.condition,
           "eval_seed_base": int(EVAL_SEED_BASE), "episodes": a.episodes,
           "controller": {"policy": "Phase7Policy(arm=T1_projection_cap, v_cap=0.96, "
                                    "use_planner=True)",
                          "recovery": "RecoveryLadder(mode='ladder') as committed at 91bb969"},
           "integrity": checks, "summary": summ, "paired": paired,
           "episodes_updated": upd}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"final_{a.condition}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; results are never overwritten")
    p.write_text(json.dumps(res, indent=2))

    keys = ["success_rate", "collision_rate", "dynamic_collision_rate", "static_collision_rate",
            "wall_collision_rate", "timeout_rate", "min_clearance_mean", "min_clearance_worst",
            "mean_spl", "collisions_while_feasible", "collisions_after_infeasible",
            "n_infeasible", "n_solver_fail", "planner_failures", "mean_u_dev",
            "min_cbc_solved", "mean_step_time_ms"]
    labels = list(summ)
    print(f"\n{'metric':28s}" + "".join(f"{l[:16]:>18s}" for l in labels))
    for k in keys:
        row = [summ[l].get(k) for l in labels]
        print(f"{k:28s}" + "".join(f"{v:>18.4g}" if isinstance(v, (int, float))
                                   else f"{'-':>18s}" for v in row))
    print("\npaired (positive diff = updated DR-CBF higher):")
    for k, v in paired.items():
        print(f"  {k:52s} diff {v['diff']:+.4f}  CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
              f"  n10={v['only_a']} n01={v['only_b']}  p={v['p']:.4f}"
              f"  {'SIGNIFICANT' if v['p'] < 0.05 else 'not resolved'}")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
