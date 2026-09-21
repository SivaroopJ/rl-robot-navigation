"""Track B / B0 - diagnose WHERE and WHY the frozen DR-CBF QP goes infeasible.

Small diagnostic collection with the FROZEN controllers (no supervisor, nothing modified). Logs
per-step runtime features (LiDAR / tracker / controller-internal) and ground-truth diagnostic
features side by side, plus the infeasibility label, so we can ask which runtime-observable
quantity predicts infeasibility a few steps ahead.

Arms: "original" = frozen Original CLF-DR-CBF (reference params, no recovery)
      "random"   = frozen Tuned + Random CLF-DR-CBF (the audited Random-CLF-DR-CBF)
Seeds: Track B DEV block. Canonical M0 with stochastic obstacles.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from continuation.params import REFERENCE
from continuation.policy import attach_recovery, build_arm, policy_kwargs
from continuation.seeds import controller_rng
from continuation.supervisor.features import diagnostic_features, runtime_features
from experiments.exp6_ppo_comparison import make_env
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/week6_trackb/B0")
DEV_BASE = 12_000_000        # Track B development block (fresh, disjoint)


def run_episode(arm, seed, tuned, spec, env):
    obs, _ = env.reset(seed=seed)
    if arm == "original":
        pol = build_arm("original", env)
        params = REFERENCE
    else:
        pol = build_arm("random", env, params=tuned)
        params = tuned
    pol.reset(obs, env.agent_position)
    wrap = attach_recovery("random", pol, params=tuned, random_spec=spec,
                           rng=controller_rng(seed)) if arm == "random" else None
    rows, hist = [], []
    term = trunc = False
    info = {}
    steps = 0
    # capture the controller's internal record without touching frozen code
    orig = pol.ctrl.generate_controller
    box = {}

    def wrapped(p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u = orig(p, gamma, xi, u_nom=u_nom, record=rec)
        box["rec"] = dict(rec)
        box["u"] = np.asarray(u, float).copy()
        return u
    pol.ctrl.generate_controller = wrapped

    while not (term or trunc):
        p = env.agent_position.copy()
        ranges = obs[4:4 + env.N_LIDAR_RAYS] * env.LIDAR_RANGE
        gt = diagnostic_features(env)
        action, _ = pol.predict(obs, p)
        rec = box.get("rec", {})
        kept = np.atleast_2d(np.asarray(rec.get("xi_kept", [[0, 0, 0, 0]]), float))
        m_u0 = float(np.min(kept[:, 0] + pol.ctrl.rateh * kept[:, 1]))
        tracks = [t for t in pol.src.tracker.tracks
                  if (t.confirmed or not pol.src.tracker.use_confirmation)]
        rt = runtime_features(ranges, n_rays=env.N_LIDAR_RAYS, lidar_range=env.LIDAR_RANGE,
                              u_nom=rec.get("u_nom", np.zeros(2)), tracks=tracks, p=p,
                              hist_infeasible=hist, h_crit=rec.get("h_crit"), m_u0=m_u0,
                              speed=float(np.linalg.norm(env.agent_velocity)))
        infeasible = int("infeasible" in str(rec.get("status")))
        hist.append(infeasible)
        row = {"seed": seed, "arm": arm, "step": steps, "infeasible": infeasible,
               "status": str(rec.get("status")), "recovered": int(bool(rec.get("recovered", False))
                                                                  or rec.get("mode") in ("event", "persist"))}
        row.update(rt)
        row.update(gt)
        rows.append(row)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
    pol.ctrl.generate_controller = orig
    if wrap is not None:
        wrap.detach()
    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    # label each step with infeasibility in the NEXT k steps (prediction target)
    inf = [r["infeasible"] for r in rows]
    for i, r in enumerate(rows):
        for k in (1, 3, 5):
            r[f"y_infeasible_next{k}"] = int(any(inf[i + 1:i + 1 + k]))
        r["outcome"] = outcome
        r["collision_type"] = info.get("collision_type") if outcome == "collision" else None
    return rows, {"seed": seed, "arm": arm, "outcome": outcome, "steps": steps,
                  "n_infeasible": int(sum(inf)),
                  "collision_type": info.get("collision_type") if outcome == "collision" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["original", "random"])
    a = ap.parse_args()
    tuned, _ = load_tuned()
    spec, _ = load_random()
    env = make_env(True)                      # canonical M0, stochastic obstacles
    assert type(env.motion_model).__name__ == "SmoothStochasticMotion"
    seeds = [DEV_BASE + i for i in range(a.episodes)]
    t0 = time.time()
    allrows, eps = [], []
    for arm in a.arms:
        for i, s in enumerate(seeds, 1):
            r, e = run_episode(arm, s, tuned, spec, env)
            allrows += r
            eps.append(e)
            if i % 10 == 0:
                print(f"  {arm} {i}/{len(seeds)}  {time.time() - t0:.0f}s", flush=True)
    env.close()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"b0_steps_{a.episodes}.jsonl"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text("".join(json.dumps(r) + "\n" for r in allrows))
    (OUT / f"b0_episodes_{a.episodes}.json").write_text(json.dumps(
        {"seeds": [seeds[0], seeds[-1]], "arms": a.arms, "wall_s": time.time() - t0,
         "episodes": eps}, indent=1))
    print(f"wrote {p} ({len(allrows)} steps, {len(eps)} episodes) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
