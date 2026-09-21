"""Track B / B4 - factorial evaluation: {Original, Random} x {supervisor off, on}.

Paired on environment seeds. Controller families are analysed separately and NEVER pooled.
--mode pilot uses the dev block; --mode eval uses the evaluation block (frozen config only).
"""
from __future__ import annotations

import argparse
import json
import time
from math import comb
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from continuation.supervisor.harness import build_arms, run_episode
from continuation.supervisor.risk import RISK_OFF, RISK_ON
from continuation.supervisor.supervisor import COOLDOWN, MAX_ENGAGED, MIN_SCALE
from experiments.exp6_ppo_comparison import make_env
from experiments.week6_continuation.common import load_random, load_tuned
from evaluation.shortest_path import ShortestPathOracle

OUT = Path("results/week6_trackb")
DEV_PILOT_BASE = 12_000_500          # dev block, disjoint from the B0 diagnosis seeds
EVAL_BASE = 12_100_000               # evaluation block, never used for tuning
_W = {}


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return {"only_a": n10, "only_b": n01, "p": float(p)}


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def boot(x, n_boot=10000, seed=12345):
    x = np.asarray([v for v in x if v is not None and v == v], float)
    if not len(x):
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cluster_ratio(x, y, num, den="steps", n_boot=10000, seed=12345):
    nx = np.array([r[num] for r in x], float); dx = np.array([r[den] for r in x], float)
    ny = np.array([r[num] for r in y], float); dy = np.array([r[den] for r in y], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nx), size=(n_boot, len(nx)))
    d = nx[idx].sum(1) / np.maximum(dx[idx].sum(1), 1) - ny[idx].sum(1) / np.maximum(dy[idx].sum(1), 1)
    return {"x": float(nx.sum() / dx.sum()), "y": float(ny.sum() / dy.sum()),
            "diff": float(nx.sum() / dx.sum() - ny.sum() / dy.sum()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def _worker(job):
    arm, seed = job
    if "env" not in _W:
        env = make_env(True)
        _W["env"] = env
        _W["oracle"] = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    return arm.name, seed, run_episode(arm, seed, env=_W["env"], oracle=_W["oracle"])


def summarise(recs, label):
    n = len(recs)
    steps = sum(r["steps"] for r in recs)
    s = {"arm": label, "episodes": n, "steps_total": steps}
    for k, key in (("goal", "success"), ("collision", "collision"), ("timeout", "timeout")):
        c = int(sum(r[key] for r in recs))
        s[f"{k}_rate"] = c / n
        s[f"{k}_ci95"] = wilson(c, n)
        s[f"{k}_count"] = c
    for kind in ("dynamic", "static", "wall"):
        s[f"{kind}_collision_rate"] = float(np.mean([r["collision_type"] == kind for r in recs]))
    def m(k):
        v = [r[k] for r in recs if r.get(k) is not None and r.get(k) == r.get(k)]
        return float(np.mean(v)) if v else float("nan")
    s["infeasible_step_rate"] = sum(r["n_infeasible"] for r in recs) / max(steps, 1)
    s["infeasible_per_episode"] = m("n_infeasible")
    s["infeasible_runs_per_episode"] = m("n_infeasible_runs")
    s["infeasible_run_len_mean"] = m("infeasible_run_len_mean")
    s["infeasible_run_len_max"] = int(max(r["infeasible_run_len_max"] for r in recs))
    s["fallback_rate"] = sum(r["n_fallback"] for r in recs) / max(steps, 1)
    s["recovery_events_total"] = int(sum(r["n_recovery_events"] for r in recs))
    s["recovery_fraction"] = m("recovery_fraction")
    for k in ("min_clearance", "spl", "path_length", "steps", "step_ms_mean", "step_ms_p95",
              "episode_time_s", "sup_engaged_fraction", "sup_unnecessary_fraction",
              "sup_mean_risk", "sup_mean_scale_when_engaged"):
        s[k] = m(k)
    s["min_clearance_ci95"] = boot([r["min_clearance"] for r in recs])
    s["collisions_per_1k_steps"] = 1000.0 * sum(r["collision"] for r in recs) / max(steps, 1)
    return s


def paired(x, y):
    assert [r["seed"] for r in x] == [r["seed"] for r in y]
    out = {}
    for k, key in (("goal", "success"), ("collision", "collision"), ("timeout", "timeout")):
        a = [int(r[key]) for r in x]; b = [int(r[key]) for r in y]
        out[k] = {**mcnemar(a, b), "diff": float(np.mean(a) - np.mean(b)),
                  "ci95": boot(np.asarray(a, float) - np.asarray(b, float))}
    for kind in ("dynamic", "static"):
        a = [int(r["collision_type"] == kind) for r in x]; b = [int(r["collision_type"] == kind) for r in y]
        out[f"{kind}_collision"] = {**mcnemar(a, b), "diff": float(np.mean(a) - np.mean(b)),
                                    "ci95": boot(np.asarray(a, float) - np.asarray(b, float))}
    for k in ("min_clearance", "spl", "steps", "path_length", "step_ms_mean"):
        d = np.asarray([r[k] for r in x], float) - np.asarray([r[k] for r in y], float)
        out[k] = {"diff": float(np.nanmean(d)), "ci95": boot(d)}
    out["infeasible_step_rate"] = cluster_ratio(x, y, "n_infeasible")
    out["fallback_rate"] = cluster_ratio(x, y, "n_fallback")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "eval"], required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    arms = build_arms(tuned, spec)
    n = a.episodes or (20 if a.mode == "pilot" else 200)
    base = DEV_PILOT_BASE if a.mode == "pilot" else EVAL_BASE
    seeds = [base + i for i in range(n)]
    jobs = [(arm, s) for arm in arms for s in seeds]
    t0 = time.time()
    got = {arm.name: {} for arm in arms}
    with get_context("fork").Pool(a.workers, maxtasksperchild=40) as pool:
        for i, (name, seed, rec) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            got[name][seed] = rec
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes  {time.time() - t0:.0f}s", flush=True)
    res = {k: [got[k][s] for s in seeds] for k in got}
    for k in res:
        for r1, r2 in zip(res["O-off"], res[k]):
            assert r1["seed"] == r2["seed"] and r1["start"] == r2["start"], "pairing broken"
    summ = {k: summarise(res[k], k) for k in res}
    cmp = {"original: on vs off": paired(res["O-on"], res["O-off"]),
           "random: on vs off": paired(res["R-on"], res["R-off"])}
    out = {"label": f"Track B factorial / {a.mode}", "mode": a.mode,
           "seeds": [seeds[0], seeds[-1]], "episodes": n,
           "config": {"RISK_ON": RISK_ON, "RISK_OFF": RISK_OFF, "MIN_SCALE": MIN_SCALE,
                      "MAX_ENGAGED": MAX_ENGAGED, "COOLDOWN": COOLDOWN},
           "tuned": tf["params"], "random": rf["spec"],
           "arms": [x.describe() for x in arms], "summary": summ, "paired": cmp,
           "wall_s": time.time() - t0, "records": res}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"trackB_{a.mode}_{n}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text(json.dumps(out, indent=1, default=float))
    keys = ("goal_rate", "collision_rate", "timeout_rate", "infeasible_step_rate",
            "fallback_rate", "infeasible_runs_per_episode", "min_clearance", "spl", "steps",
            "sup_engaged_fraction", "step_ms_mean", "collisions_per_1k_steps")
    print(f"\n{'metric':30s}" + "".join(f"{k:>10s}" for k in res))
    for k in keys:
        print(f"{k:30s}" + "".join(f"{summ[c][k]:>10.4g}" for c in res))
    for lab, c in cmp.items():
        print(f"\n{lab} (positive = supervisor-on higher):")
        for k in ("goal", "collision", "timeout", "dynamic_collision", "static_collision"):
            v = c[k]
            print(f"  {k:20s} diff {v['diff']:+.3f} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
                  f"n10={v['only_a']} n01={v['only_b']} p={v['p']:.4f}")
        for k in ("infeasible_step_rate", "fallback_rate"):
            v = c[k]
            print(f"  {k:20s} {v['x']:.4f} vs {v['y']:.4f} diff {v['diff']:+.4f} "
                  f"CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
        for k in ("min_clearance", "spl", "steps"):
            v = c[k]
            print(f"  {k:20s} diff {v['diff']:+.4f} CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
