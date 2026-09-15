"""P3 - canonical dynamic-obstacle pursuit: the primary experiment (plan sections 6-9).

Canonical M0 with the STOCHASTIC obstacle motion model, 6 obstacles at 0.675 m/s, plus the hunter.
Conditions are paired on the same environment seeds:
    P-Random  protagonist = frozen Tuned + Random CLF-DR-CBF   (primary)
    P-Tuned   protagonist = frozen Tuned, Random recovery off  (ablation)
The hunter is identical in both. No LADDER anywhere.

--mode smoke  runs the 10-episode smoke block; --mode main runs the predeclared evaluation block.
--hunter-speed 1.25 runs the separate, clearly-labelled speed sweep on the same seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from math import comb
from pathlib import Path

import numpy as np

from continuation.pursuit.config import PursuitConfig, pursuit_seeds
from continuation.pursuit.harness import make_pursuit_env, run_episode
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/pursuit/P3")
CONDITIONS = ("P-Random", "P-Tuned")


def mcnemar(a, b):
    """Exact McNemar, the project's own implementation (experiments/exp6_ppo_comparison.py)."""
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return {"only_a": n10, "only_b": n01, "p": float(p)}


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; reported for every outcome rate."""
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def boot_ci(x, n_boot=10_000, seed=12345):
    x = np.asarray([v for v in x if v is not None and v == v], float)
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


RATES = ("goal", "captured", "protagonist_collision", "hunter_collision", "timeout",
         "agent_contact")


def summarise(recs, label):
    n = len(recs)
    s = {"arm": label, "episodes": n}
    for k in RATES:
        c = int(sum(r[k] for r in recs))
        s[f"{k}_rate"] = c / n
        s[f"{k}_ci95"] = wilson(c, n)
    for kind in ("static", "dynamic", "wall"):
        s[f"prot_{kind}_collision_rate"] = float(np.mean(
            [r["protagonist_collision_type"] == kind for r in recs]))
        s[f"hunter_{kind}_collision_rate"] = float(np.mean(
            [r["hunter_collision_type"] == kind for r in recs]))
    def m(k):
        v = [r[k] for r in recs if r[k] is not None and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    for k in ("time_to_goal_s", "time_to_capture_s", "min_hunter_distance", "min_clearance",
              "spl", "path_length", "prot_frac_infeasible", "prot_recovery_fraction",
              "hunter_frac_infeasible", "hunter_frac_nominal_unsafe",
              "hunter_frac_materially_filtered", "hunter_mean_u_dev", "prot_mean_u_dev",
              "prot_ms_mean", "prot_ms_p95", "hunter_ms_mean", "hunter_ms_p95",
              "episode_time_s", "steps"):
        s[k] = m(k)
    s["time_to_goal_ci95"] = boot_ci([r["time_to_goal_s"] for r in recs])
    s["time_to_capture_ci95"] = boot_ci([r["time_to_capture_s"] for r in recs])
    s["min_hunter_distance_ci95"] = boot_ci([r["min_hunter_distance"] for r in recs])
    s["prot_recovery_events_total"] = int(sum(r["prot_n_recovery_events"] for r in recs))
    s["hunter_fallback_total"] = int(sum(r["hunter_n_fallback"] for r in recs))
    s["prot_ms_p95_worst"] = float(np.max([r["prot_ms_p95"] for r in recs]))
    s["hunter_ms_p95_worst"] = float(np.max([r["hunter_ms_p95"] for r in recs]))
    return s


def paired(x, y):
    """P-Random vs P-Tuned on identical env seeds. Positive diff = x higher."""
    assert [r["seed"] for r in x] == [r["seed"] for r in y], "unpaired"
    out = {}
    for k in RATES:
        a = [int(r[k]) for r in x]
        b = [int(r[k]) for r in y]
        d = np.asarray(a, float) - np.asarray(b, float)
        out[k] = {**mcnemar(a, b), "diff": float(d.mean()), "ci95": boot_ci(d)}
    for k in ("min_hunter_distance", "min_clearance", "spl", "steps"):
        d = np.asarray([r[k] for r in x], float) - np.asarray([r[k] for r in y], float)
        out[k] = {"diff": float(np.nanmean(d)), "ci95": boot_ci(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    ap.add_argument("--hunter-speed", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    block = "P3_smoke" if a.mode == "smoke" else "P3_main"
    seeds = pursuit_seeds(block)
    res, t0 = {}, time.time()
    for cond in CONDITIONS:
        cfg = PursuitConfig(condition=cond, hunter_max_speed=a.hunter_speed)
        env = make_pursuit_env(cfg)
        assert env.n_dynamic_obstacles == 6 and len(env.static_obstacles) == 5
        assert type(env.motion_model).__name__ == "SmoothStochasticMotion"
        rows = []
        for i, s in enumerate(seeds, 1):
            rows.append(run_episode(cfg, s, tuned=tuned, spec=spec, env=env))
            if i % 25 == 0 or i == len(seeds):
                print(f"  {cond} {i}/{len(seeds)}  {time.time() - t0:.0f}s", flush=True)
        env.close()
        res[cond] = rows
    # pairing: identical layouts across conditions
    for r1, r2 in zip(res["P-Random"], res["P-Tuned"]):
        assert r1["seed"] == r2["seed"] and r1["start"] == r2["start"] \
            and r1["goal_xy"] == r2["goal_xy"] and r1["hunter_start"] == r2["hunter_start"], \
            "pairing broken"
    summ = {c: summarise(res[c], c) for c in CONDITIONS}
    cmp = paired(res["P-Random"], res["P-Tuned"])
    out = {"label": f"Pursuit P3 / {a.mode} / hunter speed {a.hunter_speed:g}",
           "config": PursuitConfig(hunter_max_speed=a.hunter_speed).as_dict(),
           "tuned": tf["params"], "random": rf["spec"], "seed_block": block,
           "seeds": [seeds[0], seeds[-1]], "episodes": len(seeds),
           "summary": summ, "paired_P-Random_vs_P-Tuned": cmp,
           "wall_s": time.time() - t0, "records": res}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"P3_{a.mode}_hunter{a.hunter_speed:g}{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; results are never overwritten")
    p.write_text(json.dumps(out, indent=1, default=float))

    keys = ("goal_rate", "captured_rate", "protagonist_collision_rate", "hunter_collision_rate",
            "timeout_rate", "min_hunter_distance", "time_to_goal_s", "prot_frac_infeasible",
            "prot_recovery_fraction", "hunter_frac_infeasible", "prot_ms_p95", "hunter_ms_p95")
    print(f"\n{'metric':32s}" + "".join(f"{c:>14s}" for c in CONDITIONS))
    for k in keys:
        print(f"{k:32s}" + "".join(f"{summ[c][k]:>14.4g}" for c in CONDITIONS))
    print("\npaired P-Random vs P-Tuned (positive = P-Random higher):")
    for k in RATES:
        v = cmp[k]
        print(f"  {k:24s} diff {v['diff']:+.3f} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
              f"  n10={v['only_a']} n01={v['only_b']} p={v['p']:.4f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
