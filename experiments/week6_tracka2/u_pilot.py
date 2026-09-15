"""Track A continuation / Stage C - development pilot for the unlabelled-threat condition.

DEBUGGING AND FEASIBILITY ONLY. Not a statistically powered comparison; no claim is drawn from it.

Scenarios (S1-S3 are scripted diagnostics, S4 is the representative stochastic environment):
    S1 hunter approaching, NO moving decoys           (dynamic obstacles parked far away)
    S2 hunter initially occluded by a rectangle, then emerging
    S3 hunter among several moving obstacles
    S4 the full canonical environment, all six stochastic dynamic obstacles
Arms: U0_oracle (audited baseline), U1_unlabelled, U2_visible_only.
Seeds: fresh development block 13 200 000+, disjoint from every prior range (verified in the
repository audit: 10.0-10.9M navigation, 11.0-11.3M pursuit, 12.0-12.1M Track B,
13 000 000-13 000 002 Track A A-P1).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from continuation.unlabelled_threat.harness import ARMS, arm_config, make_env, run_episode
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/week6_tracka2/pilot")
DEV_BASE = 13_200_700          # clamped-gate validation block (gated pre-clamp used 13 200 300)
FAR = np.array([9.5, 9.5], dtype=np.float32)


def park_obstacles(env):
    env.obstacle_positions = np.tile(FAR, (len(env.obstacle_positions), 1))
    env.obstacle_velocities = np.zeros_like(env.obstacle_velocities)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tuned, tf = load_tuned()
    spec, rf = load_random()
    seeds = [DEV_BASE + i for i in range(a.episodes)]
    t0 = time.time()
    out = {"label": "Track A continuation / Stage C pilot (DEVELOPMENT ONLY)",
           "seeds": [seeds[0], seeds[-1]], "episodes_per_arm": len(seeds),
           "tuned": tf["params"], "random": rf["spec"], "scenarios": {}}

    # S4: the representative stochastic environment (the one that matters for a future evaluation)
    for scen, n_obs in (("S4_full_stochastic", 6), ("S1_no_decoys", 0), ("S3_several_movers", 3)):
        res = {}
        for arm in ARMS:
            cfg = arm_config(arm)
            env = make_env(cfg, n_dynamic_obstacles=n_obs)
            rows = []
            for s in seeds:
                rows.append(run_episode(arm, s, tuned=tuned, spec=spec, env=env))
                rows[-1]["scenario"] = scen
            env.close()
            res[arm] = rows
        out["scenarios"][scen] = res
        print(f"  {scen} done  {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"u_pilot{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; results are never overwritten")
    out["wall_s"] = time.time() - t0
    p.write_text(json.dumps(out, indent=1, default=float))

    from collections import Counter
    for scen, res in out["scenarios"].items():
        print(f"\n=== {scen}  ({len(seeds)} episodes/arm)")
        print(f"{'arm':16s}{'outcomes':>42s}{'rows/step':>11s}{'rowsmax':>9s}"
              f"{'infeas':>9s}{'recov':>7s}{'minHd':>8s}{'ms p95':>8s}")
        for arm in ARMS:
            r = res[arm]
            c = Counter(x["outcome"] for x in r)
            oc = " ".join(f"{k}:{v}" for k, v in sorted(c.items()))
            print(f"{arm:16s}{oc:>42s}"
                  f"{np.mean([x['track_rows_mean'] for x in r]):>11.2f}"
                  f"{max(x['track_rows_max'] for x in r):>9d}"
                  f"{np.mean([x['prot_frac_infeasible'] for x in r]):>9.4f}"
                  f"{np.mean([x['prot_n_recovery_events'] for x in r]):>7.1f}"
                  f"{np.mean([x['min_hunter_distance'] for x in r]):>8.3f}"
                  f"{np.mean([x['prot_ms_p95'] for x in r]):>8.1f}")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
