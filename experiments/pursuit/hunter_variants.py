"""Hunter variants B1 / B1.5 / P2 / P2.5 on canonical M0 (paired on identical env seeds).

Protagonist: the audited P3 protagonist (frozen Tuned + Random CLF-DR-CBF, A* carrot, known-state
hunter row), condition P-Random unless --condition says otherwise. Hunter speed 1.0 m/s.
Only the hunter's pursuit target and Protag-LiDAR handling differ between variants.

    python -m experiments.pursuit.hunter_variants --mode smoke
    python -m experiments.pursuit.hunter_variants --mode main --workers 12
    python -m experiments.pursuit.hunter_variants --mode smoke --variants B1.5 P2.5
    python -m experiments.pursuit.hunter_variants --mode main --static --workers 12   # no moving obstacles

Outputs are never overwritten; a per-episode checkpoint lets an interrupted run resume.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.env import PursuitEnv
from continuation.pursuit.harness import make_pursuit_env, run_episode
from continuation.pursuit.hunter_policies import VARIANTS, variant_seeds
from experiments.pursuit.p3_main import boot_ci, mcnemar, summarise
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/pursuit_hunter_variants")
PAIRS = (("B1.5", "B1"), ("P2", "B1"), ("P2.5", "B1.5"), ("P2.5", "P2"))
RATES = ("goal", "captured", "protagonist_collision", "hunter_collision", "timeout")
#: Terminal outcomes: exactly one per episode. `captured`/`goal` above are event FLAGS and can
#: co-occur with a Protag collision on the same step; claims about capture use these.
OUTCOMES = ("capture", "goal", "protagonist_collision", "timeout")
_W = {}


def make_env(cfg, static):
    """M0 as audited; `static` keeps the 5 rectangles and removes every moving obstacle
    (same construction as experiments/pursuit/p2_static.py)."""
    if not static:
        return make_pursuit_env(cfg)
    return PursuitEnv(pursuit=cfg, config_path="config.json", n_dynamic_obstacles=0,
                      obstacle_speed=0.675, render_mode=None, use_reward_shaping=True,
                      randomize_dynamic_obstacles=(cfg.motion == "randomized"))


def _init(condition, speed, static):
    warnings.filterwarnings("ignore")
    _W["cfg"] = PursuitConfig(condition=condition, hunter_max_speed=speed)
    _W["env"] = make_env(_W["cfg"], static)
    assert len(_W["env"].static_obstacles) == 5
    assert _W["env"].n_dynamic_obstacles == (0 if static else 6)
    _W["tuned"], _ = load_tuned()
    _W["spec"], _ = load_random()


def _task(job):
    name, seed = job
    rec = run_episode(_W["cfg"], seed, tuned=_W["tuned"], spec=_W["spec"], env=_W["env"],
                      variant=VARIANTS[name])
    return name, seed, rec


def summarise_variant(recs, name):
    s = summarise(recs, name)
    for o in OUTCOMES:
        s[f"terminal_{o}_rate"] = sum(r["outcome"] == o for r in recs) / len(recs)
    caps = [r["capture_distance_at_capture"] for r in recs if r["capture_distance_at_capture"]]
    s["capture_distance_mean"] = float(np.mean(caps)) if caps else float("nan")
    s["hunter_solver_fail_rate"] = (sum(r["hunter_n_solver_fail"] for r in recs)
                                    / max(sum(r["steps"] for r in recs), 1))
    s["hunter_infeasible_step_rate_pooled"] = (sum(r["hunter_n_infeasible"] for r in recs)
                                               / max(sum(r["steps"] for r in recs), 1))
    s["prot_infeasible_step_rate_pooled"] = (sum(r["prot_n_infeasible"] for r in recs)
                                             / max(sum(r["steps"] for r in recs), 1))
    s["hunter_no_barrier_rows_total"] = int(sum(r["hunter_n_no_barrier_rows"] for r in recs))
    if "prediction_error_mean" in recs[0]:
        pe = [r["prediction_error_mean"] for r in recs if r["prediction_error_mean"] is not None]
        s["prediction_error_mean_of_episode_means"] = float(np.mean(pe)) if pe else float("nan")
        s["prediction_error_ci95"] = boot_ci(pe)
        s["prediction_error_median_of_medians"] = float(np.median(
            [r["prediction_error_median"] for r in recs
             if r["prediction_error_median"] is not None]))
        s["prediction_fallback_total"] = int(sum(r["prediction_n_fallback"] for r in recs))
    if "filter_stats" in recs[0]:
        tot = {k: int(sum(r["filter_stats"][k] for r in recs)) for k in recs[0]["filter_stats"]}
        s["filter_stats_total"] = tot
        s["episodes_with_env_false_positive"] = int(sum(
            r["filter_stats"]["filtered_env_false_positive"] > 0 for r in recs))
    return s


def paired(x, y):
    """x vs y on identical env seeds. Positive diff = x higher."""
    assert [r["seed"] for r in x] == [r["seed"] for r in y], "unpaired"
    out = {}
    for k in RATES:
        a, b = [int(r[k]) for r in x], [int(r[k]) for r in y]
        d = np.asarray(a, float) - np.asarray(b, float)
        out[k] = {**mcnemar(a, b), "diff": float(d.mean()), "ci95": boot_ci(d)}
    for o in OUTCOMES:
        a, b = [int(r["outcome"] == o) for r in x], [int(r["outcome"] == o) for r in y]
        d = np.asarray(a, float) - np.asarray(b, float)
        out[f"terminal_{o}"] = {**mcnemar(a, b), "diff": float(d.mean()), "ci95": boot_ci(d)}
    for k in ("min_hunter_distance", "steps", "hunter_frac_infeasible"):
        d = np.asarray([r[k] for r in x], float) - np.asarray([r[k] for r in y], float)
        out[k] = {"diff": float(np.nanmean(d)), "ci95": boot_ci(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--condition", default="P-Random", choices=["P-Random", "P-Tuned"])
    ap.add_argument("--hunter-speed", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=None, help="first n seeds of the block")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--static", action="store_true", help="remove all moving obstacles")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    block = "HV_smoke" if a.mode == "smoke" else "HV_main"
    seeds = variant_seeds(block, a.n)
    stem = (f"HV_{a.mode}_{a.condition}_hunter{a.hunter_speed:g}"
            f"{'_static' if a.static else ''}{a.tag}")
    out_dir = OUT / a.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{stem}.json"
    if final.exists():
        raise SystemExit(f"{final} exists; results are never overwritten")
    ck = out_dir / f"{stem}_partial.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            n, s, r = json.loads(line)
            done[(n, s)] = r
    jobs = [(n, s) for n in a.variants for s in seeds if (n, s) not in done]
    print(f"[{stem}] {len(a.variants)} variants x {len(seeds)} seeds; {len(done)} resumed, "
          f"{len(jobs)} to run, {a.workers} workers", flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.condition, a.hunter_speed, a.static),
              maxtasksperchild=40) as pool, ck.open("a") as fh:
        for i, (n, s, r) in enumerate(pool.imap_unordered(_task, jobs), 1):
            done[(n, s)] = r
            fh.write(json.dumps([n, s, r], default=float) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    res = {n: [json.loads(json.dumps(done[(n, s)], default=float)) for s in seeds]
           for n in a.variants}
    ref = res[a.variants[0]]
    for n in a.variants:                                  # pairing: identical layouts
        for r0, r in zip(ref, res[n]):
            assert (r0["seed"], r0["start"], r0["goal_xy"], r0["hunter_start"]) == \
                (r["seed"], r["start"], r["goal_xy"], r["hunter_start"]), ("pairing broken", n)
    summ = {n: summarise_variant(res[n], n) for n in a.variants}
    cmp = {f"{x}_vs_{y}": paired(res[x], res[y]) for x, y in PAIRS
           if x in res and y in res}
    cfg = PursuitConfig(condition=a.condition, hunter_max_speed=a.hunter_speed)
    tuned, tf = load_tuned()
    spec, rf = load_random()
    out = {"label": f"Hunter variants / {a.mode} / {a.condition} / hunter {a.hunter_speed:g}",
           "map": ("M0 static (config.json: 5 rects, NO moving obstacles)" if a.static else
                   "M0 (config.json: 5 rects, 6 stochastic dyn obstacles @0.675 m/s)"),
           "pursuit_config": cfg.as_dict(), "variants": {n: VARIANTS[n].as_dict()
                                                         for n in a.variants},
           "tuned": tf["params"], "random": rf["spec"], "seed_block": block,
           "seeds": [seeds[0], seeds[-1]], "episodes_per_variant": len(seeds),
           "summary": summ, "paired": cmp, "wall_s_this_invocation": time.time() - t0,
           "records": res}
    final.write_text(json.dumps(out, indent=1, default=float))
    ck.unlink()

    keys = ("terminal_capture_rate", "terminal_goal_rate", "terminal_protagonist_collision_rate",
            "terminal_timeout_rate", "time_to_capture_s", "captured_rate", "hunter_collision_rate",
            "hunter_infeasible_step_rate_pooled",
            "prot_infeasible_step_rate_pooled", "capture_distance_mean", "min_hunter_distance")
    print(f"\n{'metric':36s}" + "".join(f"{n:>10s}" for n in a.variants))
    for k in keys:
        print(f"{k:36s}" + "".join(f"{summ[n][k]:>10.4g}" for n in a.variants))
    for n in a.variants:
        if "prediction_error_mean_of_episode_means" in summ[n]:
            print(f"{n}: prediction error (tau=0.5 s) mean {summ[n]['prediction_error_mean_of_episode_means']:.3f} m")
        if "filter_stats_total" in summ[n]:
            print(f"{n}: filter {summ[n]['filter_stats_total']}")
    for key, c in cmp.items():
        print(f"\n{key} (positive = first higher):")
        for k in [f"terminal_{o}" for o in OUTCOMES] + ["hunter_collision"]:
            v = c[k]
            print(f"  {k:24s} diff {v['diff']:+.3f} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                  f" n10={v['only_a']} n01={v['only_b']} p={v['p']:.4f}")
    print("wrote", final)


if __name__ == "__main__":
    main()
