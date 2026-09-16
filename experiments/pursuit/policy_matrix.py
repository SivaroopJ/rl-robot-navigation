"""Policy matrix: hunter policies 1-3.5 x controller configurations A/B/C x static/dynamic M0.

    python -m experiments.pursuit.policy_matrix --phase repro  --config A          # exact-match gate
    python -m experiments.pursuit.policy_matrix --phase smoke  --config B
    python -m experiments.pursuit.policy_matrix --phase main   --config B --workers 12
    python -m experiments.pursuit.policy_matrix --phase dev    --config B --policies 1 3
    python -m experiments.pursuit.policy_matrix --phase smoke  --config C \
        --thresholds results/pursuit_policy_matrix/calibration/cd_thresholds.json
    python -m experiments.pursuit.policy_matrix --phase latency --config C --workers 1 --thresholds ...

Outputs go to results/pursuit_policy_matrix/<phase>_<config><tag>/ : records.json (episode records
per map and policy), summary.json, steps.jsonl.gz (per-step logs), and are never overwritten.
A per-episode checkpoint lets an interrupted invocation resume.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from continuation.congestion.detector import DetectorConfig
from continuation.congestion.wrapper import CDConfig
from continuation.pursuit.config import PursuitConfig
from continuation.pursuit.harness import CONTROLLER_SETUPS, run_episode
from continuation.pursuit.hunter_policies import (ALIASES, POLICY_IDS, policy_by_id,
                                                  variant_seeds)
from continuation.pursuit.mpc import MPCConfig
from experiments.pursuit.hunter_variants import make_env, paired, summarise_variant
from experiments.pursuit.p3_main import boot_ci, mcnemar, wilson
from experiments.week6_continuation.common import load_random, load_tuned

OUT = Path("results/pursuit_policy_matrix")
MAPS = ("static", "dynamic")
PHASES = {"repro": ("HV_main", 10), "smoke": ("HV_smoke", None), "main": ("HV_main", None),
          "dev": ("HV_dev", None), "latency": ("HV_smoke", None),
          "check": ("HV_check", None)}
WITHIN_PAIRS = (("3", "1"), ("3.5", "1.5"), ("3", "2"), ("3.5", "2.5"), ("1.5", "1"), ("2", "1"))
HISTORICAL = {"dynamic": "results/pursuit_hunter_variants/main/HV_main_P-Random_hunter1.json",
              "static": "results/pursuit_hunter_variants/main/HV_main_P-Random_hunter1_static.json"}
TIMING_SUFFIXES = ("_ms_mean", "_ms_p95", "episode_time_s")
OUTCOMES = ("capture", "goal", "protagonist_collision", "timeout")
_W = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_thresholds(path):
    p = Path(path)
    want = p.with_suffix(".sha256").read_text().split()[0]
    if sha256(p) != want:
        raise SystemExit(f"{p} changed after it was frozen")
    d = json.loads(p.read_text())
    return DetectorConfig(**d["detector"]), d


def _init(config, detector_dict, cd_mode, cd_opts=None):
    warnings.filterwarnings("ignore")
    cfg = PursuitConfig(condition="P-Random", hunter_max_speed=1.0)
    _W["cfg"] = cfg
    _W["envs"] = {"dynamic": make_env(cfg, False), "static": make_env(cfg, True)}
    assert _W["envs"]["dynamic"].n_dynamic_obstacles == 6
    assert _W["envs"]["static"].n_dynamic_obstacles == 0
    _W["tuned"], _ = load_tuned()
    _W["spec"], _ = load_random()
    setup = CONTROLLER_SETUPS[config]
    if cd_mode is not None:
        setup = replace(setup, cd_mode=cd_mode)
    _W["setup"] = setup
    _W["mpc"] = MPCConfig()
    _W["cd"] = build_cd(detector_dict, cd_opts)


def build_cd(detector_dict, cd_opts):
    """CDConfig with the congestion-fix flags; defaults reproduce the pre-registered detector."""
    kw = dict(cd_opts or {})
    if detector_dict:
        kw["detector"] = DetectorConfig(**detector_dict)
    return CDConfig(**kw)


def _task(job):
    mp, pid, seed = job
    rec = run_episode(_W["cfg"], seed, tuned=_W["tuned"], spec=_W["spec"], env=_W["envs"][mp],
                      variant=policy_by_id(pid), controllers=_W["setup"], mpc_cfg=_W["mpc"],
                      cd_cfg=_W["cd"], step_log=True)
    steps = rec.pop("step_log")
    return mp, pid, seed, json.loads(json.dumps(rec, default=float)), steps


# --------------------------------------------------------------------------- summaries
def _pool_ratio(recs, num, den):
    return sum(r[num] for r in recs) / max(sum(r[den] for r in recs), 1)


def summarise_cell(recs, pid):
    s = summarise_variant(recs, pid)
    n = len(recs)
    for o in OUTCOMES:
        c = sum(r["outcome"] == o for r in recs)
        s[f"terminal_{o}_ci95"] = wilson(c, n)
    ne = sum(r["outcome"] in ("capture", "protagonist_collision") for r in recs)
    s["not_escaped_rate"] = ne / n
    s["not_escaped_ci95"] = wilson(ne, n)
    s["hunter_recovery_events_total"] = int(sum(r["hunter_n_recovery_events"] for r in recs))
    s["hunter_u0_step_rate_pooled"] = _pool_ratio(recs, "hunter_n_u0_steps", "steps")
    s["prot_u0_step_rate_pooled"] = _pool_ratio(recs, "prot_n_u0_steps", "steps")
    s["prot_recovery_events_total"] = int(sum(r["prot_n_recovery_events"] for r in recs))
    for agent in ("prot", "hunter"):
        lat = {}
        for comp in recs[0][f"{agent}_latency_ms"]:
            xs = [r[f"{agent}_latency_ms"][comp] for r in recs]
            lat[comp] = {"mean_of_episode_means": float(np.nanmean([x["mean"] for x in xs])),
                         "mean_of_episode_p95": float(np.nanmean([x["p95"] for x in xs])),
                         "max": float(np.nanmax([x["max"] for x in xs]))}
        s[f"{agent}_latency_ms"] = lat
        s[f"{agent}_frac_steps_over_budget_pooled"] = float(
            sum(r[f"{agent}_frac_steps_over_budget"] * r["steps"] for r in recs)
            / max(sum(r["steps"] for r in recs), 1))
    if "mpc_n_steps" in recs[0]:
        s["mpc_candidates"] = recs[0]["mpc_config"]["n_candidates"]
        s["mpc_horizon_steps"] = recs[0]["mpc_config"]["horizon_steps"]
        s["mpc_fallback_rate_pooled"] = _pool_ratio(recs, "mpc_n_fallback", "mpc_n_steps")
        s["mpc_pred_capture_rate_pooled"] = _pool_ratio(recs, "mpc_n_pred_capture", "mpc_n_steps")
        s["mpc_mean_rejected"] = float(np.nanmean([r["mpc_mean_rejected"] for r in recs]))
        s["mpc_ms_mean_of_means"] = float(np.nanmean([r["mpc_ms"]["mean"] for r in recs]))
        s["mpc_ms_max"] = float(np.nanmax([r["mpc_ms"]["max"] for r in recs]))
        pe = [r["mpc_pred_err_1s_mean"] for r in recs if r["mpc_pred_err_1s_mean"] is not None]
        s["mpc_pred_err_1s_mean"] = float(np.mean(pe)) if pe else float("nan")
    for agent in ("prot", "hunter"):
        key = f"{agent}_cd_alarms"
        if key in recs[0]:
            tot = {k: int(sum(r[key][k] for r in recs)) for k in
                   ("warning_onsets", "true_alarms", "false_alarms", "infeasible_onsets",
                    "missed", "warning_steps", "active_steps")}
            tot["state_counts"] = [int(sum(r[key]["state_counts"][i] for r in recs))
                                   for i in range(3)]
            leads = [x for r in recs for x in r[f"{agent}_cd_lead_steps"]]
            tot["lead_steps_mean"] = float(np.mean(leads)) if leads else None
            tot["coverage"] = (1 - tot["missed"] / tot["infeasible_onsets"]
                               if tot["infeasible_onsets"] else None)
            tot["warning_fraction"] = tot["warning_steps"] / max(tot["active_steps"], 1)
            tot["precision"] = (tot["true_alarms"] / tot["warning_onsets"]
                                if tot["warning_onsets"] else None)
            tot["mean_scale"] = float(np.mean([r[f"{agent}_cd_mean_scale"] for r in recs]))
            tot["direction_changes"] = int(sum(r[f"{agent}_cd_n_direction_changes"]
                                               for r in recs))
            s[f"{agent}_cd"] = tot
    return s


def paired_cells(x, y):
    out = paired(x, y)
    a = [int(r["outcome"] in ("capture", "protagonist_collision")) for r in x]
    b = [int(r["outcome"] in ("capture", "protagonist_collision")) for r in y]
    d = np.asarray(a, float) - np.asarray(b, float)
    out["not_escaped"] = {**mcnemar(a, b), "diff": float(d.mean()), "ci95": boot_ci(d)}
    return out


# --------------------------------------------------------------------------- repro gate
def repro_check(res):
    """Config A vs the historical records: every non-timing field must match exactly."""
    inv = {v: k for k, v in ALIASES.items()}
    bad = []
    for mp in res:
        hist = json.loads(Path(HISTORICAL[mp]).read_text())["records"]
        for pid, recs in res[mp].items():
            ref = {r["seed"]: r for r in hist[inv[pid]]}
            for r in recs:
                h = ref[r["seed"]]
                for k, v in h.items():
                    if k.endswith(TIMING_SUFFIXES):
                        continue
                    got = r.get(k)
                    if k == "hunter_variant":
                        v = {**v, "name": None}
                        got = {**got, "name": None}
                    same = got == v or (isinstance(v, float) and v != v and got != got)
                    if not same:
                        bad.append((mp, pid, r["seed"], k, v, got))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), required=True)
    ap.add_argument("--config", choices=list(CONTROLLER_SETUPS), required=True)
    ap.add_argument("--policies", nargs="+", default=None, choices=list(POLICY_IDS))
    ap.add_argument("--maps", nargs="+", default=list(MAPS), choices=list(MAPS))
    ap.add_argument("--thresholds", default=None, help="frozen cd_thresholds.json (config C)")
    ap.add_argument("--cd-mode", default=None, choices=["off", "shadow", "active"])
    ap.add_argument("--cd-clearance", default="sensed", choices=["sensed", "rows"],
                    help="congestion fix 1: clearance from the controller's own barrier rows")
    ap.add_argument("--cd-lookahead", default="nominal", choices=["nominal", "executed"],
                    help="congestion fix 2: look ahead along the executed (filtered) velocity")
    ap.add_argument("--cd-tracks", default="confirmed", choices=["confirmed", "strict"],
                    help="congestion fix 3: only tracks updated this step with >= 2 points")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    policies = a.policies or (["1", "1.5", "2", "2.5"] if a.config == "A"
                              else list(POLICY_IDS))
    if a.config == "A" and any(p in ("3", "3.5") for p in policies):
        raise SystemExit("config A is the historical mixed configuration: policies 1-2.5 only")
    if a.phase == "dev" and a.config != "B":
        raise SystemExit("development (calibration) runs are config B shadow only")
    block, n_default = PHASES[a.phase]
    seeds = variant_seeds(block, a.n if a.n is not None else n_default)
    detector_dict, thr_meta = None, None
    setup = CONTROLLER_SETUPS[a.config]
    if a.cd_mode is not None:
        setup = replace(setup, cd_mode=a.cd_mode)
    if setup.cd_mode == "active":
        if not a.thresholds:
            raise SystemExit("active congestion detection needs --thresholds (frozen file)")
        det, thr_meta = load_thresholds(a.thresholds)
        detector_dict = det.as_dict()
    elif a.thresholds:
        det, thr_meta = load_thresholds(a.thresholds)
        detector_dict = det.as_dict()

    cd_opts = {"clearance_source": a.cd_clearance, "lookahead_velocity": a.cd_lookahead,
               "track_gate": a.cd_tracks}
    label = f"{a.phase}_{a.config}{a.tag}"
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "records.json"
    if final.exists():
        raise SystemExit(f"{final} exists; results are never overwritten")
    ck = out_dir / "records_partial.jsonl"
    steps_path = out_dir / "steps.jsonl.gz"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            mp, pid, s, r = json.loads(line)
            done[(mp, pid, s)] = r
    jobs = [(mp, pid, s) for mp in a.maps for pid in policies for s in seeds
            if (mp, pid, s) not in done]
    print(f"[{label}] setup={setup.as_dict()} maps={a.maps} policies={policies} "
          f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}); {len(done)} resumed, {len(jobs)} to run, "
          f"{a.workers} workers", flush=True)

    t0 = time.time()
    if jobs:
        with Pool(a.workers, initializer=_init,
                  initargs=(a.config, detector_dict, a.cd_mode, cd_opts),
                  maxtasksperchild=40) as pool, ck.open("a") as fh, \
                gzip.open(steps_path, "at") as gz:
            for i, (mp, pid, s, r, st) in enumerate(pool.imap_unordered(_task, jobs), 1):
                gz.write(json.dumps({"map": mp, "policy": pid, "seed": s, "steps": st},
                                    default=float) + "\n")
                gz.flush()
                fh.write(json.dumps([mp, pid, s, r]) + "\n")
                fh.flush()
                done[(mp, pid, s)] = r
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    res = {mp: {pid: [done[(mp, pid, s)] for s in seeds] for pid in policies} for mp in a.maps}
    for mp in a.maps:                                          # pairing: identical layouts
        ref = res[mp][policies[0]]
        for pid in policies:
            for r0, r in zip(ref, res[mp][pid]):
                assert (r0["seed"], r0["start"], r0["goal_xy"], r0["hunter_start"]) == \
                    (r["seed"], r["start"], r["goal_xy"], r["hunter_start"]), (mp, pid)
    summ = {mp: {pid: summarise_cell(res[mp][pid], pid) for pid in policies} for mp in a.maps}
    cmp = {mp: {f"{x}-{y}": paired_cells(res[mp][x], res[mp][y]) for x, y in WITHIN_PAIRS
                if x in res[mp] and y in res[mp]} for mp in a.maps}
    tuned, tf = load_tuned()
    spec, rf = load_random()
    meta = {"label": label, "phase": a.phase, "config": setup.as_dict(), "maps": a.maps,
            "policies": {p: policy_by_id(p).as_dict() for p in policies},
            "seed_block": block, "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds),
            "mpc": MPCConfig().as_dict(),
            "cd": build_cd(detector_dict, cd_opts).as_dict() if setup.cd_mode != "off" else None,
            "thresholds_file": a.thresholds, "thresholds_meta": thr_meta,
            "tuned": tf["params"], "random": rf["spec"], "workers": a.workers,
            "wall_s_this_invocation": time.time() - t0, "argv": sys.argv}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summ,
                                                      "paired_within": cmp}, indent=1,
                                                     default=float))
    final.write_text(json.dumps({"meta": meta, "records": res}, default=float))
    ck.unlink()

    for mp in a.maps:
        print(f"\n== {label} / {mp}")
        keys = ("terminal_capture_rate", "terminal_goal_rate", "terminal_protagonist_collision_rate",
                "terminal_timeout_rate", "not_escaped_rate", "hunter_collision_rate",
                "hunter_infeasible_step_rate_pooled", "prot_infeasible_step_rate_pooled",
                "hunter_u0_step_rate_pooled")
        print(f"{'metric':38s}" + "".join(f"{p:>9s}" for p in policies))
        for k in keys:
            print(f"{k:38s}" + "".join(f"{summ[mp][p][k]:>9.3f}" for p in policies))
        print(f"{'hunter ms mean / p95 (mean of eps)':38s}" + "".join(
            f"{summ[mp][p]['hunter_latency_ms']['total']['mean_of_episode_means']:>5.0f}/"
            f"{summ[mp][p]['hunter_latency_ms']['total']['mean_of_episode_p95']:<3.0f}"
            for p in policies))
        for p in policies:
            if "mpc_fallback_rate_pooled" in summ[mp][p]:
                s = summ[mp][p]
                print(f"  policy {p}: MPC fallback {s['mpc_fallback_rate_pooled']:.3f}, "
                      f"mean rejected {s['mpc_mean_rejected']:.1f}/{s['mpc_candidates']}, "
                      f"MPC ms {s['mpc_ms_mean_of_means']:.1f} (max {s['mpc_ms_max']:.1f})")
    if a.phase == "repro":
        bad = repro_check(res)
        (out_dir / "repro_check.json").write_text(json.dumps(
            {"n_mismatch": len(bad), "mismatches": bad[:200]}, default=str, indent=1))
        if bad:
            print(f"REPRO GATE FAILED: {len(bad)} mismatching fields; first: {bad[:5]}")
            raise SystemExit(2)
        print("REPRO GATE PASSED: every non-timing field matches the historical records")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
