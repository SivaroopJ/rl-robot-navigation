"""Calibrate and freeze the congestion thresholds (c1, c2) from DEVELOPMENT shadow-mode runs.

    python -m experiments.pursuit.cd_calibrate --run results/pursuit_policy_matrix/dev_B

Rule (MD_files/pursuit/MPC_CD_PREREGISTRATION.md, verified against its recorded hash):
replay the detector over the logged (margin, predicted clearance) series of both agents on active
steps, for every (c1, c2) on the declared grid; pooled coverage = 1 - missed / infeasibility onsets
(W = 10) and warning fraction = warning steps / active steps. Choose the lowest warning fraction
among pairs with coverage >= 0.80 and warning fraction <= 0.25 (ties: smaller c1, then c2); if none
qualify, maximise coverage subject to warning fraction <= 0.25 (ties: lower warning fraction, c1, c2).
Writes results/pursuit_policy_matrix/calibration/cd_thresholds.json + .sha256 (never overwritten).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from continuation.congestion.detector import DetectorConfig, alarm_stats, replay

PREREG = Path("MD_files/pursuit/MPC_CD_PREREGISTRATION.md")
PREREG_HASH = Path("results/pursuit_policy_matrix/prereg.sha256")
OUT = Path("results/pursuit_policy_matrix/calibration")
C1_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5)
C2_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
COVERAGE_TARGET, WARN_CAP, WINDOW = 0.80, 0.25, 10


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_series(run_dir, *, require_dev=True):
    """[(agent, map, policy, seed, margins, clrs, infeasible, active)] from steps.jsonl.gz.

    `require_dev=False` is for REPORTING only (replaying frozen thresholds on evaluation logs);
    calibration itself always requires development seeds in shadow mode."""
    meta = json.loads((Path(run_dir) / "summary.json").read_text())["meta"]
    if require_dev and (meta["seed_block"] != "HV_dev" or meta["config"]["cd_mode"] != "shadow"):
        raise SystemExit("calibration must use development seeds in shadow mode")
    rows = {}
    with gzip.open(Path(run_dir) / "steps.jsonl.gz", "rt") as fh:
        for line in fh:
            e = json.loads(line)
            rows[(e["map"], e["policy"], e["seed"])] = e["steps"]      # last write wins
    out = []
    for (mp, pid, seed), st in sorted(rows.items()):
        p = st["p_cd_log"]
        out.append(("protag", mp, pid, seed, p["margin"], p["clr_look"], st["p_inf"], p["active"]))
        h = st["h_cd_log"]
        inf = [x for x, ran in zip(st["h_inf"], st["h_cd_ran"]) if ran]
        out.append(("hunter", mp, pid, seed, h["margin"], h["clr_look"], inf, h["active"]))
    return out, meta


def evaluate(series, det: DetectorConfig):
    tot = {"protag": [0, 0, 0, 0, 0, 0], "hunter": [0, 0, 0, 0, 0, 0]}
    # onsets, missed, warn steps, active steps, warning onsets, true alarms
    for agent, _, _, _, m, c, inf, act in series:
        c = [5.0 if x is None else x for x in c]
        st = replay(det, m, c)
        al = alarm_stats(st, inf, act, window=WINDOW)
        t = tot[agent]
        t[0] += al["infeasible_onsets"]
        t[1] += al["missed"]
        t[2] += al["warning_steps"]
        t[3] += al["active_steps"]
        t[4] += al["warning_onsets"]
        t[5] += al["true_alarms"]
    on = sum(v[0] for v in tot.values())
    miss = sum(v[1] for v in tot.values())
    warn = sum(v[2] for v in tot.values())
    act = sum(v[3] for v in tot.values())
    wo = sum(v[4] for v in tot.values())
    ta = sum(v[5] for v in tot.values())
    per = {a: {"onsets": v[0], "missed": v[1],
               "coverage": (1 - v[1] / v[0]) if v[0] else None,
               "warning_fraction": v[2] / max(v[3], 1),
               "precision": (v[5] / v[4]) if v[4] else None} for a, v in tot.items()}
    return {"coverage": (1 - miss / on) if on else None, "warning_fraction": warn / max(act, 1),
            "onsets": on, "missed": miss, "warning_onsets": wo, "true_alarms": ta,
            "precision": (ta / wo) if wo else None, "per_agent": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="", help="suffix for the frozen threshold file")
    a = ap.parse_args()
    if sha256(PREREG) != PREREG_HASH.read_text().split()[0]:
        raise SystemExit("pre-registration changed after it was hashed")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"cd_thresholds{a.tag}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; thresholds are frozen once")
    series, meta = load_series(a.run)
    base = DetectorConfig()
    grid = []
    for c1 in C1_GRID:
        for c2 in C2_GRID:
            r = evaluate(series, replace(base, c1=c1, c2=c2))
            grid.append({"c1": c1, "c2": c2, **r})
    ok = [g for g in grid if g["coverage"] is not None and g["coverage"] >= COVERAGE_TARGET
          and g["warning_fraction"] <= WARN_CAP]
    if ok:
        best = min(ok, key=lambda g: (g["warning_fraction"], g["c1"], g["c2"]))
        rule = "coverage>=0.80 and warning<=0.25: lowest warning fraction"
    else:
        capped = [g for g in grid if g["warning_fraction"] <= WARN_CAP]
        if capped:
            best = min(capped, key=lambda g: (-(g["coverage"] or 0.0), g["warning_fraction"],
                                              g["c1"], g["c2"]))
            rule = "fallback: no pair met both targets; max coverage under the 25% cap"
        else:
            # Deviation D1 (MD_files/pursuit/MPC_CD_DEVIATIONS.md): the cap is unattainable.
            best = min(grid, key=lambda g: (g["warning_fraction"], -(g["coverage"] or 0.0),
                                            g["c1"], g["c2"]))
            rule = ("deviation D1: no pair keeps warnings <= 25%; lowest warning fraction "
                    "selected; calibration targets NOT met")
    det = replace(base, c1=best["c1"], c2=best["c2"])
    frozen = {"detector": det.as_dict(), "selected_by": rule, "achieved": best,
              "coverage_target": COVERAGE_TARGET, "warning_cap": WARN_CAP, "window": WINDOW,
              "source_run": str(a.run), "source_seeds": meta["seeds"],
              "cd_options": {k: meta["cd"][k] for k in
                             ("clearance_source", "lookahead_velocity", "track_gate")}
              if meta.get("cd") else None,
              "source_policies": list(meta["policies"]), "source_maps": meta["maps"],
              "prereg_sha256": sha256(PREREG)}
    target.write_text(json.dumps(frozen, indent=1))
    target.with_suffix(".sha256").write_text(f"{sha256(target)}  {target.name}\n")
    (OUT / f"calibration_grid{a.tag}.json").write_text(json.dumps(grid, indent=1))
    print(f"selected c1={det.c1} c2={det.c2} ({rule})")
    print(f"  coverage {best['coverage']}, warning fraction {best['warning_fraction']:.3f}, "
          f"precision {best['precision']}, onsets {best['onsets']}, missed {best['missed']}")
    print(f"  TARGETS coverage>=0.80 and warning<=0.25: "
          f"{'MET' if (best['coverage'] or 0) >= COVERAGE_TARGET and best['warning_fraction'] <= WARN_CAP else 'NOT MET'}")
    for ag, v in best["per_agent"].items():
        print(f"  {ag}: {v}")
    print("wrote", target)


if __name__ == "__main__":
    main()
