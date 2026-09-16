"""Post-hoc detector search (fix 4): calibrate the CRITICAL thresholds and the hysteresis too.

    python -m experiments.pursuit.cd_sweep --run results/pursuit_policy_matrix/dev_B_fixall

EXPLORATORY, and labelled as such: the pre-registered rule fixed Critical at (margin < 0, clr < 0.05)
and calibrated only (c1, c2), which is why its 25 % warning cap was unreachable. Here all four
thresholds plus the hysteresis band and hold length are searched.

No simulation is run: in shadow mode the detector is a pure function of the logged (margin,
clearance) series, so every configuration is replayed offline over the development logs.

PROTOCOL, declared before running
    split      development seeds by half: SELECT on the first 25 seeds, REPORT on the held-out 25
    grid       crit_margin in {0, -0.05, -0.1, -0.2}; crit_clear in {0.05, 0, -0.05};
               c1 = crit_margin + d1, d1 in {0, 0.05, 0.1, 0.2}; c2 = crit_clear + d2, d2 in {0, 0.05, 0.1};
               hysteresis in {0, 0.02, 0.05}; hold_steps in {1, 2, 3}   (1 296 configurations)
    rule       among configs with coverage >= 0.80 AND warning fraction <= 0.25 on the SELECT half,
               maximise precision (ties: higher coverage, then lower warning fraction, then the
               tighter thresholds). If none qualify, report the best coverage under the cap.
    report     the selected config's numbers on the held-out half; a large select/held-out gap
               means the search overfitted and the result should not be used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from continuation.congestion.detector import DetectorConfig
from experiments.pursuit.cd_calibrate import evaluate, load_series

OUT = Path("results/pursuit_policy_matrix/calibration")
CRIT_MARGIN = (0.0, -0.05, -0.1, -0.2)
CRIT_CLEAR = (0.05, 0.0, -0.05)
D1 = (0.0, 0.05, 0.1, 0.2)
D2 = (0.0, 0.05, 0.1)
HYST = (0.0, 0.02, 0.05)
HOLD = (1, 2, 3)
COVERAGE_TARGET, WARN_CAP = 0.80, 0.25


def split(series):
    seeds = sorted({s for _, _, _, s, *_ in series})
    cut = seeds[len(seeds) // 2]
    return ([x for x in series if x[3] < cut], [x for x in series if x[3] >= cut])


def configs():
    for cm in CRIT_MARGIN:
        for cc in CRIT_CLEAR:
            for d1 in D1:
                for d2 in D2:
                    for h in HYST:
                        for k in HOLD:
                            yield DetectorConfig(c1=round(cm + d1, 4), c2=round(cc + d2, 4),
                                                 crit_margin=cm, crit_clear=cc,
                                                 hysteresis=h, hold_steps=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="_fix4")
    a = ap.parse_args()
    series, meta = load_series(a.run)
    sel, held = split(series)
    print(f"select {len({x[3] for x in sel})} seeds / {len(sel)} series; "
          f"held-out {len({x[3] for x in held})} seeds / {len(held)} series", flush=True)
    rows = []
    for i, det in enumerate(configs(), 1):
        r = evaluate(sel, det)
        rows.append({**det.as_dict(), **{k: r[k] for k in
                                         ("coverage", "warning_fraction", "precision")}})
        if i % 200 == 0:
            print(f"  {i} configs", flush=True)
    ok = [r for r in rows if (r["coverage"] or 0) >= COVERAGE_TARGET
          and r["warning_fraction"] <= WARN_CAP]
    if ok:
        best = max(ok, key=lambda r: (r["precision"] or 0, r["coverage"], -r["warning_fraction"],
                                      -r["c1"], -r["c2"]))
        rule = "targets met on the select half; highest precision"
    else:
        capped = [r for r in rows if r["warning_fraction"] <= WARN_CAP]
        best = max(capped or rows, key=lambda r: ((r["coverage"] or 0), r["precision"] or 0))
        rule = "targets NOT met on the select half; best coverage under the cap"
    det = DetectorConfig(**{k: best[k] for k in
                            ("c1", "c2", "crit_margin", "crit_clear", "hysteresis", "hold_steps")})
    ho = evaluate(held, det)
    print(f"\nselected {det.as_dict()}\n  rule: {rule}")
    print(f"  SELECT   coverage {best['coverage']:.3f} warning {best['warning_fraction']:.3f} "
          f"precision {best['precision']:.3f}")
    print(f"  HELD-OUT coverage {ho['coverage']:.3f} warning {ho['warning_fraction']:.3f} "
          f"precision {ho['precision']:.3f}")
    met = (ho["coverage"] or 0) >= COVERAGE_TARGET and ho["warning_fraction"] <= WARN_CAP
    print(f"  TARGETS on held-out data: {'MET' if met else 'NOT MET'}")
    for ag, v in ho["per_agent"].items():
        print(f"    {ag}: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"cd_thresholds{a.tag}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.write_text(json.dumps({"detector": det.as_dict(), "selected_by": rule,
                             "select_half": best, "held_out": ho, "targets_met_held_out": met,
                             "source_run": str(a.run), "cd_options": meta.get("cd"),
                             "exploratory": True, "grid_size": len(rows)}, indent=1, default=float))
    p.with_suffix(".sha256").write_text(__import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                        + f"  {p.name}\n")
    (OUT / f"sweep_grid{a.tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
