"""Track B / B0 analysis: where infeasibility happens and what predicts it.

Reads the immutable step log written by b0_diagnose.py and writes a report. Recomputes everything
from the raw rows; the raw file is never modified.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path("results/week6_trackb/B0")


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney). 0.5 = no predictive value; <0.5 = inversely predictive."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def runs(flags):
    out, r = [], 0
    for f in flags:
        if f:
            r += 1
        elif r:
            out.append(r); r = 0
    if r:
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="results/week6_trackb/B0/b0_steps_40.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.steps).read_text().splitlines()]
    report = {"source": a.steps, "n_steps": len(rows)}
    for arm in sorted({r["arm"] for r in rows}):
        R = [r for r in rows if r["arm"] == arm]
        inf = [r["infeasible"] for r in R]
        eps = sorted({r["seed"] for r in R})
        per_ep = {s: [r for r in R if r["seed"] == s] for s in eps}
        rl = [x for s in eps for x in runs([r["infeasible"] for r in per_ep[s]])]
        A = {"steps": len(R), "episodes": len(eps),
             "infeasible_steps": int(sum(inf)), "infeasible_step_rate": float(np.mean(inf)),
             "episodes_with_any": int(sum(1 for s in eps if any(r["infeasible"] for r in per_ep[s]))),
             "n_runs": len(rl), "runs_len1": int(sum(1 for x in rl if x == 1)),
             "run_len_mean": float(np.mean(rl)) if rl else 0.0,
             "run_len_max": int(max(rl)) if rl else 0}
        # state at infeasible vs feasible steps (ground truth, diagnostic only)
        for key in ("gt_true_clearance", "gt_dyn_clearance", "gt_static_clearance",
                    "gt_n_dyn_within_2m", "gt_wall_clearance", "rt_min_range",
                    "rt_min_corridor_width", "rt_n_tracks_within_2m", "rt_worst_closing_speed",
                    "rt_h_crit", "rt_m_u0", "rt_range_along_nominal"):
            i = [r[key] for r in R if r["infeasible"] and np.isfinite(r.get(key, np.nan))]
            f = [r[key] for r in R if not r["infeasible"] and np.isfinite(r.get(key, np.nan))]
            A[f"{key}|infeasible_mean"] = float(np.mean(i)) if i else float("nan")
            A[f"{key}|feasible_mean"] = float(np.mean(f)) if f else float("nan")
        # predictive value, 3 steps ahead, computed ONLY on currently-feasible steps
        feas = [r for r in R if not r["infeasible"]]
        y = [r["y_infeasible_next3"] for r in feas]
        A["base_rate_next3_from_feasible"] = float(np.mean(y))
        A["auc_next3"] = {}
        for key in [k for k in feas[0] if k.startswith("rt_")]:
            A["auc_next3"][key] = auc([r.get(key, np.nan) for r in feas], y)
        A["auc_next3_gt_reference"] = {
            k: auc([r.get(k, np.nan) for r in feas], y)
            for k in ("gt_true_clearance", "gt_dyn_clearance", "gt_n_dyn_within_2m")}
        # what follows an infeasible run
        after = Counter()
        for s in eps:
            E = per_ep[s]
            for i, r in enumerate(E):
                if r["infeasible"] and (i == 0 or not E[i - 1]["infeasible"]):
                    win = E[i:i + 10]
                    if E[-1]["outcome"] == "collision" and win[-1] is E[-1]:
                        after["collision_within_10"] += 1
                    elif any(x["recovered"] for x in win):
                        after["recovery_used"] += 1
                    elif all(not x["infeasible"] for x in win[1:]):
                        after["escaped_next_step"] += 1
                    else:
                        after["still_struggling"] += 1
        A["after_first_infeasible_of_a_run"] = dict(after)
        report[arm] = A
    p = OUT / "b0_analysis.json"
    p.write_text(json.dumps(report, indent=1, default=float))
    for arm in [k for k in report if k not in ("source", "n_steps")]:
        A = report[arm]
        print(f"\n===== {arm}: {A['steps']} steps / {A['episodes']} episodes, "
              f"infeasible {A['infeasible_steps']} ({A['infeasible_step_rate']:.4f}), "
              f"runs {A['n_runs']} (len1 {A['runs_len1']}, mean {A['run_len_mean']:.2f}, max {A['run_len_max']})")
        print(f"  base rate of infeasibility within 3 steps, from a feasible step: {A['base_rate_next3_from_feasible']:.4f}")
        print("  state at infeasible vs feasible steps:")
        for k in ("gt_true_clearance", "gt_dyn_clearance", "gt_n_dyn_within_2m", "rt_min_range",
                  "rt_min_corridor_width", "rt_n_tracks_within_2m", "rt_worst_closing_speed"):
            print(f"    {k:26s} {A[k+'|infeasible_mean']:>8.3f} vs {A[k+'|feasible_mean']:>8.3f}")
        top = sorted(A["auc_next3"].items(), key=lambda kv: -abs(kv[1] - 0.5) if kv[1] == kv[1] else 0)[:8]
        print("  most predictive RUNTIME features (AUC, 3-step horizon):")
        for k, v in top:
            print(f"    {k:26s} {v:.3f}")
        print("  ground-truth reference AUCs:", {k: round(v, 3) for k, v in A["auc_next3_gt_reference"].items()})
        print("  what follows the start of an infeasible run:", A["after_first_infeasible_of_a_run"])
    print("\nwrote", p)


if __name__ == "__main__":
    main()
