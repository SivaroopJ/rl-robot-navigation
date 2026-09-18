"""HD Experiment 1, phases 1-2: floor and ceiling on HD_DEV and HD_FINAL, then the stop check.

Pre-registration: MD_files/highdim/HD_DESIGN.md sections 4-7. Run order, fixed there:

    1. floor   (goal_random)   HD_DEV     100 seeds x {fixed, randomized}
    2. floor   (goal_random)   HD_FINAL   200 seeds x {fixed, randomized}
    3. ceiling (astar_random)  HD_FINAL   same seeds
    4. ceiling (astar_random)  HD_DEV

Frozen SCS throughout (enforced by highdim.harness). Then, on HD_FINAL pooled over conditions:
STOP if S_ceiling - S_floor < 0.05 (highdim.gates.stop_check). Paired statistics are
continuation.stats.compare_all (exact McNemar + paired bootstrap), descriptive only here.

This is one of the two entry points permitted to open HD_FINAL (tests/test_highdim.py).

    python -m experiments.highdim.hd0_baselines --workers 14
    python -m experiments.highdim.hd0_baselines --smoke --out DIR  # 2 HD_DEV seeds, 60 steps
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from continuation.stats import compare_all, paired_binary, summaries
from experiments.highdim import protect_manifest as PM
from highdim import blocks as HB
from highdim.gates import stop_check
from highdim.seeds import seed_block

REPO = PM.REPO
RESULTS = REPO / "results/highdim/HD0"
REPORT = REPO / "MD_files/highdim/HD0_BASELINES_REPORT.md"
ORDER = [("goal_random", "HD_DEV"), ("goal_random", "HD_FINAL"),
         ("astar_random", "HD_FINAL"), ("astar_random", "HD_DEV")]
LABEL = {"goal_random": "floor", "astar_random": "ceiling"}
FROZEN_PARAM_FILES = ["results/week6_continuation/H2/tuned_frozen.json",
                      "results/week6_continuation/R3/random_frozen.json"]


def seeds_for(block, n=None):
    if block == "HD_FINAL":
        return seed_block("HD_FINAL", n, allow_final=True)
    return seed_block(block, n)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def provenance():
    """Code identity for the run. `code_changes` lists tracked changes AND untracked files under
    the code paths, so a run cannot record a commit that does not contain the code it ran."""
    modified, missing = PM.verify()
    import cvxpy
    return {"commit": _git("rev-parse", "HEAD"),
            "code_changes": _git("status", "--porcelain", "--", "highdim", "experiments",
                                 "continuation", "dr_control", "robot_env",
                                 "optional_navigation", "evaluation", "config.json"),
            "manifest": {"modified": modified, "missing": missing,
                         "sha256": PM._sha(PM.MANIFEST)},
            "frozen_params_sha256": {f: PM._sha(REPO / f) for f in FROZEN_PARAM_FILES},
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cvxpy": cvxpy.__version__}


def run(out_dir, *, workers, n=None, max_steps=None, progress=25, smoke=False, tag=""):
    """{(kind, block): {condition: [light records]}}, in the pre-registered order.

    smoke=True never opens HD_FINAL: its slot is filled with HD_DEV seeds.
    """
    out_dir = Path(out_dir)
    res = {}
    for kind, block in ORDER:
        stem = f"{LABEL[kind]}_{block}"
        print(f"[{stem}]", flush=True)
        seeds = seeds_for("HD_DEV" if smoke else block, n)
        res[(kind, block)] = HB.run_block(
            kind, seeds, workers=workers, max_steps=max_steps, progress=progress, tag=tag,
            checkpoint=out_dir / f"{stem}.jsonl", traces=out_dir / f"{stem}.traces.jsonl.gz")
    for block in ("HD_DEV", "HD_FINAL"):
        HB.check_paired(res[("goal_random", block)], res[("astar_random", block)])
    return res


def _pooled(r):
    return [x for c in r for x in r[c]]


def _with_stuck(summ, res_arm):
    for c, recs in list(res_arm.items()) + [("pooled", _pooled(res_arm))]:
        summ[c]["stuck_rate"] = float(np.mean([r["stuck"] for r in recs]))
    return summ


def analyse(res):
    out = {"summaries": {}, "compare_ceiling_vs_floor": {}}
    for block in ("HD_DEV", "HD_FINAL"):
        floor, ceil = res[("goal_random", block)], res[("astar_random", block)]
        out["summaries"][block] = {
            "floor": _with_stuck(summaries(floor, f"floor {block}"), floor),
            "ceiling": _with_stuck(summaries(ceil, f"ceiling {block}"), ceil)}
        cmp = compare_all(ceil, floor)
        for c in cmp:
            x = ceil[c] if c != "pooled" else _pooled(ceil)
            y = floor[c] if c != "pooled" else _pooled(floor)
            cmp[c]["stuck"] = paired_binary(x, y, "stuck")
        out["compare_ceiling_vs_floor"][block] = cmp
    out["stop_check"] = stop_check(_pooled(res[("goal_random", "HD_FINAL")]),
                                   _pooled(res[("astar_random", "HD_FINAL")]))
    return out


# --------------------------------------------------------------------------- report
ROWS = [("success", "success_rate"), ("collision", "collision_rate"),
        ("dynamic", "dynamic_collision_rate"), ("static", "static_collision_rate"),
        ("wall", "wall_collision_rate"), ("timeout", "timeout_rate"), ("stuck", "stuck_rate"),
        ("min clearance (mean)", "min_clearance_mean"), ("SPL", "mean_spl"),
        ("infeasible-step rate", "infeasible_step_rate"),
        ("Random events / ep", "infeasible_events_per_episode"),
        ("planner failures", "planner_failures"), ("step ms (mean)", "step_time_ms_mean_ep")]
TESTS = ["success", "collision", "timeout", "stuck", "dynamic_collision", "spl",
         "min_clearance"]


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return "—" if v is None or v != v else f"{v:.3f}"


def _table(summ):
    head = "| metric | " + " | ".join(f"{a} {c}" for a in ("floor", "ceiling")
                                        for c in ("fixed", "randomized", "pooled")) + " |"
    lines = [head, "|---" * 7 + "|"]
    for name, key in ROWS:
        cells = [_fmt(summ[a][c].get(key)) for a in ("floor", "ceiling")
                 for c in ("fixed", "randomized", "pooled")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tests(cmp):
    lines = ["| ceiling − floor | condition | diff | 95% CI | discordant (ceiling-only / floor-only) | McNemar p |",
             "|---|---|---|---|---|---|"]
    for k in TESTS:
        for c in ("fixed", "randomized", "pooled"):
            v = cmp[c][k]
            disc = (f"{v['only_a']} / {v['only_b']}" if "only_a" in v else "—")
            p = f"{v['p']:.4g}" if "p" in v else "—"
            lines.append(f"| {k} | {c} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, "
                         f"{v['ci95'][1]:+.3f}] | {disc} | {p} |")
    return "\n".join(lines)


def report(analysis, prov, trace_sha):
    v = analysis["stop_check"]
    tuned = json.loads((REPO / FROZEN_PARAM_FILES[0]).read_text())["params"]
    spec = json.loads((REPO / FROZEN_PARAM_FILES[1]).read_text())["spec"]
    n_final = analysis["summaries"]["HD_FINAL"]["floor"]["fixed"]["episodes"]
    n_dev = analysis["summaries"]["HD_DEV"]["floor"]["fixed"]["episodes"]
    verdict = ("**STOP.** A* contributes less than 5 pp on M0; phases 3–7 are cancelled."
               if v["stop"] else
               "**PROCEED.** A* contributes at least 5 pp on M0; phase 3 (fast-solver check) "
               "may start.")
    s = analysis["summaries"]
    return f"""# HD Experiment 1 — phases 1–2: floor and ceiling baselines, stop check

Pre-registration: [[HD_DESIGN]] §4–7. Generated by `experiments/highdim/hd0_baselines.py`.

- Commit `{prov['commit'][:7]}`; code tree clean: {"yes" if not prov['code_changes'] else "NO"}
- Frozen-file manifest: modified {prov['manifest']['modified']}, missing {prov['manifest']['missing']}
- Frozen parameters: tuned α {tuned['alpha']}, τ_eff {tuned['tau_eff']}; Random K {spec['K']}, H {spec['H']}, speeds {spec['speeds']}
- Controller: frozen SCS (`ClfCbfDrccpController`) in every episode
- Environment: M0 (`exp6.make_env`), 6 dynamic obstacles at 0.675 m/s, 500 steps
- Arms: **floor** = goal-only + Random (planning off, no map, γ = goal);
  **ceiling** = A* + Random (the Week 6 F-D arm, re-run)

## Stop check (HD_FINAL, pooled, {v['n']} paired episodes per arm)

| S_floor | S_ceiling | gap | threshold | verdict |
|---|---|---|---|---|
| {v['S_floor']:.4f} | {v['S_ceiling']:.4f} | {v['gap']:+.4f} | {v['threshold']:.2f} | {"STOP" if v['stop'] else "PROCEED"} |

{verdict}

The historical F-D result (pooled success 0.910, collision 0.090) is context only.

## HD_FINAL ({n_final} seeds × 2 conditions)

{_table(s['HD_FINAL'])}

### Paired, ceiling vs floor (descriptive)

{_tests(analysis['compare_ceiling_vs_floor']['HD_FINAL'])}

## HD_DEV ({n_dev} seeds × 2 conditions)

These results are the reference for the pilot rule (§9.1: pilot dev success ≥ floor dev success).

{_table(s['HD_DEV'])}

### Paired, ceiling vs floor (descriptive)

{_tests(analysis['compare_ceiling_vs_floor']['HD_DEV'])}

## Data

- Records: `results/highdim/HD0/{{floor,ceiling}}_{{HD_DEV,HD_FINAL}}.jsonl`
- Per-step traces are gitignored (derived bulk; each episode re-runs deterministically from its
  seed). sha256:
{chr(10).join(f"  - `{k}`: `{h}`" for k, h in sorted(trace_sha.items()))}

## Disclosure

Before the entry point was reviewed, one development smoke run of this script used 2 HD_FINAL
seeds (14 100 000–14 100 001), truncated to 60 steps, for the floor and ceiling arms only. Those
arms are frozen, deterministic and untuned. No decision was, or could be, fitted to that output,
and it was discarded. The smoke mode has used HD_DEV seeds only since then (enforced by a test).
"""


def write_all_new(texts):
    """Write {path: text} only if NONE of the paths exists (results are never overwritten).

    Plain text, unlike continuation.harness.write_new (which JSON-encodes non-.jsonl objects),
    and all-or-nothing: every text is built before the first write."""
    clash = [p for p in texts if Path(p).exists()]
    if clash:
        raise SystemExit(f"{clash} exist; HD results are never overwritten")
    for p, text in texts.items():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--smoke", action="store_true",
                    help="2 HD_DEV seeds per block, 60 steps, into --out; never opens HD_FINAL")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    prov = provenance()
    if a.smoke:
        if a.out is None:
            raise SystemExit("--smoke needs --out DIR")
        out = a.out
        res = run(out, workers=a.workers, n=2, max_steps=60, progress=0, smoke=True)
        analysis = analyse(res)
        print(json.dumps(analysis["stop_check"]))
        return analysis

    def clean(p):
        return not (p["code_changes"] or p["manifest"]["modified"] or p["manifest"]["missing"])
    if not clean(prov):
        raise SystemExit(f"refusing to run: code changes or manifest failure\n{prov}")
    res = run(RESULTS, workers=a.workers, tag=prov["commit"])
    analysis = analyse(res)
    after = provenance()
    if not clean(after) or after["commit"] != prov["commit"]:
        raise SystemExit(f"code changed during the run; results not written\n{after}")
    trace_sha = {p.name: PM._sha(p) for p in sorted(RESULTS.glob("*.traces.jsonl.gz"))}
    write_all_new({
        RESULTS / "provenance.json": json.dumps(prov, indent=1),
        RESULTS / "analysis.json": json.dumps(analysis, indent=1, default=float),
        RESULTS / "traces.sha256": "".join(f"{h}  results/highdim/HD0/{k}\n"
                                           for k, h in sorted(trace_sha.items())),
        REPORT: report(analysis, prov, trace_sha)})
    print(json.dumps(analysis["stop_check"], indent=1))
    return analysis


if __name__ == "__main__":
    main()
