"""RS Experiment 1, phase 6: the sealed final evaluation (RS_DESIGN.md 6, 9 and 10; ticket 13).

**This is the only file allowed to open RS_FINAL** (`robustsuite.seeds.FINAL_ENTRY_POINT`, asserted
by tests/test_robustsuite.py). It runs ONCE: the frozen baseline on every cell (20 cells x 2
motions x 200 seeds) plus the M0 anchor, and, when phase 5 produced a winner, that winner on the
same episodes, paired. Phase 5 found no winner, so the sealed run is the baseline's
characterisation of the suite (RS_DESIGN 8, spec story 47) and the section 9 gate has nothing to
apply.

The analysis is the phase 2 diagnostic's, on the sealed block: `robustsuite.diagnostic` gives
every failed episode one failure mode, and `rs2_diagnostic` contributes the per-cell grouping,
the pooling and the report's table formatters, so the two reports are read the same way. The gate
itself is `robustsuite.gates` (pure, tested at its thresholds).

    python -m experiments.robustsuite.rs6_final --workers 14
    python -m experiments.robustsuite.rs6_final --stage report     # re-analyse finished blocks
    python -m experiments.robustsuite.rs6_final --smoke --out DIR  # 1 seed, 40 steps

Outputs: MD_files/robustsuite/RS6_FINAL_REPORT.md and results/robustsuite/RS6/ (one checkpoint
and one trace sidecar per arm and cell, analysis.json, gate.json, provenance.json and
traces.sha256). With --out, everything goes under DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.robustsuite import protect_manifest as PM
from experiments.robustsuite import rs2_diagnostic as D
from experiments.robustsuite import rs5_selection as S5
from robustsuite import blocks as RB
from robustsuite import gates as GT
from robustsuite import scenario as SC
from robustsuite import seeds as RS

REPO = PM.REPO
#: The frozen baseline. A winner, if phase 5 had produced one, joins it.
BASELINE = "astar_random"
#: Where phase 5 recorded its verdict.
SELECTION = REPO / "results/robustsuite/RS5/selection.json"
#: Report and results.
REPORT_DIR = REPO / "MD_files/robustsuite"
REPORT_NAME = "RS6_FINAL_REPORT.md"
RESULTS = REPO / "results/robustsuite/RS6"
#: Shared with the phase 2 entry point: the cell's file name, git, the report's table writer and
#: the formatters of its outcome and safety-layer blocks.
cell_name, _git, _table, _f = D.cell_name, D._git, D._table, D._f
#: A --smoke run: seeds per cell and steps per episode.
SMOKE_SEEDS, SMOKE_STEPS = D.SMOKE_SEEDS, D.SMOKE_STEPS

#: Departures from the spec's wording, all fixed before the run that produced them, with the
#: section of RS_DESIGN that records each. Printed as the report's Deviations section
#: (RS_DESIGN 10, spec story 54).
DEVIATIONS = """\
Every entry was written before the run it affects, in the section of [[RS_DESIGN]] named. None
changes a pre-registered threshold.

| Date | Departure | Recorded in |
|---|---|---|
| 2026-09-19 | Start and goal clearance 0.6 m, not 0.45 m, after the DCPError finding; randomized-pedestrian steering outside the turn-rate clip | §14.1 (approved) |
| 2026-09-19 | The triggered block keeps ≥ 0.8 m from the trigger centre (§4.6 asked only that it not overlap the disc) | §14.2 (approved) |
| 2026-09-19 | The triggered spawn ignores the block zone, and its pedestrian's noise has its own stream | §14.3 (approved) |
| 2026-09-19 | The candidate parameter search runs on RS_DIAG, not RS_TUNE (spec story 43 says the tuning block); the cost is disclosed | §7.3 |
| 2026-09-20 | The candidate list was revised before any candidate was built, so that A* is never run after reset | "Revision of the candidate list", re-approved |
| 2026-09-20 | Readings of open wording in the candidates' mechanisms | "Readings before any tuning run (tickets 11a–11c)" |
| 2026-09-20 | Readings of open wording in §§7.2, 7.3 and 8, including that p95 is pooled over every control step of a block | "Readings before the phase 5 run (ticket 12)" |

**Corrections** (not rule changes): §14.1 item 1 said h < 0 happens near "an obstacle"; it happens
within 0.6 m of the outer wall only, which the environment's LiDAR reads 0.3 m short.
"""


# --------------------------------------------------------------------------- the sealed block
def open_sealed_block(n=None):
    """The sealed RS_FINAL seeds. `robustsuite.seeds` refuses this call from any other file."""
    return RS.seed_block("RS_FINAL", n, allow_final=True)


def winner_of_phase_5(path=SELECTION):
    """(name, configuration) of the phase 5 winner, or (None, None) for "no improvement found"."""
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing: phase 5 has not been run, so phase 6 cannot start")
    sel = json.loads(Path(path).read_text())
    name = sel["selection"]["winner"]
    return (None, None) if name is None else (name, sel["arms"][name]["config"])


# --------------------------------------------------------------------------- running
def provenance(arms, seeds, max_steps):
    modified, missing = PM.verify()
    import cvxpy
    return {"commit": _git("rev-parse", "HEAD"),
            "code_changes": _git("status", "--porcelain", "--", "robustsuite", "experiments",
                                 "highdim", "continuation", "dr_control", "robot_env",
                                 "optional_navigation", "evaluation",
                                 "config.json").splitlines(),
            "manifest": {"modified": modified, "missing": missing,
                         "sha256": PM._sha(PM.MANIFEST)},
            "frozen_params_sha256": {f: PM._sha(REPO / f) for f in S5.FROZEN_PARAM_FILES},
            "generator_version": SC.GENERATOR_VERSION, "seed_block": "RS_FINAL",
            "seeds": seeds, "max_steps": max_steps,
            "arms": {a: c for a, c in arms}, "selection": str(SELECTION),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cvxpy": cvxpy.__version__}


def run_arm(arm, config, cells, seeds, res_dir, *, workers, max_steps, tag, report_only=False):
    """{cell: ({motion: [light records]}, {(motion, seed): trace row})} for one arm."""
    out = {}
    d = res_dir / arm
    for cell in cells:
        ck = d / f"{cell_name(cell)}.jsonl"
        tr = d / f"{cell_name(cell)}.traces.jsonl.gz"
        if not report_only:
            print(f"[{arm}] {cell_name(cell)}", flush=True)
            RB.run_block(arm, cell, seeds, workers=workers, checkpoint=ck, traces=tr, tag=tag,
                         max_steps=max_steps, progress=50, config=config)
        out[cell] = RB.load_block(arm, cell, seeds, checkpoint=ck, traces=tr, tag=tag,
                                  max_steps=max_steps, config=config)
    return out


def flat(res, cells):
    """{cell name: [records in one shared order]}, as robustsuite.gates pairs them."""
    return {cell_name(c): [r for m in RS.MOTIONS for r in res[c][0][m]] for c in cells}


def trace_digest(res_dir):
    """SHA-256 of every trace sidecar (derived bulk, kept out of the repository)."""
    shas = {str(tr.relative_to(res_dir)): hashlib.sha256(tr.read_bytes()).hexdigest()
            for tr in sorted(res_dir.rglob("*.traces.jsonl.gz"))}
    return shas, hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- report
def worst_cells(analysis, n=3):
    """The n cells with the lowest success (RS_DESIGN 10: worst-cell success)."""
    rows = [(name, c["all"]["success_rate"]) for name, c in analysis["cells"].items()
            if name != cell_name(RS.ANCHOR)]
    return sorted(rows, key=lambda r: r[1])[:n]


def write_report(arms, analyses, gate, prov, trace_sha, path, *, cells, n_seeds):
    lines = []
    add = lines.append
    base = analyses[BASELINE]
    names = [a for a, _ in arms]
    code_clean = not prov["code_changes"]
    man_ok = not prov["manifest"]["modified"] and not prov["manifest"]["missing"]
    grid = [c for c in cells if c != RS.ANCHOR]

    add("# RS Experiment 1, phase 6: the sealed final evaluation\n")
    add("Pre-registration: [[RS_DESIGN]] §6 (phase 6), §9 (the GO gate) and §10. Ticket 13. "
        "Generated by `experiments/robustsuite/rs6_final.py`, the only entry point that opens "
        "the sealed block.\n")
    add(f"- Commit `{prov['commit'][:7]}`; code tree clean: {'yes' if code_clean else 'NO'}"
        + (f"; report regenerated at `{prov['report_commit'][:7]}`"
           if prov.get("report_commit", prov["commit"]) != prov["commit"] else ""))
    add(f"- Frozen-file manifest verified: {'yes' if man_ok else 'NO'} "
        f"(`{prov['manifest']['sha256'][:12]}`)")
    add("- Frozen parameters: " + ", ".join(f"`{Path(f).name}` `{h[:12]}`"
                                            for f, h in prov["frozen_params_sha256"].items()))
    add(f"- Generator `{prov['generator_version']}`, seed block RS_FINAL ({n_seeds} seeds), "
        f"both motion conditions, {len(grid)} cells plus the M0 anchor, "
        f"{prov['max_steps'] or 500} steps")
    add("- Arms: " + ", ".join(f"`{a}`" + (f" (`{c}`)" if c else "") for a, c in arms)
        + "; frozen SCS controller in every episode")
    add(f"- Trace sidecars (not committed, regenerable from the seeds): SHA-256 in "
        f"`traces.sha256` (combined `{trace_sha[:12]}`)\n")

    # ---------------------------------------------------------------- 0. verdict
    add("## 0. Verdict\n")
    if gate is None:
        p = base["pooled"]
        add("Phase 5 found **no improvement** ([[RS5_SELECTION_REPORT]]): no candidate met the "
            "§8 rule, so there is no winner and the §9 GO gate has nothing to apply. This run "
            "is therefore what §8 and spec story 47 ask for — **the frozen stack's "
            "characterisation of the suite on layouts no candidate was tuned on**.\n")
        add(f"On the sealed block the baseline reaches **{_f(p['success_rate'])} pooled "
            f"success**, with {_f(p['collision_rate'])} collision and {_f(p['timeout_rate'])} "
            "timeout, over 20 cells weighted equally.\n")
    else:
        add(f"**{gate['verdict']}**: {_gate_sentence(gate)}\n")

    # ---------------------------------------------------------------- 1. pooled
    add(f"## 1. Pooled result ({len(grid)} cells, equal weights, both motions)\n")
    head = ["arm"] + D.OUTCOME_HEAD
    add(_table(head, [[f"`{a}`"] + D._outcomes(analyses[a]["pooled"]) for a in names]))
    add(_table(["arm"] + D.SAFETY_HEAD,
               [[f"`{a}`"] + D._safety(analyses[a]["pooled"]) for a in names]))

    # ---------------------------------------------------------------- 2. per family / cell
    add("## 2. Per family (equal weights within the family)\n")
    add(_table(["family", "arm"] + D.OUTCOME_HEAD,
               [[fam, f"`{a}`"] + D._outcomes(analyses[a]["families"][fam])
                for fam in RS.FAMILIES if fam in base["families"] for a in names]))

    add("## 3. Per cell\n")
    add(_table(["cell", "arm"] + D.OUTCOME_HEAD + ["most common failure"],
               [[cell_name(c), f"`{a}`"] + D._outcomes(analyses[a]["cells"][cell_name(c)]["all"])
                + [analyses[a]["cells"][cell_name(c)]["all"]["most_common_mode"] or "—"]
                for c in grid for a in names]))

    add("## 4. Worst cells (lowest success)\n")
    add(_table(["cell"] + [f"`{a}`" for a in names],
               [[name] + [_f(analyses[a]["cells"][name]["all"]["success_rate"]) for a in names]
                for name, _ in worst_cells(base)]))

    # ---------------------------------------------------------------- 5. failure modes
    add("## 5. Failure modes\n")
    add("Every failed episode gets exactly one mode, the first that matches "
        "(`robustsuite.diagnostic`). Counts are over the pooled cells.\n")
    add(_table(["mode", "count", "share of episodes", "by condition"],
               [[m["mode"], m["count"], _f(m["share_episodes"]),
                 ", ".join(f"{k} {v}" for k, v in m["by_condition"].items() if v)]
                for m in base["modes"]]))

    # ---------------------------------------------------------------- 6. triggered events
    add("## 6. Triggered cells, overall and conditional on the trigger having fired\n")
    add("A triggered cell's episode can end before the robot reaches the trigger region; the "
        "`fired` rows are the episodes in which the event actually happened (spec story 35).\n")
    rows = []
    for c in grid:
        if c.obstacles not in ("trigger_block", "trigger_spawn"):
            continue
        for a in names:
            cell = analyses[a]["cells"][cell_name(c)]
            rows.append([cell_name(c), f"`{a}`", "all"] + D._outcomes(cell["all"]))
            if cell.get("fired"):
                rows.append(["", "", f"fired ({_f(cell['event']['fired_rate'])})"]
                            + D._outcomes(cell["fired"]))
    add(_table(["cell", "arm", "episodes"] + D.OUTCOME_HEAD, rows))

    add("### 6.1 Where and when the failures happen relative to the event\n")
    add("Median [q1, q3] over the episodes in which the trigger fired.\n")
    add(_table(["cell", "arm", "failures before / after firing", "steps from firing to a "
                "collision", "collision distance to the block", "collision distance to the "
                "spawn point"],
               [[cell_name(c), f"`{a}`",
                 f"{analyses[a]['cells'][cell_name(c)]['event']['failures_before_fire']} / "
                 f"{analyses[a]['cells'][cell_name(c)]['event']['failures_after_fire']}",
                 D._quartiles(analyses[a]["cells"][cell_name(c)]["event"]
                              ["collision_steps_after_fire"]),
                 D._quartiles(analyses[a]["cells"][cell_name(c)]["event"]
                              ["collision_dist_block"], 2),
                 D._quartiles(analyses[a]["cells"][cell_name(c)]["event"]
                              ["collision_dist_spawn_start"], 2)]
                for c in grid if c.obstacles in ("trigger_block", "trigger_spawn")
                for a in names]))

    # ---------------------------------------------------------------- 7. the anchor
    if "anchor_check" in base:
        add("## 7. The M0 anchor\n")
        add("The anchor keeps M0's own map and motion and is outside the pooled rates (§5). It "
            "is compared with HD0's re-run of the same arm on M0 (an unpaired Fisher exact "
            "test: different seeds).\n")
        add(_table(["arm", "motion", "successes", "rate", "95% CI", "HD0 reference", "p",
                    "consistent"],
                   [[f"`{a}`", m, f"{v['k']} / {v['n']}", _f(v["rate"]),
                     f"[{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]",
                     f"{v['k_ref']} / {v['n_ref']} ({_f(v['rate_ref'])})", _f(v["p"]),
                     "yes" if v["consistent"] else "**no**"]
                    for a in names for m, v in analyses[a]["anchor_check"].items()]))

    # ---------------------------------------------------------------- 8. the gate
    add("## 8. The GO gate (§9)\n")
    if gate is None:
        add("**Not applicable.** §9 compares a winner with the baseline, and phase 5 produced "
            "none ([[RS5_SELECTION_REPORT]]). The baseline ran alone, as §8 requires when no "
            "candidate qualifies. Nothing in this report was used to choose an arm: the arm was "
            "fixed before the sealed block was opened.\n")
    else:
        add(_gate_tables(gate))

    # ---------------------------------------------------------------- 9. deviations
    add("## 9. Deviations\n")
    add(DEVIATIONS)
    path.write_text("\n".join(lines) + "\n")
    return path


def _fr(v, nd=3):
    """Format an exact pooled rate: the gate keeps them as Fractions, the formatter wants floats."""
    return _f(None if v is None else float(v), nd)


def _gate_sentence(gate):
    s, c, a = gate["success"], gate["safety"], gate["anchor"]
    parts = [f"pooled success {_fr(s['winner'])} against {_fr(s['base'])} "
             f"({100 * s['diff']:+.2f} pp, McNemar p {s['p']:.4g})",
             f"collision {_fr(c['winner'])} against {_fr(c['base'])} "
             f"({100 * c['diff']:+.2f} pp, p {c['p']:.4g})"]
    if a is not None:
        parts.append(f"the M0 anchor {_fr(a['winner'])} against {_fr(a['base'])} (p {a['p']:.4g})")
    return "; ".join(parts) + "."


def _gate_tables(gate):
    s, c, a = gate["success"], gate["safety"], gate["anchor"]
    rows = [["1. success higher, p < 0.05", _fr(s["base"]), _fr(s["winner"]),
             f"{100 * s['diff']:+.2f} pp", f"{s['p']:.4g}",
             "pass" if s["pass"] else "**fail**"],
            ["2. collision not significantly higher", _fr(c["base"]), _fr(c["winner"]),
             f"{100 * c['diff']:+.2f} pp", f"{c['p']:.4g}",
             "**fail**" if c["fail"] else "pass"]]
    if a is not None:
        rows.append(["3. M0 anchor not significantly lower", _fr(a["base"]), _fr(a["winner"]),
                     f"{100 * (a['winner'] - a['base']):+.2f} pp", f"{a['p']:.4g}",
                     "**fail**" if a["fail"] else "pass"])
    out = _table(["condition", "baseline", "winner", "difference", "McNemar p", "verdict"], rows)
    out += f"\n**{gate['verdict']}**.\n\n"
    n_cells = len(gate["per_cell"])
    per_cell_n = next(iter(gate["per_cell"].values()))["n"]
    out += (f"### 8.1 Per cell (reported, not gated)\n\nEach cell is {per_cell_n} paired "
            f"episodes. {n_cells} cell-level tests at p < 0.05 produce about "
            f"{n_cells * 0.05:.0f} false positive(s) by chance, so a single significant cell is "
            "not evidence on its own; the gate is the pooled comparison above.\n\n")
    out += _table(["cell", "baseline", "winner", "difference", "McNemar p"],
                  [[name, _f(v["base"]), _f(v["winner"]), f"{100 * v['d_success']:+.2f} pp",
                    f"{v['p']:.4g}"] for name, v in gate["per_cell"].items()])
    return out


# --------------------------------------------------------------------------- entry point
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("run", "report"), default="run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_SEEDS} seed per cell, {SMOKE_STEPS} steps per episode")
    ap.add_argument("--cells", default=None,
                    help="comma-separated family/obstacles, for a smoke run")
    ap.add_argument("--winner", default=None,
                    help="smoke runs only: exercise the gate with this arm as the winner")
    args = ap.parse_args(argv)
    if args.winner and not args.smoke:
        raise SystemExit("--winner is for smoke runs only: the real winner is phase 5's, read "
                         f"from {SELECTION}")

    res_dir = (args.out / "results") if args.out else RESULTS
    report = (args.out / REPORT_NAME) if args.out else (REPORT_DIR / REPORT_NAME)
    res_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    cells = [*RS.CELLS, RS.ANCHOR]
    if args.cells:
        cells = [RS.Cell(*t.split("/")) for t in args.cells.split(",")]
    n_seeds = SMOKE_SEEDS if args.smoke else None
    max_steps = SMOKE_STEPS if args.smoke else None
    seeds = open_sealed_block(n_seeds)

    if args.winner:
        winner, config = args.winner, None
    else:
        winner, config = winner_of_phase_5()
    arms = [(BASELINE, None)] + ([(winner, config)] if winner else [])
    print(f"arms: {arms}; {len(seeds)} sealed seeds x {len(cells)} cells x 2 motions", flush=True)

    prov_path = res_dir / "provenance.json"
    report_only = args.stage == "report"
    if report_only:
        prov = json.loads(prov_path.read_text())
        prov["report_commit"] = _git("rev-parse", "HEAD")
    else:
        prov = provenance(arms, seeds, max_steps)
        if not args.smoke and (prov["code_changes"] or prov["manifest"]["modified"]
                               or prov["manifest"]["missing"]):
            raise SystemExit("commit the code (and keep the frozen manifest intact) before the "
                             "sealed run: the record must name the code it ran")
        prov_path.write_text(json.dumps(prov, indent=1))
    tag = prov["commit"]

    res = {a: run_arm(a, c, cells, seeds, res_dir, workers=args.workers, max_steps=max_steps,
                      tag=tag, report_only=report_only) for a, c in arms}
    analyses = {a: D.analyse(res[a])[0] for a, _ in arms}
    grid = [c for c in cells if c != RS.ANCHOR]
    gate = None
    if winner:
        anchor = ((flat(res[BASELINE], [RS.ANCHOR])[cell_name(RS.ANCHOR)],
                   flat(res[winner], [RS.ANCHOR])[cell_name(RS.ANCHOR)])
                  if RS.ANCHOR in cells else None)
        gate = GT.go_gate(flat(res[BASELINE], grid), flat(res[winner], grid), anchor=anchor,
                          cells=[cell_name(c) for c in grid])
        (res_dir / "gate.json").write_text(json.dumps(D._nan_to_null(gate), indent=1,
                                                      default=float, allow_nan=False))
    shas, combined = trace_digest(res_dir)
    (res_dir / "traces.sha256").write_text(
        "".join(f"{h}  {n}\n" for n, h in sorted(shas.items())))
    prov["trace_sha256"] = combined
    prov_path.write_text(json.dumps(prov, indent=1))
    (res_dir / "analysis.json").write_text(json.dumps(
        D._nan_to_null({"arms": analyses, "gate": gate, "provenance": prov}), indent=1,
        default=float, allow_nan=False))
    write_report(arms, analyses, gate, prov, combined, report, cells=cells, n_seeds=len(seeds))
    print(f"{'no winner: baseline characterisation' if gate is None else gate['verdict']}\n"
          f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
