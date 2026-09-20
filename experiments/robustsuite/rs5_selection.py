"""RS Experiment 1, phase 5: the parameter search and the tuning-block selection (RS_DESIGN.md
7.2, 7.3 and 8; ticket 12).

Two stages, in this order:

    search   each candidate's pre-registered grid (7.3) on the FIRST 20 RS_DIAG seeds of every
             cell, both motions, against the baseline on the same episodes. The chosen
             configuration is the highest pooled success whose pooled collision rate is at most
             the baseline's + 0.01, else the default. `detour_yield`'s two configurations are
             both defaults and the pair the searches of `detour` and `yield` chose, so it runs
             last.
    tune     the baseline and the three candidates, each in its chosen configuration, on RS_TUNE
             (20 cells x 2 motions x 50 seeds), paired. The 7.2 rule disqualifies a candidate
             whose p95 control-step time over the whole block exceeds 100 ms; the section 8 rule
             then picks a winner, or reports "no improvement found".

The M0 anchor is not part of either stage: section 8 pools the 20 generated cells only.

    python -m experiments.robustsuite.rs5_selection --stage search --workers 14
    python -m experiments.robustsuite.rs5_selection --stage tune --workers 14
    python -m experiments.robustsuite.rs5_selection --stage report          # re-analyse only
    python -m experiments.robustsuite.rs5_selection --smoke --out DIR       # 1 seed, 40 steps

Outputs: MD_files/robustsuite/RS5_SELECTION_REPORT.md and results/robustsuite/RS5/ (one
checkpoint and one trace sidecar per arm, configuration and cell, plus search.json,
selection.json and provenance.json). With --out, everything goes under DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from experiments.robustsuite import protect_manifest as PM
from experiments.robustsuite import rs2_diagnostic as D
from robustsuite import blocks as RB
from robustsuite import candidates as CA
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite import selection as SEL

REPO = PM.REPO
#: The frozen baseline, the arm every candidate is measured against.
BASELINE = "astar_random"
#: The candidates in the pre-registered list order (7.4): the last tie-break of section 8.
CANDIDATES = ("detour", "yield", "detour_yield")
#: Changed components for section 8's tie-break (7.4, revised list): one each.
COMPONENTS = {"detour": 1, "yield": 1, "detour_yield": 1}
#: Seeds of RS_DIAG the 7.3 search runs on.
SEARCH_SEEDS = 20

#: Report and results.
REPORT_DIR = REPO / "MD_files/robustsuite"
REPORT_NAME = "RS5_SELECTION_REPORT.md"
RESULTS = REPO / "results/robustsuite/RS5"
#: The frozen parameter files whose hashes go into the provenance.
FROZEN_PARAM_FILES = D.FROZEN_PARAM_FILES
#: A --smoke run: seeds per cell and steps per episode.
SMOKE_SEEDS, SMOKE_STEPS = D.SMOKE_SEEDS, D.SMOKE_STEPS

#: What phase 5 shows, written after the run from this report's own tables. Printed as part of
#: section 0.
READING = """\
Written after the run, from the tables below.

1. **No candidate qualifies.** The best of the three, `detour`, gains +1.00 pp pooled success and
   costs +1.50 pp pooled collision, so it fails both conditions of §8 (≥ +2 pp and ≤ +1 pp). The
   other two are large regressions. Phase 6 therefore runs the baseline alone on RS_FINAL.
2. **`detour` does what it was built for, and pays for it in collisions.** Every trigger_block
   cell rises: open_clutter 0.17 → 0.28, rooms 0.04 → 0.08, aisles 0.07 → 0.14, corridors
   0.06 → 0.13, dense_clutter 0.32 → 0.37 (mean 0.132 → 0.200, worth +1.7 pp pooled over the 20
   cells). The collisions rise in the same five cells (mean 0.154 → 0.208, +1.35 pp pooled),
   which is essentially the whole of its +1.50 pp. A robot that leaves the standstill and follows
   a wall past the block meets the pedestrians that walk through that zone; the baseline's
   standstill was also its shelter. Outside the blocked cells `detour` changes almost nothing.
3. **The expected +6 to +14 pp did not appear.** The candidate list's estimate assumed the clutter
   families' short detours would clear their blocks (about +6.9 pp pooled from those two cells
   alone). Realised: +1.1 pp from open_clutter and dense_clutter trigger_block together. The way
   round is found more often than before, but far from always, and it costs collisions.
4. **`yield` is a large regression (−11.65 pp)**, exactly as the phase 4 checks predicted: it
   steps aside for tracks the frozen velocity tracker builds out of static geometry. It loses
   most in cells that hold no pedestrian at all (aisles_static 1.00 → 0.72, rooms_static
   0.96 → 0.82), and its timeout rate (0.198 → 0.316) and stuck rate (0.325 → 0.645) both roughly
   double. Its collision rate is slightly *lower* than the baseline's: it is a timid arm, not an
   unsafe one. `detour_yield` (−12.15 pp) is `yield` with detour's gains buried under it.
5. **The §7.3 search under-estimated `detour`'s collision cost.** Every `stall on` configuration
   went over the search's collision limit and the chosen `K8_L1.0_stall_off` was +0.12 pp there,
   against +1.50 pp here. Twenty seeds per cell is a small block, and the search's own cells are
   the ones the diagnostic already showed failing; §7.3 disclosed that cost in advance.
6. **Real time is not the binding constraint.** All three candidates pass §7.2 (p95 47.6–49.6 ms
   against the baseline's 46.1 ms). The per-step maxima (1.0–8.5 s) are contention between the
   14 worker processes, not the mechanisms: they appear in the baseline too.
"""


# --------------------------------------------------------------------------- running
#: Shared with the phase 2 entry point, as rs4_candidate_checks.py shares them: the cell's file
#: name, git, and the report's table writer.
cell_name, _git, _table = D.cell_name, D._git, D._table


def provenance(stage, search_seeds, tune_seeds, max_steps):
    modified, missing = PM.verify()
    import cvxpy
    return {"commit": _git("rev-parse", "HEAD"),
            "code_changes": _git("status", "--porcelain", "--", "robustsuite", "experiments",
                                 "highdim", "continuation", "dr_control", "robot_env",
                                 "optional_navigation", "evaluation",
                                 "config.json").splitlines(),
            "manifest": {"modified": modified, "missing": missing,
                         "sha256": PM._sha(PM.MANIFEST)},
            "frozen_params_sha256": {f: PM._sha(REPO / f) for f in FROZEN_PARAM_FILES},
            "generator_version": SC.GENERATOR_VERSION, "stage": stage,
            "seed_blocks": {"search": f"RS_DIAG[0:{len(search_seeds)}]",
                            "tune": f"RS_TUNE[0:{len(tune_seeds)}]"},
            "search_seeds": search_seeds, "tune_seeds": tune_seeds, "max_steps": max_steps,
            "arms": [BASELINE, *CANDIDATES],
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cvxpy": cvxpy.__version__}


def block_dir(res_dir, stage, arm, config=None):
    d = res_dir / stage / arm
    return d if config is None else d / config.replace("/", "_")


def run_arm(arm, config, seeds, res_dir, stage, *, workers, max_steps, tag, cells,
            report_only=False):
    """{cell: {motion: [light records]}} plus {cell: {(motion, seed): trace row}}."""
    recs, rows = {}, {}
    d = block_dir(res_dir, stage, arm, config)
    for cell in cells:
        ck = d / f"{cell_name(cell)}.jsonl"
        tr = d / f"{cell_name(cell)}.traces.jsonl.gz"
        if not report_only:
            print(f"[{stage}] {arm} {config or 'baseline'} {cell_name(cell)}", flush=True)
            RB.run_block(arm, cell, seeds, workers=workers, checkpoint=ck, traces=tr, tag=tag,
                         max_steps=max_steps, progress=40, config=config,
                         keep_step_times=(stage == "tune"))
        recs[cell], rows[cell] = RB.load_block(arm, cell, seeds, checkpoint=ck, traces=tr,
                                               tag=tag, max_steps=max_steps, config=config)
    return recs, rows


# --------------------------------------------------------------------------- counting
def counts(recs):
    """{cell name: {"success", "collision", "episodes"}}, both motions pooled within a cell."""
    out = {}
    for cell, by_motion in recs.items():
        flat = [r for m in RS.MOTIONS for r in by_motion[m]]
        out[cell_name(cell)] = {"success": sum(int(r["success"]) for r in flat),
                                "collision": sum(int(r["collision"]) for r in flat),
                                "episodes": len(flat)}
    return out


def step_times_ms(rows):
    """Every control step of an arm's block, in ms (the raw times from the trace sidecars)."""
    return np.asarray([1e3 * t for cell in rows for row in rows[cell].values()
                       for t in row.get("step_times", [])], float)


def timing(rows):
    """The §7.2 p95, over every control step of the block.

    Every episode must carry its step times. `keep_step_times` is outside the block fingerprint
    (robustsuite.blocks), so a block half-run without it would otherwise hand the rule a biased
    subset of its own steps.
    """
    episodes = [row for cell in rows for row in rows[cell].values()]
    without = [row for row in episodes if not row.get("step_times")]
    if without:
        raise SystemExit(f"{len(without)} of {len(episodes)} episodes have no step times: "
                         "the block was run without keep_step_times; delete it and rerun")
    t = step_times_ms(rows)
    if not len(t):
        raise SystemExit("no step times in the trace sidecars; rerun the tune stage")
    return {"steps": int(len(t)), "mean": float(t.mean()),
            "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "max": float(t.max())}


def outcome_rates(recs):
    """Per-cell timeout and stuck rates, for the report's context columns."""
    out = {}
    for cell, by_motion in recs.items():
        flat = [r for m in RS.MOTIONS for r in by_motion[m]]
        n = len(flat)
        out[cell_name(cell)] = {"timeout": sum(int(r["timeout"]) for r in flat) / n,
                                "stuck": sum(int(r["stuck"]) for r in flat) / n}
    return out


# --------------------------------------------------------------------------- stages
def _frac(x):
    return {"exact": f"{x.numerator}/{x.denominator}", "value": float(x)}


def _jsonable(o):
    if isinstance(o, Fraction):
        return _frac(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return o


def search_stage(res_dir, seeds, *, workers, max_steps, tag, cells, report_only):
    """The 7.3 grids of the three candidates, in list order, against the baseline."""
    base, _ = run_arm(BASELINE, None, seeds, res_dir, "search", workers=workers,
                      max_steps=max_steps, tag=tag, cells=cells, report_only=report_only)
    base_counts = counts(base)
    out, chosen = {}, {}
    for arm in CANDIDATES:
        pair = (chosen["detour"], chosen["yield"]) if arm == "detour_yield" else None
        grid = CA.search_grid(arm, chosen=pair)
        cfgs = []
        for name in grid:
            recs, _ = run_arm(arm, name, seeds, res_dir, "search", workers=workers,
                              max_steps=max_steps, tag=tag, cells=cells,
                              report_only=report_only)
            cfgs.append({"name": name, "counts": counts(recs)})
        out[arm] = SEL.choose_config(base_counts, cfgs, default=CA.DEFAULTS[arm].name,
                                     cells=[cell_name(c) for c in cells])
        chosen[arm] = out[arm]["chosen"]
        print(f"  {arm}: chose {chosen[arm]}"
              + (" (fallback to the default)" if out[arm]["fell_back"] else ""), flush=True)
    return {"seeds": seeds, "baseline_counts": base_counts, "arms": out, "chosen": chosen}


def tune_stage(res_dir, seeds, chosen, *, workers, max_steps, tag, cells, report_only):
    """The baseline and the three candidates on RS_TUNE, then the section 8 rule."""
    base, base_rows = run_arm(BASELINE, None, seeds, res_dir, "tune", workers=workers,
                              max_steps=max_steps, tag=tag, cells=cells,
                              report_only=report_only)
    arms = {BASELINE: {"config": None, "counts": counts(base), "timing": timing(base_rows),
                       "rates": outcome_rates(base)}}
    cands = []
    for arm in CANDIDATES:
        recs, rows = run_arm(arm, chosen[arm], seeds, res_dir, "tune", workers=workers,
                             max_steps=max_steps, tag=tag, cells=cells, report_only=report_only)
        t = timing(rows)
        arms[arm] = {"config": chosen[arm], "counts": counts(recs), "timing": t,
                     "rates": outcome_rates(recs)}
        cands.append({"name": arm, "counts": counts(recs), "p95_ms": t["p95"],
                      "components": COMPONENTS[arm]})
    verdict = SEL.select(counts(base), cands, cells=[cell_name(c) for c in cells])
    return {"seeds": seeds, "arms": arms, "selection": verdict}


# --------------------------------------------------------------------------- report
def _val(x):
    """A rate as a float, whether it is a Fraction in memory or its JSON form."""
    return x["value"] if isinstance(x, dict) else float(x)


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{_val(v):.{nd}f}"


def _pp(x):
    return f"{100 * _val(x):+.2f} pp"


def _arm_label(arm, config):
    return f"`{arm}`" + (f" (`{config}`)" if config else "")


def write_report(search, tune, prov, path, *, cells):
    sel = tune["selection"]
    base = sel["baseline"]
    lines = []
    add = lines.append
    code_clean = not prov["code_changes"]
    man_ok = not prov["manifest"]["modified"] and not prov["manifest"]["missing"]
    add("# RS Experiment 1, phase 5: parameter search and tuning-block selection\n")
    add("Pre-registration: [[RS_DESIGN]] §7.2, §7.3 and §8. Ticket 12. Generated by "
        "`experiments/robustsuite/rs5_selection.py`.\n")
    add(f"- Commit `{prov['commit'][:7]}`; code tree clean: {'yes' if code_clean else 'NO'}"
        + (f"; report regenerated at `{prov['report_commit'][:7]}`"
           if prov.get("report_commit", prov["commit"]) != prov["commit"] else ""))
    add(f"- Frozen-file manifest verified: {'yes' if man_ok else 'NO'} "
        f"(`{prov['manifest']['sha256'][:12]}`)")
    add("- Frozen parameters: " + ", ".join(f"`{Path(f).name}` `{h[:12]}`"
                                            for f, h in prov["frozen_params_sha256"].items()))
    add(f"- Generator `{prov['generator_version']}`; search on "
        f"{prov['seed_blocks']['search']}, selection on {prov['seed_blocks']['tune']}, "
        f"{len(cells)} cells x 2 motions")
    add("- Frozen SCS controller in every episode; every arm is the frozen `astar_random` stack "
        "with a behaviour layer (§7.4)")
    add(f"- Trace sidecars (not committed, regenerable from the seeds): SHA-256 in "
        f"`traces.sha256` (combined `{prov.get('trace_sha256', '')[:12]}`)\n")

    add("## 0. Verdict and reading\n")
    if sel["winner"] is None:
        add("**No improvement found.** No candidate met both conditions of §8, so there is no "
            "winner and phase 6 runs the baseline alone on RS_FINAL, as a characterisation of "
            "the suite (§8, spec story 47).\n")
    else:
        w = next(c for c in sel["candidates"] if c["name"] == sel["winner"])
        add(f"**Winner: `{sel['winner']}`** (`{tune['arms'][sel['winner']]['config']}`), "
            f"pooled success {_f(w['pooled']['success'])} against the baseline's "
            f"{_f(base['success'])} ({_pp(w['qualify']['d_success'])}), collision "
            f"{_f(w['pooled']['collision'])} against {_f(base['collision'])} "
            f"({_pp(w['qualify']['d_collision'])}). Phase 6 runs it against the baseline on "
            "RS_FINAL under the §9 gate.\n")
    add(READING)

    add("## 1. The selection rule on RS_TUNE (§8)\n")
    add(f"Pooled rates weight all {len(cells)} cells equally and pool both motions within a "
        "cell; the anchor is excluded. Counts are compared in exact rational arithmetic. A candidate "
        "qualifies at ≥ +2 pp pooled success and ≤ +1 pp pooled collision.\n")
    rows = [[_arm_label(BASELINE, None), base["episodes"],
             _f(base["success"]), "—", _f(base["collision"]), "—",
             _f(tune["arms"][BASELINE]["timing"]["p95"], 1), "—", "baseline"]]
    for c in sel["candidates"]:
        q = c["qualify"]
        rows.append([_arm_label(c["name"], tune["arms"][c["name"]]["config"]),
                     c["pooled"]["episodes"], _f(c["pooled"]["success"]),
                     _pp(q["d_success"]) if q else "—", _f(c["pooled"]["collision"]),
                     _pp(q["d_collision"]) if q else "—", _f(c["p95_ms"], 1),
                     c["components"],
                     c["disqualified"] or ("qualifies" if c["qualifies"] else
                                           "no: " + ", ".join(
                                               s for s, ok in (("success", q["pass_success"]),
                                                               ("collision",
                                                                q["pass_collision"]))
                                               if not ok))])
    add(_table(["arm", "episodes", "success", "Δ success", "collision", "Δ collision",
                "p95 (ms)", "components", "§8"], rows))
    add(f"Verdict: **{sel['verdict']}**. Tie-breaks, had there been a tie: fewer changed "
        "components, then the §7.4 list order (`detour`, `yield`, `detour_yield`; one component "
        "each).\n")

    add("## 2. Real time (§7.2)\n")
    add("The p95 is over every control step of the arm's whole tuning block, as in "
        "[[RS4_CANDIDATE_CHECKS]]. The limit is 100 ms.\n")
    add(_table(["arm", "control steps", "mean (ms)", "p50 (ms)", "p95 (ms)", "max (ms)", "§7.2"],
               [[_arm_label(a, v["config"]), v["timing"]["steps"], _f(v["timing"]["mean"], 1),
                 _f(v["timing"]["p50"], 1), _f(v["timing"]["p95"], 1),
                 _f(v["timing"]["max"], 1),
                 "pass" if SEL.passes_real_time(v["timing"]["p95"]) else "**DISQUALIFIED**"]
                for a, v in tune["arms"].items()]))

    add("## 3. Per-cell success and collision on RS_TUNE\n")
    arms = [BASELINE, *CANDIDATES]
    head = ["cell"] + [f"{a} S" for a in arms] + [f"{a} C" for a in arms]
    rows = []
    for cell in cells:
        n = cell_name(cell)
        row = [n]
        for key in ("success", "collision"):
            for a in arms:
                c = tune["arms"][a]["counts"][n]
                row.append(_f(c[key] / c["episodes"]))
        rows.append(row)
    add(_table(head, rows))

    add("## 4. Per family (equal-weight over its four cells)\n")
    fam_head = ["family"] + [f"{a} S" for a in arms]
    frows = []
    for fam in RS.FAMILIES:
        names = [cell_name(c) for c in cells if c.family == fam]
        if not names:                                  # only a smoke run narrows the grid
            continue
        row = [fam]
        for a in arms:
            cc = tune["arms"][a]["counts"]
            row.append(_f(float(sum(Fraction(cc[n]["success"], cc[n]["episodes"])
                                    for n in names) / len(names))))
        frows.append(row)
    add(_table(fam_head, frows))

    add("## 5. The parameter search (§7.3)\n")
    base_search = SEL.pooled(search["baseline_counts"], cells=[cell_name(c) for c in cells])
    add(f"Each candidate's pre-registered grid on the first {len(search['seeds'])} RS_DIAG "
        "seeds of every cell, both motions, against the baseline on the same episodes "
        f"(pooled success {_f(base_search['success'])}, collision "
        f"{_f(base_search['collision'])}). The chosen configuration "
        "is the highest pooled success whose pooled collision rate is at most the baseline's + "
        "0.01; a tie goes to the earlier configuration of the grid, whose first member is the "
        "default. The cost is the one §7.3 discloses: these are the seeds whose failures "
        "motivated the candidates.\n")
    for arm in CANDIDATES:
        s = search["arms"][arm]
        add(f"**`{arm}`** — chosen `{s['chosen']}`"
            + (" (no configuration met the collision limit, so the default stands)"
               if s["fell_back"] else "") + f", collision limit {_f(s['collision_limit'])}\n")
        add(_table(["configuration", "success", "Δ success", "collision", "Δ collision",
                    "≤ limit", "chosen"],
                   [[f"`{c['name']}`" + (" (default)" if c["order"] == 0 else ""),
                     _f(c["pooled"]["success"]), _pp(c["d_success"]),
                     _f(c["pooled"]["collision"]), _pp(c["d_collision"]),
                     "yes" if c["within_collision_limit"] else "no",
                     "**yes**" if c["name"] == s["chosen"] else ""]
                    for c in s["configurations"]]))

    add("## 6. Timeout and stuck rates on RS_TUNE (context, not gated)\n")
    add(_table(["arm", "timeout (pooled)", "stuck (pooled)"],
               [[_arm_label(a, v["config"]),
                 _f(float(np.mean([v["rates"][cell_name(c)]["timeout"] for c in cells]))),
                 _f(float(np.mean([v["rates"][cell_name(c)]["stuck"] for c in cells])))]
                for a, v in tune["arms"].items()]))

    path.write_text("\n".join(lines) + "\n")
    return path


def trace_digest(res_dir):
    """SHA-256 of every file under `res_dir` that .gitignore keeps out of the repository, and one
    combined digest of them all: the trace sidecars, and the search stage's per-configuration
    records. All of it is derived bulk that re-runs deterministically from its seeds; these
    digests are what proves what this run produced.
    """
    bulk = [*res_dir.rglob("*.traces.jsonl.gz"), *(res_dir / "search").rglob("*.jsonl")]
    shas = {str(tr.relative_to(res_dir)): hashlib.sha256(tr.read_bytes()).hexdigest()
            for tr in sorted(bulk)}
    combined = hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()
    return shas, combined


# --------------------------------------------------------------------------- entry point
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("search", "tune", "all", "report"), default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="write results, the report and its figures under this directory")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_SEEDS} seed per cell, {SMOKE_STEPS} steps per episode")
    ap.add_argument("--cells", default=None,
                    help="comma-separated family/obstacles, for a smoke run")
    args = ap.parse_args(argv)

    out_dir = args.out or REPO
    res_dir = (args.out / "results") if args.out else RESULTS
    report = (args.out / REPORT_NAME) if args.out else (REPORT_DIR / REPORT_NAME)
    res_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    cells = list(RS.CELLS)
    if args.cells:
        cells = []
        for text in args.cells.split(","):
            family, obstacles = text.split("/")
            cells.append(RS.Cell(family, obstacles))
    n_seeds = SMOKE_SEEDS if args.smoke else None
    max_steps = SMOKE_STEPS if args.smoke else None
    search_seeds = RS.seed_block("RS_DIAG", n_seeds or SEARCH_SEEDS)
    tune_seeds = RS.seed_block("RS_TUNE", n_seeds)
    prov_path = res_dir / "provenance.json"
    report_only = args.stage == "report"
    if report_only:
        prov = json.loads(prov_path.read_text())      # the run's own identity, not today's
        prov["report_commit"] = _git("rev-parse", "HEAD")
    else:
        prov = provenance(args.stage, search_seeds, tune_seeds, max_steps)
        if not args.smoke and (prov["code_changes"] or prov["manifest"]["modified"]
                               or prov["manifest"]["missing"]):
            raise SystemExit("commit the code (and keep the frozen manifest intact) before a "
                             "phase 5 run: the record must name the code it ran")
        prov_path.write_text(json.dumps(prov, indent=1))
    tag = prov["commit"]

    search_path, sel_path = res_dir / "search.json", res_dir / "selection.json"
    if args.stage in ("search", "all"):
        search = search_stage(res_dir, search_seeds, workers=args.workers, max_steps=max_steps,
                              tag=tag, cells=cells, report_only=report_only)
        search_path.write_text(json.dumps({**_jsonable(search), "provenance": prov}, indent=1))
    else:                     # the tune and report stages read the search back, not its blocks
        search = json.loads(search_path.read_text())
    if args.stage == "search":
        print(f"chosen: {search['chosen']}\nwrote {search_path}")
        return 0

    tune = tune_stage(res_dir, tune_seeds, search["chosen"], workers=args.workers,
                      max_steps=max_steps, tag=tag, cells=cells, report_only=report_only)
    shas, combined = trace_digest(res_dir)
    (res_dir / "traces.sha256").write_text(
        "".join(f"{h}  {n}\n" for n, h in sorted(shas.items())))
    prov["trace_sha256"] = combined
    prov_path.write_text(json.dumps(prov, indent=1))
    sel_path.write_text(json.dumps({**_jsonable(tune), "provenance": prov}, indent=1))
    write_report(search, tune, prov, report, cells=cells)
    print(f"{tune['selection']['verdict']}\nwrote {report} and {sel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
