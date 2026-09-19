"""RS Experiment 1, phase 1 exit: the map validation report (RS_DESIGN.md 4.8, ticket 08).

Generates the suite on the RS_DIAG seed indices and measures it. NO ROBOT EPISODE IS RUN: only
the generator, the frozen oracle and pedestrian-only rollouts (tests/test_robustsuite.py checks
this). The researcher approves the report before phase 2 (the baseline diagnostic).

All four obstacle conditions and both motions of a family share one family draw (4.3, 5), so each
(family, seed) is generated once and every cell is a view of it; the draw statistics are
reported per family and hold for each of its cells. The consistency check makes that visible:
on the first seed of every family it regenerates all 4 conditions x 2 motions and compares them
with the shared draw.

    python -m experiments.robustsuite.rs1_map_validation --workers 14
    python -m experiments.robustsuite.rs1_map_validation --smoke --out DIR   # 2 seeds per family

Outputs: the report MD_files/robustsuite/RS1_MAP_VALIDATION_REPORT.md, its figures under
MD_files/robustsuite/figures/rs1/, and results/robustsuite/RS1/stats.json (per-draw metrics and
provenance). With --out, all three go under DIR.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.robustsuite import protect_manifest as PM
from robustsuite import events as EV
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite import validation as RV
from robustsuite.geometry import clearances

REPO = PM.REPO
REPORT_DIR = REPO / "MD_files/robustsuite"
REPORT_NAME = "RS1_MAP_VALIDATION_REPORT.md"
STATS_PATH = REPO / "results/robustsuite/RS1/stats.json"
#: Figures, relative to the report.
FIG_DIR = "figures/rs1"
#: Seeds rendered per cell.
SAMPLES = 4
#: Pedestrian-only rollouts: steps measured per draw and motion, and steps drawn in the renders.
ROLLOUT_STEPS, RENDER_STEPS = 500, 300
#: Seeds per family in a --smoke run.
SMOKE_SEEDS = 2
#: Trigger-fraction bins for the distribution table: four equal bins over the drawn range.
FRACTION_BINS = tuple(np.linspace(*EV.TRIGGER_FRACTION, 5))
#: A block whose detour is shorter than this leaves a quick way round (ticket 06 note).
SHORT_DETOUR = 1.0
#: Pedestrians per family (RS_DESIGN 4.2).
PED = {f: SC.FAMILIES[f].pedestrians for f in RS.FAMILIES}
#: Per-draw metrics summarised as distributions, with their table labels.
DISTRIBUTIONS = {
    "start_goal_clearance": "start/goal", "route_clearance": "start→goal route",
    "pedestrian_clearance": "pedestrian routes (block zone included)",
    "spawn_clearance": "spawn start", "spawn_entry_clearance": "spawn entry",
    "passages": "structural passages", "block_passage": "block passage",
    "route_length": "route length", "straight": "straight distance",
    "pedestrian_loop_length": "pedestrian loop length",
    "trigger_fraction": "trigger fraction", "block_offset": "block offset",
    "block_length": "block length", "detour": "detour", "spawn_distance": "spawn distance",
}
#: Metrics that hold a list per draw (pooled across the family's draws).
LISTED = ("passages", "pedestrian_loop_length")
KINDS = ("wall-wall", "wall-obstacle", "obstacle-obstacle")


# --------------------------------------------------------------------------- measurement
def _measure(task):
    family, seed = task
    spec = SC.generate(RS.Cell(family, "dynamic"), "fixed", seed)
    return spec, RV.draw_metrics(spec), {m: RV.rollout_metrics(spec, m, ROLLOUT_STEPS)
                                         for m in RS.MOTIONS}


def _consistency(task):
    """(specs checked, mismatches) between every (condition, motion) spec of a seed and its
    shared family draw."""
    family, seed = task
    base = SC.generate(RS.Cell(family, "dynamic"), "fixed", seed)
    bad = [(o, m) for o in RS.OBSTACLE_CONDITIONS for m in RS.MOTIONS
           if SC.generate(RS.Cell(family, o), m, seed) != replace(base, obstacles=o)]
    return len(RS.OBSTACLE_CONDITIONS) * len(RS.MOTIONS), bad


def anchor_rows(seeds):
    """M0's own start and goal for each seed (the canonical sampler; checked identical in both
    motions), measured like a generated draw, plus its obstacle count."""
    rows = []
    for seed in seeds:
        got = {}
        for m in RS.MOTIONS:
            layout, start, goal, track = RV.anchor_trails(seed, m, 0)
            got[m] = (tuple(map(float, start)), tuple(map(float, goal)), track.shape[1])
        start, goal, n_obstacles = got["fixed"]
        dist = float(np.linalg.norm(np.subtract(goal, start)))
        rows.append({"seed": seed, "start": list(start), "goal": list(goal),
                     "same_in_both_motions": got["fixed"] == got["randomized"],
                     "n_obstacles": n_obstacles, "straight": dist,
                     "route_length": SC.oracle(layout).path_length(start, goal),
                     "distance_bin": next((i for i, (lo, hi) in enumerate(SC.DISTANCE_BINS)
                                           if lo <= dist <= hi), None),
                     "start_goal_clearance": float(clearances([start, goal], layout).min())})
    return rows


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def provenance(seeds):
    """The commit, uncommitted changes under the RS code paths, and the frozen manifest."""
    modified, missing = PM.verify()
    return {"commit": _git("rev-parse", "HEAD"),
            "code_changes": _git("status", "--porcelain", "--", "robustsuite",
                                 "experiments/robustsuite").splitlines(),
            "manifest_ok": not modified and not missing,
            "seed_block": "RS_DIAG", "seeds": seeds}


def family_stats(ms, rolls):
    """Everything the report tabulates for one family, from its per-draw metrics and rollouts."""
    col = lambda k: [m[k] for m in ms]              # noqa: E731
    draws = len(ms)
    crossing = [m for m in ms if m["variant"] == "crossing"]
    return {
        "draws": draws, "layout_draws": sum(col("layout_draws")),
        "max_layout_draws": max(col("layout_draws")),
        "redraws_per_draw": {k: sum(col(f"{k}_redraws")) / draws
                             for k in ("bin", "event", "pedestrian")},
        "bins": [col("distance_bin").count(b) for b in range(len(SC.DISTANCE_BINS))],
        "fraction_bins": [int(c) for c in np.histogram(col("trigger_fraction"),
                                                      bins=FRACTION_BINS)[0]],
        "block_kinds": {k: col("block_kind").count(k) for k in KINDS},
        "short_detours": sum(d < SHORT_DETOUR for d in col("detour")),
        "variants": {v: col("variant").count(v) for v in EV.VARIANTS},
        "crossing": len(crossing),
        "crossing_zero": sum(bool(m["crossing_zero"]) for m in crossing),
        "route_ratio": float(np.median(np.divide(col("route_length"), col("straight")))),
        "summaries": {k: RV.summary([v for m in ms for v in m[k]] if k in LISTED else col(k))
                      for k in DISTRIBUTIONS},
        "rollouts": {mo: {"regular_min_clearance": min(r[mo]["regular_min_clearance"]
                                                       for r in rolls),
                          "spawn_min_clearance": min(r[mo]["spawn_min_clearance"]
                                                     for r in rolls),
                          "max_step": max(r[mo]["max_step"] for r in rolls)}
                     for mo in RS.MOTIONS},
    }


# --------------------------------------------------------------------------- figures
def _grid(n, cols=2, size=3.2):
    """A figure with n panels on `cols` columns; unused panels are hidden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(cols, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return plt, fig, axes.ravel()[:n]


def _save(plt, fig, out_dir, name):
    path = f"{FIG_DIR}/{name}.png"
    (out_dir / FIG_DIR).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / path, dpi=80)
    plt.close(fig)
    return path


def render_cells(specs, out_dir):
    """One figure per cell: the first SAMPLES draws of its family, as the cell sees them."""
    figs = {}
    for cell in RS.CELLS:
        sample = specs[cell.family][:SAMPLES]
        plt, fig, axes = _grid(len(sample))
        for ax, sp in zip(axes, sample):
            RV.draw_spec(ax, sp, cell.obstacles)
            ax.set_title(f"seed {sp.seed}", fontsize=8)
        fig.suptitle(f"{cell.family} / {cell.obstacles}", fontsize=10)
        figs[cell] = _save(plt, fig, out_dir, f"cell_{cell.family}_{cell.obstacles}")
    return figs


def render_anchor(rows, out_dir):
    """M0 samples (start and goal), then an obstacle-only rollout of the first seed in both
    motions: M0's obstacles pass through its rectangles, which is why the suite has pedestrians."""
    sample = rows[:SAMPLES]
    plt, fig, axes = _grid(len(sample) + len(RS.MOTIONS))
    for ax, r in zip(axes, sample):
        layout, *_ = RV.anchor_trails(r["seed"], "fixed", 0)
        RV.draw_anchor(ax, layout, r["start"], r["goal"])
        ax.set_title(f"seed {r['seed']}", fontsize=8)
    for ax, motion in zip(axes[len(sample):], RS.MOTIONS):
        layout, start, goal, track = RV.anchor_trails(rows[0]["seed"], motion, RENDER_STEPS)
        RV.draw_anchor(ax, layout, start, goal)
        for i in range(track.shape[1]):
            ax.plot(*track[:, i].T, lw=0.9)
        ax.set_title(f"{motion} obstacles, {RENDER_STEPS} steps", fontsize=8)
    fig.suptitle("M0 anchor: M0's map, sampler and motion models", fontsize=10)
    return _save(plt, fig, out_dir, "anchor_m0")


def render_rollouts(specs, out_dir):
    """Per family: RENDER_STEPS of pedestrian-only walking on the first draw, both motions. The
    regular pedestrians keep off the dashed block zone; the spawned one (purple) walks from its
    start at step 0 and ignores the zone."""
    figs = {}
    for fam in RS.FAMILIES:
        sp = specs[fam][0]
        plt, fig, axes = _grid(len(RS.MOTIONS), size=4.0)
        for ax, motion in zip(axes, RS.MOTIONS):
            regular, spawned = RV.pedestrian_trails(sp, motion, RENDER_STEPS)
            RV.draw_frame(ax, sp.layout)
            RV.draw_rect(ax, sp.block.rect, fill=False, ls="--", lw=0.8, color="tab:red")
            for i in range(regular.shape[1]):
                ax.plot(*regular[:, i].T, lw=0.9)
                ax.plot(*regular[-1, i], "o", ms=4, color="0.2")
            ax.plot(*spawned.T, lw=1.2, color="tab:purple")
            ax.plot(*spawned[-1], "o", ms=4, color="tab:purple")
            ax.set_title(f"{motion}, {RENDER_STEPS} steps", fontsize=8)
        fig.suptitle(f"{fam}: pedestrian-only rollout, seed {sp.seed}", fontsize=10)
        figs[fam] = _save(plt, fig, out_dir, f"rollout_{fam}")
    return figs


def render_block_kinds(specs, metrics, out_dir):
    """Open clutter: up to 2 blocks of each kind, fired, with their detour."""
    pairs = list(zip(specs["open_clutter"], metrics["open_clutter"]))
    picks = [pm for k in KINDS for pm in [p for p in pairs if p[1]["block_kind"] == k][:2]]
    if not picks:
        return None
    plt, fig, axes = _grid(len(picks), cols=3)
    for ax, (sp, m) in zip(axes, picks):
        RV.draw_spec(ax, sp, "trigger_block")
        ax.set_title(f"{m['block_kind']}, detour {m['detour']:.2f} m", fontsize=8)
    fig.suptitle("open_clutter: blocks by what bounds their passage", fontsize=10)
    return _save(plt, fig, out_dir, "open_clutter_block_kinds")


# --------------------------------------------------------------------------- report
def _table(header, rows):
    """Markdown table lines."""
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(c) for c in r) + " |" for r in rows), ""]


def _dist(s, nd=2):
    """A summary as min / p5 / median / p95 / max."""
    if not s["n"]:
        return "–"
    return " / ".join(f"{s[k]:.{nd}f}" for k in ("min", "p5", "median", "p95", "max"))


def _counts(values):
    return " / ".join(map(str, values))


def write_report(stats, figs, path):
    """The report, from `stats` and the figure paths alone."""
    fams, prov, consistency = stats["families"], stats["provenance"], stats["consistency"]
    anchor = stats["anchor"]
    cap, r_c = SC.RESAMPLE_CAP, SC.R_C
    lines = []
    add = lines.append

    def dist_table(keys):
        return _table(["family"] + [DISTRIBUTIONS[k] for k in keys],
                      [[f] + [_dist(s["summaries"][k]) for k in keys] for f, s in fams.items()])

    add("# RS Experiment 1: map validation report (phase 1 exit)\n")
    add(f"Generated by `experiments/robustsuite/rs1_map_validation.py` at commit "
        f"`{prov['commit'][:7]}` (uncommitted RS code changes at run time: "
        f"{len(prov['code_changes'])}; frozen manifest "
        f"{'verified' if prov['manifest_ok'] else 'NOT VERIFIED'}). Generator "
        f"`{stats['generator_version']}`, seed block RS_DIAG, {len(prov['seeds'])} seeds per "
        f"family. Pre-registration: RS_DESIGN §4.8. **No robot episode was run**: this is the "
        f"generator, the frozen oracle and pedestrian-only rollouts.\n")
    add("**Status: awaiting the researcher's approval** (RS_DESIGN §6: phase 1 → 2).\n")

    add("## For the researcher: what to judge\n")
    n_all = sum(s["draws"] for s in fams.values())
    fraction_counts = np.sum([s["fraction_bins"] for s in fams.values()], axis=0)
    worst = max(s["max_layout_draws"] for s in fams.values())
    rolls_ok = all(r["regular_min_clearance"] >= RV.FLOOR[mo] - 1e-5
                   and r["spawn_min_clearance"] >= RV.FLOOR[mo] - 1e-5
                   and r["max_step"] <= RV.MAX_STEP + 1e-5
                   for s in fams.values() for mo, r in s["rollouts"].items())
    add(f"- **Per family, not per cell.** Ticket 08 asks for statistics per cell. The four cells "
        f"of a family are views of one draw, so their draw statistics are identical and are "
        f"given once per family. Consistency check: {consistency['checked']} specs "
        f"regenerated, {consistency['mismatches']} mismatches.")
    add(f"- **Feasibility.** The most layout draws any one seed needed was {worst} (cap {cap}).")
    add(f"- **Pedestrians.** The pedestrian-only rollouts "
        f"{'meet' if rolls_ok else 'BREAK'} the clearance floors and the speed limit in every "
        f"family and motion (section 6).")
    add(f"- **Accepted trigger fractions lean early.** f is drawn uniformly on "
        f"[{EV.TRIGGER_FRACTION[0]}, {EV.TRIGGER_FRACTION[1]}], but the accepted draws per "
        f"bin of 0.1 are {_counts(fraction_counts)} of {n_all} (section 5). So events happen "
        f"earlier on the route than a uniform fraction would place them. Why later fractions "
        f"fail more is not measured here.")
    add(f"- **Blocks with a quick way round** (the ticket 06 question). This is the detour "
        f"count, not the block-end kinds: in the clutter families every block is bounded by a "
        f"box at one end or both. Detour under {SHORT_DETOUR:.0f} m: "
        + ", ".join(f"{f} {s['short_detours']} / {s['draws']}" for f, s in fams.items())
        + ".")
    empty = [f"{f} (bin {i + 1})" for f, s in fams.items() for i, b in enumerate(s["bins"])
             if b == 0]
    if empty:
        add("- **Empty distance bins:** " + ", ".join(empty) + " (section 3).")
    add("")

    add("## 1. Acceptance and resampling\n")
    add("Rates are per accepted draw. Acceptance is accepted draws / layout draws. A layout is "
        "redrawn after a failed bin, event or pedestrian draw (§4.7, §14.1 item 5).\n")
    lines += _table(
        ["family", "draws", "layout draws", "acceptance", f"max layout draws (cap {cap})",
         "bin redraws / draw", "event redraws / draw", "pedestrian redraws / draw"],
        [[f, s["draws"], s["layout_draws"], f"{s['draws'] / s['layout_draws']:.2f}",
          s["max_layout_draws"]] + [f"{s['redraws_per_draw'][k]:.2f}"
                                    for k in ("bin", "event", "pedestrian")]
         for f, s in fams.items()])

    add("## 2. Clearance\n")
    add(f"Five-number summaries: min / p5 / median / p95 / max, in metres. Floors: start and goal "
        f"{SC.START_GOAL_CLEARANCE} (§14.1 item 1); everything else {r_c} (r_c).\n")
    lines += dist_table(["start_goal_clearance", "route_clearance", "pedestrian_clearance",
                         "spawn_clearance", "spawn_entry_clearance"])

    add("## 3. Passages, routes and distance bins\n")
    add("Structural passages are the doorways (rooms), the aisles and cross-aisles (aisles) and "
        "the corridor strips (corridors). For the clutter families, the value is the "
        f"narrowest gap between two rectangles of a layout. Block passages are at most "
        f"{EV.MAX_PASSAGE} m.\n")
    lines += dist_table(["passages", "block_passage", "route_length", "straight",
                         "pedestrian_loop_length"])
    lines += _table(
        ["family", "route / straight (median)",
         "distance bins " + " / ".join(f"{lo}–{hi}" for lo, hi in SC.DISTANCE_BINS)],
        [[f, f"{s['route_ratio']:.2f}", _counts(s["bins"])] for f, s in fams.items()])

    add("## 4. Cells: pedestrians and events\n")
    rows = []
    for cell in RS.CELLS:
        n = PED[cell.family]
        rows.append([f"{cell.family} / {cell.obstacles}",
                     {"static": 0, "trigger_spawn": f"{n} + 1 spawned"}.get(cell.obstacles, n),
                     {"trigger_block": "block", "trigger_spawn": "spawn"}.get(cell.obstacles,
                                                                               "–")])
    rows.append(["m0 / anchor", f"{anchor['n_obstacles']} (M0's motion models)", "–"])
    lines += _table(["cell", "pedestrians", "event"], rows)

    add("## 5. Triggers and events\n")
    add(f"The pre-registered ranges are: trigger fraction {EV.TRIGGER_FRACTION}; block offset "
        f"{EV.BLOCK_OFFSET} m; block length at most {EV.MAX_PASSAGE + 2 * EV.BLOCK_OVERLAP} m; "
        f"spawn distance {EV.SPAWN_DISTANCE} m. The detour is the oracle path from the trigger "
        f"centre to the goal with the block in place, minus the same path without it (at r_c).\n")
    lines += dist_table(["trigger_fraction", "block_offset", "block_length", "detour",
                         "spawn_distance"])
    lines += _table(
        ["family", "trigger fraction per bin " + " / ".join(
            f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(FRACTION_BINS, FRACTION_BINS[1:])),
         "block ends " + " / ".join(KINDS), f"detour < {SHORT_DETOUR:.0f} m",
         "variants " + " / ".join(EV.VARIANTS), "crossing walks of length 0"],
        [[f, _counts(s["fraction_bins"]), _counts(s["block_kinds"][k] for k in KINDS),
          f"{s['short_detours']} / {s['draws']}",
          _counts(s["variants"][v] for v in EV.VARIANTS),
          f"{s['crossing_zero']} / {s['crossing']}"] for f, s in fams.items()])
    if figs["kinds"]:
        add(f"![open_clutter block kinds]({figs['kinds']})\n")

    add("## 6. Pedestrian-only rollouts\n")
    add(f"Every draw, both motions, {ROLLOUT_STEPS} steps, no robot. The regular pedestrians "
        f"are measured against the layout plus the block zone. The spawned one is measured "
        f"against the static layout (§14.3 item 1). Floors: {RV.FLOOR['fixed']} m fixed, "
        f"{RV.FLOOR['randomized']} m randomized (§4.4). The largest step must be at most "
        f"{RV.MAX_STEP:.4f} m.\n")
    lines += _table(["family", "motion", "regular: min clearance", "spawned: min clearance",
                     "max step"],
                    [[f, mo, f"{r['regular_min_clearance']:.3f}",
                      f"{r['spawn_min_clearance']:.3f}", f"{r['max_step']:.4f}"]
                     for f, s in fams.items() for mo, r in s["rollouts"].items()])
    for f in RS.FAMILIES:
        add(f"![rollout {f}]({figs['rollouts'][f]})\n")

    add("## 7. M0 anchor\n")
    add(f"M0's own map and stratified sampler on the same {len(anchor['rows'])} seeds. Start "
        f"and goal are identical in both motions for {anchor['same_in_both_motions']} / "
        f"{len(anchor['rows'])} seeds. M0 has {anchor['n_obstacles']} dynamic obstacles, which "
        f"pass through its rectangles, as the rollout panels show.\n")
    summ = anchor["summaries"]
    lines += _table(["start/goal clearance", "straight distance", "route length",
                     "distance bins"],
                    [[_dist(summ["start_goal_clearance"]), _dist(summ["straight"]),
                      _dist(summ["route_length"]), _counts(anchor["bins"])]])
    add(f"![M0 anchor]({figs['anchor']})\n")

    add("## 8. Rendered samples of every cell\n")
    add("Key to the renders:\n")
    add("- **Grey:** the static layout.")
    add("- **Blue:** the start→goal route (A*'s static plan runs close to it). The green dot "
        "is the start and the green star is the goal.")
    add("- **Thin lines:** pedestrian loops; grey dots are their starts.")
    add("- **Dashed red:** the block zone the pedestrians keep clear of (dynamic cells).")
    add("- **Solid red:** the fired block.")
    add("- **Orange disc:** the trigger.")
    add("- **Purple:** the spawn start (×), its scripted entry (thick) and its cycle (dotted).")
    add("\nThe block is not drawn in trigger_spawn cells, where it never appears and the spawn "
        "ignores it.\n")
    for cell in RS.CELLS:
        add(f"### {cell.family} / {cell.obstacles}\n")
        add(f"![{cell.family} {cell.obstacles}]({figs['cells'][cell]})\n")

    add("## Approval\n")
    add("Pending. The researcher's approval, or the changes they ask for, is recorded in ticket "
        "08 and in RS_DESIGN §6 before ticket 09 starts.\n")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_SEEDS} seeds per family")
    ap.add_argument("--out", type=Path, help="write the report, figures and stats under DIR")
    args = ap.parse_args(argv)
    seeds = RS.seed_block("RS_DIAG", SMOKE_SEEDS if args.smoke else None)
    report_dir = args.out or REPORT_DIR
    stats_path = (args.out / "stats.json") if args.out else STATS_PATH

    tasks = [(f, s) for f in RS.FAMILIES for s in seeds]
    firsts = [(f, seeds[0]) for f in RS.FAMILIES]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            measured = list(ex.map(_measure, tasks))
            checks = list(ex.map(_consistency, firsts))
    else:
        measured = [_measure(t) for t in tasks]
        checks = [_consistency(t) for t in firsts]

    specs, metrics, rolls = ({f: [] for f in RS.FAMILIES} for _ in range(3))
    for (f, _), (sp, m, r) in zip(tasks, measured):
        specs[f].append(sp)
        metrics[f].append(m)
        rolls[f].append(r)
    anchor = anchor_rows(seeds)

    report_dir.mkdir(parents=True, exist_ok=True)
    figs = {"cells": render_cells(specs, report_dir),
            "anchor": render_anchor(anchor, report_dir),
            "rollouts": render_rollouts(specs, report_dir),
            "kinds": render_block_kinds(specs, metrics, report_dir)}

    stats = {
        "provenance": provenance(seeds), "generator_version": SC.GENERATOR_VERSION,
        "consistency": {"checked": sum(n for n, _ in checks),
                        "mismatches": sum(len(b) for _, b in checks),
                        "failed": [[f, o, m] for (f, _), (_, b) in zip(firsts, checks)
                                   for o, m in b]},
        "families": {f: family_stats(metrics[f], rolls[f]) for f in RS.FAMILIES},
        "anchor": {"family": "m0", "rows": anchor, "n_obstacles": anchor[0]["n_obstacles"],
                   "same_in_both_motions": sum(r["same_in_both_motions"] for r in anchor),
                   "bins": [[r["distance_bin"] for r in anchor].count(b)
                            for b in range(len(SC.DISTANCE_BINS))],
                   "summaries": {k: RV.summary([r[k] for r in anchor])
                                 for k in ("start_goal_clearance", "straight",
                                           "route_length")}},
        "draws": metrics, "rollouts": rolls,
        "figures": [*figs["cells"].values(), figs["anchor"], *figs["rollouts"].values(),
                    *([figs["kinds"]] if figs["kinds"] else [])],
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, default=float))
    report = report_dir / REPORT_NAME
    write_report(stats, figs, report)
    print(f"report: {report}\nstats: {stats_path}")
    return {"draws": len(tasks), "report": report, "stats": stats_path}


if __name__ == "__main__":
    main()
