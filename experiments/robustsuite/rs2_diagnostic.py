"""RS Experiment 1, phase 2: the baseline diagnostic on RS_DIAG (RS_DESIGN.md 6, 10; ticket 09).

The frozen `astar_random` arm on every cell (20 cells x 2 motions x 50 seeds) plus the M0 anchor,
frozen SCS throughout (enforced by robustsuite.harness). One checkpointed, resumable block per
cell (robustsuite.blocks), tagged with the commit, so a resume after a code change is refused.
The analysis (robustsuite.diagnostic) gives every failed episode one failure mode and counts
DCPError freezes; the report ranks the modes and renders the most common one in each cell.

    python -m experiments.robustsuite.rs2_diagnostic --workers 14
    python -m experiments.robustsuite.rs2_diagnostic --report-only     # re-analyse finished blocks
    python -m experiments.robustsuite.rs2_diagnostic --smoke --out DIR # 1 seed, 40 steps

Outputs: MD_files/robustsuite/RS2_BASELINE_DIAGNOSTIC_REPORT.md with figures under
MD_files/robustsuite/figures/rs2/, and results/robustsuite/RS2/ (one .jsonl checkpoint and one
.traces.jsonl.gz per cell, analysis.json, traces.sha256). With --out, everything goes under DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from continuation.stats import summary as cont_summary
from experiments.robustsuite import protect_manifest as PM
from robustsuite import blocks as RB
from robustsuite import diagnostic as RD
from robustsuite import harness as RH
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite import validation as RV
from robustsuite.scenario_env import RSScenarioEnv

REPO = PM.REPO
#: The frozen baseline, the only arm of phase 2.
ARM = "astar_random"
#: Report, results and figures (figures relative to the report).
REPORT_DIR = REPO / "MD_files/robustsuite"
REPORT_NAME = "RS2_BASELINE_DIAGNOSTIC_REPORT.md"
RESULTS = REPO / "results/robustsuite/RS2"
FIG_DIR = "figures/rs2"
#: The frozen parameter files whose hashes go into the provenance.
FROZEN_PARAM_FILES = ["results/week6_continuation/H2/tuned_frozen.json",
                      "results/week6_continuation/R3/random_frozen.json"]
#: A --smoke run: seeds per cell and steps per episode.
SMOKE_SEEDS, SMOKE_STEPS = 1, 40
#: Examples of the most common failure rendered per cell.
EXAMPLES = 2
#: The known F-D numbers on M0: HD0's re-run of the arm on HD_FINAL (MD_files/highdim/
#: HD0_BASELINES_REPORT.md, results/highdim/HD0/analysis.json), successes / episodes.
REF_M0 = {"fixed": (186, 200), "randomized": (175, 200), "pooled": (361, 400)}
#: Rates pooled over cells with equal weights.
POOL_KEYS = ("success_rate", "collision_rate", "dynamic_collision_rate",
             "static_collision_rate", "wall_collision_rate", "timeout_rate", "stuck_rate",
             "mean_spl", "min_clearance_mean", "infeasible_step_rate",
             "infeasible_events_per_episode", "dcp_step_rate", "random_events_per_episode",
             "random_margin_nonneg", "random_margin_median", "step_time_ms_mean_ep",
             "step_time_ms_p95_ep")

#: The likely mechanism of each failure mode. Written after the run, against the evidence
#: columns the report prints next to them (section 2) and the measurements in READING.
MECHANISM = {
    "collision_dynamic": "A pedestrian walks into the robot: pedestrians ignore it, and in a "
                         "passage the robot has no room to step aside. Random ran in the last "
                         "second of 44% of these episodes, and it rarely finds a direction that "
                         "satisfies every constraint (15% of events pooled, section 7). About "
                         "half of these collisions are in trigger_block cells, where the robot "
                         "is waiting in front of the block (section 6).",
    "collision_spawned": "The spawned pedestrian comes out from behind an occluder 2–3 m from "
                         "the trigger and LiDAR sees it late. It is rare: 11 of 500 "
                         "trigger_spawn episodes.",
    "collision_static": "The robot touches a layout rectangle, almost always in trigger_block "
                        "cells and with Random running just before (the 'Random near end' "
                        "column): the robot is being pushed around in a tight spot next to "
                        "the block.",
    "collision_block": "The robot touches the fired block itself, with Random running just "
                       "before (every case).",
    "collision_wall": "The robot touches the outer wall.",
    "timeout_at_block": "A* never replans, so the carrot stays on the far side of the block. "
                        "The CLF pulls the robot into the block and the CBF stops it about "
                        "0.45 m from the block's face (0.15 m clearance). The QP stays "
                        "feasible, with no Random and no DCPError, so the robot stands still "
                        "until the timeout: a CLF–CBF standstill.",
    "timeout_freeze_start": "h < 0 at the start, so the QP raises DCPError, u = 0 is "
                            "executed and Random never runs.",
    "timeout_freeze_midroute": "The same DCPError freeze, entered mid-route: in practice "
                               "within 0.6 m of the outer wall, which LiDAR reads 0.3 m short.",
    "timeout_stuck": "The same CLF–CBF standstill without a block: the robot drifts off its "
                     "route and wedges between obstacles, with the carrot on the far side "
                     "(rooms / static and dense_clutter / static, renders in section 9). The "
                     "failing static layouts fail in both motions.",
    "timeout_slow": "The robot kept moving but did not reach the goal within 500 steps.",
}

#: What the diagnostic shows, written after the run from this report's own tables. Printed as
#: section 0 of the report.
READING = """\
Written after the run, from the tables below. The ranking of candidates is for ticket 10 and is
not decided here.

1. **The triggered block is the dominant failure.** The trigger_block cells succeed 0.04–0.31
   (dynamic cells 0.78–0.99). `timeout_at_block` alone is 321 of 2000 episodes (16%), and every
   family has it. At the end of these episodes the robot stands 0.41–0.48 m (5th–95th
   percentile) from the block's face, and 92% of them have not moved in the last 5 s. The QP is
   feasible:
   there is no Random and no DCPError. A* never replans, so the carrot stays past the block.
   Dense clutter suffers least (0.31), which fits its many short-detour blocks (14 / 50,
   [[RS1_MAP_VALIDATION_REPORT]]).
2. **Waiting at the block also causes collisions.** The trigger_block cells have 88 pedestrian
   collisions against 43 in the dynamic cells on the same draws. Most collisions after firing
   are within 1 m of the block (section 6), a median of 140–240 steps after firing. The robot
   stands where the pedestrians walk past the block zone.
3. **Pedestrian collisions are the second mode** (183 episodes, 9%). They grow with narrowness:
   corridors 0.20 and dense clutter 0.12 in the dynamic cell, against 0.01 in open clutter.
   Random's chosen direction satisfies every constraint in only 15% of events pooled, close to
   the ~12% seen on M0.
4. **The triggered spawn is a minor threat.** The spawned pedestrian caused 11 collisions in 500
   episodes. The trigger_spawn cells succeed 2–8 pp below their dynamic cells. Their failures
   split about evenly: 37 come before firing, where the episode is the dynamic cell's, and 34
   after.
5. **DCPError freezes are almost absent:** 2 of 2000 episodes, both mid-route and both ending
   0.56 m from the outer wall. This corrects the ticket 08 expectation. The env's LiDAR reads
   rectangles at their true distance and only the outer wall 0.3 m short, so h < 0 (DCPError)
   happens within 0.6 m of the outer wall, not of every obstacle as [[RS_DESIGN]] §14.1 item 1
   words it. The 0.6 m start clearance keeps starts out of that band. Random on DCPError steps
   (the phase 3 idea) would therefore have almost nothing to act on.
6. **The same standstill without a block.** The static cells lose only 4 episodes (`timeout_stuck`,
   two layouts, each in both motions). The robot is wedged between obstacles off its route.
7. **The M0 anchor matches the known F-D numbers** (0.87 against 0.9025, p 0.36; section 8).
8. **Real time:** p95 step time is about 40–47 ms in every cell, well inside the 100 ms limit
   that candidates must meet.
"""


# --------------------------------------------------------------------------- running
def cell_name(cell):
    return f"{cell.family}_{cell.obstacles}"


def parse_cell(text):
    family, obstacles = text.split("/")
    cell = RS.Cell(family, obstacles)
    RH.check_cell(cell)
    return cell


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def provenance(seeds, max_steps):
    """Code identity for the run: the commit, uncommitted changes under the code paths, the
    frozen manifest and parameter hashes, the generator version and the seed block."""
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
            "generator_version": SC.GENERATOR_VERSION, "seed_block": "RS_DIAG",
            "seeds": seeds, "max_steps": max_steps, "arm": ARM,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cvxpy": cvxpy.__version__}


def _paths(res_dir, cell):
    return (res_dir / f"{cell_name(cell)}.jsonl",
            res_dir / f"{cell_name(cell)}.traces.jsonl.gz")


def run(cells, seeds, res_dir, *, workers, max_steps, tag, report_only=False):
    """{cell: ({motion: [light records]}, {(motion, seed): trace row})}."""
    out = {}
    for cell in cells:
        ck, tr = _paths(res_dir, cell)
        if not report_only:
            print(f"[{cell.family} / {cell.obstacles}]", flush=True)
            RB.run_block(ARM, cell, seeds, workers=workers, checkpoint=ck, traces=tr, tag=tag,
                         max_steps=max_steps, progress=25)
        out[cell] = RB.load_block(ARM, cell, seeds, checkpoint=ck, traces=tr, tag=tag,
                                  max_steps=max_steps)
    return out


# --------------------------------------------------------------------------- analysis
def _flat(recs):
    return [r for m in RS.MOTIONS for r in recs[m]]


def _views(recs, rows):
    return [RD.episode_view(r, rows[(m, r["seed"])]) for m in RS.MOTIONS for r in recs[m]]


def _group(recs, views):
    """The diagnostic summary merged with continuation.stats' metrics for one group."""
    s = RD.cell_summary(recs, views)
    c = cont_summary(recs, "")
    margins = [e["m"] for r in recs for e in r.get("events", [])]
    s.update({k: c[k] for k in (
        "dynamic_collision_rate", "static_collision_rate", "wall_collision_rate", "mean_spl",
        "min_clearance_mean", "min_clearance_worst", "min_clearance_p05",
        "infeasible_step_rate", "infeasible_events_per_episode", "n_solver_fail",
        "recovery_success_rate", "step_time_ms_mean_ep", "step_time_ms_p95_ep",
        "step_time_ms_worst") if k in c})
    s.update({"random_events": len(margins),
              "random_events_per_episode": len(margins) / max(len(recs), 1),
              "random_margin_nonneg": (float(np.mean([m >= 0 for m in margins]))
                                       if margins else float("nan")),
              "random_margin_median": (float(np.median(margins)) if margins
                                       else float("nan"))})
    return s


def analyse_cell(cell, recs, rows):
    flat, views = _flat(recs), _views(recs, rows)
    motions = [m for m in RS.MOTIONS for _ in recs[m]]
    out = {"all": _group(flat, views),
           "by_motion": {m: _group(recs[m], [v for mm, v in zip(motions, views) if mm == m])
                         for m in RS.MOTIONS}}
    if cell.obstacles in RD.TRIGGERED:
        fired = [(r, v) for r, v in zip(flat, views) if v["event"]["fired"]]
        out["fired"] = _group([r for r, _ in fired], [v for _, v in fired]) if fired else None
        out["event"] = RD.event_summary(flat, views)
    return out, views


def mode_evidence(views_by_cell, n_episodes):
    """Each mode's count, share and evidence over the 20 cells, ranked by frequency."""
    rows = []
    allv = [(c, v) for c, vs in views_by_cell.items() for v in vs]
    for mode in RD.MODES:
        vs = [(c, v) for c, v in allv if v["mode"] == mode]
        if not vs:
            continue
        n = len(vs)
        fam = Counter(c.family for c, _ in vs)
        cond = Counter(c.obstacles for c, _ in vs)
        rows.append({"mode": mode, "count": n, "share_episodes": n / n_episodes,
                     "terminal_freeze": sum(v["terminal_freeze"] for _, v in vs) / n,
                     "random_near_end": sum(v["random_near_end"] for _, v in vs) / n,
                     "left_start": sum(v["left_start"] for _, v in vs) / n,
                     "still_at_end": sum(v["still_at_end"] for _, v in vs) / n,
                     "end_dist_block": RD.order_stats(
                         v["event"]["dist_block"] for _, v in vs
                         if v["event"] and v["event"]["fired"]),
                     "by_family": {f: fam[f] for f in RS.FAMILIES},
                     "by_condition": {o: cond[o] for o in RS.OBSTACLE_CONDITIONS}})
    rows.sort(key=lambda r: (-r["count"], RD.MODES.index(r["mode"])))
    return rows


def analyse(res):
    cells = {c: analyse_cell(c, *res[c]) for c in res}
    grid = [c for c in RS.CELLS if c in cells]
    out = {"cells": {cell_name(c): cells[c][0] for c in cells}}
    if grid:
        def pool(group):
            summ = [cells[c][0]["all"] for c in group]
            return {**RD.pooled(summ, POOL_KEYS),
                    "modes": dict(sum((Counter(s["modes"]) for s in summ), Counter()))}
        out["pooled"] = pool(grid)
        out["families"] = {f: pool([c for c in grid if c.family == f])
                           for f in RS.FAMILIES if any(c.family == f for c in grid)}
        out["modes"] = mode_evidence({c: cells[c][1] for c in grid}, out["pooled"]["episodes"])
    if RS.ANCHOR in cells:
        recs = res[RS.ANCHOR][0]
        groups = {**recs, "pooled": _flat(recs)}
        out["anchor_check"] = {m: RD.anchor_check(sum(r["success"] for r in g), len(g),
                                                  *REF_M0[m])
                               for m, g in groups.items()}
    return out, {c: cells[c][1] for c in cells}


# --------------------------------------------------------------------------- renders
class _Recording(RSScenarioEnv):
    """The scenario env plus the dynamic obstacles' positions after every step (render only)."""

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.track = [self.obstacle_positions.copy()]
        return out

    def step(self, action):
        out = super().step(action)
        self.track.append(self.obstacle_positions.copy())
        return out


def _replay(job):
    """Re-run one episode exactly, recording the pedestrians (checked against the stored run)."""
    cell, motion, seed, max_steps, trajectory = job
    env = _Recording(motion)
    if max_steps is not None:
        env.MAX_STEPS = max_steps
    try:
        rec = RH.run_episode(ARM, cell, motion, seed, env=env)
    finally:
        env.close()
    if rec["trajectory"] != trajectory:
        raise RuntimeError(f"replay of {cell} {motion} {seed} differs from the stored episode")
    return [np.asarray(p, float).tolist() for p in env.track]


def pick_examples(res_cell, views, mode):
    """The first EXAMPLES episodes (fixed, then randomized, by seed) that failed in `mode`."""
    recs, rows = res_cell
    flat = _flat(recs)
    motions = [m for m in RS.MOTIONS for _ in recs[m]]
    got = [(m, r, rows[(m, r["seed"])]) for m, r, v in zip(motions, flat, views, strict=True)
           if v["mode"] == mode]
    return got[:EXAMPLES]


def _draw_episode(ax, cell, rec, row, track, mode):
    from matplotlib.patches import Circle
    traj = np.asarray(row["trajectory"])
    if cell == RS.ANCHOR:
        env = RSScenarioEnv(rec["motion"])
        RV.draw_anchor(ax, env.m0_static, rec["start"], rec["goal"])
        env.close()
    else:
        RV.draw_spec(ax, SC.generate(cell, rec["motion"], rec["seed"]), cell.obstacles)
    ax.plot(*traj.T, lw=1.6, color="black")
    ax.plot(*traj[-1], "x", ms=9, mew=2.5, color="tab:red")
    for ev in rec["scenario_events"]:
        if ev["event"] == "trigger_fired":
            ax.plot(*ev["position"], "D", ms=5, color="tab:orange", mec="black")
    for t in range(max(0, len(track) - 30), len(track)):          # the last 3 s, fading in
        for p in track[t]:
            ax.plot(*p, ".", ms=2, color="tab:brown", alpha=0.25)
    for p in track[-1]:
        ax.add_patch(Circle(p, 0.3, fill=False, lw=1.2, color="tab:brown"))
    ax.set_title(f"{mode}\n{rec['motion']} seed {rec['seed']}, step {rec['steps']}",
                 fontsize=8)


def render(res, analysis, views, out_dir, max_steps, workers):
    """{cell: (figure path, mode)} for every cell with a failure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = {}
    for cell in res:
        mode = analysis["cells"][cell_name(cell)]["all"]["most_common_mode"]
        if mode is not None:
            picks[cell] = (mode, pick_examples(res[cell], views[cell], mode))
    jobs = [(c, m, r["seed"], max_steps, row["trajectory"])
            for c, (_, ex) in picks.items() for m, r, row in ex]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            tracks = list(ex.map(_replay, jobs))
    else:
        tracks = [_replay(j) for j in jobs]
    it = iter(tracks)
    figs = {}
    (out_dir / FIG_DIR).mkdir(parents=True, exist_ok=True)
    for cell, (mode, ex) in picks.items():
        fig, axes = plt.subplots(1, len(ex), figsize=(3.4 * len(ex), 3.6), squeeze=False)
        for ax, (_, rec, row) in zip(axes[0], ex):
            _draw_episode(ax, cell, rec, row, next(it), mode)
        path = f"{FIG_DIR}/fail_{cell_name(cell)}.png"
        fig.tight_layout()
        fig.savefig(out_dir / path, dpi=80)
        plt.close(fig)
        figs[cell] = (path, mode)
    return figs


# --------------------------------------------------------------------------- report
def _f(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]) + "\n"


def _quartiles(s, nd=1):
    return "—" if not s else (f"{s['median']:.{nd}f} [{s['q1']:.{nd}f}, {s['q3']:.{nd}f}] "
                              f"(n {s['n']})")


def _modes(modes):
    return ", ".join(f"{m.replace('collision_', 'c:').replace('timeout_', 't:')} {n}"
                     for m, n in modes.items()) or "—"


#: Column heads of the outcome and safety-layer blocks of the report tables.
OUTCOME_HEAD = ["success", "collision", "dyn / static / wall", "timeout", "stuck", "SPL",
                "min clear. (mean)"]


def _outcomes(s):
    return [_f(s["success_rate"]), _f(s["collision_rate"]),
            f"{_f(s['dynamic_collision_rate'])} / {_f(s['static_collision_rate'])} / "
            f"{_f(s['wall_collision_rate'])}",
            _f(s["timeout_rate"]), _f(s["stuck_rate"]), _f(s["mean_spl"]),
            _f(s["min_clearance_mean"])]


SAFETY_HEAD = ["infeasible-step rate", "Random events / ep", "Random margin ≥ 0",
               "Random margin (median)", "DCPError step rate", "step ms (mean)",
               "step ms p95 (ep mean)"]


def _safety(s):
    return [_f(s["infeasible_step_rate"], 4), _f(s["random_events_per_episode"], 2),
            _f(s["random_margin_nonneg"], 2), _f(s["random_margin_median"], 3),
            _f(s["dcp_step_rate"], 4), _f(s["step_time_ms_mean_ep"], 1),
            _f(s["step_time_ms_p95_ep"], 1)]


def write_report(analysis, figs, prov, trace_sha, path, n_seeds):
    lines = []
    add = lines.append
    a = analysis
    cells = [c for c in RS.CELLS if cell_name(c) in a["cells"]]
    code_clean = not prov["code_changes"]
    man_ok = not prov["manifest"]["modified"] and not prov["manifest"]["missing"]
    add("# RS Experiment 1, phase 2: baseline diagnostic\n")
    add("Pre-registration: [[RS_DESIGN]] §6 (phase 2) and §10. Ticket 09. The suite is the one "
        "approved in [[RS1_MAP_VALIDATION_REPORT]]. Generated by "
        "`experiments/robustsuite/rs2_diagnostic.py`.\n")
    add(f"- Commit `{prov['commit'][:7]}`; code tree clean: {'yes' if code_clean else 'NO'}"
        + (f"; report regenerated at `{prov['report_commit'][:7]}`"
           if prov.get("report_commit", prov["commit"]) != prov["commit"] else ""))
    add("- Every file in the results folder belongs to this run: the per-cell checkpoints are "
        "fingerprinted with the commit, and `provenance.json` holds the manifest and "
        "parameter hashes")
    add(f"- Frozen-file manifest verified: {'yes' if man_ok else 'NO'} "
        f"(`{prov['manifest']['sha256'][:12]}`)")
    add("- Frozen parameters: " + ", ".join(f"`{Path(f).name}` `{h[:12]}`"
                                            for f, h in prov["frozen_params_sha256"].items()))
    add(f"- Generator `{prov['generator_version']}`, seed block {prov['seed_block']} "
        f"({n_seeds} seeds), both motion conditions, {prov['max_steps'] or 500} steps")
    add(f"- Arm `{ARM}` (frozen A* + Random-CLF-DR-CBF), frozen SCS controller in every episode")
    add(f"- Trace sidecars: SHA-256 in `traces.sha256` (combined `{trace_sha[:12]}`)\n")
    add("## 0. Reading\n")
    add(READING)

    if "pooled" in a:
        p = a["pooled"]
        add(f"## 1. Pooled result ({len(cells)} cells, equal weights, both motions)\n")
        add(_table(["episodes"] + OUTCOME_HEAD + ["DCPError step rate"],
                   [[p["episodes"]] + _outcomes(p) + [_f(p["dcp_step_rate"], 4)]]))

        add("## 2. Failure modes, ranked by frequency\n")
        add("Each failed episode has exactly one mode, the first that matches in the order of "
            "`robustsuite.diagnostic.MODES`. The evidence columns give the share of the mode's "
            f"episodes that end in a DCPError freeze (≥ {RD.FREEZE_STEPS} steps to the end), "
            f"that ran Random in their last {RD.NEAR_END} steps, and whose robot ever moved "
            f"more than {RD.LEFT_START} m from its start, and that moved less than "
            f"{RD.STILL_DIST} m in their last {RD.STILL_STEPS} steps. For episodes whose block "
            "fired, the robot's final distance from the block's face is given too.\n")
        rows = []
        for i, m in enumerate(a["modes"], 1):
            rows.append([i, f"`{m['mode']}`", m["count"], _f(m["share_episodes"]),
                         _f(m["terminal_freeze"], 2), _f(m["random_near_end"], 2),
                         _f(m["left_start"], 2), _f(m["still_at_end"], 2),
                         _quartiles(m["end_dist_block"], 2),
                         " / ".join(str(m["by_family"][f]) for f in RS.FAMILIES),
                         " / ".join(str(m["by_condition"][o]) for o in RS.OBSTACLE_CONDITIONS)])
        add(_table(["rank", "mode", "episodes", "share of all", "DCP freeze at end",
                    "Random near end", "left start", "still at end",
                    "m from block face at end (fired block)",
                    "by family (" + " / ".join(RS.FAMILIES) + ")",
                    "by condition (" + " / ".join(RS.OBSTACLE_CONDITIONS) + ")"], rows))
        add("**Likely mechanisms:**\n")
        for m in a["modes"]:
            add(f"- `{m['mode']}`: {MECHANISM[m['mode']]}")
        add("")

        add("## 3. DCPError freezes per cell\n")
        add(f"A freeze is a run of ≥ {RD.FREEZE_STEPS} consecutive DCPError steps (h < 0: u = 0, "
            "Random not called). Counts are episodes out of each cell's "
            f"{2 * n_seeds}.\n")
        rows = []
        for c in cells:
            s = a["cells"][cell_name(c)]["all"]
            rows.append([f"{c.family} / {c.obstacles}", s["dcp_episodes"],
                         s["dcp_at_start_episodes"], s["freeze_start_episodes"], s["freeze_midroute_episodes"],
                         s["terminal_freeze_episodes"], _f(s["dcp_step_rate"], 4)])
        add(_table(["cell", "any DCPError", "DCPError on step 0", "freeze at start", "freeze mid-route",
                    "freeze to the end", "DCPError step rate"], rows))

        add("## 4. Per family (its cells weighted equally)\n")
        add(_table(["family"] + OUTCOME_HEAD + ["modes"],
                   [[f] + _outcomes(s) + [_modes(s["modes"])]
                    for f, s in a["families"].items()]))

        add("## 5. Per cell\n")
        add(f"Both motions pooled ({2 * n_seeds} episodes per cell). Mode abbreviations: "
            "`c:` collision, `t:` timeout.\n")
        add(_table(["cell"] + OUTCOME_HEAD + ["modes"],
                   [[f"{c.family} / {c.obstacles}"] + _outcomes(a["cells"][cell_name(c)]["all"])
                    + [_modes(a["cells"][cell_name(c)]["all"]["modes"])] for c in cells]))
        add("### By motion condition\n")
        rows = []
        for c in cells:
            bm = a["cells"][cell_name(c)]["by_motion"]
            rows.append([f"{c.family} / {c.obstacles}"]
                        + [f"{_f(bm[m]['success_rate'], 2)} / {_f(bm[m]['collision_rate'], 2)}"
                           f" / {_f(bm[m]['timeout_rate'], 2)}" for m in RS.MOTIONS])
        add(_table(["cell"] + [f"{m}: success / collision / timeout" for m in RS.MOTIONS],
                   rows))

        add("## 6. Triggered cells: overall and conditional on the trigger having fired\n")
        rows = []
        for c in cells:
            if c.obstacles not in RD.TRIGGERED:
                continue
            x = a["cells"][cell_name(c)]
            for label, s in (("all", x["all"]), ("fired", x["fired"])):
                rows.append([f"{c.family} / {c.obstacles}", label, s["episodes"] if s else 0]
                            + (_outcomes(s) + _safety(s) if s
                               else ["—"] * (len(OUTCOME_HEAD) + len(SAFETY_HEAD))))
        add(_table(["cell", "subset", "n"] + OUTCOME_HEAD + SAFETY_HEAD, rows))
        add("### Outcome timing and location relative to the event\n")
        add("Steps from the firing step to the end of the failed episode, and, for collisions "
            "after firing, the robot's distance from the trigger centre and from the block's "
            "face: median [q1, q3] (n).\n")
        rows = []
        for c in cells:
            if c.obstacles not in RD.TRIGGERED:
                continue
            e = a["cells"][cell_name(c)]["event"]
            rows.append([f"{c.family} / {c.obstacles}", _f(e["fired_rate"], 2),
                         e["failures_before_fire"], e["failures_after_fire"],
                         _quartiles(e["steps_after_fire"], 0),
                         _quartiles(e["collision_steps_after_fire"], 0),
                         _quartiles(e["collision_dist_trigger"], 2),
                         _quartiles(e["collision_dist_block"], 2)
                         if c.obstacles == "trigger_block"
                         else _quartiles(e["collision_dist_spawn_start"], 2),
                         e["spawned_hits"] if c.obstacles == "trigger_spawn"
                         else f"{e['collisions_near_block']} within {RD.BLOCK_NEAR:g} m"])
        add(_table(["cell", "fired rate", "failures before firing", "failures after firing",
                    "failure: steps after firing", "collision: steps after firing",
                    "collision: m from trigger",
                    "collision: m from block face / from spawn start",
                    "hit by the spawn / collisions near the block"], rows))

        add("## 7. Safety layer: infeasibility, Random recovery, step time\n")
        add("Random's margin is the minimum CBC of the direction it selected; ≥ 0 means it "
            "found a direction that satisfies every constraint.\n")
        rows = [["**pooled**"] + _safety(a["pooled"])]
        rows += [[f"**{f}**"] + _safety(s) for f, s in a["families"].items()]
        rows += [[f"{c.family} / {c.obstacles}"] + _safety(a["cells"][cell_name(c)]["all"])
                 for c in cells]
        add("Pooled and family rows weight their cells equally; the margin median is the "
            "mean of the cells' medians there.\n")
        add(_table(["group"] + SAFETY_HEAD, rows))

    if "anchor_check" in a:
        add("## 8. M0 anchor\n")
        s = a["cells"][cell_name(RS.ANCHOR)]["all"]
        add(_table(["episodes"] + OUTCOME_HEAD + ["modes"],
                   [[s["episodes"]] + _outcomes(s) + [_modes(s["modes"])]]))
        add("Against the known F-D numbers on M0 (HD0's re-run on HD_FINAL, "
            "[[HD0_BASELINES_REPORT]]), with a two-sided Fisher exact test on success (the "
            "seeds differ, so the test is unpaired). Consistent means p ≥ 0.05.\n")
        add("The anchor's 95% interval is exact (Clopper–Pearson). With 50–100 anchor episodes "
            "the test detects only large departures, about 10 pp or more.\n")
        add(_table(["motion", "anchor success", "95% CI", "F-D success (HD0)", "p",
                    "consistent"],
                   [[m, f"{x['k']}/{x['n']} = {x['rate']:.3f}",
                     f"[{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]",
                     f"{x['k_ref']}/{x['n_ref']} = {x['rate_ref']:.4f}", f"{x['p']:.3g}",
                     "yes" if x["consistent"] else "**NO**"]
                    for m, x in a["anchor_check"].items()]))

    add("## 9. The most common failure in each cell\n")
    add(f"Up to {EXAMPLES} examples of each cell's most common mode: the first in seed order, "
        "fixed motion first. Each is a replay of the stored episode, checked equal to its "
        "stored trajectory.\n")
    add("Key:")
    add("- **Black:** the robot's trajectory. The red × is where it ended.")
    add("- **Brown:** the pedestrians. Circles show their final positions (radius 0.3 m), "
        "and dots their last 3 s.")
    add("- **Orange diamond:** where the trigger fired.")
    add("- The rest is as in [[RS1_MAP_VALIDATION_REPORT]]: grey layout, blue route, green "
        "start and goal, red block, orange trigger disc, purple spawn.\n")
    for c in cells + ([RS.ANCHOR] if RS.ANCHOR in figs else []):
        if c in figs:
            fig, mode = figs[c]
            name = "M0 anchor" if c == RS.ANCHOR else f"{c.family} / {c.obstacles}"
            add(f"### {name}: `{mode}`\n")
            add(f"![{name}]({fig})\n")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- main
def _trace_digest(res_dir, cells):
    shas = {}
    for c in cells:
        _, tr = _paths(res_dir, c)
        shas[tr.name] = hashlib.sha256(tr.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()
    return shas, combined


def _nan_to_null(v):
    """Strict JSON: NaN (no value, e.g. no Random event) becomes null."""
    if isinstance(v, dict):
        return {k: _nan_to_null(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nan_to_null(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if v != v else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_SEEDS} seed, {SMOKE_STEPS}-step episodes")
    ap.add_argument("--out", type=Path, help="write results, report and figures under DIR")
    ap.add_argument("--cells", nargs="+", type=parse_cell,
                    help="family/obstacles (m0/anchor for the anchor); default: all + anchor")
    ap.add_argument("--report-only", action="store_true",
                    help="re-analyse finished blocks; run no new episodes (the renders "
                         "still replay their examples)")
    args = ap.parse_args(argv)
    seeds = RS.seed_block("RS_DIAG", SMOKE_SEEDS if args.smoke else None)
    max_steps = SMOKE_STEPS if args.smoke else None
    cells = args.cells or [*RS.CELLS, RS.ANCHOR]
    res_dir = args.out or RESULTS
    report_dir = args.out or REPORT_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prov_path = res_dir / "provenance.json"
    if args.report_only:
        prov = json.loads(prov_path.read_text())
        prov["report_commit"] = _git("rev-parse", "HEAD")
    else:
        prov = provenance(seeds, max_steps)
        if not args.smoke and (prov["code_changes"] or prov["manifest"]["modified"]
                               or prov["manifest"]["missing"]):
            raise SystemExit("commit the code (and keep the frozen manifest intact) before "
                             "the diagnostic run: the record must name the code it ran")
        prov_path.write_text(json.dumps(prov, indent=1))
    res = run(cells, seeds, res_dir, workers=args.workers, max_steps=max_steps,
              tag=prov["commit"], report_only=args.report_only)
    analysis, views = analyse(res)
    figs = render(res, analysis, views, report_dir, max_steps, args.workers)
    shas, combined = _trace_digest(res_dir, cells)
    (res_dir / "traces.sha256").write_text(
        "".join(f"{h}  {n}\n" for n, h in sorted(shas.items())))
    analysis.update({"provenance": prov, "trace_sha256": combined,
                     "figures": [p for p, _ in figs.values()]})
    (res_dir / "analysis.json").write_text(json.dumps(_nan_to_null(analysis), indent=1,
                                                      default=float, allow_nan=False))
    report = report_dir / REPORT_NAME
    write_report(analysis, figs, prov, combined, report, len(seeds))
    n = sum(len(_flat(res[c][0])) for c in res)
    print(f"{n} episodes\nreport: {report}\nanalysis: {res_dir / 'analysis.json'}")
    return {"episodes": n, "report": report}


if __name__ == "__main__":
    main()
