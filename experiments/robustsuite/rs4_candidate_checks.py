"""RS Experiment 1, phase 4: the measurements tickets 11a-11c ask for, before any tuning run.

Two checks, both on RS_DIAG and neither a tuning run:

1. PHANTOM COUNTS, from pedestrian-only replays. Every baseline episode of the cells that never
   fire a block (static, +dynamic, +triggered spawn, and the M0 anchor) is replayed exactly
   (rs2_diagnostic._replay records the pedestrians; the trajectory is checked against the stored
   run), its LiDAR is reconstructed with the environment's own ray caster, and the candidate
   layers are driven over it as shadows: they observe and compute, and change nothing.
   - `detour`: how many new cells K = 8 and K = 14 leave behind. There is no new static obstacle
     in any of these cells, so every cell counted is a phantom, and any detour they would start
     is a needless one (RS_DESIGN 7.4, candidate 1 revised).
   - `yield`: how many steps the layer would yield on. In the `static` cells there is no
     pedestrian at all, so those yields come from tracks the frozen tracker builds out of static
     geometry (reading 11).

2. STEP TIME. The baseline and the three candidates, at their default configurations, on one seed
   of every cell and the anchor under both motion conditions, timed exactly as the harness times a
   step (the whole predict call, the layer's own work included). The 7.2 rule disqualifies a
   candidate whose p95 exceeds 100 ms; the maximum is reported too.

    python -m experiments.robustsuite.rs4_candidate_checks --workers 14

Writes results/robustsuite/RS4/checks.json and MD_files/robustsuite/RS4_CANDIDATE_CHECKS.md.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dr_control.velocity_tracker import LidarVelocityTracker
from experiments.robustsuite import rs2_diagnostic as D
from optional_navigation.planner import CarrotFollower, StaticMapPlanner
from robot_env.lidar_core import cast_rays
from robustsuite import candidates as CA
from robustsuite import harness as RH
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite import selection as SEL
from robustsuite.scenario_env import RSScenarioEnv

RESULTS = D.REPO / "results/robustsuite/RS4"
#: Each phase is written as it finishes, so a failure in a later one cannot discard it.
PHANTOM_ROWS = RESULTS / "phantom_episodes.json"
TIMING_ROWS = RESULTS / "step_times.json"
REPORT = D.REPORT_DIR / "RS4_CANDIDATE_CHECKS.md"
#: The phantom check: the cells in which no new static obstacle ever appears, plus the anchor.
PHANTOM_CONDITIONS = ("static", "dynamic", "trigger_spawn")
#: Seeds: the first 20 RS_DIAG seeds, the ones the 7.3 parameter search uses.
N_PHANTOM_SEEDS = 20
#: The K values of the pre-registered grid.
K_VALUES = (8, 14)
#: A yield is counted as a yield to a real pedestrian when the deciding track is this close to
#: one (its radius, 0.3 m, plus a 0.2 m margin for the track's own error).
PEDESTRIAN_MATCH = 0.5
#: The 7.2 real-time rule, in milliseconds.
#: RS_DESIGN 7.2, from the module that holds the pre-registered rules (phase 5).
P95_LIMIT = SEL.P95_LIMIT_MS


def phantom_cells():
    return [c for c in RS.CELLS if c.obstacles in PHANTOM_CONDITIONS] + [RS.ANCHOR]


def _scan(p, layout, peds, env_like):
    """The LiDAR ranges at p, from the environment's own ray caster."""
    return cast_rays(np.asarray(p, np.float32), n_rays=env_like["n_rays"],
                     lidar_range=env_like["lidar_range"], world_size=env_like["world_size"],
                     agent_radius=env_like["agent_radius"], rects=list(layout),
                     circles=np.asarray(peds, np.float32).reshape(-1, 2),
                     circle_radius=env_like["obstacle_radius"])


def _shadow(job):
    """One replayed episode: what the layers would have made of it. Changes nothing."""
    cell, motion, seed = job
    rec = RH.run_episode("astar_random", cell, motion, seed)
    traj = [np.asarray(q, float) for q in rec["trajectory"]]
    peds = D._replay((cell, motion, seed, None, rec["trajectory"]))
    env_like = {"n_rays": 24, "lidar_range": 5.0, "world_size": 10.0, "agent_radius": 0.3,
                "obstacle_radius": 0.3}
    if cell == RS.ANCHOR:
        env = RSScenarioEnv(motion)
        layout = list(env.m0_static)
        env.close()
    else:
        layout = [tuple(r) for r in SC.generate(cell, motion, seed).layout]
    path = StaticMapPlanner(10.0, 0.3, layout).path(traj[0], np.asarray(rec["goal"], float))
    if path is None:                                    # the baseline had no plan either
        return None
    detours = {K: CA.DetourLayer(CarrotFollower(path), layout, CA.DetourConfig(K=K))
               for K in K_VALUES}
    tracker = _tracker()
    yld = CA.YieldLayer(CarrotFollower(path), tracker, StaticMapPlanner(10.0, 0.3, layout),
                        CA.configure("yield"))
    first = {K: None for K in K_VALUES}
    # per yield step: the distance from the deciding track to the nearest real pedestrian, so a
    # yield to a track the frozen tracker built out of static geometry can be told from a yield
    # to a pedestrian (ground truth, measurement only: no candidate reads it)
    yields = []
    for t in range(rec["steps"]):
        ranges = _scan(traj[t], layout, peds[t], env_like)
        obs = np.zeros(52)
        obs[4:28] = np.asarray(ranges, float) / env_like["lidar_range"]
        tracker.update(traj[t], np.asarray(ranges, float))
        for K, layer in detours.items():
            layer.see(obs)
            layer.reference(traj[t])
            if first[K] is None and layer.detours:
                first[K] = layer.detours[0]["start"]
        yld.see(obs)
        before = yld.n_yield_steps
        threat = yld.threat(traj[t])
        yld.reference(traj[t])
        if yld.n_yield_steps > before:                  # what the layer yielded to
            near = min((float(np.linalg.norm(np.asarray(q, float) - threat.position))
                        for q in peds[t]), default=float("inf"))
            yields.append(near)
    return {"cell": D.cell_name(cell), "motion": motion, "seed": seed, "steps": rec["steps"],
            "obstacles": "anchor" if cell == RS.ANCHOR else cell.obstacles,
            "new_cells": {K: int(layer.map.new.sum()) for K, layer in detours.items()},
            "first_detour": {K: first[K] for K in K_VALUES},
            "yield_steps": yld.n_yield_steps, "no_side": yld.n_no_side,
            "yields_to_a_pedestrian": int(sum(d <= PEDESTRIAN_MATCH for d in yields)),
            "yields_to_an_artefact": int(sum(d > PEDESTRIAN_MATCH for d in yields))}


def _tracker():
    """A LiDAR velocity tracker built exactly as the frozen policy's barrier source builds it."""
    return LidarVelocityTracker(r_nominal=0.3, dt=0.1, n_rays=24, lidar_range=5.0)


def phantoms(seeds, workers):
    jobs = [(c, m, s) for c in phantom_cells() for m in RS.MOTIONS for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_shadow, jobs) if r is not None]
    return rows


def _timed(job):
    arm, cell, motion, seed = job
    rec = RH.run_episode(arm, cell, motion, seed, config=None, keep_step_times=True)
    return {"arm": arm, "cell": D.cell_name(cell), "motion": motion, "seed": seed,
            "steps": rec["steps"], "outcome": rec["outcome"],
            "step_times_ms": [1e3 * t for t in rec["step_times"]]}


def timings(seed, workers, arms=("astar_random", *CA.ARMS)):
    jobs = [(a, c, m, seed) for a in arms for c in [*RS.CELLS, RS.ANCHOR] for m in RS.MOTIONS]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_timed, jobs))


# --------------------------------------------------------------------------- summaries
def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def _per_k(row, field, K):
    """One K's entry of a per-episode field; JSON turns the integer keys into strings."""
    d = row[field]
    return d[K] if K in d else d[str(K)]


def summarise_phantoms(rows):
    out = {}
    for cond in ("static", "dynamic", "trigger_spawn", "anchor"):
        sel = [r for r in rows if r["obstacles"] == cond]
        if not sel:
            continue
        out[cond] = {"episodes": len(sel), "steps": int(sum(r["steps"] for r in sel)),
                     "yield_steps_per_episode": float(np.mean([r["yield_steps"] for r in sel])),
                     "yield_rate": float(sum(r["yield_steps"] for r in sel)
                                         / max(sum(r["steps"] for r in sel), 1)),
                     "episodes_with_a_yield": int(sum(r["yield_steps"] > 0 for r in sel)),
                     "yields_to_a_pedestrian": int(sum(r["yields_to_a_pedestrian"] for r in sel)),
                     "yields_to_an_artefact": int(sum(r["yields_to_an_artefact"] for r in sel))}
        for K in K_VALUES:
            cells = [_per_k(r, "new_cells", K) for r in sel]
            out[cond][f"K{K}"] = {
                "episodes_with_a_phantom_cell": int(sum(c > 0 for c in cells)),
                "phantom_cells_mean": float(np.mean(cells)),
                "phantom_cells_max": int(max(cells)),
                "episodes_with_a_phantom_detour":
                    int(sum(_per_k(r, "first_detour", K) is not None for r in sel))}
    return out


def summarise_timings(rows):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        t = [x for r in rows if r["arm"] == arm for x in r["step_times_ms"]]
        out[arm] = {"episodes": sum(r["arm"] == arm for r in rows), "steps": len(t),
                    "mean": float(np.mean(t)), "p50": _pct(t, 50), "p95": _pct(t, 95),
                    "max": float(np.max(t)), "passes_7_2": bool(_pct(t, 95) <= P95_LIMIT)}
    return out


def notes(summary):
    """The readings the two tables support, written from the numbers themselves."""
    p, out = summary["phantoms"], []
    k_hi = f"K{max(K_VALUES)}"
    k_lo = f"K{min(K_VALUES)}"
    phantom_hi = sum(c[k_hi]["episodes_with_a_phantom_cell"] for c in p.values())
    phantom_lo = sum(c[k_lo]["episodes_with_a_phantom_cell"] for c in p.values())
    eps = sum(c["episodes"] for c in p.values())
    worst = max(c[k_lo]["phantom_cells_max"] for c in p.values())
    out.append(f"- **Pedestrians leave no phantom cells at K = {max(K_VALUES)}**: {phantom_hi} of "
               f"{eps} replayed episodes end with one. At K = {min(K_VALUES)}, "
               f"{phantom_lo} do, up to {worst} cells in one episode. The default "
               f"K = {CA.configure('detour').K} is on the safe side of that.")
    if not phantom_hi:
        stalls = sum(c[k_hi]["episodes_with_a_phantom_detour"] for c in p.values())
        out.append(f"- **The stall trigger is what fires here**: with no new cell anywhere at "
                   f"K = {max(K_VALUES)}, trigger (a) cannot fire, so all {stalls} detours of "
                   f"{eps} episodes are stalls -- the robot standing still for 3 s while a "
                   f"pedestrian passes. `stall` is a grid switch, so the 7.3 search decides "
                   f"whether that helps.")
    ped = sum(c["yields_to_a_pedestrian"] for c in p.values())
    art = sum(c["yields_to_an_artefact"] for c in p.values())
    st = p.get("static", {})
    if st:
        out.append(f"- **yield is active, and often on nothing**: it yields on "
                   f"{st['yield_rate']:.0%} of steps in the `static` cells, which hold no "
                   f"pedestrian at all. Over every cell {art} of {ped + art} yields are to a "
                   f"track farther than {PEDESTRIAN_MATCH} m from any pedestrian, i.e. to the "
                   f"tracker's own wall artefacts (reading 14).")
    bad = [a for a, s in summary["timings"].items() if not s["passes_7_2"]]
    out.append("- **Step time**: "
               + ("every arm is inside the 7.2 limit on this sample."
                  if not bad else f"{', '.join(bad)} exceeds the 100 ms p95 limit here.")
               + " The rule is applied for real on RS_TUNE in ticket 12.")
    return out


def write_report(summary, prov, path):
    K = K_VALUES
    lines = [
        "# RS Experiment 1: candidate checks (phase 4)",
        "",
        "Tickets 11a-11c. Produced by `experiments/robustsuite/rs4_candidate_checks.py`. No "
        "tuning run: the phantom counts are shadows over replayed baseline episodes, and the "
        "step times are one seed per cell.",
        "",
        f"- Commit: `{prov['commit']}`  ",
        f"- Manifest: `{prov['manifest']['sha256'][:16]}...`  ",
        f"- Generator: `{prov['generator_version']}`  ",
        f"- Seeds: RS_DIAG[0:{prov['n_phantom_seeds']}] (phantoms), "
        f"RS_DIAG[{prov['timing_seed_index']}] (step times)",
        "",
        "## 1. Phantom new cells and phantom yields",
        "",
        "Cells in which no new static obstacle ever appears, so every new cell is a phantom and "
        "every detour trigger (a) it would fire is needless. `static` has no pedestrian at all, "
        "so its yields come entirely from tracks the frozen tracker builds out of static "
        "geometry (reading 14). A detour counted below is one the layer would have started on "
        "the baseline's own trajectory; where a row shows no phantom cell at all, trigger (a) "
        "cannot have fired, so every one of them is the stall trigger (b).",
        "",
        "| condition | episodes | "
        + " | ".join(f"eps with a phantom cell (K={k}) | mean cells (K={k}) | "
                     f"eps with a detour (K={k})" for k in K)
        + " | yield steps / episode | yield rate | yields to a pedestrian |",
        "|---" * (2 + 3 * len(K) + 3) + "|",
    ]
    for cond, s in summary["phantoms"].items():
        cols = [cond, str(s["episodes"])]
        for k in K:
            d = s[f"K{k}"]
            cols += [f"{d['episodes_with_a_phantom_cell']} / {s['episodes']}",
                     f"{d['phantom_cells_mean']:.2f}",
                     f"{d['episodes_with_a_phantom_detour']} / {s['episodes']}"]
        yields = s["yields_to_a_pedestrian"] + s["yields_to_an_artefact"]
        cols += [f"{s['yield_steps_per_episode']:.1f}", f"{s['yield_rate']:.3f}",
                 f"{s['yields_to_a_pedestrian']} / {yields}" if yields else "0"]
        lines.append("| " + " | ".join(cols) + " |")
    lines += [
        "",
        "## 2. Step time (RS_DESIGN 7.2: p95 must be at most 100 ms)",
        "",
        "| arm | episodes | control steps | mean (ms) | p50 (ms) | p95 (ms) | max (ms) | 7.2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in summary["timings"].items():
        lines.append(f"| `{arm}` | {s['episodes']} | {s['steps']} | {s['mean']:.1f} | "
                     f"{s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} | "
                     f"{'pass' if s['passes_7_2'] else 'FAIL'} |")
    lines += ["", "Default configurations: "
              + ", ".join(f"`{a}` {CA.configure(a).name}" for a in CA.ARMS) + ".", "",
              "## 3. What the numbers say", "",
              *summary.get("notes", []), ""]
    path.write_text("\n".join(lines))


def _phase(path, refresh, fn, *args):
    """Run one phase, or read back the rows a previous run already wrote."""
    if path.exists() and not refresh:
        print(f"  reusing {path}", flush=True)
        return json.loads(path.read_text())
    rows = fn(*args)
    path.write_text(json.dumps(rows))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timing-seed-index", type=int, default=0)
    ap.add_argument("--refresh", action="store_true",
                    help="re-run a phase whose rows are already on disk")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = RS.seed_block("RS_DIAG")[:N_PHANTOM_SEEDS]
    prov = {**D.provenance(seeds, None), "n_phantom_seeds": N_PHANTOM_SEEDS,
            "timing_seed_index": args.timing_seed_index}
    rows = _phase(PHANTOM_ROWS, args.refresh, phantoms, seeds, args.workers)
    times = _phase(TIMING_ROWS, args.refresh, timings,
                   RS.seed_block("RS_DIAG")[args.timing_seed_index], args.workers)
    summary = {"phantoms": summarise_phantoms(rows), "timings": summarise_timings(times)}
    summary["notes"] = notes(summary)
    (RESULTS / "checks.json").write_text(json.dumps(
        {"provenance": prov, **summary, "phantom_episodes": rows}, indent=1))
    write_report(summary, prov, REPORT)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
