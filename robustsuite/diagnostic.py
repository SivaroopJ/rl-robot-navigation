"""The baseline diagnostic's per-episode reading and cell aggregates (RS_DESIGN 6 phase 2, 10).

Pure: it reads a light episode record plus its trace-sidecar row (`trace`, `trajectory`, `spec`)
and never runs anything. Ground truth (what was hit, where the block is) is used here only to
DESCRIBE a finished episode, never by any policy.

A failed episode gets exactly one failure mode, the first in MODES that matches:

    collision_dynamic        hit a regular pedestrian (or, on the anchor, an M0 obstacle)
    collision_spawned        hit the triggered spawn (the slot after the regular pedestrians)
    collision_static         hit a layout rectangle
    collision_block          hit the fired block (robot within its radius of the block)
    collision_wall           hit the outer wall
    timeout_at_block         trigger_block fired and the robot ends within BLOCK_NEAR of it
    timeout_freeze_start     a terminal DCPError freeze, and the robot never left its start
    timeout_freeze_midroute  a terminal DCPError freeze after the robot had moved
    timeout_stuck            the Phase 5 stuck flag (highdim.harness.stuck)
    timeout_slow             any other timeout

A DCPError step is one whose QP status is "DCPError": h < 0 makes the reference objective
non-convex, CVXPY raises, and the frozen policy executes u = 0 without calling Random (RS_DESIGN
14.1 item 1). h is the distance from the robot's centre to the nearest LiDAR surface point minus
0.3 m. The env's LiDAR reads rectangles and pedestrians at their true distance but the outer
wall 0.3 m short, so h < 0 within 0.6 m of the outer wall and, elsewhere, only on contact
(measured in ticket 09). A freeze is a run of at least
FREEZE_STEPS consecutive DCPError steps; it is "at the start" if the run starts on step 0 and
"terminal" if it lasts to the episode's end.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import binomtest, fisher_exact

from robustsuite.events import AGENT_RADIUS

#: A freeze: at least this many consecutive DCPError steps (1 s at dt 0.1).
FREEZE_STEPS = 10
#: A timeout "at the block": the robot's final centre within this distance of the fired block.
BLOCK_NEAR = 1.0
#: The robot "left its start" once it was ever this far from it.
LEFT_START = 0.3
#: Random "near the end": a Random event within this many steps of the last one.
NEAR_END = 10
#: "Still at the end": the robot moved less than STILL_DIST in its last STILL_STEPS steps.
STILL_STEPS, STILL_DIST = 50, 0.1
#: The obstacle conditions with an event.
TRIGGERED = ("trigger_block", "trigger_spawn")

#: The failure modes, in classification (and tie-break) order.
MODES = ("collision_dynamic", "collision_spawned", "collision_static", "collision_block",
         "collision_wall", "timeout_at_block", "timeout_freeze_start",
         "timeout_freeze_midroute", "timeout_stuck", "timeout_slow")


def dcp_runs(trace):
    """[(first step, length)] of every maximal run of DCPError steps."""
    runs, start = [], None
    for t, s in enumerate(trace):
        if s["qp_status"] == "DCPError":
            start = t if start is None else start
        elif start is not None:
            runs.append((start, t - start))
            start = None
    if start is not None:
        runs.append((start, len(trace) - start))
    return runs


def freeze_profile(trace):
    runs = dcp_runs(trace)
    long = [(s, n) for s, n in runs if n >= FREEZE_STEPS]
    return {"dcp_steps": sum(n for _, n in runs),
            "dcp_at_start": bool(runs) and runs[0][0] == 0,
            "freeze_start": any(s == 0 for s, _ in long),
            "freeze_midroute": any(s > 0 for s, _ in long),
            "terminal_freeze": any(s + n == len(trace) for s, n in long)}


def rect_distance(p, rect):
    """Distance from point p to the axis-aligned rectangle (cx, cy, hw, hh); 0 inside."""
    cx, cy, hw, hh = rect
    q = np.clip(p, [cx - hw, cy - hh], [cx + hw, cy + hh])
    return float(np.linalg.norm(np.asarray(p, float) - q))


def _n_regular(rec, spec):
    """Regular pedestrians in the env's slots (Spec.active_pedestrians, on the dict form)."""
    return 0 if rec["obstacles"] == "static" else len(spec["pedestrians"])


def episode_view(rec, row):
    """The diagnostic reading of one episode: failure mode, freezes, and the event relation."""
    trace, spec = row["trace"], row["spec"]
    traj = np.asarray(row["trajectory"], float)
    end = traj[-1]
    fired_block = rec["obstacles"] == "trigger_block" and rec["trigger_fired"]
    d_block = rect_distance(end, spec["block"]["rect"]) if fired_block else None
    view = freeze_profile(trace)
    view.update({
        "left_start": bool(np.max(np.linalg.norm(traj - traj[0], axis=1)) > LEFT_START),
        "random_near_end": any(s["random_event"] for s in trace[-NEAR_END:]),
        "still_at_end": bool(len(traj) > STILL_STEPS
                             and np.linalg.norm(end - traj[-1 - STILL_STEPS]) < STILL_DIST),
        "hit_slot": None})

    mode = None
    if rec["outcome"] == "collision":
        ctype = rec["collision_type"]
        if ctype == "dynamic":
            peds = np.asarray(rec["final_obstacle_positions"], float).reshape(-1, 2)
            if not len(peds):
                raise ValueError("a dynamic collision with no dynamic obstacle recorded")
            slot = int(np.argmin(np.linalg.norm(peds - end, axis=1)))
            view["hit_slot"] = slot
            # the anchor has no spec: all of its obstacles are regular
            mode = ("collision_spawned" if spec is not None and slot >= _n_regular(rec, spec)
                    else "collision_dynamic")
        elif ctype == "static":
            mode = ("collision_block" if d_block is not None and d_block < AGENT_RADIUS + 1e-6
                    else "collision_static")
        else:
            mode = "collision_wall"
    elif rec["outcome"] == "timeout":
        if d_block is not None and d_block <= BLOCK_NEAR:
            mode = "timeout_at_block"
        elif view["terminal_freeze"]:
            mode = "timeout_freeze_midroute" if view["left_start"] else "timeout_freeze_start"
        else:
            mode = "timeout_stuck" if rec["stuck"] else "timeout_slow"
    view["mode"] = mode

    view["event"] = None
    if rec["obstacles"] in TRIGGERED:
        fired = bool(rec["trigger_fired"])
        view["event"] = {
            "fired": fired,
            "steps_after_fire": rec["steps"] - rec["trigger_step"] if fired else None,
            "dist_trigger": float(np.linalg.norm(end - np.asarray(spec["trigger"]["centre"]))),
            "dist_block": (rect_distance(end, spec["block"]["rect"])
                           if rec["obstacles"] == "trigger_block" else None),
            "dist_spawn_start": (float(np.linalg.norm(end - np.asarray(spec["spawn"]["start"])))
                                 if rec["obstacles"] == "trigger_spawn" else None)}
    return view


def _rate(n, d):
    return n / d if d else float("nan")


def cell_summary(recs, views):
    """Outcome rates, failure-mode counts and DCPError freezes of one group of episodes."""
    n = len(recs)
    modes = Counter(v["mode"] for v in views if v["mode"] is not None)
    steps = sum(r["steps"] for r in recs)
    top = min(modes, key=lambda m: (-modes[m], MODES.index(m))) if modes else None
    return {
        "episodes": n,
        "success_rate": _rate(sum(r["success"] for r in recs), n),
        "collision_rate": _rate(sum(r["collision"] for r in recs), n),
        "timeout_rate": _rate(sum(r["timeout"] for r in recs), n),
        "stuck_rate": _rate(sum(r["stuck"] for r in recs), n),
        "modes": {m: modes[m] for m in MODES if modes[m]},
        "most_common_mode": top,
        "dcp_episodes": sum(v["dcp_steps"] > 0 for v in views),
        "dcp_at_start_episodes": sum(v["dcp_at_start"] for v in views),
        "dcp_step_rate": _rate(sum(v["dcp_steps"] for v in views), steps),
        "freeze_start_episodes": sum(v["freeze_start"] for v in views),
        "freeze_midroute_episodes": sum(v["freeze_midroute"] for v in views),
        "terminal_freeze_episodes": sum(v["terminal_freeze"] for v in views),
    }


def order_stats(values):
    """n, min, quartiles and max of the non-None values; None if there are none."""
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "min": float(v.min()), "q1": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
            "max": float(v.max())}


def event_summary(recs, views):
    """Triggered cells: the fired rate, and failures timed and located relative to the event."""
    fails = [(r, v) for r, v in zip(recs, views, strict=True) if v["mode"] is not None]
    after = [(r, v) for r, v in fails if v["event"]["fired"]]
    coll = [v for r, v in after if r["outcome"] == "collision"]
    return {
        "fired": sum(v["event"]["fired"] for v in views),
        "fired_rate": _rate(sum(v["event"]["fired"] for v in views), len(views)),
        "failures_before_fire": len(fails) - len(after),
        "failures_after_fire": len(after),
        "steps_after_fire": order_stats(v["event"]["steps_after_fire"] for _, v in after),
        "collision_steps_after_fire": order_stats(v["event"]["steps_after_fire"] for v in coll),
        "collision_dist_trigger": order_stats(v["event"]["dist_trigger"] for v in coll),
        "collision_dist_block": order_stats(v["event"]["dist_block"] for v in coll),
        "collision_dist_spawn_start": order_stats(v["event"]["dist_spawn_start"] for v in coll),
        "spawned_hits": sum(v["mode"] == "collision_spawned" for v in views),
        "collisions_near_block": sum(v["event"]["dist_block"] is not None
                                     and v["event"]["dist_block"] <= BLOCK_NEAR for v in coll),
    }


def pooled(summaries, keys):
    """Every cell weighted equally (RS_DESIGN 8, 9): the mean of the cells' rates. A cell with no
    value for a key (NaN: e.g. no Random event, so no margin) is left out of that key's mean."""
    def mean(k):
        v = [s[k] for s in summaries if s[k] == s[k]]
        return float(np.mean(v)) if v else float("nan")
    out = {k: mean(k) for k in keys}
    out["episodes"] = sum(s["episodes"] for s in summaries)
    return out


def anchor_check(k, n, k_ref, n_ref, alpha=0.05):
    """The anchor's success count against a reference (unpaired: different seeds), two-sided
    Fisher exact test. Consistent (within sampling noise) when p >= alpha. The anchor's exact
    (Clopper-Pearson) 95% interval is returned too: with n = 100 it is about +-7 pp wide, so
    "consistent" rules out only large departures."""
    p = float(fisher_exact([[k, n - k], [k_ref, n_ref - k_ref]])[1])
    ci = binomtest(k, n).proportion_ci(0.95, method="exact")
    return {"k": k, "n": n, "rate": k / n, "ci95": [float(ci.low), float(ci.high)],
            "k_ref": k_ref, "n_ref": n_ref, "rate_ref": k_ref / n_ref, "p": p,
            "consistent": p >= alpha}
