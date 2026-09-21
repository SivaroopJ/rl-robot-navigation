"""Week 6 / the suite manifest. DECLARATIVE -- every configuration, seed and selection rule is
fixed here, before any result exists, exactly as approved in MAP_SUITE_SPEC.md.

COUNTS, which the tests assert rather than trust:
    64 configurations   Level 1: 42 (F1' 8, F2a 4, F2b 4, F3 3, F4 4, F5 4, F6 5, F8 5, F9 5)
                        Level 2: 11 (F10 5, F12 6)      Level 3: 4 (F7)
                        Level 4:  7 (X1-X4 4, X5-X7 3)
    118 screening cells   64 configs x motion models (2 where free, 1 where scripted).
                          UNCHANGED by Amendment 1: the whole screening suite was completed and
                          F2b_n9 keeps its screening record.
    109 final cells        99 benchmark @ 200 episodes + 10 stress @ 50
    11,800 + 40,600 = 52,400 episodes
                          Amendment 1 (2026-09-09) removed F2b_n9's 2 final cells: 111 -> 109,
                          41,400 -> 40,600, 53,200 -> 52,400. See FINAL_EXCLUDED for the reason.

M0 is the CONTROL and is not a member of the 64. F11 (moving goal) is DROPPED and has no
configuration, no code and no budget.

SELECTION RULE, fixed before any data (spec 4.3):
    every declared level of F1', F2a, F2b, F3, F4, F5, F6, F8, F9 -> final tier
    F10 and F12 -> first three declared variants, BY MANIFEST INDEX
    F7 -> all four; X1-X7 -> all seven
Nothing is dropped for performing badly, and screening never selects anything.
"""
from __future__ import annotations

import copy

from generalization import maps

# ----------------------------------------------------------------- seed blocks (spec 4.4)
GEN_SEED_BASE = 7_000_000        # map layout sampling only; never an episode seed
SCREEN_SEED_BASE = 8_000_000     # +0..49
GEN_EVAL_SEED_BASE = 9_000_000   # +0..199
SCREEN_EPISODES = 50
FINAL_EPISODES_BENCH = 200
FINAL_EPISODES_STRESS = 50
#: EVAL_SEED_BASE = 1_000_000 belongs to Phases 1-7 and is used ONLY by the M0 gate.

BOTH = ("deterministic", "stochastic")
SCRIPTED = ("scripted",)

#: AMENDMENT 1 to the Week 6 preregistration (2026-09-09), recorded after screening and BEFORE
#: the final tier. `F2b_n9` is EXCLUDED FROM THE FINAL TIER ONLY. It stays in the manifest and
#: keeps its complete screening record.
#:
#: REASON -- STRUCTURAL FEASIBILITY, NOT CONTROLLER PERFORMANCE. Only 68 % of its declared
#: screening start/goal pairs have an A* path (368/400 family-wide, but 34/50 per cell in BOTH
#: motion models), against the preregistered >= 90 % criterion. The same 16 seeds fail for both
#: arms, both arms degrade identically to direct-to-goal when the planner fails, and neither arm
#: gains or loses by it -- so the exclusion cannot favour either controller. The preregistered
#: remedy ("advance the generation seed") cannot be applied because this is a FIXED DECLARED
#: layout, not a seed-generated one: there is no seed to advance. No replacement map is invented,
#: because inventing one after seeing screening results is exactly what the preregistration
#: forbids. The configuration does not satisfy the intended navigable-map assumption.
FINAL_EXCLUDED = {"F2b_n9"}

#: F8 start/goal pairs. Every pair verified collision-free and A*-feasible on the canonical
#: static map before being written here; `tests/test_generalization_maps.py` re-checks it.
F8_PAIRS = {
    "F8a_open_goal":          ([1.0, 4.8], [5.6, 4.6]),
    "F8b_goal_behind_rect":   ([1.2, 8.6], [7.0, 1.6]),
    "F8c_goal_confined_slot": ([1.0, 2.0], [6.0, 7.0]),
    "F8d_goal_in_traffic":    ([0.9, 1.0], [5.0, 5.0]),
    "F8e_long_diagonal":      ([0.8, 0.8], [9.2, 9.2]),
}

#: F10 reactive walls. Trigger disc radius 0.5 m unless stated; rect is axis-aligned.
F10_WALLS = {
    "F10a_behind":         {"trigger": [5.0, 5.0], "radius": 0.5, "rect": [3.6, 5.0, 0.5, 1.2]},
    "F10b_beside":         {"trigger": [5.0, 5.0], "radius": 0.5, "rect": [5.0, 6.4, 1.2, 0.4]},
    "F10c_blocks_route":   {"trigger": [4.0, 5.0], "radius": 0.5, "rect": [6.2, 5.0, 0.4, 1.4]},
    "F10d_at_waypoint":    {"trigger": [3.0, 5.0], "radius": 0.5, "rect": [5.0, 5.0, 0.5, 1.0]},
    "F10e_before_arrival": {"trigger": [7.0, 5.0], "radius": 0.5, "rect": [8.4, 5.0, 0.4, 1.2]},
}
F10_START_GOAL = ([0.8, 5.0], [9.2, 5.0])

_S = 0.675     # canonical obstacle speed, reused by the scripted specs


def _sc(**kw):
    return dict(kw)


#: F12 structured trajectories. Six obstacles each, declared in full, deterministic.
F12_SPECS = {
    "F12a_crossing":  [_sc(start=[5.0, 1.0], vel=[0.0,  _S]), _sc(start=[1.0, 5.0], vel=[_S, 0.0]),
                       _sc(start=[5.0, 9.0], vel=[0.0, -_S]), _sc(start=[9.0, 5.0], vel=[-_S, 0.0]),
                       _sc(start=[3.0, 3.0], vel=[_S, _S]),   _sc(start=[7.0, 7.0], vel=[-_S, -_S])],
    "F12b_head_on":   [_sc(start=[9.0, 5.0], vel=[-_S, 0.0]), _sc(start=[9.0, 4.4], vel=[-_S, 0.0]),
                       _sc(start=[9.0, 5.6], vel=[-_S, 0.0]), _sc(start=[8.0, 4.7], vel=[-_S, 0.0]),
                       _sc(start=[8.0, 5.3], vel=[-_S, 0.0]), _sc(start=[7.0, 5.0], vel=[-_S, 0.0])],
    "F12c_parallel":  [_sc(start=[1.0, 3.5], vel=[_S, 0.0]),  _sc(start=[2.0, 3.5], vel=[_S, 0.0]),
                       _sc(start=[3.0, 3.5], vel=[_S, 0.0]),  _sc(start=[1.0, 6.5], vel=[_S, 0.0]),
                       _sc(start=[2.0, 6.5], vel=[_S, 0.0]),  _sc(start=[3.0, 6.5], vel=[_S, 0.0])],
    "F12d_converging":[_sc(start=[1.0, 1.0], vel=[_S,  _S]),  _sc(start=[9.0, 1.0], vel=[-_S,  _S]),
                       _sc(start=[1.0, 9.0], vel=[_S, -_S]),  _sc(start=[9.0, 9.0], vel=[-_S, -_S]),
                       _sc(start=[5.0, 1.0], vel=[0.0,  _S]), _sc(start=[5.0, 9.0], vel=[0.0, -_S])],
    "F12e_circling_goal":
                      [_sc(start=[6.5, 5.0], centre=[5.0, 5.0], omega=0.45),
                       _sc(start=[5.0, 6.5], centre=[5.0, 5.0], omega=0.45),
                       _sc(start=[3.5, 5.0], centre=[5.0, 5.0], omega=0.45),
                       _sc(start=[5.0, 3.5], centre=[5.0, 5.0], omega=0.45),
                       _sc(start=[7.0, 5.0], centre=[5.0, 5.0], omega=-0.35),
                       _sc(start=[3.0, 5.0], centre=[5.0, 5.0], omega=-0.35)],
    "F12f_corridor_traffic":
                      [_sc(start=[5.0, 2.0], waypoints=[[5.0, 8.0], [5.0, 2.0]], speed=_S),
                       _sc(start=[5.0, 3.0], waypoints=[[5.0, 8.0], [5.0, 2.0]], speed=_S),
                       _sc(start=[5.0, 8.0], waypoints=[[5.0, 2.0], [5.0, 8.0]], speed=_S),
                       _sc(start=[5.0, 7.0], waypoints=[[5.0, 2.0], [5.0, 8.0]], speed=_S),
                       _sc(start=[2.0, 5.0], waypoints=[[8.0, 5.0], [2.0, 5.0]], speed=_S),
                       _sc(start=[8.0, 5.0], waypoints=[[2.0, 5.0], [8.0, 5.0]], speed=_S)],
}

#: X1-X4 feasibility stress. Characterisation only -- no success rate is ever reported.
X_SPECS = {
    "X1_corridor_blocked": {
        "corridor": 1.0,
        "scripted": [_sc(start=[5.0, 5.0], waypoints=[[5.0, 5.4], [5.0, 4.6]], speed=0.2),
                     _sc(start=[5.0, 4.6], waypoints=[[5.0, 5.4], [5.0, 4.6]], speed=0.2)],
        "start_goal": ([1.0, 5.0], [9.0, 5.0])},
    "X2_passage_closes": {
        "corridor": 1.2,
        "scripted": [_sc(start=[5.0, 8.5], waypoints=[[5.0, 5.0]], speed=0.5),
                     _sc(start=[5.0, 1.5], waypoints=[[5.0, 5.0]], speed=0.5)],
        "start_goal": ([1.0, 5.0], [9.0, 5.0])},
    "X3_wall_plus_obstacle": {
        "corridor": 1.0,
        "scripted": [_sc(start=[4.2, 5.0], waypoints=[[4.2, 5.0]], speed=0.0)],
        "start_goal": ([1.0, 5.0], [9.0, 5.0])},
    "X4_goal_enclosed": {
        "corridor": None,
        "scripted": [_sc(start=[6.2, 5.0], centre=[5.0, 5.0], omega=0.9),
                     _sc(start=[5.0, 6.2], centre=[5.0, 5.0], omega=0.9),
                     _sc(start=[3.8, 5.0], centre=[5.0, 5.0], omega=0.9),
                     _sc(start=[5.0, 3.8], centre=[5.0, 5.0], omega=0.9)],
        "start_goal": ([1.0, 1.0], [5.0, 5.0])},
}


def _entry(cid, family, level, cfg, meta, *, cls, difficulty, models=BOTH,
           env="base", scenario=None, stress=False):
    return {"id": cid, "family": family, "level": level, "config": cfg, "meta": meta,
            "classification": cls, "difficulty": difficulty, "motion_models": list(models),
            "env": env, "scenario": scenario or {}, "stress": bool(stress),
            "final_excluded": cid in FINAL_EXCLUDED,
            "final_episodes": FINAL_EPISODES_STRESS if stress else FINAL_EPISODES_BENCH}


def build_manifest():
    """The 64 configurations, in declared order. Pure; called by the tests and the runner."""
    base = maps.canonical_config()
    E = []

    # ---- F1' 8: five single-obstacle rotations, then three all-obstacle rotations
    for deg in (15, 30, 45, 60, 90):
        cfg, m = maps.f1_rotated(base, {0}, deg)
        E.append(_entry(f"F1_rot{deg}_obs0", "F1'", 1, cfg, m,
                        cls="interface", difficulty="in-distribution" if deg <= 30 else "moderate"))
    for deg in (30, 45, 90):
        cfg, m = maps.f1_rotated(base, set(range(len(base["environment"]["static_obstacles"]))), deg)
        E.append(_entry(f"F1_rot{deg}_all", "F1'", 1, cfg, m,
                        cls="interface", difficulty="moderate"))

    # ---- F2a 4 / F2b 4
    for f in (0.5, 0.75, 1.25, 1.5):
        cfg, m = maps.f2a_static_scale(base, f)
        E.append(_entry(f"F2a_scale{f}", "F2a", 1, cfg, m, cls="supported",
                        difficulty="in-distribution" if 0.75 <= f <= 1.25 else "moderate"))
    for n in (2, 3, 7, 9):
        cfg, m = maps.f2b_static_count(base, n)
        E.append(_entry(f"F2b_n{n}", "F2b", 1, cfg, m, cls="supported",
                        difficulty="in-distribution" if n in (3, 7) else "moderate"))

    # ---- F3 3
    for w in (7.0, 15.0, 20.0):
        cfg, m = maps.f3_world(base, w)
        E.append(_entry(f"F3_world{int(w)}", "F3", 1, cfg, m, cls="supported",
                        difficulty="moderate" if w <= 15 else "ood"))

    # ---- F4 4
    for r in (0.15, 0.20, 0.45, 0.60):
        cfg, m = maps.f4_obstacle_radius(base, r)
        E.append(_entry(f"F4_r{r}", "F4", 1, cfg, m,
                        cls="interface" if m["tracker_rejects_as_static"] else "supported",
                        difficulty="in-distribution" if r >= 0.20 else "moderate"))

    # ---- F5 4
    for n in (2, 4, 8, 10):
        cfg, m = maps.f5_obstacle_count(base, n)
        E.append(_entry(f"F5_n{n}", "F5", 1, cfg, m, cls="supported",
                        difficulty="in-distribution" if n in (4, 8) else "moderate"))

    # ---- F6 5
    for s in (0.30, 0.45, 0.85, 0.96, 1.10):
        cfg, m = maps.f6_speed(base, s)
        E.append(_entry(f"F6_v{s}", "F6", 1, cfg, m,
                        cls="interface" if s > 0.96 else "supported",
                        difficulty=("in-distribution" if 0.45 <= s <= 0.85
                                    else "moderate" if s <= 0.96 else "ood")))

    # ---- F8 5
    for name, (s, g) in F8_PAIRS.items():
        E.append(_entry(name, "F8", 1, copy.deepcopy(base),
                        {"start": s, "goal": g,
                         "straight_line_distance": float(sum((a - b) ** 2 for a, b in zip(s, g)) ** 0.5)},
                        cls="supported", difficulty="moderate",
                        env="scenario", scenario={"start_goal": (s, g)}))

    # ---- F9 5
    for w in (1.6, 1.2, 1.0, 0.85, 0.75):
        cfg, m = maps.f9_corridor(base, w)
        E.append(_entry(f"F9_w{w}", "F9", 1, cfg, m, cls="supported",
                        difficulty="moderate" if w >= 1.0 else "ood",
                        env="scenario", scenario={"start_goal": ([1.0, 5.0], [9.0, 5.0])}))

    # ---- F10 5  (Level 2) -- NO REPLANNING for either arm (spec 1.4)
    for name, w in F10_WALLS.items():
        E.append(_entry(name, "F10", 2, copy.deepcopy(base),
                        {"trigger": w["trigger"], "trigger_radius": w["radius"],
                         "rect": w["rect"], "replanning": "none, both arms"},
                        cls="interface", difficulty="ood",
                        env="scenario",
                        scenario={"start_goal": F10_START_GOAL,
                                  "reactive_walls": [dict(w, name=name)]}))

    # ---- F12 6  (Level 2) -- scripted, so a single motion model
    for name, spec in F12_SPECS.items():
        cfg, _ = maps.f5_obstacle_count(base, len(spec))
        E.append(_entry(name, "F12", 2, cfg, {"n_scripted": len(spec), "pattern": name},
                        cls="supported", difficulty="moderate",
                        models=SCRIPTED, env="scenario", scenario={"scripted": spec}))

    # ---- F7 4  (Level 3)
    for cid, kw, diff in (("F7a_r45_v96", dict(radius=0.45, speed=0.96), "ood"),
                          ("F7b_n10_v96", dict(n=10, speed=0.96), "ood"),
                          ("F7c_r45_n10", dict(radius=0.45, n=10), "ood"),
                          ("F7d_r45_n10_v96", dict(radius=0.45, n=10, speed=0.96), "ood")):
        cfg, m = maps.f7_combo(base, **kw)
        E.append(_entry(cid, "F7", 3, cfg, m,
                        cls="interface" if kw.get("radius") else "supported", difficulty=diff))

    # ---- X1-X4 4  (Level 4, stress, scripted)
    for name, s in X_SPECS.items():
        cfg = copy.deepcopy(base)
        meta = {"scenario": name}
        if s["corridor"] is not None:
            cfg, m = maps.f9_corridor(cfg, s["corridor"])
            meta.update(m)
        cfg, _ = maps.f5_obstacle_count(cfg, len(s["scripted"]))
        E.append(_entry(name, "X", 4, cfg, meta, cls="feasibility-stress", difficulty="stress",
                        models=SCRIPTED, env="scenario", stress=True,
                        scenario={"scripted": s["scripted"], "start_goal": s["start_goal"]}))

    # ---- X5-X7 3  (Level 4, stress) -- the 1.20/1.35/1.50 levels MOVED here from F6
    for v in (1.20, 1.35, 1.50):
        cfg, m = maps.f6_speed(base, v)
        E.append(_entry(f"X_v{v}", "X", 4, cfg, m, cls="feasibility-stress",
                        difficulty="stress", stress=True))
    return E


def screening_cells(manifest=None):
    """(entry, motion_model) for every configuration -- 118 cells."""
    return [(e, mm) for e in (manifest or build_manifest()) for mm in e["motion_models"]]


def final_cells(manifest=None):
    """The §4.3 selection rule, applied by DECLARED INDEX. Never by controller performance."""
    man = manifest or build_manifest()
    seen, out = {}, []
    for e in man:
        fam = e["family"]
        seen[fam] = seen.get(fam, 0) + 1
        if fam in ("F10", "F12") and seen[fam] > 3:
            continue                                  # first three declared variants only
        if e["final_excluded"]:
            continue                                  # AMENDMENT 1: F2b_n9, see FINAL_EXCLUDED
        for mm in e["motion_models"]:
            out.append((e, mm))
    return out


def budget(manifest=None):
    man = manifest or build_manifest()
    scr = len(screening_cells(man))
    fin = final_cells(man)
    fin_ep = sum(e["final_episodes"] * 2 for e, _ in fin)
    return {"configurations": len(man), "screening_cells": scr,
            "final_excluded_configs": sorted(FINAL_EXCLUDED),
            "final_cells": len(fin),
            "final_benchmark_cells": sum(1 for e, _ in fin if not e["stress"]),
            "final_stress_cells": sum(1 for e, _ in fin if e["stress"]),
            "screening_episodes": scr * SCREEN_EPISODES * 2,
            "final_episodes": fin_ep,
            "total_episodes": scr * SCREEN_EPISODES * 2 + fin_ep}
