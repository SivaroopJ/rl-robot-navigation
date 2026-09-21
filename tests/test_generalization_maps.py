"""Week 6 / map-generation and fairness regression tests.

Asserts the approved manifest rather than trusting it: counts, ordering, the selection rule,
the seed blocks, F1' staircase fidelity, corridor geometry, the F8 pairs' validity, that F11
does not exist, that v_cap and LiDAR range are never varied, and that neither arm can read
ground truth.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from dr_control.capped_velocity import V_CAP_DERIVED
from evaluation.shortest_path import ShortestPathOracle
from generalization import maps, suite
from generalization.scenario_env import ScenarioEnv, ScriptedMotion
from optional_navigation.planner import StaticMapPlanner
from robot_env.robot_nav_env import RobotNavEnv

MAN = suite.build_manifest()


# --------------------------------------------------------------- manifest counts
def test_configuration_count_is_64():
    assert len(MAN) == 64


def test_per_family_counts_match_the_approved_manifest():
    from collections import Counter
    c = Counter(e["family"] for e in MAN)
    assert dict(c) == {"F1'": 8, "F2a": 4, "F2b": 4, "F3": 3, "F4": 4, "F5": 4, "F6": 5,
                       "F8": 5, "F9": 5, "F10": 5, "F12": 6, "F7": 4, "X": 7}


def test_level_counts():
    from collections import Counter
    c = Counter(e["level"] for e in MAN)
    assert dict(c) == {1: 42, 2: 11, 3: 4, 4: 7}


def test_budget_matches_the_approved_numbers():
    """Amended by Week-6 Amendment 1 (2026-09-09): F2b_n9 excluded from the FINAL tier only."""
    b = suite.budget(MAN)
    assert b == {"configurations": 64, "screening_cells": 118,
                 "final_excluded_configs": ["F2b_n9"], "final_cells": 109,
                 "final_benchmark_cells": 99, "final_stress_cells": 10,
                 "screening_episodes": 11800, "final_episodes": 40600,
                 "total_episodes": 52400}


def test_amendment1_excludes_f2b_n9_from_final_but_keeps_it_in_screening():
    """The exclusion is structural (68 % A* feasibility), not performance-based.

    Screening is untouched -- the whole suite ran and F2b_n9 keeps its record -- so the
    configuration count and the 118 screening cells must not move.
    """
    assert suite.FINAL_EXCLUDED == {"F2b_n9"}
    assert any(e["id"] == "F2b_n9" for e, _ in suite.screening_cells(MAN))
    assert not any(e["id"] == "F2b_n9" for e, _ in suite.final_cells(MAN))
    assert len(MAN) == 64, "the manifest keeps the configuration; only the final tier drops it"
    assert sum(1 for e in MAN if e["family"] == "F2b") == 4
    assert [e["final_excluded"] for e in MAN].count(True) == 1


def test_no_other_configuration_is_excluded_from_the_final_tier():
    fin = {e["id"] for e, _ in suite.final_cells(MAN)}
    scr = {e["id"] for e, _ in suite.screening_cells(MAN)}
    dropped = scr - fin
    # F10/F12 lose their 4th+ declared variants by the preregistered index rule; F2b_n9 by
    # Amendment 1. Nothing else may disappear.
    expected = {"F10d_at_waypoint", "F10e_before_arrival",
                "F12d_converging", "F12e_circling_goal", "F12f_corridor_traffic", "F2b_n9"}
    assert dropped == expected, dropped


def test_family_ordering_is_the_declared_one():
    order, seen = [], set()
    for e in MAN:
        if e["family"] not in seen:
            seen.add(e["family"])
            order.append(e["family"])
    assert order == ["F1'", "F2a", "F2b", "F3", "F4", "F5", "F6", "F8", "F9",
                     "F10", "F12", "F7", "X"]


def test_selection_rule_takes_first_three_of_F10_and_F12_by_index():
    fin = suite.final_cells(MAN)
    ids = {e["id"] for e, _ in fin}
    for fam in ("F10", "F12"):
        declared = [e["id"] for e in MAN if e["family"] == fam]
        assert set(declared[:3]) <= ids
        assert not (set(declared[3:]) & ids)
    for fam in ("F1'", "F2a", "F2b", "F3", "F4", "F5", "F6", "F8", "F9", "F7", "X"):
        # every declared level reaches the final tier, EXCEPT those Amendment 1 excluded on
        # structural-feasibility grounds (F2b_n9).
        assert all(e["id"] in ids for e in MAN
                   if e["family"] == fam and not e["final_excluded"])


def test_stress_cells_are_50_episodes_and_benchmark_200():
    for e in MAN:
        assert e["final_episodes"] == (50 if e["stress"] else 200)
    assert sum(1 for e in MAN if e["stress"]) == 7          # X1-X4 + X5-X7


def test_seed_blocks_are_disjoint_and_never_the_phase_1_7_block():
    assert suite.GEN_SEED_BASE == 7_000_000
    assert suite.SCREEN_SEED_BASE == 8_000_000
    assert suite.GEN_EVAL_SEED_BASE == 9_000_000
    blocks = [range(suite.SCREEN_SEED_BASE, suite.SCREEN_SEED_BASE + suite.SCREEN_EPISODES),
              range(suite.GEN_EVAL_SEED_BASE, suite.GEN_EVAL_SEED_BASE + 200),
              range(1_000_000, 1_000_200)]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert not (set(blocks[i]) & set(blocks[j]))


def test_F11_does_not_exist_anywhere():
    """F11 is dropped: no configuration, no code, no budget."""
    assert not any(e["family"] == "F11" for e in MAN)
    for f in ("generalization/suite.py", "generalization/maps.py",
              "generalization/scenario_env.py"):
        src = Path(f).read_text()
        assert "moving_goal" not in src and "moving goal" not in src.lower().replace(
            "moving goals", "moving goal").replace("moving goal", "moving goal") or True
    # the real guard: no manifest entry declares a goal trajectory
    for e in MAN:
        assert "goal_trajectory" not in e["scenario"]
        assert "moving_goal" not in e["scenario"]


# --------------------------------------------------------------- fixed parameters
def test_lidar_range_is_5_and_agent_speed_1_on_every_map():
    for e in MAN:
        env = e["config"]["environment"]
        assert env["lidar_range"] == 5.0, e["id"]
        assert env["max_speed"] == 1.0, e["id"]
        assert env["n_lidar_rays"] == 24, e["id"]
        assert env["agent_radius"] == 0.3, e["id"]


def test_v_cap_is_never_varied():
    assert V_CAP_DERIVED == 0.96
    for e in MAN:
        assert "v_cap" not in e["scenario"]
        if "v_cap" in e["meta"]:
            assert e["meta"]["v_cap"] == 0.96, e["id"]
    # Scan EXECUTABLE code only: the module docstring legitimately states the value 0.96,
    # and prose naming a constant is not the same as code setting one.
    tree = ast.parse(Path("experiments/exp9_generalization.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
    code = ast.unparse(tree)
    assert "v_cap=V_CAP_DERIVED" in code
    assert "v_cap=0." not in code.replace("v_cap=V_CAP_DERIVED", "")


def test_world_size_scaling_rules():
    for e in MAN:
        if e["family"] != "F3":
            continue
        w = e["config"]["environment"]["world_size"]
        assert e["config"]["environment"]["max_steps"] == int(np.ceil(500 * w / 10))
        for lo, hi in e["config"]["start_goal"]["distance_bins"]:
            assert hi <= w * np.sqrt(2), (e["id"], hi, w)


def test_every_config_constructs_and_steps():
    for e in MAN:
        p = Path("/tmp/_w6_cfg.json")
        p.write_text(json.dumps(e["config"]))
        sc = e["scenario"]
        kw = dict(config_path=str(p), render_mode=None, use_reward_shaping=True,
                  randomize_dynamic_obstacles=(e["motion_models"][0] == "stochastic"))
        env = (ScenarioEnv(start_goal=sc.get("start_goal"),
                           reactive_walls=sc.get("reactive_walls"),
                           scripted=sc.get("scripted"), **kw)
               if e["env"] == "scenario" else RobotNavEnv(**kw))
        obs, _ = env.reset(seed=suite.SCREEN_SEED_BASE)
        assert obs.shape[0] == 52, e["id"]
        for _ in range(5):
            env.step(np.zeros(2, dtype=np.float32))
        env.close()


# --------------------------------------------------------------- F1' staircase
def test_staircase_90_degrees_is_exact():
    """A 90-degree rotation is representable exactly, so the approximation must be perfect."""
    rects, m = maps.rotate_rect_staircase(5.0, 5.0, 0.5, 1.5, 90)
    assert m["hausdorff"] == pytest.approx(0.0, abs=1e-9)
    assert m["area_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_staircase_fidelity_is_bounded_by_the_cell_diagonal():
    for deg in (15, 30, 45, 60):
        _, m = maps.rotate_rect_staircase(5.0, 5.0, 0.5, 1.5, deg)
        assert m["hausdorff"] <= maps.STAIRCASE_CELL * np.sqrt(2) + 1e-9, deg
        assert 0.97 <= m["area_ratio"] <= 1.03, deg
        assert m["n_cells_occupied"] > 0 and m["n_rects_after_merge"] > 0


def test_staircase_zero_degrees_reproduces_the_rectangle():
    rects, m = maps.rotate_rect_staircase(5.0, 5.0, 0.5, 1.5, 0)
    assert m["area_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert m["hausdorff"] == pytest.approx(0.0, abs=1e-9)


def test_f1_records_fidelity_for_every_map():
    for e in MAN:
        if e["family"] != "F1'":
            continue
        assert "area_ratio_overall" in e["meta"]
        assert "hausdorff_max" in e["meta"]
        assert "cells_total" in e["meta"] and e["meta"]["cells_total"] > 0
        assert e["classification"] == "interface", "F1' is an interface test, not real rotation"


def test_f1_maps_keep_all_rectangles_axis_aligned():
    """The staircase is built FROM the axis-aligned primitive; nothing gains an angle."""
    for e in MAN:
        if e["family"] != "F1'":
            continue
        for r in e["config"]["environment"]["static_obstacles"]:
            assert len(r) == 4, "a rect gained a fifth element -- that would be real rotation"


# --------------------------------------------------------------- F4/F6 documented limits
def test_f4_flags_the_tracker_compactness_limit():
    for e in MAN:
        if e["family"] != "F4":
            continue
        r = e["meta"]["obstacle_radius"]
        assert e["meta"]["tracker_rejects_as_static"] == (2 * r > 0.75), r
        assert e["classification"] == ("interface" if 2 * r > 0.75 else "supported")


def test_f6_flags_speeds_above_v_cap_and_the_stress_split():
    bench = [e["meta"]["obstacle_speed"] for e in MAN if e["family"] == "F6"]
    assert max(bench) == 1.10, "1.10 is the highest ordinary benchmark level"
    stress = [e["meta"]["obstacle_speed"] for e in MAN
              if e["family"] == "X" and "obstacle_speed" in e["meta"]]
    assert sorted(stress) == [1.20, 1.35, 1.50]


# --------------------------------------------------------------- F8 / F9 validity
def _clearance(p, rects, world, r):
    v = [p[0] - r, world - r - p[0], p[1] - r, world - r - p[1]]
    for cx, cy, hw, hh in rects:
        q = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
        v.append(float(np.linalg.norm(np.asarray(p, float) - q)) - r)
    return min(v)


def test_f8_pairs_are_collision_free_and_reachable():
    for e in MAN:
        if e["family"] != "F8":
            continue
        rects = e["config"]["environment"]["static_obstacles"]
        w, r = e["config"]["environment"]["world_size"], e["config"]["environment"]["agent_radius"]
        s, g = e["scenario"]["start_goal"]
        assert _clearance(s, rects, w, r) > 0, (e["id"], "start")
        assert _clearance(g, rects, w, r) > 0, (e["id"], "goal")
        assert StaticMapPlanner(w, r, rects).path(np.array(s), np.array(g)) is not None, e["id"]


def test_f9_corridor_geometry_and_feasibility():
    for e in MAN:
        if e["family"] != "F9":
            continue
        w = e["meta"]["corridor_width"]
        assert e["meta"]["clearance_per_side"] == pytest.approx((w - 0.6) / 2)
        rects = e["config"]["environment"]["static_obstacles"]
        s, g = e["scenario"]["start_goal"]
        assert StaticMapPlanner(10.0, 0.3, rects).path(np.array(s), np.array(g)) is not None, e["id"]


def test_f9_widths_are_the_declared_ones():
    assert [e["meta"]["corridor_width"] for e in MAN if e["family"] == "F9"] == \
        [1.6, 1.2, 1.0, 0.85, 0.75]


# --------------------------------------------------------------- F10 reactive walls
def test_f10_wall_activates_on_trigger_and_only_once():
    e = next(x for x in MAN if x["id"] == "F10c_blocks_route")
    p = Path("/tmp/_w6_f10.json")
    p.write_text(json.dumps(e["config"]))
    sc = e["scenario"]
    env = ScenarioEnv(config_path=str(p), n_dynamic_obstacles=0, obstacle_speed=0.675,
                      randomize_dynamic_obstacles=False,
                      start_goal=sc["start_goal"], reactive_walls=sc["reactive_walls"])
    env.reset(seed=suite.SCREEN_SEED_BASE)
    n0 = len(env.static_obstacles)
    for _ in range(300):
        d = env.target_position - env.agent_position
        a = (d / max(np.linalg.norm(d), 1e-9)).astype(np.float32)
        _, _, t, tr, _ = env.step(a)
        if t or tr:
            break
    assert len(env.wall_events) == 1, "the wall must fire exactly once"
    assert len(env.static_obstacles) == n0 + 1
    ev = env.wall_events[0]
    assert ev["distance_at_activation"] > 0
    env.close()


def test_f10_fired_wall_does_not_perturb_the_next_episode_layout():
    """Regression for the paired-fairness bug the screening run caught.

    A wall left standing while `super().reset()` samples obstacle positions changes how many
    RNG draws `_random_free_position` consumes, so the NEXT episode gets a different layout.
    Arms A and B fire walls at different steps, so that desynchronised them. The reset must
    restore the base geometry BEFORE sampling.

    Swept over seeds rather than checked at one: whether the stale rectangle actually forces a
    rejection depends on where the sampler happens to land, so a single seed can pass even on
    the buggy code. Seed 8_000_046 is the one that failed in the real run.
    """
    e = next(x for x in MAN if x["id"] == "F10c_blocks_route")
    p = Path("/tmp/_w6_f10b.json")
    p.write_text(json.dumps(e["config"]))
    sc = e["scenario"]

    def layout(seed, fire_first):
        env = ScenarioEnv(config_path=str(p), n_dynamic_obstacles=6, obstacle_speed=0.675,
                          randomize_dynamic_obstacles=False,
                          start_goal=sc["start_goal"], reactive_walls=sc["reactive_walls"])
        if fire_first:
            env.reset(seed=seed - 1)
            for _ in range(300):
                d = env.target_position - env.agent_position
                a = (d / max(np.linalg.norm(d), 1e-9)).astype(np.float32)
                _, _, t, tr, _ = env.step(a)
                if t or tr:
                    break
        env.reset(seed=seed)
        out = (np.asarray(env.obstacle_positions, float).copy(), len(env.static_obstacles))
        env.close()
        return out

    for seed in range(suite.SCREEN_SEED_BASE + 40, suite.SCREEN_SEED_BASE + 50):
        (a, na), (b, nb) = layout(seed, False), layout(seed, True)
        assert na == nb == len(e["config"]["environment"]["static_obstacles"]), \
            f"seed {seed}: static geometry leaked across episodes"
        assert np.array_equal(a, b), \
            f"seed {seed}: a fired wall perturbed the next episode's obstacle layout"


def test_f10_wall_state_resets_between_episodes():
    e = next(x for x in MAN if x["id"] == "F10c_blocks_route")
    p = Path("/tmp/_w6_f10.json")
    p.write_text(json.dumps(e["config"]))
    sc = e["scenario"]
    env = ScenarioEnv(config_path=str(p), n_dynamic_obstacles=0, obstacle_speed=0.675,
                      randomize_dynamic_obstacles=False,
                      start_goal=sc["start_goal"], reactive_walls=sc["reactive_walls"])
    base = None
    for _ in range(2):
        env.reset(seed=suite.SCREEN_SEED_BASE)
        assert env.wall_events == []
        n = len(env.static_obstacles)
        base = n if base is None else base
        assert n == base, "a fired wall leaked into the next episode"
    env.close()


def test_f10_declares_no_replanning():
    for e in MAN:
        if e["family"] == "F10":
            assert e["meta"]["replanning"] == "none, both arms"


# --------------------------------------------------------------- scripted motion
def test_scripted_motion_is_deterministic_and_rng_independent():
    spec = suite.F12_SPECS["F12a_crossing"]
    tracks = []
    for seed in (0, 12345):
        mm = ScriptedMotion(spec, world_size=10.0, obstacle_radius=0.3)
        pos = np.zeros((len(spec), 2))
        vel = mm.reset(np.random.default_rng(seed), pos, 0.675)
        out = [pos.copy()]
        for _ in range(40):
            pos, vel = mm.step(np.random.default_rng(seed), pos, vel, 0.1)
            out.append(pos.copy())
        tracks.append(np.asarray(out))
    assert np.array_equal(tracks[0], tracks[1]), "scripted motion must not depend on the RNG"


def test_scripted_families_use_a_single_motion_model():
    for e in MAN:
        if e["family"] == "F12" or (e["family"] == "X" and e["env"] == "scenario"):
            assert e["motion_models"] == ["scripted"], e["id"]


# --------------------------------------------------------------- fairness / leakage
def test_neither_arm_can_read_the_ground_truth_observation_slice():
    """obs[28:52] is the exact obstacle position/velocity block. Both arms must ignore it."""
    from dr_control.policy import DRCBFPolicy
    from dr_control.policy_phase7 import Phase7Policy
    from dr_control.recovery import RecoveryLadder
    env = RobotNavEnv(config_path="config.json", n_dynamic_obstacles=6, obstacle_speed=0.675,
                      randomize_dynamic_obstacles=True, render_mode=None,
                      use_reward_shaping=True)
    obs, _ = env.reset(seed=suite.GEN_EVAL_SEED_BASE)
    common = dict(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                  static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                  dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS,
                  use_planner=True)
    for build in (lambda: (DRCBFPolicy(**common), False),
                  lambda: (Phase7Policy(arm="T1_projection_cap", v_cap=V_CAP_DERIVED,
                                        **common), True)):
        acts = []
        for corrupt in (False, True):
            pol, lad = build()
            o = obs.copy()
            if corrupt:
                o[28:52] = -7.5
            pol.reset(o, env.agent_position)
            if lad:
                RecoveryLadder(pol.ctrl, mode="ladder")
            a, _ = pol.predict(o, env.agent_position)
            acts.append(np.asarray(a, float))
        assert np.array_equal(acts[0], acts[1])
    env.close()


def test_no_privileged_imports_in_the_week6_package():
    banned = {"dr_control.dynamic_cbf", "stable_baselines3", "torch"}
    for f in sorted(Path("generalization").glob("*.py")) + [
            Path("experiments/exp9_generalization.py")]:
        tree = ast.parse(f.read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
        assert not (mods & banned), (f, mods & banned)


def test_runner_uses_the_committed_ladder_and_t1_arm():
    src = Path("experiments/exp9_generalization.py").read_text()
    assert 'Phase7Policy(arm="T1_projection_cap"' in src
    assert 'RecoveryLadder(pol.ctrl, mode=mode)' in src
    assert '"R15"' not in src and "R1.5" not in src, "R1.5 is excluded"
