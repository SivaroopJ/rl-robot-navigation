"""RS Experiment 1: robust A*-Random on a fresh map suite.

Pre-registration: MD_files/robustsuite/RS_DESIGN.md. Decision: docs/adr/0002.
"""
from __future__ import annotations

import ast
import gzip
import json
import shutil

import numpy as np
import pytest
from scipy import ndimage

from continuation import seeds as CS
from continuation.pursuit.config import PURSUIT_BLOCKS
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import make_env as m0_env
from experiments.robustsuite import protect_manifest as PM
from highdim import harness as HDH
from highdim import seeds as HDS
from robustsuite import blocks as RB
from robustsuite import families as RF
from robustsuite import harness as RH
from robustsuite import pedestrians as RP
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite.scenario_env import RSScenarioEnv

# =========================================================================== ticket 01
# Frozen-file manifest and seed blocks.

# The frozen side of the experiment, by role. Each must be in the manifest.
MUST_COVER = [
    "robot_env/robot_nav_env.py",                     # dynamics, collision, LiDAR, reward
    "robot_env/dynamic_obstacles.py",                 # M0 anchor motion
    "optional_navigation/planner.py",                 # A* + carrot follower
    "evaluation/shortest_path.py",
    "dr_control/drccp_controller.py",                 # CLF, DR-CCP QP
    "dr_control/estimated_cbf.py",                    # LiDAR barrier source
    "dr_control/velocity_tracker.py",
    "dr_control/policy.py",
    "continuation/policy.py",                         # tuned policy + Random arm construction
    "continuation/random_recovery.py",
    "continuation/seeds.py",
    "continuation/harness.py",
    "continuation/stats.py",
    "highdim/policy.py",                              # the astar_random baseline arm
    "highdim/harness.py",                             # the episode loop RS reuses
    "highdim/blocks.py",
    "highdim/subgoal.py",
    "highdim/gates.py",
    "highdim/seeds.py",                               # reservations the RS blocks avoid
    "continuation/pursuit/config.py",
    "dr_control/fast_drccp.py",
    "experiments/week6_continuation/common.py",       # frozen parameter loaders
    "experiments/exp6_ppo_comparison.py",
    "config.json",
    "results/week6_continuation/H2/tuned_frozen.json",
    "results/week6_continuation/R3/random_frozen.json",
]


def test_frozen_files_are_untouched():
    modified, missing = PM.verify()
    assert not modified and not missing, (modified, missing)


@pytest.mark.parametrize("rel", MUST_COVER)
def test_manifest_covers_the_frozen_side(rel):
    assert rel in PM.read()


def test_manifest_never_covers_new_robustsuite_code():
    assert not [k for k in PM.read() if "robustsuite" in k]


def test_verify_detects_a_modified_or_missing_frozen_file(tmp_path):
    for rel in PM.read():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PM.REPO / rel, dst)
    assert PM.verify(root=tmp_path) == ([], [])
    target = tmp_path / "dr_control/drccp_controller.py"
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    (tmp_path / "config.json").unlink()
    assert PM.verify(root=tmp_path) == (["dr_control/drccp_controller.py"], ["config.json"])


# --------------------------------------------------------------------------- seed blocks
def test_rs_blocks_have_the_preregistered_ranges():
    assert RS.seed_block("RS_DIAG") == list(range(15_000_000, 15_000_050))
    assert RS.seed_block("RS_TUNE") == list(range(15_100_000, 15_100_050))
    assert RS.seed_block("RS_FINAL", allow_final=True) == list(range(15_200_000, 15_200_200))


def test_rs_blocks_are_disjoint_from_every_earlier_reservation():
    RS.check_disjoint()
    rs = set()
    for name in RS.BLOCKS:
        rs |= set(RS.seed_block(name, allow_final=True))
    earlier = [(b, b + CS.BLOCK_STRIDE) for b, _ in CS.BLOCKS.values()]
    earlier += list(CS.FORBIDDEN_RANGES)
    earlier += [(b, b + s) for b, s in PURSUIT_BLOCKS.values()]
    earlier += [(11_000_000, 14_000_000)]                            # pursuit, Track B, Track A
    earlier += [(b, b + s) for b, s in HDS.BLOCKS.values()]
    earlier += [(HDS.TRAIN_BASE, HDS.TRAIN_BASE + 1000 * HDS.TRAIN_RUNS)]
    for lo, hi in earlier:
        assert all(not lo <= s < hi for s in rs), (lo, hi)


def test_rs_blocks_are_disjoint_from_each_other():
    seen = set()
    for name in RS.BLOCKS:
        block = set(RS.seed_block(name, allow_final=True))
        assert not block & seen
        seen |= block


def test_rs_blocks_refuse_more_seeds_than_they_hold():
    with pytest.raises(ValueError):
        RS.seed_block("RS_DIAG", 51)
    with pytest.raises(KeyError):
        RS.seed_block("RS_NOPE")


def test_final_block_is_sealed():
    with pytest.raises(PermissionError):
        RS.seed_block("RS_FINAL")


def test_only_the_final_evaluation_entry_point_opens_rs_final():
    # every Python file in the repo outside the venv and the tests: a call that passes
    # allow_final to RS's seed_block must come from experiments/robustsuite/rs6_final.py
    users = []
    skip = {".venv", "tests", ".git"}
    files = [f for f in PM.REPO.rglob("*.py") if not skip & set(f.relative_to(PM.REPO).parts)]
    for f in files:
        text = f.read_text(errors="ignore")
        if "RS_FINAL" not in text and "robustsuite" not in text:
            continue
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and any(
                    k.arg == "allow_final" and not (isinstance(k.value, ast.Constant)
                                                    and k.value.value is False)
                    for k in n.keywords):
                users.append(str(f.relative_to(PM.REPO)))
    assert set(users) <= {f"experiments/robustsuite/{RS.FINAL_ENTRY_POINT}"}, users


def test_cells_are_the_full_factorial_plus_the_anchor():
    assert len(RS.CELLS) == 20
    assert {c.family for c in RS.CELLS} == set(RS.FAMILIES)
    assert {c.obstacles for c in RS.CELLS} == set(RS.OBSTACLE_CONDITIONS)
    assert len({(c.family, c.obstacles) for c in RS.CELLS}) == 20
    assert RS.ANCHOR not in RS.CELLS
    assert RS.MOTIONS == ("fixed", "randomized")


def test_one_family_draw_is_shared_by_every_obstacle_condition_of_a_family():
    seed = RS.seed_block("RS_DIAG")[0]
    for fam in RS.FAMILIES:
        got = {RS.generator_entropy(RS.Cell(fam, o), seed) for o in RS.OBSTACLE_CONDITIONS}
        assert len(got) == 1, fam
    per_family = {RS.generator_entropy(RS.Cell(f, "static"), seed) for f in RS.FAMILIES}
    assert len(per_family) == len(RS.FAMILIES)
    a = RS.CELLS[0]
    assert RS.generator_entropy(a, seed) != RS.generator_entropy(a, seed + 1)


def test_the_generator_signature_has_no_motion_condition():
    # both motion conditions share one family draw by construction, not by convention
    import inspect
    assert list(inspect.signature(RS.generator_entropy).parameters) == ["cell", "seed"]


def test_the_m0_anchor_is_not_a_generated_cell():
    with pytest.raises(ValueError):
        RS.generator_entropy(RS.ANCHOR, RS.seed_block("RS_DIAG")[0])


# =========================================================================== ticket 02
# Tracer: the open-clutter / static cell end to end. Seam 2 (the scenario generator) and seam 1
# (the RS episode harness), plus the M0 anchor through the scenario environment.

OPEN_STATIC = RS.Cell("open_clutter", "static")
DIAG = RS.seed_block("RS_DIAG")


def test_spec_is_deterministic_given_the_seed_and_identical_across_motions():
    for seed in DIAG[:5]:
        a = SC.generate(OPEN_STATIC, "fixed", seed)
        assert a == SC.generate(OPEN_STATIC, "fixed", seed)
        assert a == SC.generate(OPEN_STATIC, "randomized", seed)
    assert SC.generate(OPEN_STATIC, "fixed", DIAG[0]) != \
        SC.generate(OPEN_STATIC, "fixed", DIAG[1])


def _point_rect_distance(p, r):
    cx, cy, hw, hh = r
    q = np.array([np.clip(p[0], cx - hw, cx + hw), np.clip(p[1], cy - hh, cy + hh)])
    return float(np.linalg.norm(np.asarray(p, float) - q))


def _rect_rect_gap(a, b):
    # sample b's boundary densely; exact enough for a 1.2 m rule checked at 1e-6 slack below
    xs = np.linspace(b[0] - b[2], b[0] + b[2], 81)
    ys = np.linspace(b[1] - b[3], b[1] + b[3], 81)
    edge = ([(x, b[1] - b[3]) for x in xs] + [(x, b[1] + b[3]) for x in xs]
            + [(b[0] - b[2], y) for y in ys] + [(b[0] + b[2], y) for y in ys])
    return min(_point_rect_distance(p, a) for p in edge)


SAMPLE = DIAG + RS.seed_block("RS_TUNE")          # 100 family draws


@pytest.fixture(scope="module")
def open_specs():
    return [SC.generate(OPEN_STATIC, "fixed", s) for s in SAMPLE]


def test_open_clutter_layouts_follow_the_family_rules(open_specs):
    for sp in open_specs:
        assert 5 <= len(sp.layout) <= 7
        for r in sp.layout:
            cx, cy, hw, hh = r
            assert 0.25 <= hw <= 1.0 and 0.25 <= hh <= 1.0
            assert min(cx - hw, cy - hh, 10 - cx - hw, 10 - cy - hh) >= 1.0 - 1e-9
        for i, a in enumerate(sp.layout):
            for b in sp.layout[i + 1:]:
                assert _rect_rect_gap(a, b) >= 1.2 - 1e-6


def test_start_and_goal_are_in_free_space_with_clearance(open_specs):
    # 0.6 m, M0's wall margin: closer, the frozen controller's h < 0 (RS_DESIGN 14.1)
    for sp in open_specs:
        for p in (sp.start, sp.goal):
            assert min(p[0], p[1], 10 - p[0], 10 - p[1]) >= 0.6
            assert all(_point_rect_distance(p, r) >= 0.6 for r in sp.layout)


def test_start_goal_distance_is_inside_its_m0_bin(open_specs):
    bins = [(4.0, 5.5), (5.5, 7.0), (7.0, 8.5), (8.5, 10.5)]
    assert [tuple(b) for b in json.loads(PM.REPO.joinpath("config.json").read_text())
            ["start_goal"]["distance_bins"]] == bins == list(SC.DISTANCE_BINS)
    for sp in open_specs:
        lo, hi = bins[sp.distance_bin]
        assert lo <= np.linalg.norm(np.subtract(sp.goal, sp.start)) <= hi
    assert {sp.distance_bin for sp in open_specs} == {0, 1, 2, 3}


def test_a_clear_path_exists_at_the_clearance_radius(open_specs):
    for sp in open_specs:                            # 0.2 s per query
        orc = ShortestPathOracle(10.0, 0.45, sp.layout, resolution=200)
        assert orc.path_length(sp.start, sp.goal) is not None


def test_specs_record_feasibility_and_draw_counts(open_specs):
    for sp in open_specs:
        assert sp.layout_draws >= 1 and sp.bin_redraws >= 0
        assert sp.event_redraws >= 0 and sp.pedestrian_redraws >= 0
        assert sp.layout_draws + sp.bin_redraws + sp.event_redraws + sp.pedestrian_redraws \
            <= SC.RESAMPLE_CAP
        assert sp.generator_version == SC.GENERATOR_VERSION


# A layout whose free space is split in two by a full-height wall: never feasible.
SPLIT = ((5.0, 5.0, 0.2, 5.0),)


def _patch_draw(monkeypatch, draw, family="open_clutter"):
    fam = RF.FAMILIES[family]
    monkeypatch.setitem(RF.FAMILIES, family, fam._replace(draw=draw))


def test_resampling_skips_infeasible_draws_and_counts_them(monkeypatch):
    real = RF.FAMILIES["open_clutter"].draw
    calls = []

    def flaky(rng):
        calls.append(1)
        return None if len(calls) == 1 else (SPLIT, {}) if len(calls) == 2 else real(rng)
    _patch_draw(monkeypatch, flaky)
    sp = SC.generate(OPEN_STATIC, "fixed", DIAG[0])
    # the failed draw and the split layout are skipped and counted, like every later rejection
    assert sp.layout_draws == len(calls) >= 3 and sp.layout != SPLIT and sp.feasible


def test_the_resample_cap_raises_rather_than_returning_an_infeasible_spec(monkeypatch):
    _patch_draw(monkeypatch, lambda rng: (SPLIT, {}))
    with pytest.raises(SC.FeasibilityError):
        SC.generate(OPEN_STATIC, "fixed", DIAG[0])


def test_unknown_cells_and_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        SC.generate(RS.Cell("open_clutter", "earthquake"), "fixed", DIAG[0])
    with pytest.raises(ValueError):
        SC.generate(OPEN_STATIC, "wobbly", DIAG[0])
    with pytest.raises(ValueError):
        SC.generate(RS.ANCHOR, "fixed", DIAG[0])


# --------------------------------------------------------------------------- seam 1
SHORT = 60
TIMING = {"step_time_ms_mean", "step_time_ms_median", "step_time_ms_p95", "step_time_ms_max",
          "qp_time_ms_mean", "recovery_overhead_ms_mean", "episode_time_s", "mean_step_time",
          "step_times", "random_eval_ms_mean"}
#: RS-only labels and additions, absent from the HD record.
RS_ONLY = {"cell", "family", "obstacles", "motion", "spec", "scenario_events",
           "generator_version", "trigger_fired", "trigger_step"}


def _canon(rec):
    out = {k: v for k, v in rec.items() if k not in TIMING}
    out["events"] = [{k: v for k, v in e.items() if k != "t_recovery"}
                     for e in rec.get("events", [])]
    return json.dumps(out, sort_keys=True, default=float)


def _rs(cell, motion, seed, max_steps=SHORT, arm="astar_random"):
    env = RSScenarioEnv(motion)
    env.MAX_STEPS = max_steps
    return RH.run_episode(arm, cell, motion, seed, env=env)


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("seed", CS.seed_block("RANDOM_EXP_R3_VALID")[:3] + DIAG[:2])
def test_m0_anchor_through_the_scenario_env_is_bit_identical_to_the_hd_arm(seed, motion):
    env = m0_env(motion == "randomized")
    env.MAX_STEPS = SHORT
    ref = HDH.run_episode("astar_random", motion, seed, env=env)
    got = _rs(RS.ANCHOR, motion, seed)
    assert got["trajectory"] == ref["trajectory"]
    assert got["trace"] == ref["trace"]
    shared = (set(ref) & set(got)) - TIMING
    assert set(ref) - TIMING - {"condition"} <= shared
    assert _canon({k: ref[k] for k in shared}) == _canon({k: got[k] for k in shared})


def test_same_seed_reproduces_the_same_open_clutter_episode():
    seed = DIAG[1]
    a = _rs(OPEN_STATIC, "randomized", seed)
    assert _canon(a) == _canon(_rs(OPEN_STATIC, "randomized", seed))
    assert a["steps"] > 1


def test_both_motions_share_the_layout_start_and_goal():
    seed = DIAG[2]
    f, r = (_rs(OPEN_STATIC, m, seed, max_steps=5) for m in RS.MOTIONS)
    sp = SC.generate(OPEN_STATIC, "fixed", seed)
    assert f["spec"] == r["spec"] == sp.as_dict()
    assert f["start"] == r["start"] == pytest.approx(list(sp.start), abs=1e-6)
    assert f["goal"] == r["goal"] == pytest.approx(list(sp.goal), abs=1e-6)


def test_the_record_carries_the_cell_the_spec_and_the_event_log():
    rec = _rs(OPEN_STATIC, "fixed", DIAG[0], max_steps=5)
    assert rec["cell"] == ["open_clutter", "static"] and rec["motion"] == "fixed"
    assert rec["generator_version"] == SC.GENERATOR_VERSION
    assert rec["scenario_events"] == []
    assert rec["trigger_fired"] is False and rec["trigger_step"] is None
    assert len(rec["trace"]) == rec["steps"] == len(rec["trajectory"]) - 1
    assert "step_time_ms_p95" in rec


def test_the_policy_plans_on_the_generated_layout_and_the_env_holds_no_pedestrians(monkeypatch):
    seen = {}
    real = RH.HP.build_hd_arm

    def spy(arm, env, **kw):
        seen["static"] = list(env.static_obstacles)
        seen["n_dyn"] = env.n_dynamic_obstacles
        return real(arm, env, **kw)
    monkeypatch.setattr(RH.HP, "build_hd_arm", spy)
    rec = _rs(OPEN_STATIC, "fixed", DIAG[3], max_steps=3)
    sp = SC.generate(OPEN_STATIC, "fixed", DIAG[3])
    assert seen["static"] == [tuple(r) for r in sp.layout]
    assert seen["n_dyn"] == 0 and rec["planner_failed"] == 0


def test_the_anchor_restores_m0_after_a_generated_episode():
    env = RSScenarioEnv("fixed")
    env.MAX_STEPS = 3
    RH.run_episode("astar_random", OPEN_STATIC, "fixed", DIAG[0], env=env)
    got = RH.run_episode("astar_random", RS.ANCHOR, "fixed", DIAG[0], env=env)
    assert _canon(got) == _canon(_rs(RS.ANCHOR, "fixed", DIAG[0], max_steps=3))


def test_evaluation_refuses_a_non_frozen_controller(monkeypatch):
    from dr_control.fast_drccp import FastClfCbfDrccpController
    real = RH.HP.build_hd_arm

    def with_fast(arm, env, **kw):
        pol = real(arm, env, **kw)
        c = pol.ctrl
        pol.ctrl = FastClfCbfDrccpController(cbf_rate=c.rateh, clf_rate=c.rateV,
                                             wasserstein_r=c.wasserstein_r,
                                             epsilon=c.epsilon, k_v=c.k_v, max_v=c.max_v)
        return pol
    monkeypatch.setattr(RH.HP, "build_hd_arm", with_fast)
    with pytest.raises(TypeError):
        _rs(OPEN_STATIC, "fixed", DIAG[0], max_steps=3)


def test_the_harness_refuses_unknown_arms_cells_and_mismatched_envs():
    with pytest.raises(ValueError):
        _rs(OPEN_STATIC, "fixed", DIAG[0], arm="goal_random")
    with pytest.raises(ValueError):
        _rs(RS.Cell("open_clutter", "nope"), "fixed", DIAG[0])
    with pytest.raises(ValueError):
        RH.run_episode("astar_random", OPEN_STATIC, "fixed", DIAG[0],
                       env=RSScenarioEnv("randomized"))


# --------------------------------------------------------------------------- block runner

def _block(tmp_path, seeds, cell=OPEN_STATIC, workers=1, motions=("fixed",), tag=""):
    return RB.run_block("astar_random", cell, seeds, motions=motions, workers=workers,
                        checkpoint=tmp_path / "b.jsonl", traces=tmp_path / "b.traces.jsonl.gz",
                        max_steps=SHORT, tag=tag)


def test_block_records_are_light_and_traces_go_to_the_gz_file(tmp_path):
    seeds = DIAG[:2]
    recs = _block(tmp_path, seeds)["fixed"]
    assert [r["seed"] for r in recs] == seeds
    assert all("trace" not in r and "trajectory" not in r for r in recs)
    assert all(r["cell"] == list(OPEN_STATIC) for r in recs)
    with gzip.open(tmp_path / "b.traces.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert {(r["cond"], r["seed"]) for r in rows} == {("fixed", s) for s in seeds}
    assert all(len(r["trace"]) == rec["steps"] for r, rec in zip(rows, recs))


def test_block_equals_the_harness_episode_by_episode(tmp_path):
    rec = _block(tmp_path, DIAG[:1])["fixed"][0]
    ref = _rs(OPEN_STATIC, "fixed", DIAG[0])
    assert _canon(rec) == _canon({k: v for k, v in ref.items() if k not in RB.HEAVY})


def test_block_resume_skips_finished_episodes_and_reproduces_them(tmp_path):
    seeds = DIAG[:2]
    full = _block(tmp_path, seeds)
    ck = tmp_path / "b.jsonl"
    lines = ck.read_text().splitlines()
    ck.write_text("\n".join(lines[:2]) + "\n" + lines[2][:30])      # header + 1 row + torn row
    resumed = _block(tmp_path, seeds)
    assert [_canon(r) for r in full["fixed"]] == [_canon(r) for r in resumed["fixed"]]


def test_the_cell_is_part_of_the_block_fingerprint(tmp_path):
    _block(tmp_path, DIAG[:1])
    with pytest.raises(SystemExit):
        _block(tmp_path, DIAG[:1], cell=RS.ANCHOR)
    with pytest.raises(SystemExit):
        _block(tmp_path, DIAG[:1], tag="another-commit")
    fp = RB.fingerprint("astar_random", OPEN_STATIC, DIAG[:1], ("fixed",), SHORT, "")
    assert fp != RB.fingerprint("astar_random", RS.Cell("open_clutter", "dynamic"), DIAG[:1],
                                ("fixed",), SHORT, "")


def test_block_parallel_equals_serial_including_the_trace_digest(tmp_path):
    seeds = DIAG[:3]
    a = _block(tmp_path / "a", seeds, workers=1, motions=RS.MOTIONS)
    b = _block(tmp_path / "b", seeds, workers=3, motions=RS.MOTIONS)
    for m in RS.MOTIONS:
        assert [_canon(r) for r in a[m]] == [_canon(r) for r in b[m]]
    ga, gb = (tmp_path / d / "b.traces.jsonl.gz" for d in ("a", "b"))
    assert PM._sha(ga) == PM._sha(gb)
    assert not (tmp_path / "a" / "b.traces.partial.jsonl").exists()


def test_a_finished_block_loads_back_without_running(tmp_path, monkeypatch):
    seeds = DIAG[:2]
    run = _block(tmp_path, seeds, cell=RS.ANCHOR)
    monkeypatch.setattr(RB, "run_episode", None)        # loading must not run anything
    got, traces = RB.load_block("astar_random", RS.ANCHOR, seeds, motions=("fixed",),
                                checkpoint=tmp_path / "b.jsonl",
                                traces=tmp_path / "b.traces.jsonl.gz", tag="", max_steps=SHORT)
    assert [_canon(r) for r in got["fixed"]] == [_canon(r) for r in run["fixed"]]
    assert set(traces) == {("fixed", s) for s in seeds}


# =========================================================================== tickets 03-05
# The other four families (seam 2 on 50 draws each, seam 1 one episode each) and pedestrians.

FAMILY_SEEDS = DIAG                                   # 50 family draws per family
PED_COUNTS = {"open_clutter": 5, "rooms": 3, "aisles": 4, "corridors": 3, "dense_clutter": 4}


@pytest.fixture(scope="module")
def family_specs():
    return {f: [SC.generate(RS.Cell(f, "dynamic"), "fixed", s) for s in FAMILY_SEEDS]
            for f in RS.FAMILIES}


def _clear(points, layout):
    """Independent clearance: distance to the nearest rectangle or outer wall."""
    p = np.atleast_2d(np.asarray(points, float))
    out = np.min([p[:, 0], 10 - p[:, 0], p[:, 1], 10 - p[:, 1]], axis=0)
    for cx, cy, hw, hh in layout:
        q = np.stack([np.clip(p[:, 0], cx - hw, cx + hw), np.clip(p[:, 1], cy - hh, cy + hh)], 1)
        out = np.minimum(out, np.linalg.norm(p - q, axis=1))
    return out


def _polyline_clearance(poly, layout):
    pts = [np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
           for a, b in zip(poly[:-1], poly[1:]) for t in np.linspace(0, 1, 200)]
    return float(_clear(pts, layout).min())


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_every_family_is_deterministic_and_shared_by_motions_and_conditions(family):
    for seed in DIAG[:2]:
        a = SC.generate(RS.Cell(family, "dynamic"), "fixed", seed)
        assert a == SC.generate(RS.Cell(family, "dynamic"), "randomized", seed)
        s = SC.generate(RS.Cell(family, "static"), "fixed", seed)
        t = SC.generate(RS.Cell(family, "trigger_block"), "randomized", seed)
        u = SC.generate(RS.Cell(family, "trigger_spawn"), "fixed", seed)
        assert (s.obstacles, a.obstacles, t.obstacles, u.obstacles) == \
            ("static", "dynamic", "trigger_block", "trigger_spawn")
        for other in (s, t, u):
            assert {k: v for k, v in other.as_dict().items() if k != "obstacles"} == \
                {k: v for k, v in a.as_dict().items() if k != "obstacles"}
        assert s.active_pedestrians == () and len(a.active_pedestrians) == PED_COUNTS[family]
        assert t.active_pedestrians == a.active_pedestrians
        assert s.active_block is None and a.active_block is None
        assert t.active_block == t.block.rect and u.active_block is None
        assert u.active_spawn == u.spawn
        assert s.active_spawn is None and a.active_spawn is None and t.active_spawn is None


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_start_goal_clearance_bins_and_route_hold_in_every_family(family, family_specs):
    bins = [(4.0, 5.5), (5.5, 7.0), (7.0, 8.5), (8.5, 10.5)]
    for sp in family_specs[family]:
        assert _clear([sp.start, sp.goal], sp.layout).min() >= 0.6
        lo, hi = bins[sp.distance_bin]
        assert lo <= np.linalg.norm(np.subtract(sp.goal, sp.start)) <= hi
        assert sp.route[0] == sp.start and sp.route[-1] == sp.goal
        assert _polyline_clearance(sp.route, sp.layout) >= 0.45 - 1e-9
    for sp in family_specs[family][:10]:                 # the frozen oracle agrees (0.2 s each)
        orc = ShortestPathOracle(10.0, 0.45, sp.layout, resolution=200)
        assert orc.path_length(sp.start, sp.goal) == pytest.approx(sp.route_length, abs=1e-9)


# ---------------------------------------------------------------- 03: rooms
def _wall_lines(layout):
    """{(axis, coordinate): merged [lo, hi] intervals} of the 0.2 m thick wall pieces."""
    lines = {}
    for cx, cy, hw, hh in layout:
        if abs(hw - 0.1) < 1e-9:
            lines.setdefault(("v", round(cx, 9)), []).append((cy - hh, cy + hh))
        if abs(hh - 0.1) < 1e-9:
            lines.setdefault(("h", round(cy, 9)), []).append((cx - hw, cx + hw))
    merged = {}
    for key, iv in lines.items():
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1] + 1e-9:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        if out[0][0] <= 1e-9 or out[-1][1] >= 10 - 1e-9:     # a real wall reaches a boundary
            merged[key] = out
    return merged


def test_rooms_have_walls_with_one_doorway_per_segment_in_range(family_specs):
    n_rooms = set()
    for sp in family_specs["rooms"]:
        lines = _wall_lines(sp.layout)
        doors = [(key, a[1], b[0]) for key, iv in lines.items() for a, b in zip(iv, iv[1:])]
        k = sp.structure["n_rooms"]
        n_rooms.add(k)
        assert len(lines) == 2 and len(doors) == (4 if k == 4 else 3)
        for (axis, line), lo, hi in doors:
            assert 1.2 <= hi - lo <= 1.6
            c = (line, (lo + hi) / 2) if axis == "v" else ((lo + hi) / 2, line)
            for r in sp.layout:
                if min(r[2], r[3]) > 0.1 + 1e-9:                 # furniture
                    assert _point_rect_distance(c, r) >= 1.0 - 1e-9
        for (axis, line), iv in lines.items():
            assert 4.0 - 1e-9 <= line <= 6.0 + 1e-9
    assert n_rooms == {3, 4}


def test_rooms_doorways_keep_half_a_metre_from_every_wall_junction(family_specs):
    for sp in family_specs["rooms"]:
        lines = _wall_lines(sp.layout)
        for (axis, line), iv in lines.items():
            ends = [0.0, 10.0]                   # the outer walls, then crossing wall faces
            for (ax2, line2), iv2 in lines.items():
                if ax2 != axis and iv2[0][0] - 1e-9 <= line <= iv2[-1][1] + 1e-9:
                    ends += [line2 - 0.1, line2 + 0.1]
            for a, b in zip(iv, iv[1:]):
                lo, hi = a[1], b[0]
                assert min(abs(lo - e) for e in ends if e <= lo + 1e-9) >= 0.5 - 1e-9
                assert min(abs(e - hi) for e in ends if e >= hi - 1e-9) >= 0.5 - 1e-9


def test_rooms_start_and_goal_are_in_different_rooms(family_specs):
    for sp in family_specs["rooms"]:
        crossed = False
        for (axis, line), iv in _wall_lines(sp.layout).items():
            lo, hi = iv[0][0], iv[-1][1]                         # the wall's full extent
            i, j = (0, 1) if axis == "v" else (1, 0)
            a, b = sp.start, sp.goal
            if (a[i] - line) * (b[i] - line) < 0:
                t = (line - a[i]) / (b[i] - a[i])
                crossed |= lo <= a[j] + t * (b[j] - a[j]) <= hi
        assert crossed


# ---------------------------------------------------------------- 03: aisles
def _rows(sp):
    """(orientation, sorted row intervals across, along-extent, pieces per row) from the layout."""
    r0 = sp.layout[0]
    vertical = 0.3 - 1e-9 <= r0[2] <= 0.4 + 1e-9 and r0[3] > 0.4
    ax, al = (0, 1) if vertical else (1, 0)
    rows = {}
    for r in sp.layout:
        rows.setdefault(round(r[ax], 9), []).append((r[al] - r[2 + al], r[al] + r[2 + al],
                                                     r[2 + ax]))
    across = sorted((c - v[0][2], c + v[0][2]) for c, v in rows.items())
    along = (min(a for v in rows.values() for a, _, _ in v),
             max(b for v in rows.values() for _, b, _ in v))
    return vertical, across, along, rows


def test_aisles_are_parallel_rows_with_aisles_and_cross_aisles_in_range(family_specs):
    mids = set()
    for sp in family_specs["aisles"]:
        vertical, across, along, rows = _rows(sp)
        assert 3 <= len(across) <= 4
        assert all(0.6 - 1e-9 <= hi - lo <= 0.8 + 1e-9 for lo, hi in across)
        assert all(1.4 - 1e-9 <= b[0] - a[1] <= 2.0 + 1e-9 for a, b in zip(across, across[1:]))
        assert abs((across[0][0] + across[-1][1]) / 2 - 5.0) < 1e-9          # centred
        assert 1.4 - 1e-9 <= along[0] <= 2.0 + 1e-9 and 1.4 - 1e-9 <= 10 - along[1] <= 2.0 + 1e-9
        pieces = {len(v) for v in rows.values()}
        assert len(pieces) == 1                                  # a mid cross-aisle cuts every row
        mids.add(pieces.pop())
        for v in rows.values():
            v = sorted(v)
            assert all(1.4 - 1e-9 <= b[0] - a[1] <= 2.0 + 1e-9 for a, b in zip(v, v[1:]))
    assert mids == {1, 2}


def test_aisles_start_and_goal_are_in_different_aisles_or_opposite_ends(family_specs):
    for sp in family_specs["aisles"]:
        vertical, across, along, _ = _rows(sp)
        ax, al = (0, 1) if vertical else (1, 0)

        def aisle(p):
            if not along[0] <= p[al] <= along[1]:
                return None
            for i, (a, b) in enumerate(zip(across, across[1:])):
                if a[1] <= p[ax] <= b[0]:
                    return i
            return None
        i, j = aisle(sp.start), aisle(sp.goal)
        assert i is not None and j is not None
        assert i != j or abs(sp.start[al] - sp.goal[al]) >= 0.6 * (along[1] - along[0])


# ---------------------------------------------------------------- 04: corridors, dense clutter
def test_corridor_strips_are_in_range_and_everything_else_is_wall(family_specs):
    rng = np.random.default_rng(0)
    for sp in family_specs["corridors"][:20]:
        strips = list(sp.structure["strips"].values())
        kinds = set(sp.structure["strips"])
        assert {"spine", "connector", "stub"} <= kinds
        assert 2 <= sum(k.startswith("branch") for k in kinds) <= 3
        assert all(1.5 - 1e-9 <= 2 * min(s[2], s[3]) <= 2.0 + 1e-9 for s in strips)
        spine = sp.structure["strips"]["spine"]
        assert abs(2 * max(spine[2], spine[3]) - 10.0) < 1e-9       # runs the length of the world
        pts = rng.uniform(0, 10, (3000, 2))

        def inside(rects, p):
            return any(abs(p[0] - r[0]) < r[2] and abs(p[1] - r[1]) < r[3] for r in rects)
        for p in pts:
            assert inside(strips, p) != inside(sp.layout, p)


def test_corridor_networks_have_the_preregistered_topology(family_specs):
    counts = set()
    for sp in family_specs["corridors"]:
        s = sp.structure["strips"]
        branches = sorted(k for k in s if k.startswith("branch"))
        counts.add(len(branches))

        def touch(a, b):
            return _rect_rect_gap(s[a], s[b]) < 1e-9
        assert all(touch(b, "spine") for b in branches)                  # T-junctions
        joined = [b for b in branches if touch(b, "connector")]
        assert len(joined) == 2 and not touch("connector", "spine")      # the loop
        stub_on = [b for b in branches if touch(b, "stub")]
        assert len(stub_on) == 1 and not touch("stub", "spine")
        if len(branches) == 3:                                           # a true dead-end L
            assert stub_on[0] not in joined and not touch("stub", "connector")
    assert counts == {2, 3}


def _max_turn(route, arc=1.0):
    """Largest angle between the 1 m chords before and after any point of the route."""
    pts = np.asarray(route, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    def at(s):
        k = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        return pts[k] + (s - cum[k]) / seg[k] * (pts[k + 1] - pts[k])
    best = 0.0
    for s in np.arange(arc, cum[-1] - arc, 0.02):
        a, b = at(s) - at(s - arc), at(s + arc) - at(s)
        best = max(best, np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a)
                                                      / np.linalg.norm(b), -1, 1))))
    return best


def test_corridor_routes_turn_at_least_once(family_specs):
    for sp in family_specs["corridors"]:
        assert _max_turn(sp.route) >= 60.0 - 0.5


def test_dense_clutter_follows_its_rules(family_specs):
    for sp in family_specs["dense_clutter"]:
        assert 12 <= len(sp.layout) <= 18
        for cx, cy, hw, hh in sp.layout:
            assert 0.15 <= hw <= 0.4 and 0.15 <= hh <= 0.4
            assert min(cx - hw, cy - hh, 10 - cx - hw, 10 - cy - hh) >= 0.8 - 1e-9
        for i, a in enumerate(sp.layout):
            for b in sp.layout[i + 1:]:
                assert _rect_rect_gap(a, b) >= 1.1 - 1e-6


def test_every_family_draw_is_connected_at_the_clearance_radius(family_specs):
    for f in RS.FAMILIES:
        for sp in family_specs[f][:5]:
            orc = ShortestPathOracle(10.0, 0.45, sp.layout, resolution=200)
            assert ndimage.label(~orc.occupancy, structure=np.ones((3, 3)))[1] == 1


# ---------------------------------------------------------------- 05: pedestrian routes
@pytest.mark.parametrize("family", RS.FAMILIES)
def test_pedestrian_routes_follow_the_rules(family, family_specs):
    for sp in family_specs[family]:
        assert len(sp.pedestrians) == PED_COUNTS[family] and len(sp.pedestrians) + 1 <= 6
        for r in sp.pedestrians:
            assert r.speed == 0.675 <= 1.0
            assert np.linalg.norm(np.subtract(r.start, sp.start)) >= 1.5
            assert np.linalg.norm(np.subtract(r.start, sp.goal)) >= 1.0
            assert len(r.waypoints) == 8
            # the block zone is reserved in every condition (4.4)
            assert _clear([r.start, *r.waypoints], sp.pedestrian_layout).min() >= 0.45
            assert r.entry[0] == r.start and r.entry[-1] == r.waypoints[0]
            assert r.loop[0] == r.loop[-1] == r.waypoints[0]
            order = [r.loop.index(w) for w in r.waypoints]
            assert order == sorted(order)                        # the cycle, in order
            assert _polyline_clearance(r.entry, sp.pedestrian_layout) >= 0.45 - 1e-9
            assert _polyline_clearance(r.loop, sp.pedestrian_layout) >= 0.45 - 1e-9


# ---------------------------------------------------------------- 05: pedestrian motion
ROLLOUT = 1500                      # 150 s: every pedestrian walks its loop at least once


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("family", RS.FAMILIES)
def test_pedestrians_never_enter_walls_or_exceed_their_speed(family, motion, family_specs):
    for sp in family_specs[family][:2]:
        model = RP.PedestrianMotion(sp.pedestrians, sp.pedestrian_layout,
                                    randomized=motion == "randomized")
        rng = np.random.default_rng(sp.seed)
        pos = np.zeros((len(sp.pedestrians), 2), dtype=np.float32)
        vel = model.reset(rng, pos, 0.675)
        assert np.allclose(pos, [r.start for r in sp.pedestrians], atol=1e-5)
        walked = np.zeros(len(pos))
        floor = 0.45 if motion == "fixed" else 0.3
        for _ in range(ROLLOUT):
            before = pos.copy()
            pos, vel = model.step(rng, pos, vel, 0.1)
            step = np.linalg.norm(pos - before, axis=1)
            walked += step
            assert step.max() <= 0.0675 + 1e-5
            assert np.linalg.norm(vel, axis=1).max() <= 0.675 + 1e-4
            assert _clear(pos, sp.pedestrian_layout).min() >= floor - 1e-5
        assert walked.min() >= 0.5 * 0.0675 * ROLLOUT         # nobody parks


def test_fixed_pedestrians_draw_nothing_from_the_rng(family_specs):
    sp = family_specs["rooms"][0]
    model = RP.PedestrianMotion(sp.pedestrians, sp.layout, randomized=False)
    rng = np.random.default_rng(1)
    pos = np.zeros((3, 2), dtype=np.float32)
    vel = model.reset(rng, pos, 0.675)
    state = rng.bit_generator.state
    for _ in range(50):
        pos, vel = model.step(rng, pos, vel, 0.1)
    assert rng.bit_generator.state == state


# ---------------------------------------------------------------- seam 1: env and harness
def _ped_track(cell, motion, seed, actions):
    env = RSScenarioEnv(motion)
    env.install(SC.generate(cell, motion, seed))
    env.reset(seed=seed)
    out = [env.obstacle_positions.copy()]
    for a in actions:
        env.step(a)
        out.append(env.obstacle_positions.copy())
    return np.array(out)


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("family", RS.FAMILIES)
def test_pedestrians_do_not_depend_on_the_robot(family, motion):
    # paired arms see the same pedestrians: nothing the robot does reaches them
    cell = RS.Cell(family, "dynamic")
    rng = np.random.default_rng(3)
    a = _ped_track(cell, motion, DIAG[4], rng.uniform(-1, 1, (80, 2)))
    b = _ped_track(cell, motion, DIAG[4], np.zeros((80, 2)))
    assert a.shape == (81, PED_COUNTS[family], 2) and np.array_equal(a, b)


def test_the_env_runs_the_cells_pedestrians_and_none_in_static():
    seed = DIAG[5]
    env = RSScenarioEnv("randomized")
    for cond, n in (("dynamic", 5), ("static", 0), ("dynamic", 5)):
        sp = SC.generate(RS.Cell("open_clutter", cond), "randomized", seed)
        env.install(sp)
        obs, _ = env.reset(seed=seed)
        assert env.n_dynamic_obstacles == n and len(env.obstacle_positions) == n
        if n:
            assert np.allclose(env.obstacle_positions, [r.start for r in sp.pedestrians],
                               atol=1e-5)
            assert np.any(obs[28:] != 0)                     # the ground-truth block sees them
        else:
            assert not np.any(obs[28:])


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_a_baseline_episode_runs_in_every_family_on_its_layout(family, monkeypatch):
    seen = {}
    real = RH.HP.build_hd_arm

    def spy(arm, env, **kw):
        seen["static"] = list(env.static_obstacles)
        return real(arm, env, **kw)
    monkeypatch.setattr(RH.HP, "build_hd_arm", spy)
    for cond in ("static", "dynamic"):
        rec = _rs(RS.Cell(family, cond), "randomized", DIAG[6], max_steps=30)
        sp = SC.generate(RS.Cell(family, cond), "randomized", DIAG[6])
        assert seen["static"] == [tuple(r) for r in sp.layout]
        assert rec["planner_failed"] == 0 and rec["steps"] >= 1
        assert rec["spec"]["obstacles"] == cond


def test_same_seed_reproduces_a_dynamic_episode():
    cell = RS.Cell("corridors", "dynamic")
    a = _rs(cell, "randomized", DIAG[7], max_steps=40)
    b = _rs(cell, "randomized", DIAG[7], max_steps=40)
    assert _canon(a) == _canon(b)


# =========================================================================== ticket 06
# The triggered static block (RS_DESIGN 4.6). Every family draw carries the trigger and the block,
# so `family_specs` (the dynamic cells) holds them too: the checks below re-derive each rule from
# the rectangles and the route, independently of robustsuite.events.

def _arc(poly):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.asarray(poly), axis=0),
                                                           axis=1))])


def _at(poly, s):
    cum, p = _arc(poly), np.asarray(poly, float)
    return np.array([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


def _rect_dist(p, r):
    return float(_clear_rects([p], [r])[0])


def _clear_rects(points, layout):
    """Distance to the nearest rectangle only (no outer wall)."""
    p = np.atleast_2d(np.asarray(points, float))
    out = np.full(len(p), np.inf)
    for cx, cy, hw, hh in layout:
        q = np.stack([np.clip(p[:, 0], cx - hw, cx + hw), np.clip(p[:, 1], cy - hh, cy + hh)], 1)
        out = np.minimum(out, np.linalg.norm(p - q, axis=1))
    return out


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_trigger_is_on_the_route_between_30_and_70_percent(family, family_specs):
    for sp in family_specs[family]:
        tr, length = sp.trigger, _arc(sp.route)[-1]
        assert 0.3 <= tr.fraction <= 0.7 and tr.radius == 0.5
        assert tr.arc == pytest.approx(tr.fraction * length, abs=1e-9)
        assert np.allclose(tr.centre, _at(sp.route, tr.arc), atol=1e-9)
        assert np.linalg.norm(np.subtract(sp.start, tr.centre)) > 0.5    # the robot must enter


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_block_spans_the_passage_ahead_with_the_preregistered_size(family, family_specs):
    for sp in family_specs[family]:
        b, tr = sp.block, sp.trigger
        cx, cy, hw, hh = b.rect
        assert 1.5 <= b.offset <= 2.5
        p = _at(sp.route, tr.arc + b.offset)
        ax, o = b.axis, 1 - b.axis
        # the long axis is the world axis closest to the route's local perpendicular
        t = _at(sp.route, tr.arc + b.offset + 0.5) - _at(sp.route, tr.arc + b.offset - 0.5)
        assert abs(t[ax]) <= abs(t[o]) + 1e-12
        # 0.4 m thick, centred on the route point, so it crosses the route there
        assert (hw, hh)[o] == pytest.approx(0.2) and b.rect[o] == pytest.approx(p[o], abs=1e-9)
        assert _polyline_clearance(sp.route, [b.rect]) == 0.0
        # passage + 0.2 m into the wall on each side, at most 3.4 m long
        half = (hw, hh)[ax]
        assert 2 * half == pytest.approx(b.passage + 0.4) and b.passage <= 3.0
        assert 2 * half <= 3.4 + 1e-9
        # both passage edges touch a static obstacle or the outer wall; the passage is free
        edges = [b.rect[ax] - half + 0.2, b.rect[ax] + half - 0.2]
        for e in edges:
            q = np.array(p, float)
            q[ax] = e
            assert _clear([q], sp.layout)[0] == pytest.approx(0.0, abs=1e-9)
        inner = np.linspace(edges[0], edges[1], 60)[1:-1]
        line = np.repeat([p], len(inner), axis=0)
        line[:, ax] = inner
        assert _clear(line, sp.layout).min() > 0.0


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_block_keeps_clear_of_start_goal_and_the_trigger_disc(family, family_specs):
    for sp in family_specs[family]:
        r = sp.block.rect
        assert _rect_dist(sp.start, r) >= 0.45 and _rect_dist(sp.goal, r) >= 0.45
        # robot centre inside the disc => its body never touches the block as it appears
        assert _rect_dist(sp.trigger.centre, r) >= 0.5 + 0.3


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_a_clear_path_still_exists_with_the_block_added(family, family_specs):
    # seam 2, on 50 draws per family: the frozen oracle rebuilt on the layout plus the block
    for sp in family_specs[family]:
        orc = ShortestPathOracle(10.0, 0.45, sp.layout + (sp.block.rect,), resolution=200)
        assert orc.path_length(sp.trigger.centre, sp.goal) is not None
        assert orc.path_length(sp.start, sp.goal) is not None


def test_the_incremental_oracle_equals_the_oracle_rebuilt_with_the_block(family_specs):
    from robustsuite.geometry import with_rect
    for fam in RS.FAMILIES:
        sp = family_specs[fam][0]
        grown = with_rect(SC.oracle(sp.layout), sp.block.rect)
        fresh = ShortestPathOracle(10.0, 0.45, sp.layout + (sp.block.rect,), resolution=200)
        assert np.array_equal(grown.occupancy, fresh.occupancy)
        assert grown.static_obstacles == fresh.static_obstacles


def test_a_failed_event_draw_redraws_the_layout_and_counts(monkeypatch):
    calls = {"n": 0}
    real = SC.draw_events

    def flaky(*a, **kw):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(*a, **kw)
    monkeypatch.setattr(SC, "draw_events", flaky)
    sp = SC.generate(RS.Cell("open_clutter", "trigger_block"), "fixed", DIAG[0])
    assert sp.event_redraws >= 1 and sp.layout_draws >= 2
    monkeypatch.setattr(SC, "draw_events", lambda *a, **kw: None)
    with pytest.raises(SC.FeasibilityError):
        SC.generate(RS.Cell("open_clutter", "trigger_block"), "fixed", DIAG[0])


# ---------------------------------------------------------------- seam 1: the env fires it
BLOCK_CELL = RS.Cell("open_clutter", "trigger_block")


def _scenario_env(cell, motion, seed):
    """The spec for (cell, motion, seed) and a scenario env reset on it."""
    sp = SC.generate(cell, motion, seed)
    env = RSScenarioEnv(motion)
    env.install(sp)
    env.reset(seed=seed)
    return sp, env


def _drive(env, sp, wait=0):
    """Wait `wait` steps, then drive the robot exactly along the spec's route at 1 m/s, ignoring
    termination. Returns (robot position, n_dynamic_obstacles, obstacle positions) for every
    state, indexed by step number: entry 0 is the state after reset."""
    def state():
        return (env.agent_position.copy(), env.n_dynamic_obstacles,
                env.obstacle_positions.copy())
    out = [state()]
    for _ in range(wait):
        env.step(np.zeros(2))
        out.append(state())
    pts, k = np.asarray(sp.route, float), 1
    while k < len(pts):
        d = pts[k] - env.agent_position
        dist = float(np.linalg.norm(d))
        if dist < 1e-4:                                  # float32 positions
            k += 1
            continue
        env.step(d / dist * min(1.0, dist / (env.MAX_SPEED * env.dt)))
        out.append(state())
    return out


def _entry_step(states, sp):
    """The first step on which the robot's centre is inside the trigger disc."""
    c = np.asarray(sp.trigger.centre)
    return next(k for k, (q, _, _) in enumerate(states)
                if k and np.linalg.norm(q - c) <= sp.trigger.radius)


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("seed", DIAG[:3])
def test_the_block_fires_exactly_when_the_robot_enters_the_trigger(seed, motion):
    sp, env = _scenario_env(BLOCK_CELL, motion, seed)
    layout = [tuple(r) for r in sp.layout]
    before = env.static_obstacles
    states = _drive(env, sp)
    k = _entry_step(states, sp)
    assert env.scenario_events == [
        {"event": "trigger_fired", "step": k, "position": [float(v) for v in states[k][0]]},
        {"event": "block_added", "step": k, "block": list(sp.block.rect)}]   # once only
    assert env.trigger_step == k
    assert env.static_obstacles == layout + [tuple(sp.block.rect)]
    assert before == layout                          # a new list: the old one never grew


@pytest.mark.parametrize("cond", ("static", "dynamic"))
def test_the_block_never_fires_outside_its_condition(cond):
    sp, env = _scenario_env(RS.Cell("open_clutter", cond), "fixed", DIAG[0])
    _drive(env, sp)
    assert env.scenario_events == [] and env.trigger_step is None
    assert env.static_obstacles == [tuple(r) for r in sp.layout]


def _lidar(env, rects, circles=None):
    """The normalised LiDAR block of the observation, cast independently on `rects` and
    `circles` (default: the env's pedestrians)."""
    from robot_env.lidar_core import cast_rays
    circles = env.obstacle_positions if circles is None else circles
    rays = cast_rays(env.agent_position, n_rays=env.N_LIDAR_RAYS, lidar_range=env.LIDAR_RANGE,
                     world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS, rects=rects,
                     circles=circles, circle_radius=env.OBSTACLE_RADIUS)
    return np.clip(rays / env.LIDAR_RANGE, -1, 1)


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_fired_block_is_in_lidar_and_collisions_from_the_firing_step(family):
    seed = DIAG[1]
    sp = SC.generate(RS.Cell(family, "trigger_block"), "fixed", seed)
    layout, block = [tuple(r) for r in sp.layout], tuple(sp.block.rect)
    on_block = _at(sp.route, sp.trigger.arc + sp.block.offset)     # a route point inside it
    facing = _at(sp.route, sp.trigger.arc + sp.block.offset - 0.8)  # 0.6 m before its face
    env = RSScenarioEnv("fixed")
    env.install(sp)

    # before firing: the robot can stand where the block will be, and LiDAR does not see it
    env.reset(seed=seed)
    env.agent_position = on_block.astype(np.float32)
    obs, _, _, _, info = env.step(np.zeros(2))
    assert info["collision_type"] != "static" and env.trigger_step is None
    env.agent_position = facing.astype(np.float32)
    obs, *_ = env.step(np.zeros(2))
    assert np.array_equal(obs[4:28], _lidar(env, layout))

    # the firing step already sees it
    env.reset(seed=seed)
    env.agent_position = np.asarray(sp.trigger.centre, dtype=np.float32)
    obs, _, _, _, info = env.step(np.zeros(2))
    assert env.trigger_step == 1 and info["collision_type"] != "static"
    assert np.array_equal(obs[4:28],
                          _lidar(env, layout + [block]))
    env.agent_position = facing.astype(np.float32)
    obs, *_ = env.step(np.zeros(2))
    with_block = _lidar(env, layout + [block])
    assert np.array_equal(obs[4:28], with_block)
    assert not np.array_equal(with_block, _lidar(env, layout))
    env.agent_position = on_block.astype(np.float32)
    _, _, term, _, info = env.step(np.zeros(2))
    assert term and info["collision_type"] == "static"


@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_the_block_zone_keeps_pedestrians_identical_across_conditions(motion):
    # the block is the pedestrians' map in every condition, fired or not
    seed = DIAG[2]
    tracks = []
    for cond in ("dynamic", "trigger_block"):
        sp, env = _scenario_env(RS.Cell("open_clutter", cond), motion, seed)
        tracks.append(np.array([o for *_, o in _drive(env, sp)]))
    assert env.trigger_step is not None
    assert np.array_equal(*tracks)


# ---------------------------------------------------------------- seam 1: the harness
FIRES = (RS.Cell("open_clutter", "trigger_block"), DIAG[5])    # the baseline fires at step 22


def _trigger_elsewhere(sp):
    """The spec with its trigger moved where the robot never goes."""
    from dataclasses import replace
    return replace(sp, trigger=replace(sp.trigger, centre=(-50.0, -50.0), fraction=0.5,
                                       radius=0.1))


def _perturbed(sp, *, trigger=True, rect=True):
    """The spec with its block fields changed (the rectangle moved and resized if `rect`), and
    (if `trigger`) its trigger moved where the robot never goes."""
    from dataclasses import replace
    cx, cy, hw, hh = sp.block.rect
    moved = (cx + 0.37, cy - 0.21, hw + 0.3, hh + 0.1) if rect else sp.block.rect
    out = replace(sp, block=replace(sp.block, rect=moved, offset=sp.block.offset + 0.3,
                                    passage=sp.block.passage + 0.5, axis=1 - sp.block.axis))
    return _trigger_elsewhere(out) if trigger else out


@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_actions_before_firing_do_not_see_the_trigger_or_block_fields(motion, monkeypatch):
    # the information boundary: a policy acts on its observation and the layout only. The block
    # rectangle is also the randomized pedestrians' reserved zone, so moving it changes their
    # world; it is moved under fixed motion only, where pedestrians never read the layout.
    cell, seed = FIRES
    real = _rs(cell, motion, seed, max_steps=40)
    k = real["trigger_step"]
    assert real["trigger_fired"] and 1 <= k < 40
    sp = SC.generate(cell, motion, seed)
    for trig in (True, False):
        monkeypatch.setattr(RH, "cell_spec", lambda *a, t=trig: _perturbed(
            sp, trigger=t, rect=motion == "fixed"))
        other = _rs(cell, motion, seed, max_steps=40)
        # positions 0..k are the result of the actions chosen before the block appeared
        assert other["trajectory"][:k + 1] == real["trajectory"][:k + 1]
        assert json.dumps(other["trace"][:k], default=float) == \
            json.dumps(real["trace"][:k], default=float)
        assert other["trigger_fired"] is (not trig)


def test_the_policy_map_never_contains_the_block_and_the_record_says_when_it_fired(monkeypatch):
    cell, seed = FIRES
    pols, seen = [], {}
    real = RH.HP.build_hd_arm

    def spy(arm, env, **kw):
        seen["static"] = list(env.static_obstacles)
        pols.append(real(arm, env, **kw))
        return pols[-1]
    monkeypatch.setattr(RH.HP, "build_hd_arm", spy)
    rec = _rs(cell, "fixed", seed, max_steps=40)
    sp = SC.generate(cell, "fixed", seed)
    layout = [tuple(r) for r in sp.layout]
    assert rec["trigger_fired"] is True and rec["trigger_step"] >= 1
    fired = [e for e in rec["scenario_events"] if e["event"] == "trigger_fired"]
    assert [e["step"] for e in fired] == [rec["trigger_step"]]
    assert fired[0]["position"] == pytest.approx(rec["trajectory"][rec["trigger_step"]])
    assert seen["static"] == layout
    planner = pols[0].planner
    assert planner.static_obstacles == layout
    assert np.array_equal(planner.occupancy,
                          ShortestPathOracle(10.0, 0.3, layout).occupancy)


# =========================================================================== ticket 07
# The triggered spawn (RS_DESIGN 4.6, 14.3). Every family draw carries it, so `family_specs`
# holds it too. The spawn ignores the block zone (researcher decision, 14.3): its checks are
# against the static layout only.

def _polyline_distance(poly, q):
    """Distance from q to the polyline."""
    p = np.asarray(poly, float)
    best = np.inf
    for a, b in zip(p[:-1], p[1:]):
        d = b - a
        t = 0.0 if not d @ d else float(np.clip((q - a) @ d / (d @ d), 0, 1))
        best = min(best, float(np.linalg.norm(q - (a + t * d))))
    return best


def _inside_any(points, layout):
    return bool((_clear_rects(points, layout) == 0.0).any())


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_spawn_starts_occluded_2_to_3_m_from_the_trigger(family, family_specs):
    for sp in family_specs[family]:
        c, p = np.asarray(sp.trigger.centre), np.asarray(sp.spawn.start)
        dist = float(np.linalg.norm(p - c))
        assert 2.0 <= dist <= 3.0
        assert dist - sp.trigger.radius >= 1.5          # from every robot position that fires
        assert _clear([p], sp.layout)[0] >= 0.45
        assert np.linalg.norm(p - sp.start) >= 1.5 and np.linalg.norm(p - sp.goal) >= 1.0
        assert _inside_any(c + np.linspace(0, 1, 2001)[:, None] * (p - c), sp.layout)
        assert sp.spawn.route.speed == 0.675 <= 1.0


def test_the_spawn_variants_are_a_fair_coin_per_seed(family_specs):
    got = [sp.spawn.variant for f in RS.FAMILIES for sp in family_specs[f]]
    assert set(got) == {"crossing", "head_on"}
    assert 0.38 <= got.count("crossing") / len(got) <= 0.62      # 250 draws, ~3.8 sd


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_crossing_entry_crosses_the_route_ahead(family, family_specs):
    specs = [sp for sp in family_specs[family] if sp.spawn.variant == "crossing"]
    assert specs
    for sp in specs:
        sw, arc = sp.spawn, sp.trigger.arc
        tgt = np.asarray(sw.target)
        ahead = _at(sp.route, arc + np.linspace(1.0, 2.0, 2001)).T
        assert np.linalg.norm(ahead - tgt, axis=1).min() < 1e-3        # 1-2 m past the trigger
        i = [tuple(q) for q in sw.scripted].index(tuple(sw.target))
        assert sw.scripted[0] == sw.start and len(sw.scripted) - i <= 2
        if len(sw.scripted) - i == 1:
            continue                                                   # no room to walk on
        end = np.asarray(sw.scripted[-1])
        step = end - tgt
        ax = int(np.argmax(np.abs(step)))
        assert abs(step[1 - ax]) < 1e-12 and np.abs(step).max() <= 2.0 + 1e-9
        s = float(np.argmin(np.linalg.norm(ahead - tgt, axis=1))) / 2000 + 1.0 + arc
        t = _at(sp.route, s + 0.5) - _at(sp.route, s - 0.5)
        assert abs(t[ax]) <= abs(t[1 - ax]) + 1e-9                     # across the route
        arrival = tgt - np.asarray(sw.scripted[i - 1])
        assert step[ax] * arrival[ax] >= 0                             # on the way it came
        if np.abs(step).max() < 2.0 - 1e-9:                            # stopped 0.45 short
            u = step / np.linalg.norm(step)
            probe = end + np.arange(0.0, 1.0, 0.002)[:, None] * u
            hit = np.nonzero((_clear_rects(probe, sp.layout) == 0.0)
                             | (probe.min(axis=1) <= 0) | (probe.max(axis=1) >= 10))[0]
            assert 0.45 - 0.003 <= hit[0] * 0.002 <= 0.45 + 0.003


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_head_on_entry_comes_back_along_the_route(family, family_specs):
    specs = [sp for sp in family_specs[family] if sp.spawn.variant == "head_on"]
    assert specs
    for sp in specs:
        sw, arc = sp.spawn, sp.trigger.arc
        assert np.allclose(sw.target, _at(sp.route, arc + 2.0), atol=1e-9)
        i = [tuple(q) for q in sw.scripted].index(tuple(sw.target))
        walk = np.asarray(sw.scripted[i:])
        assert np.allclose(walk[-1], _at(sp.route, arc - 1.0), atol=1e-9)   # 3 m back
        assert max(_polyline_distance(sp.route, q) for q in walk) < 1e-9
        assert _arc(walk)[-1] == pytest.approx(3.0, abs=1e-9)


@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_spawn_walk_keeps_clear_and_joins_its_own_cycle(family, family_specs):
    for sp in family_specs[family]:
        sw, r = sp.spawn, sp.spawn.route
        assert _polyline_clearance(sw.scripted, sp.layout) >= 0.45 - 1e-9
        assert r.start == sw.start and r.entry[:len(sw.scripted)] == sw.scripted
        assert r.entry[-1] == r.waypoints[0] and r.loop[0] == r.loop[-1] == r.waypoints[0]
        assert len(r.waypoints) == 8
        assert _polyline_clearance(r.entry, sp.layout) >= 0.45 - 1e-9
        assert _polyline_clearance(r.loop, sp.layout) >= 0.45 - 1e-9
        assert len(sp.pedestrians) + 1 <= 6                             # the six slots


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("family", RS.FAMILIES)
def test_the_spawned_pedestrian_never_enters_walls(family, motion, family_specs):
    for sp in family_specs[family][:2]:
        model = RP.PedestrianMotion([sp.spawn.route], sp.layout,
                                    randomized=motion == "randomized")
        rng = np.random.default_rng(sp.seed)
        pos = np.zeros((1, 2), dtype=np.float32)
        vel = model.reset(rng, pos, 0.675)
        floor = 0.45 if motion == "fixed" else 0.3
        for _ in range(ROLLOUT):
            before = pos.copy()
            pos, vel = model.step(rng, pos, vel, 0.1)
            assert np.linalg.norm(pos - before) <= 0.0675 + 1e-5
            assert _clear(pos, sp.layout).min() >= floor - 1e-5


# ---------------------------------------------------------------- seam 1: the env fires it
SPAWN_CELL = RS.Cell("open_clutter", "trigger_spawn")


@pytest.mark.parametrize("motion", RS.MOTIONS)
@pytest.mark.parametrize("seed", DIAG[:3])
def test_the_spawn_fires_exactly_when_the_robot_enters_the_trigger(seed, motion):
    sp, env = _scenario_env(SPAWN_CELL, motion, seed)
    n = len(sp.pedestrians)
    states = _drive(env, sp)
    k = _entry_step(states, sp)
    assert env.scenario_events == [
        {"event": "trigger_fired", "step": k, "position": [float(v) for v in states[k][0]]},
        {"event": "spawned", "step": k, "position": list(sp.spawn.start),
         "variant": sp.spawn.variant}]
    assert all(m == n and len(o) == n for _, m, o in states[:k])      # absent before
    assert all(m == n + 1 and len(o) == n + 1 for _, m, o in states[k:])
    first = states[k][2][-1]                                           # moved on the firing step
    assert 0.0 < np.linalg.norm(first - np.asarray(sp.spawn.start)) <= 0.0675 + 1e-5
    assert env.static_obstacles == [tuple(r) for r in sp.layout]       # no block in this cell


@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_the_regular_pedestrians_are_the_dynamic_cells_through_a_spawn(motion):
    seed = DIAG[2]
    sp, env = _scenario_env(SPAWN_CELL, motion, seed)
    a = [o for *_, o in _drive(env, sp)]
    assert env.trigger_step is not None
    dsp, denv = _scenario_env(RS.Cell("open_clutter", "dynamic"), motion, seed)
    b = [o for *_, o in _drive(denv, dsp)]
    n = len(sp.pedestrians)
    assert len(a) == len(b) and all(np.array_equal(x[:n], y) for x, y in zip(a, b))


@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_the_spawned_pedestrian_walks_the_same_whenever_it_fires(motion):
    # paired arms fire at different steps; the spawn's own walk is the same from its firing on
    seed = DIAG[2]
    walks = []
    for wait in (0, 17):
        sp, env = _scenario_env(SPAWN_CELL, motion, seed)
        obst = [o for *_, o in _drive(env, sp, wait=wait)]
        k = env.trigger_step
        walks.append(np.array([o[-1] for o in obst[k:k + 60]]))
    m = min(len(w) for w in walks)
    assert m >= 20 and np.array_equal(walks[0][:m], walks[1][:m])


def test_the_spawn_is_in_lidar_collisions_and_observation_only_after_firing():
    seed = DIAG[1]
    sp, env = _scenario_env(SPAWN_CELL, "fixed", seed)
    layout = [tuple(r) for r in sp.layout]
    # before firing: the robot stands on the spawn start, and nothing is there
    env.agent_position = np.asarray(sp.spawn.start, dtype=np.float32)
    obs, *_ = env.step(np.zeros(2))
    assert env.trigger_step is None and len(env.obstacle_positions) == len(sp.pedestrians)
    assert np.array_equal(obs[4:28], _lidar(env, layout))
    # the firing step: the spawned pedestrian is the last slot, in LiDAR and the observation
    env.agent_position = np.asarray(sp.trigger.centre, dtype=np.float32)
    obs, _, _, _, info = env.step(np.zeros(2))
    assert env.trigger_step == 2 and len(env.obstacle_positions) == len(sp.pedestrians) + 1
    assert np.array_equal(obs[4:28], _lidar(env, layout))
    # 0.9 m further along its own entry path: clear of the static layout by construction
    env.agent_position = _at(sp.spawn.route.entry, 0.0675 + 0.9).astype(np.float32)
    obs, *_ = env.step(np.zeros(2))
    without = _lidar(env, layout, env.obstacle_positions[:-1])
    assert not np.array_equal(obs[4:28], without)                      # LiDAR sees it
    env.agent_position = env.obstacle_positions[-1].copy()
    _, _, term, _, info = env.step(np.zeros(2))
    assert term and info["collision_type"] == "dynamic"


def test_the_spawn_never_fires_outside_its_condition():
    for cond in ("dynamic", "trigger_block"):
        sp, env = _scenario_env(RS.Cell("open_clutter", cond), "fixed", DIAG[0])
        _drive(env, sp)
        assert not [e for e in env.scenario_events if e["event"] == "spawned"]
        assert env.n_dynamic_obstacles == len(sp.pedestrians)


# ---------------------------------------------------------------- seam 1: the harness
def _perturbed_spawn(sp):
    """The spec with every spawn field changed and the trigger moved where the robot never goes.
    The spawn is not in the world before firing, so this holds for both motions."""
    from dataclasses import replace
    sw = sp.spawn
    shift = lambda poly: tuple((x + 0.31, y - 0.17) for x, y in poly)
    route = replace(sw.route, start=shift([sw.route.start])[0], entry=shift(sw.route.entry),
                    loop=shift(sw.route.loop), waypoints=shift(sw.route.waypoints))
    moved = replace(sw, start=route.start, target=shift([sw.target])[0],
                    scripted=shift(sw.scripted), route=route,
                    variant="head_on" if sw.variant == "crossing" else "crossing")
    return _trigger_elsewhere(replace(sp, spawn=moved))


@pytest.mark.parametrize("motion", RS.MOTIONS)
def test_actions_before_firing_do_not_see_the_spawn_fields(motion, monkeypatch):
    cell, seed = SPAWN_CELL, FIRES[1]
    real = _rs(cell, motion, seed, max_steps=40)
    k = real["trigger_step"]
    assert real["trigger_fired"] and 1 <= k < 40
    assert [e["event"] for e in real["scenario_events"]] == ["trigger_fired", "spawned"]
    sp = SC.generate(cell, motion, seed)
    monkeypatch.setattr(RH, "cell_spec", lambda *a: _perturbed_spawn(sp))
    other = _rs(cell, motion, seed, max_steps=40)
    assert other["trigger_fired"] is False
    assert other["trajectory"][:k + 1] == real["trajectory"][:k + 1]
    assert json.dumps(other["trace"][:k], default=float) == \
        json.dumps(real["trace"][:k], default=float)
