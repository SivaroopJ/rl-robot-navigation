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

from continuation import seeds as CS
from continuation.pursuit.config import PURSUIT_BLOCKS
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import make_env as m0_env
from experiments.robustsuite import protect_manifest as PM
from highdim import harness as HDH
from highdim import seeds as HDS
from robustsuite import blocks as RB
from robustsuite import harness as RH
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
    for sp in open_specs:
        for p in (sp.start, sp.goal):
            assert min(p[0], p[1], 10 - p[0], 10 - p[1]) >= 0.45
            assert all(_point_rect_distance(p, r) >= 0.45 for r in sp.layout)


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
        assert sp.layout_draws + sp.bin_redraws <= SC.RESAMPLE_CAP
        assert sp.generator_version == SC.GENERATOR_VERSION


# A layout whose free space is split in two by a full-height wall: never feasible.
SPLIT = ((5.0, 5.0, 0.2, 5.0),)


def test_resampling_skips_infeasible_draws_and_counts_them(monkeypatch):
    real = SC.FAMILY_DRAWS["open_clutter"]
    calls = []

    def flaky(rng):
        calls.append(1)
        return None if len(calls) == 1 else SPLIT if len(calls) == 2 else real(rng)
    monkeypatch.setitem(SC.FAMILY_DRAWS, "open_clutter", flaky)
    sp = SC.generate(OPEN_STATIC, "fixed", DIAG[0])
    assert sp.layout_draws == 3 and sp.layout != SPLIT and sp.feasible


def test_the_resample_cap_raises_rather_than_returning_an_infeasible_spec(monkeypatch):
    monkeypatch.setitem(SC.FAMILY_DRAWS, "open_clutter", lambda rng: SPLIT)
    with pytest.raises(SC.FeasibilityError):
        SC.generate(OPEN_STATIC, "fixed", DIAG[0])


def test_unbuilt_cells_and_bad_arguments_are_refused():
    with pytest.raises(NotImplementedError):
        SC.generate(RS.Cell("open_clutter", "dynamic"), "fixed", DIAG[0])
    with pytest.raises(NotImplementedError):
        SC.generate(RS.Cell("rooms", "static"), "fixed", DIAG[0])
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
           "generator_version"}


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
