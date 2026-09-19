"""RS Experiment 1: robust A*-Random on a fresh map suite.

Pre-registration: MD_files/robustsuite/RS_DESIGN.md. Decision: docs/adr/0002.
"""
from __future__ import annotations

import ast
import shutil

import pytest

from continuation import seeds as CS
from continuation.pursuit.config import PURSUIT_BLOCKS
from experiments.robustsuite import protect_manifest as PM
from highdim import seeds as HDS
from robustsuite import seeds as RS

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
