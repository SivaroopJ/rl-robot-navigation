"""HD Experiment 1 (high-dimensionality axis): PPO subgoal source replaces A* in Random-CLF-DR-CBF.

Pre-registration: MD_files/highdim/HD_DESIGN.md. Decision: docs/adr/0001.
"""
from __future__ import annotations

import shutil

import pytest

from experiments.highdim import protect_manifest as PM

# The frozen side of the experiment, by role. Each must be in the manifest.
MUST_COVER = [
    "optional_navigation/planner.py",                 # A* + carrot follower (the baseline)
    "evaluation/shortest_path.py",                    # grid the planner inherits
    "dr_control/drccp_controller.py",                 # CLF, DR-CCP QP
    "dr_control/estimated_cbf.py",                    # LiDAR barrier source
    "dr_control/velocity_tracker.py",
    "dr_control/policy.py",                           # frozen predict(): the follower hook
    "dr_control/fast_drccp.py",                       # used unmodified, training only
    "continuation/policy.py",                         # tuned policy + arm construction
    "continuation/random_recovery.py",
    "continuation/seeds.py",
    "robot_env/robot_nav_env.py",                     # dynamics, collision, reward, observation
    "config.json",
    "experiments/exp6_ppo_comparison.py",             # make_env / episode_record reused
    "results/week6_continuation/H2/tuned_frozen.json",
    "results/week6_continuation/R3/random_frozen.json",
]


def test_frozen_files_are_untouched():
    modified, missing = PM.verify()
    assert not modified and not missing, (modified, missing)


@pytest.mark.parametrize("rel", MUST_COVER)
def test_manifest_covers_the_frozen_side(rel):
    assert rel in PM.read()


def test_manifest_never_covers_new_highdim_code():
    assert not [k for k in PM.read() if "highdim" in k]


def test_verify_detects_a_modified_frozen_file(tmp_path):
    for rel in PM.read():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PM.REPO / rel, dst)
    assert PM.verify(root=tmp_path) == ([], [])

    target = tmp_path / "optional_navigation/planner.py"
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    (tmp_path / "config.json").unlink()
    modified, missing = PM.verify(root=tmp_path)
    assert modified == ["optional_navigation/planner.py"]
    assert missing == ["config.json"]
