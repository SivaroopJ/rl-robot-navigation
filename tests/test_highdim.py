"""HD Experiment 1 (high-dimensionality axis): PPO subgoal source replaces A* in Random-CLF-DR-CBF.

Pre-registration: MD_files/highdim/HD_DESIGN.md. Decision: docs/adr/0001.
"""
from __future__ import annotations

import ast
import json
import shutil

import numpy as np
import pytest

from continuation import harness as CH
from continuation import seeds as CS
from dr_control.drccp_controller import ClfCbfDrccpController
from experiments.exp6_ppo_comparison import make_env
from experiments.highdim import protect_manifest as PM
from highdim import harness as H
from highdim import policy as HP
from highdim import seeds as HS
from optional_navigation.planner import StaticMapPlanner

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


# =========================================================================== ticket 02
# Seed blocks + seam 1 (episode harness) for the floor arm: goal-only + Random.

# Episode-timing keys: wall-clock, legitimately different between identical runs.
TIMING = {"step_time_ms_mean", "step_time_ms_median", "step_time_ms_p95", "step_time_ms_max",
          "qp_time_ms_mean", "recovery_overhead_ms_mean", "episode_time_s", "mean_step_time",
          "step_times", "random_eval_ms_mean"}
SHORT = 60            # steps; truncates test episodes so the suite stays fast
EVENT_SEED_INDEX = 5  # HD_DEV[5] has Random events inside SHORT steps in both conditions


def _episode(kind="goal_random", condition="fixed", seed=None, max_steps=SHORT):
    seed = HS.seed_block("HD_DEV")[3] if seed is None else seed
    env = make_env(condition == "randomized")
    env.MAX_STEPS = max_steps              # instance attribute only; the env file is untouched
    return H.run_episode(kind, condition, seed, env=env)


def _canon(rec):
    out = {k: v for k, v in rec.items() if k not in TIMING}
    out["events"] = [{k: v for k, v in e.items() if k != "t_recovery"}
                     for e in rec.get("events", [])]
    return json.dumps(out, sort_keys=True, default=float)


# --------------------------------------------------------------------------- seed blocks
def test_hd_blocks_have_the_preregistered_ranges():
    assert HS.seed_block("HD_DEV") == list(range(14_000_000, 14_000_100))
    assert HS.seed_block("HD_FINAL", allow_final=True) == list(range(14_100_000, 14_100_200))
    assert HS.train_env_seed(run=2, env_index=7) == 14_502_007


def test_hd_blocks_are_disjoint_from_every_earlier_reservation():
    HS.check_disjoint()
    hd = set(HS.seed_block("HD_DEV")) | set(HS.seed_block("HD_FINAL", allow_final=True))
    for base, size in CS.BLOCKS.values():
        assert not hd & set(range(base, base + CS.BLOCK_STRIDE))
    for lo, hi in CS.FORBIDDEN_RANGES + [(11_000_000, 14_000_000)]:
        assert all(not lo <= s < hi for s in hd)


def test_train_seed_range_is_clear_of_dev_final_and_earlier_blocks():
    lo = HS.train_env_seed(run=0, env_index=0)
    hi = HS.train_env_seed(run=HS.TRAIN_RUNS - 1, env_index=HS.TRAIN_ENVS - 1)
    assert (lo, hi) == (14_500_000, 14_599_999)
    for s in HS.seed_block("HD_DEV") + HS.seed_block("HD_FINAL", allow_final=True):
        assert not lo <= s <= hi


def test_train_seeds_stay_inside_their_range():
    with pytest.raises(ValueError):
        HS.train_env_seed(run=100, env_index=0)
    with pytest.raises(ValueError):
        HS.train_env_seed(run=0, env_index=1000)


def test_final_block_is_sealed():
    with pytest.raises(PermissionError):
        HS.seed_block("HD_FINAL")


def test_only_the_two_permitted_entry_points_open_hd_final():
    users = []
    files = list((PM.REPO / "experiments/highdim").glob("*.py")) + \
        list((PM.REPO / "highdim").glob("*.py"))
    for f in files:
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and any(
                    k.arg == "allow_final" and not (isinstance(k.value, ast.Constant)
                                                    and k.value.value is False)
                    for k in n.keywords):
                users.append(f.name)
    assert set(users) <= {"hd0_baselines.py", "hd4_final.py"}, users


# --------------------------------------------------------------------------- floor arm, seam 1
def test_floor_arm_returns_a_complete_episode_record():
    rec = _episode()
    assert rec["arm"] == "goal_random" and rec["condition"] == "fixed"
    assert rec["outcome"] in ("success", "collision", "timeout")
    trace = rec["trace"]
    assert len(trace) == rec["steps"] > 0
    for key in ("gamma", "V", "u_nom", "u_exec", "qp_status", "random_event", "min_cbc"):
        assert all(key in s for s in trace), key
    assert rec["controller"] == ClfCbfDrccpController.name


def test_floor_arm_sets_gamma_to_the_goal_at_every_step():
    rec = _episode()
    goal = np.asarray(rec["goal"])
    for s in rec["trace"]:
        assert np.allclose(s["gamma"], goal, atol=1e-5)


def test_floor_arm_clf_value_is_the_frozen_formula_towards_the_goal():
    rec = _episode()
    k_v = 0.10
    traj = np.asarray(rec["trajectory"])
    for t, s in enumerate(rec["trace"]):
        if s["V"] is None:
            continue                       # delegated / failed solve carries no V
        e = traj[t] - np.asarray(s["gamma"])
        assert s["V"] == pytest.approx(0.5 * k_v * float(e @ e), rel=1e-9, abs=1e-12)


def test_floor_arm_never_calls_the_planner(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("A* planner called by the floor arm")
    monkeypatch.setattr(StaticMapPlanner, "path", boom)
    _episode()


def test_floor_arm_actions_do_not_depend_on_the_static_map(monkeypatch):
    a = _episode(seed=HS.seed_block("HD_DEV")[EVENT_SEED_INDEX])
    env = make_env(False)
    monkeypatch.setattr(HP, "POLICY_MAP", tuple(env.static_obstacles))   # hand it the real map
    b = _episode(seed=HS.seed_block("HD_DEV")[EVENT_SEED_INDEX])
    assert _canon(a) == _canon(b)


def test_same_seed_reproduces_the_same_floor_episode():
    seed = HS.seed_block("HD_DEV")[EVENT_SEED_INDEX]
    a = _episode("goal_random", "randomized", seed)
    assert a["n_recovery_events"] > 0        # the Random RNG is actually drawn
    assert _canon(a) == _canon(_episode("goal_random", "randomized", seed))


def test_random_recovery_draws_from_the_episode_controller_rng():
    seed = HS.seed_block("HD_DEV")[EVENT_SEED_INDEX]
    rec = _episode(seed=seed)
    rng = CS.controller_rng(seed)
    want = [rng.uniform(0.0, 2.0 * np.pi) for _ in rec["events"]]
    assert rec["events"] and [e["phi"] for e in rec["events"]] == pytest.approx(want, abs=1e-6)
    assert sum(s["random_event"] for s in rec["trace"]) == len(rec["events"])


def test_condition_must_match_the_env_motion_model():
    with pytest.raises(ValueError):
        H.run_episode("goal_random", "randomized", HS.seed_block("HD_DEV")[0],
                      env=make_env(False))


def test_evaluation_refuses_a_non_frozen_controller(monkeypatch):
    from dr_control.fast_drccp import FastClfCbfDrccpController
    real = HP.build_hd_arm

    def with_fast(kind, env, **kw):     # the evaluation guard must reject this swap
        pol = real(kind, env, **kw)
        c = pol.ctrl
        pol.ctrl = FastClfCbfDrccpController(cbf_rate=c.rateh, clf_rate=c.rateV,
                                             wasserstein_r=c.wasserstein_r,
                                             epsilon=c.epsilon, k_v=c.k_v, max_v=c.max_v)
        return pol
    monkeypatch.setattr(HP, "build_hd_arm", with_fast)
    with pytest.raises(TypeError):
        _episode()


def test_floor_arm_uses_the_frozen_tuned_and_random_parameters():
    rec = _episode()
    assert rec["params"]["alpha"] == 0.8 and rec["params"]["k_v"] == 0.1
    assert rec["params"]["tau_eff"] == pytest.approx(0.12)
    assert rec["random_spec"] == {"K": 16, "H": 1, "speeds": [0.8, 1.0], "accept_m": 0.0,
                                  "name": "Random-CLF-DR-CBF"}


# =========================================================================== ticket 03
# Ceiling arm: frozen A* + carrot + Random-CLF-DR-CBF (the F-D arm), through the same harness.

VALID_SEEDS = CS.seed_block("RANDOM_EXP_R3_VALID")[:5]
#: Keys that legitimately differ between the two harnesses: labels and the HD-only additions.
HARNESS_ONLY = {"arm", "kind", "condition", "controller", "params", "random_spec", "trace",
                "trajectory"}


def _continuation_fd(condition, seed, max_steps=SHORT):
    """The Week 6 F-D arm through the frozen Continuation harness, plus its trajectory.

    The frozen harness does not return positions, so env.step is wrapped on this env instance
    (measurement only) to record the agent position after every step.
    """
    tuned, spec = HP.frozen_params()
    env = make_env(condition == "randomized")
    env.MAX_STEPS = max_steps
    traj = []
    step = env.step

    def recording_step(action):
        out = step(action)
        traj.append([float(x) for x in env.agent_position])
        return out
    env.step = recording_step
    arm = CH.ArmSpec("F-D_Random", "random", params=tuned, random=spec)
    rec = CH.run_episode(arm, condition, seed, env=env)
    return rec, [rec["start"]] + traj


@pytest.mark.parametrize("condition", ["fixed", "randomized"])
@pytest.mark.parametrize("seed", VALID_SEEDS)
def test_ceiling_arm_is_bit_identical_to_the_continuation_random_arm(seed, condition):
    ref, ref_traj = _continuation_fd(condition, seed)
    got = _episode("astar_random", condition, seed)
    assert got["trajectory"] == ref_traj                      # every position, exactly
    shared = (set(ref) & set(got)) - TIMING - HARNESS_ONLY
    assert {"outcome", "steps", "path_length", "min_clearance", "n_recovery_events",
            "cbf_min_margin", "mean_u_dev", "events"} <= shared
    assert _canon({k: ref[k] for k in shared}) == _canon({k: got[k] for k in shared})


def test_ceiling_arm_follows_the_astar_carrot_not_the_goal():
    rec = _episode("astar_random", "fixed", HS.seed_block("HD_DEV")[3])
    goal = np.asarray(rec["goal"])
    assert any(not np.allclose(s["gamma"], goal, atol=1e-5) for s in rec["trace"])


def test_same_seed_reproduces_the_same_ceiling_episode():
    seed = HS.seed_block("HD_DEV")[EVENT_SEED_INDEX]
    assert _canon(_episode("astar_random", "randomized", seed)) == \
        _canon(_episode("astar_random", "randomized", seed))


def test_paired_runs_share_seed_start_and_goal():
    seed = HS.seed_block("HD_DEV")[EVENT_SEED_INDEX]
    recs = H.run_paired(("goal_random", "astar_random"), "fixed", seed, max_steps=SHORT)
    assert set(recs) == {"goal_random", "astar_random"}
    assert recs["goal_random"]["start"] == recs["astar_random"]["start"]
    assert recs["goal_random"]["goal"] == recs["astar_random"]["goal"]


def test_paired_runs_reject_a_start_goal_mismatch(monkeypatch):
    real = H.run_episode

    def shifted(kind, condition, seed, **kw):
        rec = real(kind, condition, seed, **kw)
        if kind == "astar_random":
            rec["goal"] = [g + 1.0 for g in rec["goal"]]
        return rec
    monkeypatch.setattr(H, "run_episode", shifted)
    with pytest.raises(RuntimeError):
        H.run_paired(("goal_random", "astar_random"), "fixed", HS.seed_block("HD_DEV")[0],
                     max_steps=SHORT)
