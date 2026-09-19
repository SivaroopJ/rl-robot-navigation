"""HD Experiment 1 (high-dimensionality axis): PPO subgoal source replaces A* in Random-CLF-DR-CBF.

Pre-registration: MD_files/highdim/HD_DESIGN.md. Decision: docs/adr/0001.
"""
from __future__ import annotations

import ast
import gzip
import json
import shutil

import gymnasium as gym
import numpy as np
import pytest

from continuation import harness as CH
from continuation import seeds as CS
from dr_control.drccp_controller import ClfCbfDrccpController
from experiments.exp6_ppo_comparison import make_env
from experiments.highdim import protect_manifest as PM
from highdim import blocks as HB
from highdim import gates as HG
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


# =========================================================================== ticket 04
# Block runner (checkpoint/resume, pairing), stuck metric, pre-registered stop check.
def _recs(n_success, n=400, seed0=0):
    return [{"seed": seed0 + i, "success": int(i < n_success)} for i in range(n)]


def test_stop_check_stops_below_five_points_and_proceeds_at_five():
    assert HG.stop_check(_recs(320), _recs(339))["stop"] is True           # gap 19/400
    assert HG.stop_check(_recs(320), _recs(340))["stop"] is False          # gap exactly 0.05
    v = HG.stop_check(_recs(300), _recs(360))
    assert v == {"S_floor": 0.75, "S_ceiling": 0.9, "gap": pytest.approx(0.15), "n": 400,
                 "threshold": 0.05, "stop": False}


def test_stop_check_uses_exact_arithmetic_at_the_threshold():
    # 0.85 - 0.80 is 0.04999999999999993 in floating point; the rule must still proceed.
    assert HG.stop_check(_recs(320), _recs(340))["stop"] is False
    assert HG.stop_check(_recs(3400, n=4000), _recs(3200, n=4000))["stop"] is True  # negative gap


def test_stop_check_requires_paired_seeds():
    with pytest.raises(ValueError):
        HG.stop_check(_recs(300), _recs(300, seed0=1))


def test_stuck_metric_is_the_phase5_definition():
    still = [[1.0, 1.0]] * 60
    moving = [[0.005 * t, 0.0] for t in range(60)]         # 0.255 m per 51 steps
    creeping = [[0.0045 * t, 0.0] for t in range(60)]      # 0.2295 m per 51 steps
    assert H.stuck(still) and H.stuck(creeping) and not H.stuck(moving)
    assert not H.stuck(still[:51])                          # t = 50 is step 51: traj needs 52
    assert H.stuck(still[:52]) and not H.stuck(still[:52], "success")   # last step unchecked
    assert H.stuck(still[:53], "collision")


def _stuck_exp5_loop(traj, outcome):
    """exp5's loop, verbatim in structure, over a recorded trajectory."""
    out = False
    for t in range(len(traj) - 1):
        p = np.asarray(traj[t + 1])
        if t == len(traj) - 2 and outcome in ("success", "collision"):
            break
        if t >= 50 and float(np.linalg.norm(p - np.asarray(traj[t - 50]))) < 0.25:
            out = True
    return out


def test_stuck_matches_the_exp5_loop_on_random_walks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(40, 120))
        traj = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0).tolist()
        for outcome in ("timeout", "success", "collision"):
            assert H.stuck(traj, outcome) == _stuck_exp5_loop(traj, outcome)


def test_episode_record_carries_stuck_consistent_with_its_trajectory():
    rec = _episode()
    assert rec["stuck"] == int(H.stuck(rec["trajectory"], rec["outcome"]))


def _block(tmp_path, seeds, kind="goal_random", workers=1):
    return HB.run_block(kind, seeds, conditions=("fixed",), workers=workers,
                        checkpoint=tmp_path / f"{kind}.jsonl",
                        traces=tmp_path / f"{kind}.traces.jsonl.gz", max_steps=SHORT)


def test_block_records_are_light_and_traces_go_to_the_gz_file(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:2]
    res = _block(tmp_path, seeds)
    recs = res["fixed"]
    assert [r["seed"] for r in recs] == seeds
    assert all("trace" not in r and "trajectory" not in r for r in recs)
    with gzip.open(tmp_path / "goal_random.traces.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert {(r["cond"], r["seed"]) for r in rows} == {("fixed", s) for s in seeds}
    assert all(len(r["trace"]) == rec["steps"] for r, rec in zip(rows, recs))


def test_block_resume_skips_finished_episodes_and_reproduces_them(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:2]
    full = _block(tmp_path, seeds)
    ck = tmp_path / "goal_random.jsonl"
    lines = ck.read_text().splitlines()
    ck.write_text("\n".join(lines[:2]) + "\n" + lines[2][:30])      # header + 1 row + torn row
    resumed = _block(tmp_path, seeds)
    assert [_canon(r) for r in full["fixed"]] == [_canon(r) for r in resumed["fixed"]]


def test_block_refuses_a_checkpoint_from_a_different_definition(tmp_path):
    _block(tmp_path, HS.seed_block("HD_DEV")[:1])
    with pytest.raises(SystemExit):
        _block(tmp_path, HS.seed_block("HD_DEV")[1:2])


def test_block_parallel_equals_serial_including_the_trace_digest(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:3]
    a = _block(tmp_path / "a", seeds, workers=1)
    b = _block(tmp_path / "b", seeds, workers=3)
    assert [_canon(r) for r in a["fixed"]] == [_canon(r) for r in b["fixed"]]
    ga, gb = (tmp_path / d / "goal_random.traces.jsonl.gz" for d in ("a", "b"))
    assert PM._sha(ga) == PM._sha(gb)
    assert not (tmp_path / "a" / "goal_random.traces.partial.jsonl").exists()


def test_block_survives_a_kill_between_trace_and_checkpoint_writes(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:2]
    full = _block(tmp_path, seeds)
    digest = PM._sha(tmp_path / "goal_random.traces.jsonl.gz")
    # Simulate a kill mid-block: traces back in the partial file, the second episode's trace
    # torn, the checkpoint still holding both rows, and no finished gz.
    gz = tmp_path / "goal_random.traces.jsonl.gz"
    with gzip.open(gz, "rt") as f:
        rows = f.read().splitlines()
    gz.unlink()
    (tmp_path / "goal_random.traces.partial.jsonl").write_text(rows[0] + "\n" + rows[1][:40])
    resumed = _block(tmp_path, seeds)              # re-runs the episode whose trace is torn
    assert [_canon(r) for r in full["fixed"]] == [_canon(r) for r in resumed["fixed"]]
    assert PM._sha(gz) == digest


def test_block_treats_an_empty_checkpoint_as_new(tmp_path):
    (tmp_path / "goal_random.jsonl").write_text("")
    assert len(_block(tmp_path, HS.seed_block("HD_DEV")[:1])["fixed"]) == 1


def test_block_refuses_to_resume_under_a_different_code_tag(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:1]
    kw = dict(conditions=("fixed",), workers=1, checkpoint=tmp_path / "c.jsonl",
              traces=tmp_path / "c.traces.jsonl.gz", max_steps=SHORT)
    HB.run_block("goal_random", seeds, tag="commit-a", **kw)
    with pytest.raises(SystemExit):
        HB.run_block("goal_random", seeds, tag="commit-b", **kw)


def test_worker_env_cache_does_not_leak_a_step_limit(monkeypatch):
    monkeypatch.setattr(HB, "run_episode", lambda kind, cond, seed, env, oracle: env.MAX_STEPS)
    monkeypatch.setattr(HB, "_W", {})
    seed = HS.seed_block("HD_DEV")[0]
    limits = [HB._worker(("goal_random", "fixed", seed, m, None))[2]
              for m in (5, None, 7, None)]
    assert limits == [5, 500, 7, 500]


def test_stuck_rejects_an_unknown_outcome():
    with pytest.raises(ValueError):
        H.stuck([[0, 0]] * 60, "succes")


def test_stop_check_rejects_an_empty_block():
    with pytest.raises(ValueError):
        HG.stop_check([], [])


def test_arms_are_checked_for_pairing_across_blocks():
    a = {"fixed": [{"seed": 1, "start": [0, 0], "goal": [1, 1]}]}
    HB.check_paired(a, {"fixed": [{"seed": 1, "start": [0, 0], "goal": [1, 1]}]})
    with pytest.raises(RuntimeError):
        HB.check_paired(a, {"fixed": [{"seed": 1, "start": [0, 0], "goal": [2, 1]}]})
    with pytest.raises(RuntimeError):
        HB.check_paired(a, {"fixed": [{"seed": 2, "start": [0, 0], "goal": [1, 1]}]})


def test_hd0_smoke_analyses_and_reports_without_opening_hd_final(tmp_path, monkeypatch):
    from experiments.highdim import hd0_baselines as HD0
    opened = []
    real = HD0.seed_block
    monkeypatch.setattr(HD0, "seed_block",
                        lambda name, *a, **k: opened.append(name) or real(name, *a, **k))
    res = HD0.run(tmp_path, workers=1, n=2, max_steps=SHORT, progress=0, smoke=True)
    assert "HD_FINAL" not in opened
    analysis = HD0.analyse(res)
    assert set(analysis["stop_check"]) == {"S_floor", "S_ceiling", "gap", "n", "threshold",
                                           "stop"}
    text = HD0.report(analysis, HD0.provenance(), {"x.traces.jsonl.gz": "0" * 64})
    assert "## Stop check" in text and "| success |" in text and "| stuck |" in text
    assert "tuned α 0.8" in text and "K 16" in text and "(2 seeds × 2 conditions)" in text


# =========================================================================== ticket 05
# Fast-solver training-suitability check (HD_DESIGN.md section 8). Not an evaluation: the check
# episode is the only harness path that may run the fast solver.

def _check_episode(kind="goal_random", condition="fixed", seed=None, max_steps=SHORT, **kw):
    seed = HS.seed_block("HD_DEV")[3] if seed is None else seed
    env = make_env(condition == "randomized")
    env.MAX_STEPS = max_steps
    return H.run_check_episode(kind, condition, seed, env=env, **kw)


@pytest.mark.parametrize("kind", ["goal_random", "astar_random"])
def test_fast_check_episode_runs_the_unmodified_reduced_osqp_at_the_frozen_parameters(kind):
    from dr_control.fast_drccp import FastClfCbfDrccpController
    seen = []
    real = FastClfCbfDrccpController.generate_controller

    def spy(self, *a, **k):
        seen.append(self)
        return real(self, *a, **k)
    FastClfCbfDrccpController.generate_controller = spy
    try:
        rec = _check_episode(kind, solver="fast")
    finally:
        FastClfCbfDrccpController.generate_controller = real
    assert rec["controller"] == "drccp_fast" and rec["solver"] == "fast"
    c = seen[0]
    assert len(seen) == rec["steps"] and all(s is c for s in seen)
    assert c.variant == "reduced_osqp"
    assert (c.rateh, c.rateV, c.wasserstein_r, c.epsilon, c.k_v, c.n_keep, c.max_v) == \
        (0.8, 1.0, 0.012, 0.1, 0.1, 5, 1.0)
    assert c.tau == pytest.approx(0.12)
    assert rec["random_spec"]["K"] == 16 and rec["random_spec"]["speeds"] == [0.8, 1.0]


def _without_shadow(rec):
    out = {k: v for k, v in rec.items() if k not in ("shadow", "solver")}
    out["trace"] = [{k: v for k, v in s.items() if k != "shadow"} for s in rec["trace"]]
    return _canon(out)


@pytest.mark.parametrize("kind", ["goal_random", "astar_random"])
def test_shadow_leaves_the_frozen_episode_bit_identical(kind):
    seed = HS.seed_block("HD_DEV")[EVENT_SEED_INDEX]
    ref = _episode(kind, "randomized", seed)
    got = _check_episode(kind, "randomized", seed, shadow=True)
    assert got["solver"] == "frozen" and got["controller"] == "drccp"
    assert ref["n_recovery_events"] > 0 or kind == "astar_random"
    assert _without_shadow(got) == _without_shadow(ref)


def test_shadow_solves_the_identical_inputs_the_frozen_solver_saw(monkeypatch):
    from dr_control.fast_drccp import FastClfCbfDrccpController
    calls = {"frozen": [], "fast": []}

    def spy(label, real):
        def f(self, p, gamma, xi, *, u_nom=None, record=None):
            if label == "frozen" and calls["frozen"] and self is not calls["frozen"][0][0]:
                return real(self, p, gamma, xi, u_nom=u_nom, record=record)   # not the policy's
            calls[label].append((self, np.array(p, float), np.array(gamma, float),
                                 np.array(xi, float), u_nom, np.array(self.prev_u, float)))
            return real(self, p, gamma, xi, u_nom=u_nom, record=record)
        return f
    monkeypatch.setattr(ClfCbfDrccpController, "generate_controller",
                        spy("frozen", ClfCbfDrccpController.generate_controller))
    monkeypatch.setattr(FastClfCbfDrccpController, "generate_controller",
                        spy("fast", FastClfCbfDrccpController.generate_controller))
    rec = _check_episode(seed=HS.seed_block("HD_DEV")[EVENT_SEED_INDEX], shadow=True)
    assert rec["n_recovery_events"] > 0          # Random has overwritten prev_u at least once
    assert len(calls["frozen"]) == len(calls["fast"]) == rec["steps"]
    for fr, fa in zip(calls["frozen"], calls["fast"]):
        for a, b in zip(fr[1:], fa[1:]):
            assert (a is None and b is None) or np.array_equal(a, b)


@pytest.mark.parametrize("status,cat", [
    ("optimal", "optimal"), ("infeasible", "infeasible"),
    ("infeasible_inaccurate", "infeasible"), ("optimal_inaccurate", "other"),
    ("DCPError", "other"), ("solver_error", "other"), ("None", "other")])
def test_feasibility_category_follows_what_the_policy_does_with_the_status(status, cat):
    # optimal -> action used; infeasible -> Random recovery; anything else -> frozen u = 0
    assert H.feasibility_category(status) == cat


def test_shadow_record_summarises_the_per_step_comparison():
    rec = _check_episode(seed=HS.seed_block("HD_DEV")[EVENT_SEED_INDEX], shadow=True)
    sh, steps = rec["shadow"], [s["shadow"] for s in rec["trace"]]
    assert sh["steps"] == rec["steps"] == len(steps)
    both = [s for s in steps if s["cat_frozen"] == s["cat_fast"] == "optimal"]
    assert sh["both_optimal"] == len(both) > 0
    assert sh["agree"] == sum(s["cat_frozen"] == s["cat_fast"] for s in steps)
    assert sh["delegated"] == sum(s["delegated"] for s in steps)
    assert sum(sh["du_hist"].values()) == len(both)
    for s in both:
        assert s["du_inf"] == max(abs(a - b) for a, b in zip(s["u_fast"], s["u_frozen"]))
    assert sh["du_max"] == max(s["du_inf"] for s in both)
    assert sh["du_le_tol"] == sum(s["du_inf"] <= 1e-3 for s in both)
    # the frozen QP output is what executed wherever Random did not step in
    for t, s in zip(rec["trace"], steps):
        if s["cat_frozen"] == "optimal" and not t["random_event"]:
            assert t["u_exec"] == s["u_frozen"]


def test_shadow_counts_a_feasibility_disagreement(monkeypatch):
    from dr_control.fast_drccp import FastClfCbfDrccpController
    real = FastClfCbfDrccpController.generate_controller
    n = {"k": 0}

    def flip_third(self, p, gamma, xi, *, u_nom=None, record=None):
        u = real(self, p, gamma, xi, u_nom=u_nom, record=record)
        n["k"] += 1
        if n["k"] == 3:
            record["status"] = "infeasible"
        return u
    monkeypatch.setattr(FastClfCbfDrccpController, "generate_controller", flip_third)
    rec = _check_episode(shadow=True)
    sh = rec["shadow"]
    assert rec["trace"][2]["shadow"]["cat_frozen"] == "optimal"
    assert sh["agree"] == sh["steps"] - 1 and sh["crosstab"] == {"optimal->infeasible": 1}


def test_shadow_requires_the_frozen_solver_in_the_loop():
    with pytest.raises(ValueError):
        _check_episode(solver="fast", shadow=True)


def _sh(steps=1000, agree=1000, both=1000, le=1000, du_max=1e-4):
    return {"steps": steps, "agree": agree, "crosstab": {}, "both_optimal": both,
            "du_le_tol": le, "du_max": du_max, "du_hist": {"<=1e-04": both}}


def test_shadow_rule_needs_full_category_agreement_and_99_percent_within_tolerance():
    assert HG.shadow_verdict([_sh(le=990)])["pass"]                  # exactly 99 %
    assert not HG.shadow_verdict([_sh(le=989)])["pass"]
    assert not HG.shadow_verdict([_sh(agree=999, le=1000)])["pass"]  # one category flip
    v = HG.shadow_verdict([_sh(le=495, du_max=0.2), _sh(le=500, du_max=0.3)])   # pooled
    assert (v["both_optimal"], v["du_le_tol"], v["du_max"]) == (2000, 995, 0.3)
    assert not v["pass"] and v["pass_category"] and not v["pass_du"]


def test_shadow_rule_refuses_an_empty_corpus():
    with pytest.raises(ValueError):
        HG.shadow_verdict([])
    with pytest.raises(ValueError):
        HG.shadow_verdict([_sh(both=0, le=0)])


def _cl(success, cond="fixed"):
    return [{"condition": cond, "seed": i, "success": s} for i, s in enumerate(success)]


def test_closed_loop_rule_passes_matching_arms():
    frozen = _cl([1] * 150 + [0] * 50)
    fast = _cl([1] * 149 + [0] + [1] + [0] * 49)                  # one discordant pair each way
    v = HG.closed_loop_verdict(frozen, fast)
    assert v["p"] == 1.0 and v["diff"] == 0.0 and v["pass"]


def test_closed_loop_rule_fails_on_mcnemar_or_on_a_wide_interval():
    frozen = _cl([0] * 200)
    v6 = HG.closed_loop_verdict(frozen, _cl([1] * 6 + [0] * 194))   # p = 2/64
    assert v6["p"] == pytest.approx(0.03125) and not v6["pass"]
    v5 = HG.closed_loop_verdict(frozen, _cl([1] * 5 + [0] * 195))   # p = 0.0625, CI > 0.03
    assert v5["p"] == pytest.approx(0.0625) and v5["ci95"][1] > 0.03 and not v5["pass"]


def test_closed_loop_rule_requires_pairing_by_condition_and_seed():
    with pytest.raises(ValueError):
        HG.closed_loop_verdict(_cl([1, 0], "fixed"), _cl([1, 0], "randomized"))


def test_training_solver_is_fast_only_when_every_test_passes():
    assert HG.training_solver({"pass": True}, {"floor": {"pass": True},
                                               "ceiling": {"pass": True}})["solver"] == "fast"
    for ea, eb in [(False, (True, True)), (True, (True, False)), (True, (False, True))]:
        v = HG.training_solver({"pass": ea}, {"floor": {"pass": eb[0]},
                                              "ceiling": {"pass": eb[1]}})
        assert v["solver"] == "frozen" and not v["pass"]


def test_check_block_keeps_the_shadow_summary_light_and_the_steps_in_the_traces(tmp_path):
    seeds = HS.seed_block("HD_DEV")[:2]
    res = HB.run_block("goal_random", seeds, conditions=("fixed",), workers=2,
                       checkpoint=tmp_path / "c.jsonl", traces=tmp_path / "c.traces.jsonl.gz",
                       max_steps=SHORT, check=("frozen", True))
    recs = res["fixed"]
    assert all(r["solver"] == "frozen" and r["shadow"]["steps"] == r["steps"] for r in recs)
    with gzip.open(tmp_path / "c.traces.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert all("cat_fast" in s["shadow"] for r in rows for s in r["trace"])
    fast = HB.run_block("goal_random", seeds, conditions=("fixed",), workers=1,
                        checkpoint=tmp_path / "f.jsonl", traces=tmp_path / "f.traces.jsonl.gz",
                        max_steps=SHORT, check=("fast", False))
    assert all(r["solver"] == "fast" and "shadow" not in r for r in fast["fixed"])


def test_check_block_refuses_an_evaluation_checkpoint(tmp_path):
    kw = dict(conditions=("fixed",), workers=1, checkpoint=tmp_path / "c.jsonl",
              traces=tmp_path / "c.traces.jsonl.gz", max_steps=SHORT)
    seeds = HS.seed_block("HD_DEV")[:1]
    HB.run_block("goal_random", seeds, **kw)
    with pytest.raises(SystemExit):
        HB.run_block("goal_random", seeds, check=("fast", False), **kw)


def test_hd1_smoke_decides_and_reports_on_hd_dev_only(tmp_path, monkeypatch):
    from experiments.highdim import hd1_solver_check as HD1
    opened = []
    real = HD1.seed_block
    monkeypatch.setattr(HD1, "seed_block",
                        lambda name, *a, **k: opened.append(name) or real(name, *a, **k))
    res = HD1.run(tmp_path, workers=1, n=2, max_steps=SHORT, progress=0)
    assert set(opened) == {"HD_DEV"}
    assert set(res) == {(k, s) for k in ("goal_random", "astar_random")
                        for s in ("frozen_shadow", "fast")}
    analysis = HD1.analyse(res)
    d = analysis["decision"]
    assert d["solver"] in ("fast", "frozen") and set(d["E-b"]) == {"floor", "ceiling"}
    assert analysis["E-a"]["steps"] == sum(r["steps"] for c in res[("goal_random",
                                           "frozen_shadow")].values() for r in c) + \
        sum(r["steps"] for c in res[("astar_random", "frozen_shadow")].values() for r in c)
    text = HD1.report(analysis, HD1.provenance(), {"x.traces.jsonl.gz": "0" * 64})
    for must in ("## Verdict", "E-a", "E-b", "Stage 1", "not reopened",
                 "acceleration mechanism, not a formally equivalent replacement",
                 "frozen SCS", "(2 seeds × 2 conditions)", "three attempts; stopping rule applied",
                 "ε·n_keep = 0.5 < 1", "optimal vs infeasible"):
        assert must in text, must


# =========================================================================== ticket 06
# PPO-subgoal + Random (arm 3) and the filter-bypass diagnostic (arm 4), driven by an untrained
# PPO checkpoint through seam 1.

#: HD_DEV[8]: the untrained seed-0 model has Random events inside SHORT steps in both conditions
PPO_EVENT_SEED_INDEX = 8


@pytest.fixture(scope="module")
def ppo_model():
    from highdim import ppo as HPPO
    return HPPO.new_model(seed=0)


def _ppo_episode(model, kind="ppo_random", condition="fixed", seed=None, max_steps=SHORT):
    seed = HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX] if seed is None else seed
    env = make_env(condition == "randomized")
    env.MAX_STEPS = max_steps
    return H.run_episode(kind, condition, seed, env=env, model=model)


def test_ppo_arm_controller_receives_the_subgoal_reference(ppo_model, monkeypatch):
    got = []
    real = ClfCbfDrccpController.generate_controller

    def spy(self, p, gamma, xi, *, u_nom=None, record=None):
        got.append((np.array(p, float), np.array(gamma, float)))
        return real(self, p, gamma, xi, u_nom=u_nom, record=record)
    monkeypatch.setattr(ClfCbfDrccpController, "generate_controller", spy)
    rec = _ppo_episode(ppo_model)
    goal = np.asarray(rec["goal"])
    assert len(got) == rec["steps"] == len(rec["trace"])
    for (p, gamma), s in zip(got, rec["trace"]):
        a = np.asarray(s["a_disc"])
        raw = np.asarray(s["a_raw"])
        assert np.all(np.abs(raw) <= 1.0) and np.linalg.norm(a) <= 1.0 + 1e-12
        assert np.allclose(a, raw / max(1.0, np.linalg.norm(raw)), atol=0, rtol=0)
        if np.linalg.norm(goal - p) < 1.0:
            assert np.allclose(gamma, goal, atol=1e-5)    # the policy's goal, from obs[0:2]
        else:
            assert np.array_equal(gamma, p + 1.0 * a)
        assert np.allclose(s["gamma"], gamma, atol=0)


def test_ppo_arm_clf_value_is_the_frozen_formula_towards_the_subgoal(ppo_model):
    rec = _ppo_episode(ppo_model)
    traj = np.asarray(rec["trajectory"])
    checked = 0
    for t, s in enumerate(rec["trace"]):
        if s["V"] is None:
            continue
        e = traj[t] - np.asarray(s["gamma"])
        assert s["V"] == pytest.approx(0.5 * 0.10 * float(e @ e), rel=1e-9, abs=1e-12)
        checked += 1
    assert checked > 0


def test_ppo_arm_parameters_hash_match_the_frozen_files(ppo_model):
    from experiments.week6_continuation.common import RESULTS as FROZEN
    rec = _ppo_episode(ppo_model)
    for stem, key, got in (("H2/tuned_frozen.json", "params", rec["params"]),
                           ("R3/random_frozen.json", "spec", rec["random_spec"])):
        f = FROZEN / stem
        assert PM._sha(f) == f.with_suffix(".sha256").read_text().split()[0]
        want = json.loads(f.read_text())[key]
        shared = [k for k in want if k in got and k != "name"]     # name: a label only
        assert len(shared) >= 4 and {k: got[k] for k in shared} == {k: want[k] for k in shared}
    assert rec["params"]["alpha"] == 0.8 and rec["random_spec"]["K"] == 16


def test_forced_infeasible_samples_trigger_random_recovery_in_the_ppo_arm(ppo_model, monkeypatch):
    from dr_control.estimated_cbf import EstimatedLidarBarrierSource
    real = EstimatedLidarBarrierSource.samples
    calls = {"n": 0}
    FORCED = 3

    def samples(self, p):
        calls["n"] += 1
        if calls["n"] != FORCED + 1:
            return real(self, p)
        # two opposing barriers closing at 5 m/s: no |u|_inf <= 1 keeps min CBC >= tau
        h = np.array([0.05, 0.05])
        g = np.array([[1.0, 0.0], [-1.0, 0.0]])
        return h, g, np.array([-5.0, -5.0])
    monkeypatch.setattr(EstimatedLidarBarrierSource, "samples", samples)
    rec = _ppo_episode(ppo_model, seed=HS.seed_block("HD_DEV")[0])
    s = rec["trace"][FORCED]
    assert "infeasible" in s["qp_status"] and s["random_event"]
    assert FORCED in [e["t"] for e in rec["events"]]


def test_bypass_arm_never_calls_the_qp_or_random_and_executes_the_nominal_velocity(
        ppo_model, monkeypatch):
    from continuation.random_recovery import RandomRecovery
    from dr_control.drccp_controller import nominal_action

    def boom(*a, **k):
        raise AssertionError("QP or Random called by the bypass arm")
    monkeypatch.setattr(ClfCbfDrccpController, "generate_controller", boom)
    monkeypatch.setattr(RandomRecovery, "__init__", boom)
    env = make_env(False)
    env.MAX_STEPS = SHORT
    executed = []
    step = env.step
    env.step = lambda a: executed.append(np.array(a, float)) or step(a)
    rec = H.run_episode("ppo_unfiltered", "fixed", HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX],
                        env=env, model=ppo_model)
    traj = np.asarray(rec["trajectory"])
    max_v = env.MAX_SPEED
    assert rec["controller"] == "bypass" and rec["random_spec"] is None
    assert len(executed) == rec["steps"] == len(rec["trace"])
    for t, (s, a) in enumerate(zip(rec["trace"], executed)):
        want = nominal_action(traj[t], np.asarray(s["gamma"]), max_v) / max_v
        assert s["qp_status"] == "bypassed" and not s["random_event"]
        assert np.allclose(a, want.astype(np.float32), atol=0, rtol=0)


def test_bypass_arm_reuses_the_checkpoint_and_subgoal_of_the_ppo_arm(ppo_model):
    a = _ppo_episode(ppo_model, "ppo_random")
    b = _ppo_episode(ppo_model, "ppo_unfiltered")
    assert a["start"] == b["start"] and a["goal"] == b["goal"]
    # the first decision sees the same observation, so the same action and gamma
    assert a["trace"][0]["a_raw"] == b["trace"][0]["a_raw"]
    assert a["trace"][0]["gamma"] == b["trace"][0]["gamma"]


def test_ppo_arms_require_a_model_and_the_baselines_refuse_one(ppo_model):
    with pytest.raises(ValueError):
        _ppo_episode(None, "ppo_random")
    with pytest.raises(ValueError):
        _ppo_episode(ppo_model, "goal_random")


@pytest.mark.parametrize("kind", ["ppo_random", "ppo_unfiltered"])
def test_ppo_arm_actions_do_not_depend_on_the_static_map(ppo_model, monkeypatch, kind):
    a = _ppo_episode(ppo_model, kind)
    monkeypatch.setattr(HP, "POLICY_MAP", tuple(make_env(False).static_obstacles))
    b = _ppo_episode(ppo_model, kind)
    assert _canon(a) == _canon(b)


def _perturbed_env(condition):
    """Adds noise to the ground-truth obstacle block obs[28:52] of every observation."""
    env = make_env(condition == "randomized")
    env.MAX_STEPS = SHORT
    rng = np.random.default_rng(7)
    reset, step = env.reset, env.step

    def noisy(obs):
        obs = np.array(obs, copy=True)
        obs[28:] = rng.uniform(-1, 1, size=obs[28:].shape).astype(obs.dtype)
        return obs

    env.reset = lambda **k: (lambda o, i: (noisy(o), i))(*reset(**k))
    env.step = lambda a: (lambda o, *r: (noisy(o), *r))(*step(a))
    return env


@pytest.mark.parametrize("kind", ["ppo_random", "ppo_unfiltered"])
def test_ppo_arm_ignores_the_ground_truth_obstacle_block(ppo_model, kind):
    seed = HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX]
    a = _ppo_episode(ppo_model, kind, "randomized", seed)
    b = H.run_episode(kind, "randomized", seed, env=_perturbed_env("randomized"),
                      model=ppo_model)
    assert _canon(a) == _canon(b)


@pytest.mark.parametrize("kind", ["ppo_random", "ppo_unfiltered"])
def test_same_seed_and_checkpoint_reproduce_the_same_ppo_episode(ppo_model, tmp_path, kind):
    from stable_baselines3 import PPO
    ppo_model.save(tmp_path / "ckpt.zip")
    seed = HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX]
    a = _ppo_episode(PPO.load(tmp_path / "ckpt.zip", device="cpu"), kind, "randomized", seed)
    b = _ppo_episode(PPO.load(tmp_path / "ckpt.zip", device="cpu"), kind, "randomized", seed)
    c = _ppo_episode(ppo_model, kind, "randomized", seed)
    assert _canon(a) == _canon(b) == _canon(c)
    if kind == "ppo_random":
        assert a["n_recovery_events"] > 0                 # the Random RNG is actually drawn


def test_subgoal_rule_on_worked_examples():
    from highdim.subgoal import SubgoalSource
    src = SubgoalSource([5.0, 5.0])
    src.set_action([3.0, 4.0])                   # box-clipped to (1, 1), then onto the disc
    assert np.allclose(src.a_raw, [1.0, 1.0]) and np.allclose(src.a_disc, [0.5 ** 0.5] * 2)
    assert np.allclose(src.reference([1.0, 1.0]), [1.0 + 0.5 ** 0.5, 1.0 + 0.5 ** 0.5])
    src.set_action([0.3, -0.4])                  # inside the disc: unchanged, distance 0.5 m
    assert np.allclose(src.reference([1.0, 1.0]), [1.3, 0.6])
    assert np.array_equal(src.reference([4.5, 5.0]), [5.0, 5.0])     # 0.5 m from the goal
    assert np.allclose(src.reference([4.0, 5.0]), [4.3, 4.6])        # exactly L: not "< L"
    assert np.allclose(src.goal, [5.0, 5.0])


# =========================================================================== ticket 07
# PPO environment wrapper (seam 2), the subgoal hold, and the trainer.

def test_subgoal_hold_freezes_gamma_in_world_coordinates():
    from highdim.subgoal import SubgoalSource
    src = SubgoalSource([9.0, 9.0])
    src.set_action([1.0, 0.0], p=[1.0, 1.0])     # a decision at p = (1, 1)
    for p in ([1.0, 1.0], [1.4, 1.2], [3.0, 3.0]):
        assert np.array_equal(src.reference(p), [2.0, 1.0])
    assert np.array_equal(src.reference([8.5, 9.0]), [9.0, 9.0])     # the goal switch still wins
    src.set_action([0.0, 1.0])                    # a decision without p: gamma follows p again
    assert np.allclose(src.reference([3.0, 3.0]), [3.0, 4.0])


def test_harness_hold_queries_the_model_every_k_steps_and_freezes_gamma(ppo_model):
    calls = []

    class Counting:
        def predict(self, obs, deterministic=True):
            calls.append(np.array(obs))
            return ppo_model.predict(obs, deterministic=deterministic)
    env = make_env(False)
    env.MAX_STEPS = 23
    rec = H.run_episode("ppo_random", "fixed", HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX],
                        env=env, model=Counting(), hold=5)
    assert rec["steps"] == 23 and len(calls) == 5 and rec["hold"] == 5
    traj = np.asarray(rec["trajectory"])
    goal = np.asarray(rec["goal"])
    for t, s in enumerate(rec["trace"]):
        d = t - t % 5                              # the decision step
        want = traj[d] + np.asarray(rec["trace"][d]["a_disc"])
        if np.linalg.norm(goal - traj[t]) >= 1.0:
            assert np.allclose(s["gamma"], want, atol=1e-12)
        assert s["a_disc"] == rec["trace"][d]["a_disc"]


DIAG_KEYS = {"p", "gamma", "V", "u_nom", "u_exec", "u_dev", "qp_status", "random_event",
             "min_cbc", "a_raw", "a_disc"}


def _wrapper(condition="fixed", seed=None, **kw):
    from highdim.wrapper import HDSubgoalEnv
    seed = HS.train_env_seed(run=99, env_index=0) if seed is None else seed
    return HDSubgoalEnv(condition, seed=seed, **kw)


def _roll(env, actions):
    obs, _ = env.reset()
    out = [obs]
    diags = []
    for a in actions:
        obs, r, term, trunc, info = env.step(np.asarray(a, np.float32))
        out.append(obs)
        diags += info["diagnostics"]
        if term or trunc:
            break
    return out, diags


def test_wrapper_exposes_the_28d_observation_and_2d_action_box():
    env = _wrapper()
    obs, info = env.reset()
    assert env.observation_space.shape == (28,) and env.action_space.shape == (2,)
    assert obs.shape == (28,) and env.observation_space.contains(obs)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)


def test_wrapper_ignores_the_ground_truth_obstacle_block():
    acts = np.random.default_rng(3).uniform(-1.5, 1.5, size=(40, 2))
    a = _wrapper("randomized")
    b = _wrapper("randomized")
    rng = np.random.default_rng(11)
    inner_reset, inner_step = b.env.reset, b.env.step

    def noisy(o):
        o = np.array(o, copy=True)
        o[28:] = rng.uniform(-1, 1, size=o[28:].shape).astype(o.dtype)
        return o
    b.env.reset = lambda **k: (lambda o, i: (noisy(o), i))(*inner_reset(**k))
    b.env.step = lambda x: (lambda o, *r: (noisy(o), *r))(*inner_step(x))
    oa, da = _roll(a, acts)
    ob, db = _roll(b, acts)
    assert len(oa) == len(ob) and all(np.array_equal(x, y) for x, y in zip(oa, ob))
    for x, y in zip(da, db, strict=True):
        assert x["gamma"] == y["gamma"] and x["u_exec"] == y["u_exec"]


def test_wrapper_diagnostics_carry_every_key_and_the_gamma_rule():
    from highdim.subgoal import L
    env = _wrapper()
    obs, _ = env.reset()
    goal = env.policy.goal
    diags, near = [], 0
    for _ in range(120):                          # head for the goal to cross the L boundary
        p = env.env.agent_position.copy()
        d = goal - p
        obs, _, term, trunc, info = env.step((d / max(np.abs(d).max(), 1e-9)).astype(np.float32))
        assert len(info["diagnostics"]) == 1
        diags += info["diagnostics"]
        if term or trunc:
            break
    for s in diags:
        assert set(s) >= DIAG_KEYS
        p, a = np.asarray(s["p"]), np.asarray(s["a_disc"])
        assert np.linalg.norm(a) <= 1.0 + 1e-12
        if np.linalg.norm(goal - p) < L:
            near += 1
            assert np.array_equal(s["gamma"], list(goal))
        else:
            assert np.array_equal(np.asarray(s["gamma"]), p + L * a)
        if s["u_nom"] is not None:
            assert s["u_dev"] == pytest.approx(
                float(np.linalg.norm(np.subtract(s["u_exec"], s["u_nom"]))), abs=1e-12)
    assert near > 0 and diags[0]["p"] != diags[-1]["p"]


def test_wrapper_reward_and_termination_are_the_env_s_own_on_the_executed_velocity():
    a = _wrapper()
    a.reset()
    ref = make_env(False)
    ref.reset(seed=HS.train_env_seed(run=99, env_index=0))
    for x in np.random.default_rng(5).uniform(-1, 1, size=(25, 2)):
        _, r, term, trunc, info = a.step(x.astype(np.float32))
        u = np.asarray(info["diagnostics"][0]["u_exec"])
        executed = np.clip(u / ref.MAX_SPEED, -1.0, 1.0).astype(np.float32)   # as predict()
        _, r_ref, term_ref, trunc_ref, _ = ref.step(executed)
        assert (r, term, trunc) == (r_ref, term_ref, trunc_ref)
        if term or trunc:
            break


def test_wrapper_draws_the_controller_rng_from_its_own_generator():
    a, b = _wrapper(), _wrapper()
    a.reset(), b.reset()
    ra, rb = a.recovery.rng.uniform(size=3), b.recovery.rng.uniform(size=3)
    assert np.array_equal(ra, rb)                       # same wrapper seed -> same draws
    c = _wrapper(seed=HS.train_env_seed(run=99, env_index=1))
    c.reset()
    assert not np.array_equal(ra, c.recovery.rng.uniform(size=3))
    a.reset()                                           # a fresh policy and recovery per reset
    assert a.recovery is not None and not np.array_equal(a.recovery.rng.uniform(size=3), ra)


def test_wrapper_hold_repeats_the_decision_and_sums_the_reward():
    env = _wrapper(hold=5)
    env.reset()
    _, r, term, trunc, info = env.step(np.array([0.3, -0.2], np.float32))
    d = info["diagnostics"]
    assert len(d) == 5 or term or trunc
    assert all(s["a_raw"] == d[0]["a_raw"] for s in d)
    assert r == pytest.approx(sum(s["reward"] for s in d))


def test_wrapper_first_reset_uses_its_train_seed_then_auto_resets():
    a, b = _wrapper(), _wrapper()
    o1, _ = a.reset()
    o2, _ = b.reset()
    assert np.array_equal(o1, o2)
    o3, _ = a.reset()
    assert not np.array_equal(o1, o3)


class _Logging(gym.Wrapper):
    """Test-only: remembers each step's diagnostics (DummyVecEnv drops infos after the step)."""

    def __init__(self, env):
        super().__init__(env)
        self.log = []

    def step(self, action):
        out = self.env.step(action)
        self.log.append(out[4]["diagnostics"][0])
        return out


def test_sb3_rollout_stores_the_subgoal_action_and_its_log_probability_only():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = _Logging(_wrapper())
    vec = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec, n_steps=16, batch_size=16, seed=0, device="cpu",
                policy_kwargs={"net_arch": [256, 256]}, verbose=0)
    _, cb = model._setup_learn(16, callback=None)
    cb.on_training_start(locals(), globals())
    assert model.collect_rollouts(vec, cb, model.rollout_buffer, n_rollout_steps=16)
    buf = model.rollout_buffer
    obs = torch.as_tensor(buf.observations.reshape(16, 28))
    act = torch.as_tensor(buf.actions.reshape(16, 2))
    with torch.no_grad():
        _, logp, _ = model.policy.evaluate_actions(obs, act)
    assert np.allclose(logp.numpy(), buf.log_probs.reshape(16), atol=1e-5)
    clipped = np.clip(buf.actions.reshape(16, 2), -1, 1)
    raw = np.asarray([s["a_raw"] for s in env.log])
    assert np.allclose(clipped, raw, atol=1e-6)
    u_exec = np.asarray([s["u_exec"] for s in env.log])
    env_action = np.clip(u_exec / env.unwrapped.env.MAX_SPEED, -1, 1)
    differ = np.abs(raw - u_exec).max(axis=1) > 1e-3
    assert differ.sum() >= 8                    # a swap, even a partial one, would show here
    for name in ("observations", "actions", "rewards", "returns", "episode_starts", "values",
                 "log_probs", "advantages"):
        arr = np.asarray(getattr(buf, name)).reshape(16, -1)
        for col in range(arr.shape[1] - 1):     # any two adjacent columns, any row
            pair = arr[:, col:col + 2]
            for leak in (u_exec, env_action):
                assert not np.any(np.all(np.abs(pair - leak)[differ] < 1e-6, axis=1)), name
    assert buf.observations.shape[-1] == 28


def test_training_envs_are_four_fixed_and_four_randomized_on_hd_train_seeds():
    from highdim import train as HT
    specs = HT.env_specs(run=3)
    assert [c for c, _ in specs] == ["fixed"] * 4 + ["randomized"] * 4
    assert [s for _, s in specs] == [HS.train_env_seed(run=3, env_index=i) for i in range(8)]


def test_training_solver_follows_the_fast_solver_check(tmp_path, monkeypatch):
    from highdim import train as HT
    assert HT.training_solver() == "frozen"              # ticket 05 failed (4a2ec26)
    f = tmp_path / "analysis.json"
    f.write_text(json.dumps({"decision": {"solver": "fast", "pass": True}}))
    monkeypatch.setattr(HT, "SOLVER_DECISION", f)
    assert HT.training_solver() == "fast"
    f.write_text(json.dumps({"decision": {"solver": "fast", "pass": False}}))
    with pytest.raises(RuntimeError):
        HT.training_solver()


def test_model_uses_the_config_hyperparameters_and_no_observation_normalisation():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from highdim import train as HT
    model = HT.build_model(DummyVecEnv([lambda: _wrapper()]), seed=0)
    assert (model.n_steps, model.batch_size, model.n_epochs, model.gamma, model.gae_lambda,
            model.ent_coef, model.target_kl) == (2048, 256, 10, 0.99, 0.95, 0.01, 0.02)
    assert model.learning_rate(1.0) == pytest.approx(3e-4)
    assert model.learning_rate(0.5) == pytest.approx(1.5e-4)          # linear
    assert model.policy.net_arch == [256, 256]
    assert not isinstance(model.get_env(), VecNormalize)
    assert str(model.device) == "cpu"


def test_block_runner_drives_a_saved_checkpoint_like_the_harness(ppo_model, tmp_path):
    path = tmp_path / "ckpt.zip"
    ppo_model.save(path)
    seeds = HS.seed_block("HD_DEV")[PPO_EVENT_SEED_INDEX:PPO_EVENT_SEED_INDEX + 1]
    res = HB.run_block("ppo_random", seeds, conditions=("fixed",), workers=1,
                       checkpoint=tmp_path / "b.jsonl", traces=tmp_path / "b.traces.jsonl.gz",
                       max_steps=SHORT, model=str(path))
    direct = _ppo_episode(ppo_model, seed=seeds[0])
    got = res["fixed"][0]
    assert got["outcome"] == direct["outcome"] and got["steps"] == direct["steps"]
    assert got["path_length"] == direct["path_length"]
    assert HB.fingerprint("ppo_random", seeds, ("fixed",), SHORT, "", model=str(path)) != \
        HB.fingerprint("ppo_random", seeds, ("fixed",), SHORT, "")


def test_short_training_writes_checkpoints_metadata_and_dev_evaluations(tmp_path):
    from stable_baselines3 import PPO
    from highdim import train as HT
    out = tmp_path / "run"
    HT.train(run=99, seed=0, total_timesteps=64, out_dir=out, n_envs=2, every=32,
             eval_seeds=1, eval_workers=1, eval_max_steps=20,
             overrides={"n_steps": 16, "batch_size": 16}, vec="dummy")
    for name in ("ckpt_32", "ckpt_64", "final"):
        assert (out / f"{name}.zip").exists()
        meta = json.loads((out / f"{name}.json").read_text())
        assert meta["solver"] == "frozen" and meta["L"] == 1.0 and meta["hold"] == 1
        assert meta["seed"] == 0 and meta["run"] == 99
        assert meta["steps"] == (64 if name == "final" else int(name[5:]))
        assert len(meta["commit"]) == 40 and len(meta["frozen_params_sha256"]) == 2
        assert meta["env_conditions"] == ["fixed", "randomized"]
    evals = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines()]
    assert [e["steps"] for e in evals] == [32, 64]
    assert all(e["seeds"] == HS.seed_block("HD_DEV", 1) and e["n"] == 2 for e in evals)
    model = PPO.load(out / "final.zip", device="cpu")
    env = make_env(True)
    env.MAX_STEPS = 10
    rec = H.run_episode("ppo_random", "randomized", HS.seed_block("HD_DEV")[0], env=env,
                        model=model)
    assert rec["steps"] == 10 and rec["solver"] == "frozen"


def test_sb3_training_resets_each_env_on_its_hd_train_seed():
    # PPO(seed=...) re-seeds the vec env with seed + i; the trainer must restore HD_TRAIN seeds
    from highdim import train as HT
    run = 7
    venv = HT.make_vec_env(run, solver="frozen", n_envs=2, vec="dummy")
    model = HT.build_model(venv, seed=1, run=run, overrides={"n_steps": 8, "batch_size": 8})
    model._setup_learn(16, callback=None)              # what learn() does before the first rollout
    got = model._last_obs
    for i, (cond, s) in enumerate(HT.env_specs(run, 2)):
        want, _ = _wrapper(cond, seed=s).reset()
        assert np.array_equal(got[i], want), (i, s)
    venv.close()


# =========================================================================== ticket 08
# The 1M-step pilot: training health (HD_DESIGN.md section 9.1 sanity), the pilot rule and its
# single hold-5 fallback, and the pilot report on HD_DEV.

def test_short_training_logs_training_health_per_update(tmp_path):
    from highdim import train as HT
    out = tmp_path / "run"
    final = HT.train(run=99, seed=0, total_timesteps=64, out_dir=out, n_envs=2, every=1000,
                     eval_seeds=1, eval_workers=1, overrides={"n_steps": 16, "batch_size": 16},
                     vec="dummy")
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["ppo_steps"] for r in log] == [32, 64]          # one row per policy update
    for r in log:
        for k in ("approx_kl", "clip_fraction", "explained_variance", "loss", "value_loss",
                  "policy_gradient_loss", "entropy_loss", "std"):
            assert np.isfinite(r[k]), k
        full = 10 * (32 // 16)                               # n_epochs x minibatches
        assert 1 <= r["epochs"] <= 10 and 1 <= r["grad_steps"] <= full
        assert r["early_stop"] == (r["grad_steps"] < full)
        assert r["finite"] is True
    h = final["health"]
    assert h["updates"] == 2 and h["rollouts"] == 2 and h["nonfinite_rollouts"] == 0
    assert h["nonfinite_updates"] == 0 and h["diag_steps"] == 64 == final["control_steps"]
    assert h["diag_incomplete"] == 0
    assert json.loads((out / "final.json").read_text())["health"] == h


def test_rollout_check_flags_any_non_finite_buffer_entry():
    from types import SimpleNamespace
    from highdim import train as HT
    ok = {k: np.zeros((4, 2)) for k in HT.BUFFER_FIELDS}
    assert HT.buffer_finite(SimpleNamespace(**ok))
    for k in HT.BUFFER_FIELDS:
        bad = dict(ok)
        bad[k] = np.array([[0.0, np.nan]])
        assert not HT.buffer_finite(SimpleNamespace(**bad)), k


def test_diagnostic_check_flags_a_step_missing_a_key_or_with_a_non_finite_action():
    from highdim import train as HT
    from highdim.wrapper import DIAGNOSTIC_KEYS
    good = {k: 0.0 for k in DIAGNOSTIC_KEYS}
    good.update(a_raw=[0.1, 0.2], a_disc=[0.1, 0.2])
    assert HT.diagnostic_complete(good)
    assert not HT.diagnostic_complete({k: v for k, v in good.items() if k != "min_cbc"})
    assert not HT.diagnostic_complete({**good, "a_raw": [np.nan, 0.0]})
    assert not HT.diagnostic_complete({**good, "u_exec": [np.inf, 0.0]})


def test_training_health_counts_a_step_without_diagnostics_as_incomplete():
    from highdim import train as HT
    from highdim.wrapper import DIAGNOSTIC_KEYS
    d = {k: [0.0, 0.0] for k in DIAGNOSTIC_KEYS}
    h = HT.TrainingHealth("unused")
    h.locals = {"infos": [{"diagnostics": [d, d]}, {}, {"diagnostics": []}]}
    h._on_step()
    assert (h.diag_steps, h.diag_incomplete) == (2, 2)


def _health(**kw):
    h = {"updates": 62, "rollouts": 62, "nonfinite_rollouts": 0, "nonfinite_updates": 0,
         "diag_steps": 1_015_808, "diag_incomplete": 0}
    return {**h, **kw}


def _log(ev_last=0.4, n=62):
    return [{"ppo_steps": 16_384 * (i + 1), "explained_variance": 0.1 if i < n - 1 else ev_last,
             "approx_kl": 0.01, "early_stop": False, "epochs": 10, "finite": True}
            for i in range(n)]


def test_pilot_sanity_needs_finite_training_full_diagnostics_and_positive_explained_variance():
    assert HG.pilot_sanity(_health(), _log())["pass"]
    for h in (_health(nonfinite_rollouts=1), _health(nonfinite_updates=1),
              _health(diag_incomplete=1), _health(diag_steps=0)):
        assert not HG.pilot_sanity(h, _log())["pass"], h
    s = HG.pilot_sanity(_health(), _log(ev_last=0.0))
    assert not s["pass"] and s["explained_variance"] == 0.0     # > 0 is required
    assert not HG.pilot_sanity(_health(), [])["pass"]


def test_pilot_sanity_reports_but_does_not_gate_kl_and_early_stopping():
    log = _log()
    for r in log:
        r.update(approx_kl=0.09, early_stop=True, epochs=1)
    s = HG.pilot_sanity(_health(), log)
    assert s["pass"] and s["early_stop_frac"] == 1.0 and s["kl_below_target_frac"] == 0.0


def _pair(k, n=200):
    return [{"condition": "fixed" if i < n // 2 else "randomized", "seed": i % (n // 2),
             "success": int(i < k)} for i in range(n)]


def test_pilot_learning_rule_is_a_tie_passing_point_comparison_with_the_floor():
    assert HG.pilot_learning(_pair(124), _pair(124))["pass"]          # a tie passes
    assert HG.pilot_learning(_pair(125), _pair(124))["pass"]
    r = HG.pilot_learning(_pair(123), _pair(124))
    assert not r["pass"] and (r["S_ppo"], r["S_floor"], r["n"]) == (0.615, 0.62, 200)


def test_pilot_learning_rule_requires_pairing_by_condition_and_seed():
    b = _pair(10)
    b[0], b[1] = b[1], b[0]
    with pytest.raises(ValueError):
        HG.pilot_learning(_pair(10), b)


def test_pilot_verdict_allows_exactly_one_hold_5_re_pilot():
    ok = {"sanity": {"pass": True}, "learning": {"pass": True}}
    slow = {"sanity": {"pass": True}, "learning": {"pass": False}}
    assert HG.pilot_verdict({1: ok}) == {"verdict": "viable", "hold": 1}
    assert HG.pilot_verdict({1: slow}) == {"verdict": "re-pilot", "hold": 5}
    assert HG.pilot_verdict({1: slow, 5: ok}) == {"verdict": "viable", "hold": 5}
    assert HG.pilot_verdict({1: slow, 5: slow}) == {"verdict": "NO-GO", "hold": None}
    with pytest.raises(ValueError):
        HG.pilot_verdict({5: ok})                        # hold 5 only after a hold-1 failure
    with pytest.raises(ValueError):
        HG.pilot_verdict({1: ok, 5: ok})


def test_pilot_verdict_sends_a_sanity_failure_back_as_a_bug_not_a_re_pilot():
    bug = {"sanity": {"pass": False}, "learning": {"pass": True}}
    assert HG.pilot_verdict({1: bug}) == {"verdict": "fix and re-run", "hold": 1}
    slow = {"sanity": {"pass": True}, "learning": {"pass": False}}
    assert HG.pilot_verdict({1: slow, 5: bug}) == {"verdict": "fix and re-run", "hold": 5}


def test_subgoal_stats_measure_gamma_distance_and_goal_switches():
    from experiments.highdim import hd2_pilot as HD2
    goal = [5.0, 0.0]
    row = {"trajectory": [[0.0, 0.0], [1.0, 0.0], [4.5, 0.0], [5.0, 0.0]],
           "trace": [{"gamma": [0.5, 0.0], "u_nom": [1.0, 0.0], "u_exec": [1.0, 0.0],
                      "qp_status": "optimal", "random_event": False},
                     {"gamma": [2.0, 0.0], "u_nom": [1.0, 0.0], "u_exec": [0.0, 0.0],
                      "qp_status": "infeasible", "random_event": True},
                     {"gamma": [5.0 + 2e-6, 0.0], "u_nom": [1.0, 0.0], "u_exec": [0.9, 0.0],
                      "qp_status": "optimal", "random_event": False}]}
    s = HD2.step_stats([row], [{"goal": goal}])
    assert s["steps"] == 3
    assert s["goal_switch_rate"] == pytest.approx(1 / 3)
    assert s["gamma_dist_mean"] == pytest.approx((0.5 + 1.0 + 0.5) / 3, abs=1e-5)
    assert s["gamma_dist_mean_off_goal"] == pytest.approx(0.75)
    assert s["infeasible_rate"] == pytest.approx(1 / 3)
    assert s["random_event_rate"] == pytest.approx(1 / 3)
    assert s["intervention_rate"] == pytest.approx(1 / 3)       # |u_exec - u_nom| > 0.1
    assert s["u_dev_mean"] == pytest.approx((0 + 1.0 + 0.1) / 3)
    assert sum(s["gamma_dist_hist"].values()) == 3


def test_hd2_smoke_evaluates_a_checkpoint_on_hd_dev_and_reports(tmp_path, monkeypatch):
    from experiments.highdim import hd2_pilot as HD2
    from highdim import train as HT
    run_dir = tmp_path / "run"
    HT.train(run=99, seed=0, total_timesteps=64, out_dir=run_dir, n_envs=2, every=32,
             eval_seeds=1, eval_workers=1, eval_max_steps=20,
             overrides={"n_steps": 16, "batch_size": 16}, vec="dummy")
    opened = []
    real = HD2.seed_block
    monkeypatch.setattr(HD2, "seed_block",
                        lambda name, *a, **k: opened.append(name) or real(name, *a, **k))
    a = HD2.evaluate_pilot(run_dir, tmp_path / "out", hold=1, mark=64, n=2, workers=1,
                           max_steps=20)
    assert set(opened) == {"HD_DEV"}
    assert a["hold"] == 1 and a["mark"] == 64 and a["learning"]["n"] == 4
    assert set(a["summaries"]) == {"ppo", "floor", "ceiling"}
    assert a["step_stats"]["ppo"]["steps"] == sum(
        r["steps"] for c in ("fixed", "randomized") for r in a["records"][c])
    assert [e["steps"] for e in a["learning_curve"]] == [32, 64]
    text = HD2.report({1: a}, HD2.provenance())
    for must in ("## Verdict", "## Implementation sanity", "## Learning curve",
                 "goal-switch", "γ distance", "intervention", "QP infeasible",
                 "Random event", "dynamic", "static", "wall",
                 HD2.VERDICT_TEXT[HD2._verdict({1: a})["verdict"]][:12]):
        assert must in text, must
