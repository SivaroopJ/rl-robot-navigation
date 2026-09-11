"""Week 6 Continuation: implementation-safety tests (spec 'REQUIRED IMPLEMENTATION SAFETY' 1-7)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from continuation import seeds as S
from continuation.harness import ArmSpec, run_block, run_episode
from continuation.params import REFERENCE, DRCBFParams, h1_grid
from continuation.policy import TunableDRCBFPolicy, attach_recovery, build_arm, policy_kwargs
from continuation.random_recovery import (RandomRecovery, RandomSpec, candidate_directions,
                                          margins, select)
from dr_control.drccp_controller import cbc_value
from dr_control.policy import DRCBFPolicy
from experiments.exp6_ppo_comparison import make_env

REPO = Path(__file__).resolve().parents[1]
DEV = 20_005        # a dev-block seed with frozen-QP infeasibility (read-only dev block)


# --------------------------------------------------------------------------- 1-2 random velocity
@pytest.mark.parametrize("K", [4, 8, 16])
@pytest.mark.parametrize("v", [0.2, 0.6, 1.0])
def test_candidates_are_bounded_and_evenly_rotated(K, v):
    rng = np.random.default_rng(0)
    for _ in range(200):
        phi = rng.uniform(0, 2 * np.pi)
        th, d = candidate_directions(phi, K)
        U = np.clip(v * d, -1.0, 1.0)
        assert np.all(np.linalg.norm(U, axis=1) <= 1.0 + 1e-12)
        assert np.all(np.abs(U) <= 1.0)
        assert np.allclose(np.linalg.norm(U, axis=1), v)
        assert np.allclose(np.diff(th), 2 * np.pi / K)


def test_spec_rejects_speeds_above_the_action_bound():
    with pytest.raises(ValueError):
        RandomSpec(speeds=(0.2, 1.2))
    with pytest.raises(ValueError):
        RandomSpec(speeds=(0.6, 0.2))


def test_margins_match_the_frozen_cbc_value():
    rng = np.random.default_rng(1)
    for _ in range(50):
        ang = rng.uniform(0, 2 * np.pi, 5)
        kept = np.column_stack([rng.uniform(-1, 0, 5), rng.uniform(-.1, 1, 5),
                                np.cos(ang), np.sin(ang)])
        U = rng.uniform(-1, 1, (7, 2))
        m = margins(kept, U, 0.4)
        ref = [min(cbc_value(r, u, 0.4) for r in kept) for u in U]
        assert np.allclose(m, ref)


def test_selection_is_argmax_and_escalates_until_m_nonnegative():
    # one wall-like row: CBC = -0.5 + 0.4*0.1 + u_x  -> needs u_x >= 0.46
    kept = np.array([[-0.5, 0.1, 1.0, 0.0]])
    spec = RandomSpec(K=8, speeds=(0.2, 0.4, 0.6, 0.8, 1.0))
    best, acc, per = select(kept, 0.4, 0.0, spec)
    assert acc and best["v"] == 0.6 and len(per) == 3        # 0.2, 0.4 fail; 0.6 passes
    assert best["m"] >= 0
    for row in per:                                           # each speed picked its argmax
        th, d = candidate_directions(0.0, 8)
        assert np.isclose(row["m"], margins(kept, row["v"] * d, 0.4).max())


def test_fallback_is_best_over_all_speeds_when_none_is_acceptable():
    kept = np.array([[-5.0, 0.0, 1.0, 0.0]])                  # unattainable
    spec = RandomSpec(K=8, speeds=(0.2, 0.6, 1.0))
    best, acc, per = select(kept, 0.4, 0.0, spec)
    assert not acc and len(per) == 3
    assert best["v"] == 1.0 and np.isclose(best["m"], max(r["m"] for r in per))


def _wrapped_ctrl(status="infeasible"):
    class Ctrl:
        rateh, max_v, prev_u = 0.4, 1.0, np.zeros(2)
        def generate_controller(self, p, gamma, xi, *, u_nom=None, record=None):
            record.update(status=self.status, xi_kept=np.atleast_2d(xi))
            return np.zeros(2)
    c = Ctrl()
    c.status = status
    return c


def test_inert_on_feasible_steps_and_strict_persistence():
    c = _wrapped_ctrl("optimal")
    w = RandomRecovery(c, spec=RandomSpec(K=8, H=3), rng=np.random.default_rng(0), tau=0.04)
    xi = np.array([[-0.2, 0.3, 1.0, 0.0]])
    rec = {}
    assert np.array_equal(c.generate_controller(0, 0, xi, record=rec), np.zeros(2))
    assert rec["mode"] == "qp" and not w.events
    c.status = "infeasible"
    u0 = c.generate_controller(0, 0, xi, record={})
    c.status = "optimal"                                     # QP recovers, persistence holds
    u1 = c.generate_controller(0, 0, xi, record={})
    u2 = c.generate_controller(0, 0, xi, record={})
    u3 = c.generate_controller(0, 0, xi, record={})
    assert np.array_equal(u0, u1) and np.array_equal(u0, u2) and not np.array_equal(u0, u3)
    assert [r["mode"] for r in w.records] == ["qp", "event", "persist", "persist", "qp"]
    assert np.array_equal(c.prev_u, u2)          # prev_u tracks the executed action
    w.detach()


# --------------------------------------------------------------------------- 3 reproducibility
def test_controller_rng_is_reproducible_and_independent_of_env():
    a = S.controller_rng(10_400_007).uniform(size=5)
    b = S.controller_rng(10_400_007).uniform(size=5)
    c = S.controller_rng(10_400_008).uniform(size=5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    env = make_env(True)
    env.reset(seed=10_400_007)
    e = env.np_random.uniform(size=5)
    assert not np.allclose(a, e)


def test_random_arm_episode_is_bit_reproducible():
    arm = ArmSpec("R2", "random", random=RandomSpec(speeds=(0.2, 0.6, 1.0)))
    r1 = run_episode(arm, "fixed", DEV)
    r2 = run_episode(arm, "fixed", DEV)
    assert r1["n_recovery_events"] > 0, "test seed must exercise the recovery"
    for k in ("outcome", "steps", "min_clearance", "path_length", "events", "n_infeasible"):
        if k == "events":
            strip = lambda E: [{x: v for x, v in e.items() if x != "t_recovery"} for e in E]
            assert strip(r1[k]) == strip(r2[k])
        else:
            assert r1[k] == r2[k], k


# --------------------------------------------------------------------------- 4 paired fairness
@pytest.mark.parametrize("cond", ["fixed", "randomized"])
def test_obstacle_trajectory_is_independent_of_the_agent(cond):
    trajs = []
    for policy in ("zero", "random"):
        env = make_env(cond == "randomized")
        env.reset(seed=10_000_003)
        rng = np.random.default_rng(3)
        pos = []
        for _ in range(60):
            a = np.zeros(2) if policy == "zero" else rng.uniform(-1, 1, 2)
            _, _, term, trunc, _ = env.step(a)
            pos.append(env.obstacle_positions.copy())
            if term or trunc:
                break
        trajs.append(pos)
    n = min(len(t) for t in trajs)
    assert n >= 5
    assert all(np.array_equal(trajs[0][i], trajs[1][i]) for i in range(n))


def test_run_block_pairs_arms_on_identical_episodes():
    arms = [ArmSpec("A", "tuned"), ArmSpec("B", "random", random=RandomSpec())]
    res = run_block(arms, [20_003, 20_005], conditions=("fixed",), workers=1)
    for i in range(2):
        assert res["A"]["fixed"][i]["start"] == res["B"]["fixed"][i]["start"]
        assert res["A"]["fixed"][i]["goal"] == res["B"]["fixed"][i]["goal"]


# --------------------------------------------------------------------------- 5 no privileged info
FORBIDDEN_IMPORTS = ("robot_env", "dr_control.dynamic_cbf", "dr_control.cbf_sources",
                     "stable_baselines3", "torch", "sb3_contrib", "generalization")


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return tree


def _imports(path):
    tree = _strip_docstrings(ast.parse(Path(path).read_text()))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    return mods, ast.unparse(tree)


@pytest.mark.parametrize("mod", ["continuation/random_recovery.py", "continuation/policy.py",
                                 "continuation/params.py"])
def test_controller_modules_cannot_reach_ground_truth(mod):
    mods, src = _imports(REPO / mod)
    for m in mods:
        assert not any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS), m
    for name in ("push_truth", "OracleVelocitySource", "obstacle_positions",
                 "obstacle_velocities", "np_random"):
        assert name not in src, name


def test_random_recovery_imports_only_numpy_and_stdlib():
    mods, _ = _imports(REPO / "continuation/random_recovery.py")
    assert mods <= {"__future__", "time", "dataclasses", "numpy"}, mods


@pytest.mark.parametrize("kind", ["tuned", "ladder", "random"])
def test_actions_ignore_the_ground_truth_observation_block(kind):
    env = make_env(False)
    obs, _ = env.reset(seed=DEV)
    pols, wraps = [], []
    for _ in range(2):
        p = build_arm(kind, env)
        p.reset(obs, env.agent_position)
        wraps.append(attach_recovery(kind, p, random_spec=RandomSpec(),
                                     rng=S.controller_rng(DEV)))
        pols.append(p)
    n_inf = 0
    for _ in range(80):
        bad = obs.copy()
        bad[28:52] = -7.5
        a, _ = pols[0].predict(obs, env.agent_position)
        b, _ = pols[1].predict(bad, env.agent_position)
        assert np.array_equal(a, b)
        n_inf += int(pols[0].last_step_infeasible)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    assert n_inf > 0, "rollout must exercise the infeasible branch"


# --------------------------------------------------------------------------- 6 frozen code path
@pytest.mark.parametrize("cond", ["fixed", "randomized"])
def test_tunable_policy_at_reference_is_bit_identical_to_frozen(cond):
    env_a, env_b = make_env(cond == "randomized"), make_env(cond == "randomized")
    for seed in (DEV, 21_004):
        oa, _ = env_a.reset(seed=seed)
        ob, _ = env_b.reset(seed=seed)
        pa = DRCBFPolicy(**policy_kwargs(env_a))
        pb = TunableDRCBFPolicy(params=REFERENCE, **policy_kwargs(env_b))
        pa.reset(oa, env_a.agent_position)
        pb.reset(ob, env_b.agent_position)
        for _ in range(500):
            a, _ = pa.predict(oa, env_a.agent_position)
            b, _ = pb.predict(ob, env_b.agent_position)
            assert np.array_equal(a, b)
            oa, _, ta, tra, _ = env_a.step(a)
            ob, _, tb, trb, _ = env_b.step(b)
            if ta or tra:
                break
        assert pa.n_solver_fail == pb.n_solver_fail


def test_frozen_arms_refuse_tuning():
    env = make_env(False)
    with pytest.raises(ValueError):
        build_arm("original", env, params=DRCBFParams(alpha=0.8))
    with pytest.raises(ValueError):
        build_arm("c0b", env, params=DRCBFParams(alpha=0.8))


def test_tuned_arms_carry_no_t1_cap():
    env = make_env(False)
    obs, _ = env.reset(seed=DEV)
    for kind in ("tuned", "ladder", "random"):
        p = build_arm(kind, env, params=DRCBFParams(alpha=0.8))
        p.reset(obs, env.agent_position)
        assert type(p.src).__name__ == "EstimatedLidarBarrierSource"
        assert not hasattr(p.src, "v_cap")


# --------------------------------------------------------------------------- params
def test_tau_eff_includes_alpha_above_one():
    assert np.isclose(REFERENCE.tau_eff(), 0.04)
    assert np.isclose(DRCBFParams(alpha=1.2).tau_eff(), 0.048)
    assert np.isclose(DRCBFParams(alpha=2.0, wasserstein_r=0.012, epsilon=0.05).tau_eff(), 0.48)


def test_regimes_and_degeneracy():
    assert DRCBFParams(epsilon=0.2).regime == "collapsed"          # eps*N = 1.0
    assert DRCBFParams(epsilon=0.3).regime == "non_collapsed"      # eps*N = 1.5
    assert DRCBFParams(epsilon=0.3, k_scans=3).regime == "collapsed"
    a = DRCBFParams(wasserstein_r=0.002, epsilon=0.05)
    b = DRCBFParams(wasserstein_r=0.004, epsilon=0.10)
    assert a.effective_class() == b.effective_class()
    c = DRCBFParams(wasserstein_r=0.003, epsilon=0.3)
    d = DRCBFParams(wasserstein_r=0.004, epsilon=0.4)
    assert c.effective_class() != d.effective_class()


def test_h1_grid_shape_and_distinct_classes():
    g = h1_grid()
    assert len(g) == 120 and len(set(g)) == 120
    col = {p.effective_class() for p in g if p.regime == "collapsed"}
    assert len(col) == 50                                           # audit A2


def test_ladder_gets_the_correct_tau():
    env = make_env(False)
    obs, _ = env.reset(seed=DEV)
    for prm, tau in ((REFERENCE, 0.04), (DRCBFParams(alpha=2.0), 0.08)):
        p = build_arm("ladder", env, params=prm)
        p.reset(obs, env.agent_position)
        lad = attach_recovery("ladder", p, params=prm)
        assert np.isclose(lad.tau, tau)
        lad.detach()
    with pytest.raises(ValueError):
        attach_recovery("random", build_arm("random", env, params=DRCBFParams(epsilon=0.3)),
                        params=DRCBFParams(epsilon=0.3), random_spec=RandomSpec(),
                        rng=S.controller_rng(0))


# --------------------------------------------------------------------------- seeds / 7 protection
def test_final_block_is_sealed():
    with pytest.raises(PermissionError):
        S.seed_block("FINAL")
    assert S.seed_block("FINAL", allow_final=True)[0] == 10_900_000


def test_only_cont_final_opens_the_final_block():
    users = []
    for f in list((REPO / "experiments/week6_continuation").glob("*.py")) + \
            list((REPO / "continuation").glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and any(
                    k.arg == "allow_final" and not (isinstance(k.value, ast.Constant)
                                                    and k.value.value is False)
                    for k in n.keywords):
                users.append(f.name)
    assert set(users) <= {"cont_final.py"}, users


def test_no_continuation_block_touches_an_earlier_reservation():
    S._check_disjoint()
    for name in S.BLOCKS:
        if name == "FINAL":
            continue
        for s in S.seed_block(name):
            assert s >= 10_000_000


def test_week5_and_week55_artefacts_are_untouched():
    from experiments.week6_continuation.protect_manifest import verify
    modified, missing = verify()
    assert not modified and not missing, (modified, missing)


def test_checkpoint_resume_reproduces_the_uninterrupted_block(tmp_path):
    arms = [ArmSpec("A", "tuned")]
    seeds = [20_003, 20_005]
    ck = tmp_path / "b.jsonl"
    full = run_block(arms, seeds, conditions=("fixed",), workers=1, checkpoint=ck)
    lines = ck.read_text().splitlines()
    ck.write_text("\n".join(lines[:2]) + "\n" + lines[2][:40])     # header + 1 row + torn row
    resumed = run_block(arms, seeds, conditions=("fixed",), workers=1, checkpoint=ck)
    skip = ("step_time_ms_mean", "step_time_ms_median", "step_time_ms_p95", "step_time_ms_max",
            "qp_time_ms_mean", "recovery_overhead_ms_mean", "episode_time_s", "mean_step_time")
    for a, b in zip(full["A"]["fixed"], resumed["A"]["fixed"]):
        norm = lambda r: json.dumps({k: v for k, v in r.items() if k not in skip},
                                    sort_keys=True)       # NaN-safe structural equality
        assert norm(a) == norm(b)
    with pytest.raises(SystemExit):                                   # different definition
        run_block([ArmSpec("A", "original")], seeds, conditions=("fixed",), workers=1,
                  checkpoint=ck)
