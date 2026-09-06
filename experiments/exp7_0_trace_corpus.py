"""Week5-Phase7 / Stage 0 / trace corpus.

WHY THIS EXISTS
    Direction 2 must prove that an accelerated controller is EQUIVALENT to the frozen one
    before any runtime number is reported (gate G1, criteria E1/E2). "Equivalent" is only
    checkable against a fixed, replayable record of what the frozen controller actually did.
    This module produces that record. Nothing is optimised, changed or tuned here; Stage 0
    only observes.

    Direction 1's Stage 2 diagnosis also runs entirely off this corpus -- it re-analyses
    recorded infeasible steps offline rather than re-rolling episodes -- and Direction 3's
    Stage 5 estimator metrics reuse the ground-truth diagnostic channel recorded here.

HOW THE FROZEN CONTROLLER IS OBSERVED WITHOUT BEING TOUCHED
    `ClfCbfDrccpController.generate_controller` depends on internal state ONLY through
    `self.prev_u` (verified by reading the frozen source: every other input is an argument or
    a constructor parameter). So a step is fully replayable from
    `(p, gamma, xi, u_prev, constructor params)`.

    Rather than reimplement `DRCBFPolicy.predict` -- which would risk diverging from the
    behaviour it is supposed to record -- this module WRAPS the bound method on the live
    instance. The frozen call is made verbatim and its inputs, its `record` dict and its
    return value are copied out. No frozen file is imported-and-edited, monkeypatched at class
    level, or subclassed.

    `verify_replayable()` then closes the loop: it rebuilds a fresh controller from the
    recorded constructor parameters, sets `prev_u` to the recorded value, replays recorded
    steps, and requires the action to match BIT-FOR-BIT. If that fails, the corpus is not a
    faithful record and Stage 1 cannot use it.

INFORMATION LEDGER -- UNCHANGED FROM PHASES 2-6
    The control path still reads only obs[0:2] and obs[4:28]. Ground truth (obstacle states,
    true clearance) is recorded in a SEPARATE, explicitly labelled `truth_*` channel that is
    written to disk for later metrics and never reaches the controller. This is the same
    arrangement Phases 3 and 4 used for their metrics channels.

SCOPE NOTE (flagged, not silent)
    The plan said "Phase 5 and Phase 6 configurations". Stage 0 records the PHASE 6
    configuration only -- the real RobotNavEnv under the frozen DRCBFPolicy, in both the fixed
    and randomized obstacle-motion conditions. That is the configuration Stage 7 evaluates and
    Stage 2 diagnoses, so it is the corpus every gate consumes. Adding the Phase 5 standalone
    harness would roughly double the corpus and the code surface without serving a gate; if a
    later stage needs it, it is added then as a plan amendment.

SEED HYGIENE
    dev        = DEV_SEED_BASE            (20_000)   iterating
    validation = DEV_SEED_BASE + 1_000    (21_000)   gates, ablations, sensitivity curves
    final      = EVAL_SEED_BASE           (1_000_000) equivalence checking ONLY
    The final tier is a RECORDING of frozen behaviour on the reserved block. It is used to
    check that an accelerated controller reproduces the frozen one; no method is selected, and
    no parameter is tuned, from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dr_control.drccp_controller import ClfCbfDrccpController
from dr_control.policy import DRCBFPolicy
from evaluation.evaluate_week4 import EVAL_SEED_BASE
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp0_analytic import DEV_SEED_BASE
from experiments.exp6_ppo_comparison import make_env, true_clearance
from robot_env.robot_nav_env import RobotNavEnv

LABEL = "Week5-Phase7 / Stage 0 / trace corpus"
OUT_DIR = Path("results/week5_phase7/stage0_trace_corpus")

TIERS = {
    "dev": (DEV_SEED_BASE, 20),
    "validation": (DEV_SEED_BASE + 1_000, 50),
    "final": (EVAL_SEED_BASE, 200),
}
CONDITIONS = {"fixed": False, "randomized": True}   # -> randomize_dynamic_obstacles

#: Status strings are recorded as small ints plus this table, so the npz stays numeric.
STATUS_CODES = {}


def _status_code(status):
    return STATUS_CODES.setdefault(str(status), len(STATUS_CODES))


class ControllerTrace:
    """Wraps ONE live controller instance and copies out every call. Frozen code untouched."""

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self._orig = ctrl.generate_controller
        ctrl.generate_controller = self._traced
        self.rows = []

    def _traced(self, p, gamma, xi, *, u_nom=None, record=None):
        rec = {} if record is None else record
        u_prev = np.asarray(self.ctrl.prev_u, dtype=float).copy()
        xi_in = np.atleast_2d(np.asarray(xi, dtype=float)).copy()
        u = self._orig(p, gamma, xi_in, u_nom=u_nom, record=rec)
        self.rows.append({
            "p": np.asarray(p, dtype=float).reshape(2).copy(),
            "gamma": np.asarray(gamma, dtype=float).reshape(2).copy(),
            "xi": xi_in,
            "u_prev": u_prev,
            "u": np.asarray(u, dtype=float).reshape(2).copy(),
            "rec": dict(rec),
        })
        return u

    def detach(self):
        self.ctrl.generate_controller = self._orig


def sample_provenance(src, p):
    """(age, track_id) per xi row -- METRICS ONLY, needed by Stage 2's staleness attribution.

    `EstimatedLidarBarrierSource.samples` iterates the scan buffer newest-first and emits one
    row per buffered scan, so ROW INDEX IS SCAN AGE by construction. The owning track id is
    not returned by the frozen method, so the nearest-point argmin is recomputed here. That
    duplication is checked, not trusted: `h_check` recomputes the barrier value the same way
    and the caller asserts it matches the frozen output exactly.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    ages, tids, h_check = [], [], []
    for age, (pts, (ids, _conf)) in enumerate(zip(src.buffer, src.track_buffer)):
        d = np.linalg.norm(pts - p[None, :], axis=1)
        j = int(np.argmin(d))
        ages.append(age)
        tids.append(int(ids[j]))
        dist = float(d[j])
        h_check.append(-src.r_robot if dist < 1e-12 else dist - src.r_robot)
    return np.array(ages, dtype=np.int16), np.array(tids, dtype=np.int32), np.array(h_check)


def run_episode(env, oracle, seed, ep_index):
    """One frozen Phase 6 episode, fully instrumented. Returns (steps, episode record)."""
    obs, _ = env.reset(seed=seed)
    start, goal = env.agent_position.copy(), env.target_position.copy()
    pol = DRCBFPolicy(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                      static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED,
                      dt=env.dt, lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS)
    pol.reset(obs, env.agent_position)
    trace = ControllerTrace(pol.ctrl)

    steps, provenance, truth = [], [], []
    traj, term, trunc, info = [start.copy()], False, False, {}
    n_geom_mismatch = 0
    t_step = 0
    while not (term or trunc):
        p_before = env.agent_position.copy()
        n_before = len(trace.rows)
        action, _ = pol.predict(obs, p_before)
        if len(trace.rows) == n_before:          # controller was not called: should never happen
            raise RuntimeError("policy.predict did not call the controller")

        ages, tids, h_check = sample_provenance(pol.src, p_before)
        row = trace.rows[-1]
        h_frozen = row["xi"][:, 1]
        if h_check.shape != h_frozen.shape or not np.array_equal(h_check, h_frozen):
            n_geom_mismatch += 1                 # the argmin duplication disagreed: recorded

        truth.append({
            "clearance": true_clearance(env),
            "obs_pos": np.asarray(env.obstacle_positions, dtype=float).copy(),
            "obs_vel": np.asarray(env.obstacle_velocities, dtype=float).copy(),
        })
        provenance.append((ages, tids))
        steps.append(t_step)
        t_step += 1

        obs, _, term, trunc, info = env.step(action)
        traj.append(env.agent_position.copy())

    trace.detach()
    outcome = ("success" if info.get("success") else
               "collision" if info.get("collision") else "timeout")
    traj = np.asarray(traj)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    shortest = oracle.path_length(start, goal)
    ep = {
        "ep_index": ep_index, "seed": seed, "outcome": outcome, "steps": len(steps),
        "collision_type": info.get("collision_type") if outcome == "collision" else None,
        "start": start.tolist(), "goal": goal.tolist(),
        "path_length": path_len,
        "shortest_path": float(shortest) if shortest is not None else None,
        "spl": (float(shortest / max(path_len, shortest, 1e-9))
                if (outcome == "success" and shortest is not None) else 0.0),
        "planner_failed": int(pol.planner_failed),
        "n_infeasible": int(pol.n_infeasible),
        "n_solver_fail": int(pol.n_solver_fail),
        "geometry_recompute_mismatch": n_geom_mismatch,
    }
    return trace.rows, provenance, truth, ep


def pack(rows, provenance, truth, ep_ids):
    """Ragged per-step records -> flat numeric arrays with CSR-style offsets."""
    n = len(rows)
    xi_ptr = np.zeros(n + 1, dtype=np.int64)
    kept_ptr = np.zeros(n + 1, dtype=np.int64)
    for i, r in enumerate(rows):
        kept = np.atleast_2d(np.asarray(r["rec"].get("xi_kept", r["xi"]), dtype=float))
        xi_ptr[i + 1] = xi_ptr[i] + r["xi"].shape[0]
        kept_ptr[i + 1] = kept_ptr[i] + kept.shape[0]

    def scal(key, default=np.nan):
        return np.array([float(r["rec"].get(key, default) if r["rec"].get(key) is not None
                               else default) for r in rows], dtype=float)

    out = {
        "ep_index": np.asarray(ep_ids, dtype=np.int32),
        "step": np.concatenate([np.arange(0, 0)] + [np.arange(0)]) if n == 0 else None,
        "p": np.stack([r["p"] for r in rows]),
        "gamma": np.stack([r["gamma"] for r in rows]),
        "u_prev": np.stack([r["u_prev"] for r in rows]),
        "u": np.stack([r["u"] for r in rows]),
        "u_nom": np.stack([np.asarray(r["rec"].get("u_nom", [np.nan, np.nan]),
                                      dtype=float).reshape(2) for r in rows]),
        "xi_flat": np.concatenate([r["xi"] for r in rows], axis=0),
        "xi_ptr": xi_ptr,
        "xi_kept_flat": np.concatenate(
            [np.atleast_2d(np.asarray(r["rec"].get("xi_kept", r["xi"]), dtype=float))
             for r in rows], axis=0),
        "xi_kept_ptr": kept_ptr,
        "sample_age": np.concatenate([a for a, _ in provenance]),
        "sample_track_id": np.concatenate([t for _, t in provenance]),
        "status_code": np.array([_status_code(r["rec"].get("status")) for r in rows],
                                dtype=np.int16),
        "delta": scal("delta"),
        "h_crit": scal("h_crit"),
        "box_overshoot": scal("box_overshoot"),
        "total_time": scal("total_time"),
        "solver_time": scal("solver_time"),
        "canon_time": scal("canon_time"),
        "p1": np.array([float(r["rec"].get("weights", {}).get("p1", np.nan)) for r in rows]),
        "p3": np.array([float(r["rec"].get("weights", {}).get("p3", np.nan)) for r in rows]),
        "V": scal("V"),
        "truth_clearance": np.array([t["clearance"] for t in truth], dtype=float),
        "truth_obs_pos": np.stack([t["obs_pos"] for t in truth]),
        "truth_obs_vel": np.stack([t["obs_vel"] for t in truth]),
    }
    out["step"] = np.concatenate([np.arange(c) for c in np.bincount(
        out["ep_index"] - out["ep_index"].min(), minlength=1)]) if n else np.zeros(0, np.int32)
    return out


def derived_cbc(packed, alpha):
    """min CBC over the KEPT samples and over ALL samples, per step. Plain NumPy, no solver."""
    n = len(packed["u"])
    mk, ma = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        ub = np.array([1.0, alpha, packed["u"][i, 0], packed["u"][i, 1]])
        a0, a1 = packed["xi_ptr"][i], packed["xi_ptr"][i + 1]
        k0, k1 = packed["xi_kept_ptr"][i], packed["xi_kept_ptr"][i + 1]
        ma[i] = float(np.min(packed["xi_flat"][a0:a1] @ ub))
        mk[i] = float(np.min(packed["xi_kept_flat"][k0:k1] @ ub))
    return mk, ma


def verify_replayable(packed, params, n_check, rng):
    """Rebuild the frozen controller and replay recorded steps. Requires BIT-identical actions.

    This is the gate item that makes the corpus usable: if a step cannot be reproduced from its
    recorded inputs alone, the record is incomplete and Stage 1's equivalence test would be
    comparing against something it cannot reconstruct.
    """
    n = len(packed["u"])
    idx = rng.choice(n, size=min(n_check, n), replace=False)
    mismatches = []
    for i in idx:
        ctrl = ClfCbfDrccpController(max_v=params["max_v"], cbf_rate=params["cbf_rate"])
        ctrl.prev_u = packed["u_prev"][i].copy()
        a0, a1 = packed["xi_ptr"][i], packed["xi_ptr"][i + 1]
        u = ctrl.generate_controller(packed["p"][i], packed["gamma"][i],
                                     packed["xi_flat"][a0:a1])
        if not np.array_equal(np.asarray(u, dtype=float), packed["u"][i]):
            mismatches.append((int(i), float(np.max(np.abs(np.asarray(u) - packed["u"][i])))))
    return len(idx), mismatches


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(tier, condition, *, out_dir, replay_checks, seed_for_checks):
    seed_base, n_eps = TIERS[tier]
    env = make_env(CONDITIONS[condition])
    oracle = ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.static_obstacles)
    ctrl_ref = ClfCbfDrccpController(max_v=env.MAX_SPEED)
    params = {"clf_rate": ctrl_ref.rateV, "cbf_rate": ctrl_ref.rateh,
              "wasserstein_r": ctrl_ref.wasserstein_r, "epsilon": ctrl_ref.epsilon,
              "max_v": ctrl_ref.max_v, "k_v": ctrl_ref.k_v, "p0": ctrl_ref.p0,
              "n_keep": ctrl_ref.n_keep, "solver": ctrl_ref.solver,
              "dt": float(env.dt), "n_rays": int(env.N_LIDAR_RAYS),
              "lidar_range": float(env.LIDAR_RANGE), "agent_radius": float(env.AGENT_RADIUS)}
    params["tau"] = params["wasserstein_r"] / params["epsilon"]

    all_rows, all_prov, all_truth, ep_ids, episodes = [], [], [], [], []
    t0 = time.time()
    for i in range(n_eps):
        rows, prov, truth, ep = run_episode(env, oracle, seed_base + i, i)
        all_rows += rows
        all_prov += prov
        all_truth += truth
        ep_ids += [i] * len(rows)
        episodes.append(ep)
        if (i + 1) % 25 == 0 or i + 1 == n_eps:
            print(f"   {tier}/{condition}: {i+1}/{n_eps} episodes, "
                  f"{len(all_rows)} steps, {time.time()-t0:.0f}s", flush=True)
    env.close()

    packed = pack(all_rows, all_prov, all_truth, ep_ids)
    packed["min_cbc_kept"], packed["min_cbc_all"] = derived_cbc(packed, params["cbf_rate"])

    n_checked, mismatches = verify_replayable(
        packed, params, replay_checks, np.random.default_rng(seed_for_checks))

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"trace_{tier}_{condition}.npz"
    np.savez_compressed(npz, **packed)

    status_names = {v: k for k, v in STATUS_CODES.items()}
    # "infeasible" in the status string, matching EXACTLY how the frozen policy and Phase 6
    # counted it (dr_control/policy.py: `"infeasible" in str(status)`), so infeasible_inaccurate
    # is included and the totals are comparable to results/phase6/*.json.
    infeasible_codes = {c for c, name in status_names.items() if "infeasible" in name}
    is_infeasible = np.isin(packed["status_code"], list(infeasible_codes)) if infeasible_codes \
        else np.zeros(len(packed["u"]), dtype=bool)
    # SOLVED means status == "optimal" exactly, as EpisodeDiagnostics.min_cbc_solved defines it.
    # It is NOT the complement of infeasible: `optimal_inaccurate` and solver errors are also
    # non-optimal, the frozen controller answers them with u = 0, and pooling them with solved
    # steps corrupts min CBC. (Found in Stage 0: 2 optimal_inaccurate steps in final/randomized
    # dragged the reported floor to -0.371 while the true floor over solved steps is 0.03990.)
    optimal_code = STATUS_CODES.get("optimal")
    is_solved = (packed["status_code"] == optimal_code) if optimal_code is not None \
        else np.zeros(len(packed["u"]), dtype=bool)

    meta = {
        "label": f"{LABEL} :: {tier}/{condition}",
        "tier": tier, "condition": condition,
        "seed_base": seed_base, "episodes": n_eps,
        "steps": int(len(packed["u"])),
        "controller_params": params,
        "status_table": status_names,
        "status_counts": {status_names[int(c)]: int((packed["status_code"] == c).sum())
                          for c in np.unique(packed["status_code"])},
        "n_infeasible_steps": int(is_infeasible.sum()),
        "n_solved_steps": int(is_solved.sum()),
        "n_non_optimal_non_infeasible": int((~is_solved & ~is_infeasible).sum()),
        "replay_verification": {"checked": n_checked, "mismatches": len(mismatches),
                                "worst": (max(m[1] for m in mismatches) if mismatches else 0.0),
                                "bit_identical": not mismatches},
        "geometry_recompute_mismatch": int(sum(e["geometry_recompute_mismatch"]
                                               for e in episodes)),
        "outcomes": {k: sum(1 for e in episodes if e["outcome"] == k)
                     for k in ("success", "collision", "timeout")},
        "min_cbc_solved": (float(np.nanmin(packed["min_cbc_kept"][is_solved]))
                           if is_solved.any() else None),
        "min_cbc_all_steps": float(np.nanmin(packed["min_cbc_kept"])),
        "tau": params["tau"],
        "min_cbc_solved_minus_tau": (float(np.nanmin(packed["min_cbc_kept"][is_solved])
                                           - params["tau"]) if is_solved.any() else None),
        "mean_total_time_ms": float(np.nanmean(packed["total_time"]) * 1e3),
        "mean_solver_time_ms": float(np.nanmean(packed["solver_time"]) * 1e3),
        "mean_canon_time_ms": float(np.nanmean(packed["canon_time"]) * 1e3),
        "wall_seconds": time.time() - t0,
        "episodes_detail": episodes,
        "npz": npz.name,
        "npz_sha256": sha256(npz),
    }
    (out_dir / f"meta_{tier}_{condition}.json").write_text(json.dumps(meta, indent=1))
    return packed, meta, is_infeasible


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["dev"], choices=list(TIERS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--replay-checks", type=int, default=300,
                    help="steps re-solved from their recorded inputs and required to match")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--refresh-meta", action="store_true",
                    help="recompute derived summary fields from an existing corpus and exit")
    args = ap.parse_args()

    if args.refresh_meta:
        idx = refresh_meta(Path(args.out))
        print(f"### {LABEL}\nrefreshed {len(idx['parts'])} metas in {args.out}")
        for m in idx["parts"]:
            print(f"   {m['tier']}/{m['condition']:11s} solved={m['n_solved_steps']:6d} "
                  f"infeasible={m['n_infeasible_steps']:5d} "
                  f"other={m['n_non_optimal_non_infeasible']:3d} "
                  f"minCBC|solved={m['min_cbc_solved']:.6f} "
                  f"(tau={m['tau']}, diff={m['min_cbc_solved_minus_tau']:+.2e})")
        return

    out_dir = Path(args.out)
    print(f"### {LABEL}")
    print(f"tiers={args.tiers} conditions={args.conditions} out={out_dir}\n")

    index, infeas_parts = [], []
    for tier in args.tiers:
        for cond in args.conditions:
            packed, meta, is_inf = build(tier, cond, out_dir=out_dir,
                                         replay_checks=args.replay_checks,
                                         seed_for_checks=hash((tier, cond)) % (2**31))
            index.append(meta)
            if is_inf.any():
                infeas_parts.append((tier, cond, packed, is_inf))
            r = meta["replay_verification"]
            print(f"   {tier}/{cond}: {meta['steps']} steps, "
                  f"{meta['n_infeasible_steps']} infeasible, "
                  f"replay {r['checked']} checked / {r['mismatches']} mismatched, "
                  f"min CBC|solved {meta['min_cbc_solved']}")

    # ---- enriched infeasible corpus: every infeasible step, all tiers/conditions ----------
    if infeas_parts:
        cat = {}
        # per-SAMPLE arrays are re-sliced separately below; a per-step mask cannot index them
        per_sample = ("xi_ptr", "xi_kept_ptr", "xi_flat", "xi_kept_flat",
                      "sample_age", "sample_track_id")
        keys = [k for k in infeas_parts[0][2] if k not in per_sample]
        for k in keys:
            cat[k] = np.concatenate([p[k][m] for _, _, p, m in infeas_parts])
        xi_rows, kept_rows, xp, kp, src_tier = [], [], [0], [0], []
        for tier, cond, p, m in infeas_parts:
            for i in np.nonzero(m)[0]:
                a0, a1 = p["xi_ptr"][i], p["xi_ptr"][i + 1]
                k0, k1 = p["xi_kept_ptr"][i], p["xi_kept_ptr"][i + 1]
                xi_rows.append(p["xi_flat"][a0:a1])
                kept_rows.append(p["xi_kept_flat"][k0:k1])
                xp.append(xp[-1] + (a1 - a0))
                kp.append(kp[-1] + (k1 - k0))
                src_tier.append(f"{tier}/{cond}")
        cat["xi_flat"] = np.concatenate(xi_rows, axis=0)
        cat["xi_kept_flat"] = np.concatenate(kept_rows, axis=0)
        cat["xi_ptr"] = np.array(xp, dtype=np.int64)
        cat["xi_kept_ptr"] = np.array(kp, dtype=np.int64)
        cat["source"] = np.array(src_tier)
        # sample_age / sample_track_id are per-SAMPLE, not per-step: re-slice them properly
        for name in ("sample_age", "sample_track_id"):
            parts = []
            for tier, cond, p, m in infeas_parts:
                for i in np.nonzero(m)[0]:
                    a0, a1 = p["xi_ptr"][i], p["xi_ptr"][i + 1]
                    parts.append(p[name][a0:a1])
            cat[name] = np.concatenate(parts)
        npz = out_dir / "trace_infeasible_enriched.npz"
        np.savez_compressed(npz, **cat)
        (out_dir / "meta_infeasible_enriched.json").write_text(json.dumps({
            "label": f"{LABEL} :: enriched infeasible corpus",
            "steps": int(len(cat["u"])),
            "sources": {s: int((cat["source"] == s).sum()) for s in np.unique(cat["source"])},
            "npz": npz.name, "npz_sha256": sha256(npz),
        }, indent=1))
        print(f"\n   enriched infeasible corpus: {len(cat['u'])} steps -> {npz.name}")

    (out_dir / "INDEX.json").write_text(json.dumps(
        {"label": LABEL, "parts": index}, indent=1))
    print(f"\nwrote {out_dir}/INDEX.json")

def refresh_meta(out_dir=OUT_DIR):
    """Recompute the DERIVED fields of every meta_*.json from the recorded npz.

    Stage 0 correction, metrics only: `min_cbc_solved` was originally computed over
    `~infeasible` rather than over `status == "optimal"`, which pooled two
    `optimal_inaccurate` steps (answered with u = 0 by the frozen controller) in with the
    solved ones and reported a floor of -0.371 instead of 0.03990. The recorded npz arrays are
    unaffected -- only the summary JSON was wrong -- so the corpus is re-summarised rather than
    re-recorded.
    """
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "INDEX.json").read_text())
    for meta in index["parts"]:
        d = np.load(out_dir / meta["npz"])
        table = {int(k): v for k, v in meta["status_table"].items()}
        names = np.array([table[int(c)] for c in d["status_code"]])
        is_solved = names == "optimal"
        is_infeasible = np.char.find(names.astype(str), "infeasible") >= 0
        tau = meta["controller_params"]["tau"]
        meta.update({
            "n_infeasible_steps": int(is_infeasible.sum()),
            "n_solved_steps": int(is_solved.sum()),
            "n_non_optimal_non_infeasible": int((~is_solved & ~is_infeasible).sum()),
            "min_cbc_solved": (float(np.nanmin(d["min_cbc_kept"][is_solved]))
                               if is_solved.any() else None),
            "min_cbc_all_steps": float(np.nanmin(d["min_cbc_kept"])),
            "tau": tau,
            "min_cbc_solved_minus_tau": (float(np.nanmin(d["min_cbc_kept"][is_solved]) - tau)
                                         if is_solved.any() else None),
        })
        (out_dir / f"meta_{meta['tier']}_{meta['condition']}.json").write_text(
            json.dumps(meta, indent=1))
    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=1))
    return index

if __name__ == "__main__":
    main()
