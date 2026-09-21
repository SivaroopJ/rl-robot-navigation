"""Summaries, paired statistics and the pre-declared selection rule (AUDIT.md section 6).

Binary paired outcomes: exp6's exact McNemar + exp6's paired bootstrap CI (10 000 reps, 95 %),
imported unchanged. Continuous per-episode metrics: the same paired bootstrap. Step-level rates
(e.g. infeasible steps / steps): EPISODE-CLUSTER bootstrap of a ratio of sums, so timesteps are
never treated as independent.
"""
from __future__ import annotations

import numpy as np

from experiments.exp6_ppo_comparison import mcnemar, paired_ci, summarise

N_BOOT = 10_000
BOOT_SEED = 12345           # exp6's


def _fin(v):
    v = [x for x in v if x is not None and x == x]
    return v


def summary(recs, label):
    """exp6.summarise (unchanged) + continuation-specific aggregates."""
    s = summarise(recs, label)
    n = len(recs)
    steps = sum(r["steps"] for r in recs)
    trig = sum(r["n_triggers"] for r in recs)
    trig_det = sum(r["n_trigger_success"] + r["n_trigger_fail"] for r in recs)
    ev = sum(r["n_infeasible_events"] for r in recs)
    with_inf = [r for r in recs if r["n_infeasible"] > 0]
    tiers = {k: sum(r["tier_counts"].get(k, 0) for r in recs) for k in ("0", "1", "2")}
    ntier = max(sum(tiers.values()), 1)
    def mean(k):
        v = _fin([r.get(k) for r in recs])
        return float(np.mean(v)) if v else float("nan")
    def pct(k, q):
        v = _fin([r.get(k) for r in recs])
        return float(np.percentile(v, q)) if v else float("nan")
    s.update({
        "infeasible_step_rate": sum(r["n_infeasible"] for r in recs) / max(steps, 1),
        "infeasible_events": ev, "infeasible_events_per_episode": ev / max(n, 1),
        "infeasible_duration_mean": (float(np.mean(_fin([r["infeasible_duration_mean"]
                                                         for r in recs])))
                                     if _fin([r["infeasible_duration_mean"] for r in recs])
                                     else float("nan")),
        "episodes_with_infeasibility": len(with_inf),
        "P_collision_given_infeasibility": (float(np.mean([r["collision"] for r in with_inf]))
                                            if with_inf else float("nan")),
        "P_event_collision_within_1s": (sum(r["n_events_coll_within_window"] for r in recs)
                                        / ev if ev else float("nan")),
        "recovery_fraction": sum(r["n_recovery_steps"] for r in recs) / max(steps, 1),
        "trigger_count": trig,
        "recovery_success_rate": (sum(r["n_trigger_success"] for r in recs) / trig_det
                                  if trig_det else float("nan")),
        "tier_frac_T0": tiers["0"] / ntier, "tier_frac_T1": tiers["1"] / ntier,
        "tier_frac_T2": tiers["2"] / ntier,
        "cbf_min_margin_worst": float(np.min(_fin([r["cbf_min_margin"] for r in recs]) or [np.nan])),
        "m_u0_infeasible_mean": mean("m_u0_infeasible_mean"),
        "m_exec_infeasible_mean": mean("m_exec_infeasible_mean"),
        "clf_slack_mean": mean("clf_slack_mean"),
        "action_norm_mean": mean("action_norm_mean"),
        "step_time_ms_mean_ep": mean("step_time_ms_mean"),
        "step_time_ms_median_ep": mean("step_time_ms_median"),
        "step_time_ms_p95_ep": mean("step_time_ms_p95"),
        "step_time_ms_worst": float(np.max(_fin([r["step_time_ms_max"] for r in recs]))),
        "qp_time_ms_mean": mean("qp_time_ms_mean"),
        "recovery_overhead_ms_mean": mean("recovery_overhead_ms_mean"),
        "episode_time_s_mean": mean("episode_time_s"),
        "min_clearance_p05": pct("min_clearance", 5),
    })
    if any("n_recovery_events" in r for r in recs):
        E = [e for r in recs for e in r.get("events", [])]
        s.update({
            "n_recovery_events": len(E),
            "recovery_speed_mean": float(np.mean([e["speed"] for e in E])) if E else float("nan"),
            "recovery_accept_frac": float(np.mean([e["accepted"] for e in E])) if E else float("nan"),
            "candidates_mean": float(np.mean([e["n_candidates"] for e in E])) if E else float("nan"),
            "random_eval_ms_mean": (float(np.mean([e["t_recovery"] for e in E])) * 1e3
                                    if E else float("nan")),
            "m_before_recovery_mean": float(np.mean([e["m_u0"] for e in E])) if E else float("nan"),
            "m_selected_mean": float(np.mean([e["m"] for e in E])) if E else float("nan"),
            "speed_hist": {f"{v:g}": int(sum(1 for e in E if abs(e["speed"] - v) < 1e-9))
                           for v in sorted({e["speed"] for e in E})},
        })
    return s


def summaries(res_arm, label):
    """fixed / randomized / pooled for one arm's {condition: recs}."""
    out = {c: summary(r, f"{label} [{c}]") for c, r in res_arm.items()}
    out["pooled"] = summary([x for r in res_arm.values() for x in r], f"{label} [pooled]")
    return out


# --------------------------------------------------------------------------- paired tests
def paired_binary(x, y, key):
    a = [int(r[key]) for r in x]
    b = [int(r[key]) for r in y]
    return {**mcnemar(a, b), **paired_ci(a, b, n_boot=N_BOOT, seed=BOOT_SEED)}



def paired_continuous(x, y, key):
    a = np.array([r[key] for r in x], float)
    b = np.array([r[key] for r in y], float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return {"diff": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    return {**paired_ci(a[ok], b[ok], n_boot=N_BOOT, seed=BOOT_SEED), "n": int(ok.sum())}


def paired_cluster_ratio(x, y, num, den="steps"):
    """Paired episode-cluster bootstrap of (sum num / sum den)_x - (...)_y."""
    nx = np.array([r[num] for r in x], float)
    dx = np.array([r[den] for r in x], float)
    ny = np.array([r[num] for r in y], float)
    dy = np.array([r[den] for r in y], float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(nx), size=(N_BOOT, len(nx)))
    bx = nx[idx].sum(1) / np.maximum(dx[idx].sum(1), 1)
    by = ny[idx].sum(1) / np.maximum(dy[idx].sum(1), 1)
    d = bx - by
    return {"x": float(nx.sum() / max(dx.sum(), 1)), "y": float(ny.sum() / max(dy.sum(), 1)),
            "diff": float(nx.sum() / max(dx.sum(), 1) - ny.sum() / max(dy.sum(), 1)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


BINARY = ("success", "collision", "timeout")
CONTINUOUS = ("min_clearance", "spl", "path_length", "mean_u_dev", "action_norm_mean",
              "step_time_ms_mean", "step_time_ms_p95", "episode_time_s")


def compare(x, y):
    """x vs y, paired by seed. Positive diff = x higher (exp6 convention)."""
    assert [r["seed"] for r in x] == [r["seed"] for r in y], "unpaired"
    out = {k: paired_binary(x, y, k) for k in BINARY}
    for ctype in ("dynamic", "static", "wall"):
        a = [{"v": int(r["collision_type"] == ctype)} for r in x]
        b = [{"v": int(r["collision_type"] == ctype)} for r in y]
        out[f"{ctype}_collision"] = paired_binary(a, b, "v")
    for k in CONTINUOUS:
        out[k] = paired_continuous(x, y, k)
    out["infeasible_step_rate"] = paired_cluster_ratio(x, y, "n_infeasible")
    out["recovery_fraction"] = paired_cluster_ratio(x, y, "n_recovery_steps")
    return out


def compare_all(res_x, res_y):
    out = {c: compare(res_x[c], res_y[c]) for c in res_x}
    out["pooled"] = compare([r for c in res_x for r in res_x[c]],
                            [r for c in res_y for r in res_y[c]])
    return out


# --------------------------------------------------------------------------- selection rule
def rank_key(s, fifth="infeasible_step_rate"):
    """Lexicographic: collision asc, success desc, timeout asc, clearance desc, fifth asc."""
    return (round(s["collision_rate"], 12), -round(s["success_rate"], 12),
            round(s["timeout_rate"], 12), -round(s["min_clearance_mean"], 12),
            round(s[fifth], 12))


def gates(cand, ref, *, n_per_condition, g1_success_pp=0.05, g1_timeout=0.10):
    """G1 (pooled) + G2 (per condition). cand/ref: summaries() dicts. ref may be None (screen)."""
    fails = []
    p = cand["pooled"]
    if p["timeout_rate"] > g1_timeout + 1e-12:
        fails.append(f"G1 timeout {p['timeout_rate']:.3f} > {g1_timeout}")
    if ref is not None:
        if p["success_rate"] < ref["pooled"]["success_rate"] - g1_success_pp - 1e-12:
            fails.append(f"G1 success {p['success_rate']:.3f} < ref "
                         f"{ref['pooled']['success_rate']:.3f} - {g1_success_pp}")
        delta = 0.10 if n_per_condition >= 50 else 0.15
        for c in ("fixed", "randomized"):
            if cand[c]["success_rate"] < ref[c]["success_rate"] - delta - 1e-12:
                fails.append(f"G2 {c} success {cand[c]['success_rate']:.3f} < ref "
                             f"{ref[c]['success_rate']:.3f} - {delta}")
            if cand[c]["collision_rate"] > ref[c]["collision_rate"] + delta + 1e-12:
                fails.append(f"G2 {c} collision {cand[c]['collision_rate']:.3f} > ref "
                             f"{ref[c]['collision_rate']:.3f} + {delta}")
    return fails


def select(cands, ref, *, n_per_condition, fifth="infeasible_step_rate"):
    """cands: {name: summaries()}. Returns (ranked eligible names, {name: gate failures})."""
    rejected = {n: gates(s, ref, n_per_condition=n_per_condition) for n, s in cands.items()}
    eligible = [n for n in cands if not rejected[n]]
    eligible.sort(key=lambda n: (rank_key(cands[n]["pooled"], fifth), n))
    return eligible, {n: f for n, f in rejected.items() if f}
