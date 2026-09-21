"""Pre-registered decision rules of HD Experiment 1 (HD_DESIGN.md sections 7-9), as pure functions.

The stop check takes per-episode records (dicts with at least `seed` and `success`); the section 8
rules take shadow summaries (highdim.harness.ShadowSolver.summary), paired records, or verdicts.
Rates of counts are compared in exact rational arithmetic so a rule never flips on floating-point
rounding at its threshold. The McNemar p-value and the bootstrap CI are floats by nature and are
compared as floats.
"""
from __future__ import annotations

from fractions import Fraction

STOP_THRESHOLD = Fraction(5, 100)       # section 7: S_ceiling - S_floor < 0.05 -> STOP

# Section 8, fast-solver training suitability.
DU_TOL = 1e-3                           # E-a: |u_fast - u_frozen|_inf <= 1e-3 ...
DU_SHARE = Fraction(99, 100)            # ... on >= 99 % of the steps where both are optimal
MCNEMAR_P = 0.05                        # E-b: McNemar p > 0.05 on success, and
CI_BOUND = 0.03                         # the 95 % CI of the success difference inside +-0.03

# Section 9.1, the pilot.
TARGET_KL = 0.02                        # config.json; reported against, never gated
PILOT_FALLBACK_HOLD = 5                 # the single re-pilot after a learning failure


def _paired_successes(floor, ceiling):
    if [r["seed"] for r in floor] != [r["seed"] for r in ceiling]:
        raise ValueError("floor and ceiling records are not paired by seed")
    return sum(int(r["success"]) for r in floor), sum(int(r["success"]) for r in ceiling)


def stop_check(floor, ceiling):
    """Section 7 on HD_FINAL, pooled over conditions: STOP when S_ceiling - S_floor < 0.05."""
    n = len(floor)
    if n == 0:
        raise ValueError("no episodes")
    k_floor, k_ceiling = _paired_successes(floor, ceiling)
    gap = Fraction(k_ceiling - k_floor, n)
    return {"S_floor": k_floor / n, "S_ceiling": k_ceiling / n, "gap": float(gap), "n": n,
            "threshold": float(STOP_THRESHOLD), "stop": gap < STOP_THRESHOLD}


def shadow_verdict(summaries):
    """Section 8 E-a over the pooled corpus of per-episode shadow summaries: the feasibility
    category agrees on 100 % of steps, and |du|_inf <= DU_TOL on >= 99 % of both-optimal steps."""
    steps = sum(s["steps"] for s in summaries)
    agree = sum(s["agree"] for s in summaries)
    both = sum(s["both_optimal"] for s in summaries)
    within = sum(s["du_le_tol"] for s in summaries)
    if steps == 0 or both == 0:
        raise ValueError("no steps, or no step where both solvers are optimal")
    cross, hist = {}, {}
    for s in summaries:
        for src, dst in ((s["crosstab"], cross), (s["du_hist"], hist)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v
    du_max = max(s["du_max"] for s in summaries if s["both_optimal"])
    pass_cat = agree == steps
    pass_du = Fraction(within, both) >= DU_SHARE
    return {"steps": steps, "agree": agree, "agreement": agree / steps, "crosstab": cross,
            "delegated": sum(s.get("delegated", 0) for s in summaries),
            "both_optimal": both, "du_le_tol": within, "du_share": within / both,
            "du_max": du_max, "du_hist": hist, "du_tol": DU_TOL,
            "du_share_required": float(DU_SHARE), "pass_category": pass_cat,
            "pass_du": pass_du, "pass": pass_cat and pass_du}


def closed_loop_verdict(frozen, fast):
    """Section 8 E-b for one arm, pooled over conditions: fast vs frozen paired by (condition,
    seed). Pass: exact McNemar p > 0.05 on success AND the paired-bootstrap 95 % CI of
    S_fast - S_frozen inside [-0.03, +0.03]."""
    from continuation.stats import paired_binary
    key = [(r["condition"], r["seed"]) for r in frozen]
    if not key or key != [(r["condition"], r["seed"]) for r in fast]:
        raise ValueError("fast and frozen records are not paired by (condition, seed)")
    t = paired_binary(fast, frozen, "success")
    lo, hi = t["ci95"]
    return {"n": len(key), "S_frozen": sum(int(r["success"]) for r in frozen) / len(key),
            "S_fast": sum(int(r["success"]) for r in fast) / len(key),
            "only_fast": t["only_a"], "only_frozen": t["only_b"], "p": t["p"],
            "diff": t["diff"], "ci95": [lo, hi],
            "pass_mcnemar": t["p"] > MCNEMAR_P,
            "pass_ci": -CI_BOUND <= lo and hi <= CI_BOUND,
            "pass": t["p"] > MCNEMAR_P and -CI_BOUND <= lo and hi <= CI_BOUND}


def training_solver(ea, eb):
    """Section 8 decision: "fast" only when E-a and E-b (every arm) pass, otherwise "frozen"."""
    ok = ea["pass"] and all(v["pass"] for v in eb.values())
    return {"pass": ok, "solver": "fast" if ok else "frozen",
            "E-a": ea["pass"], "E-b": {k: v["pass"] for k, v in eb.items()}}


def pilot_sanity(health, log):
    """Section 9.1 implementation sanity from the trainer's health summary and per-update log
    (highdim.train). Gated: no non-finite rollout (action, value, log-prob, reward, return,
    advantage) or update (losses, parameters); every diagnostic key present, with finite
    actions, on every control step; explained variance > 0 at the last update (the 1M mark).
    Reported only: each update's approx_kl (SB3's mean over the last epoch's minibatches)
    against the target, and the fraction of updates the target-KL stop cut short (SB3 stops at
    a minibatch whose approx_kl exceeds 1.5 x target)."""
    ev = log[-1]["explained_variance"] if log else float("nan")
    checks = {"finite_rollouts": health["nonfinite_rollouts"] == 0,
              "finite_updates": health["nonfinite_updates"] == 0
              and all(r["finite"] for r in log),
              "diagnostics_complete": health["diag_steps"] > 0
              and health["diag_incomplete"] == 0,
              "explained_variance_positive": bool(ev > 0)}
    n = len(log)
    kl = [r["approx_kl"] for r in log]
    return {**checks, "pass": bool(log) and all(checks.values()), "updates": n,
            "explained_variance": ev,
            "early_stop_frac": sum(r["early_stop"] for r in log) / n if n else float("nan"),
            "kl_below_target_frac": sum(k < TARGET_KL for k in kl) / n if n else float("nan"),
            "approx_kl_median": float(sorted(kl)[n // 2]) if n else float("nan"),
            "target_kl": TARGET_KL}


def pilot_learning(ppo, floor):
    """Section 9.1 learning rule on HD_DEV pooled over conditions: the checkpoint's deterministic
    success >= the floor's on the same episodes. Point estimates, exact counts; a tie passes."""
    key = [(r["condition"], r["seed"]) for r in ppo]
    if not key or key != [(r["condition"], r["seed"]) for r in floor]:
        raise ValueError("PPO and floor records are not paired by (condition, seed)")
    k_ppo = sum(int(r["success"]) for r in ppo)
    k_floor = sum(int(r["success"]) for r in floor)
    n = len(key)
    return {"n": n, "k_ppo": k_ppo, "k_floor": k_floor, "S_ppo": k_ppo / n,
            "S_floor": k_floor / n, "pass": k_ppo >= k_floor}


def pilot_verdict(pilots):
    """Section 9.1 transition from {hold: {"sanity", "learning"}} in the order the pilots ran.

    A sanity failure is an implementation bug: fix and re-run at the same hold (it does not use
    up the fallback). A hold-1 learning failure earns exactly one hold-5 re-pilot; a hold-5
    learning failure is NO-GO."""
    if 1 not in pilots or set(pilots) - {1, PILOT_FALLBACK_HOLD}:
        raise ValueError(f"pilots must be hold 1, then optionally hold 5: {sorted(pilots)}")
    first = pilots[1]
    if not first["sanity"]["pass"]:
        if len(pilots) > 1:
            raise ValueError("a hold-5 pilot cannot follow a hold-1 sanity failure")
        return {"verdict": "fix and re-run", "hold": 1}
    if first["learning"]["pass"]:
        if len(pilots) > 1:
            raise ValueError("a hold-5 pilot runs only after a hold-1 learning failure")
        return {"verdict": "viable", "hold": 1}
    second = pilots.get(PILOT_FALLBACK_HOLD)
    if second is None:
        return {"verdict": "re-pilot", "hold": PILOT_FALLBACK_HOLD}
    if not second["sanity"]["pass"]:
        return {"verdict": "fix and re-run", "hold": PILOT_FALLBACK_HOLD}
    if second["learning"]["pass"]:
        return {"verdict": "viable", "hold": PILOT_FALLBACK_HOLD}
    return {"verdict": "NO-GO", "hold": None}
