"""HD Experiment 1, phase 3: fast-solver training-suitability check.

Pre-registration: MD_files/highdim/HD_DESIGN.md section 8. One question only: is the existing
FastClfCbfDrccpController (variant reduced_osqp, unmodified) faithful enough to the frozen SCS
controller to speed up PPO training? It never selects an evaluation solver: evaluation and every
reported number use frozen SCS. Phase 7 Stage 1 is background and is not reopened.

Corpus: HD_DEV, 100 seeds x {fixed, randomized}, both baseline arms (floor = goal_random,
ceiling = astar_random), frozen tuned + Random parameters.

    E-a  the frozen solver drives the loop; at every step the fast solver solves the identical
         inputs (p, gamma, xi, u_prev) in shadow (highdim.harness.ShadowSolver). Pooled over the
         corpus: feasibility category agrees on 100 % of steps, and |u_fast - u_frozen|_inf
         <= 1e-3 on >= 99 % of both-optimal steps.
    E-b  the fast solver drives the loop; paired by seed against the frozen episodes of E-a
         (the shadow never changes them; tests/test_highdim.py), pooled over conditions, one
         test per arm: McNemar p > 0.05 and the success-difference 95 % CI inside +-0.03.

Both pass -> train with the fast solver; otherwise -> train with frozen SCS
(highdim.gates.training_solver). The fast solver is never modified. HD_FINAL is never opened.

    python -m experiments.highdim.hd1_solver_check --workers 14
    python -m experiments.highdim.hd1_solver_check --smoke --out DIR   # 2 seeds, 60 steps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuation.stats import summaries
from experiments.highdim import protect_manifest as PM
from experiments.highdim.hd0_baselines import (FROZEN_PARAM_FILES, _fmt, provenance,
                                               write_all_new)
from highdim import blocks as HB
from highdim.gates import closed_loop_verdict, shadow_verdict, training_solver
from highdim.seeds import seed_block

REPO = PM.REPO
RESULTS = REPO / "results/highdim/HD1"
HD0 = REPO / "results/highdim/HD0"
REPORT = REPO / "MD_files/highdim/HD1_SOLVER_CHECK_REPORT.md"
ARMS = [("goal_random", "floor"), ("astar_random", "ceiling")]
#: mode -> (solver, shadow) for highdim.blocks.run_block(check=...)
MODES = {"frozen_shadow": ("frozen", True), "fast": ("fast", False)}


def run(out_dir, *, workers, n=None, max_steps=None, progress=25, tag=""):
    """{(kind, mode): {condition: [light records]}} on HD_DEV."""
    out_dir = Path(out_dir)
    seeds = seed_block("HD_DEV", n)
    res = {}
    for kind, label in ARMS:
        for mode, check in MODES.items():
            stem = f"{label}_{mode}"
            print(f"[{stem}]", flush=True)
            res[(kind, mode)] = HB.run_block(
                kind, seeds, workers=workers, max_steps=max_steps, progress=progress, tag=tag,
                check=check, checkpoint=out_dir / f"{stem}.jsonl",
                traces=out_dir / f"{stem}.traces.jsonl.gz")
        HB.check_paired(res[(kind, "frozen_shadow")], res[(kind, "fast")])
    return res


def _pooled(r):
    return [x for c in r for x in r[c]]


def analyse(res):
    shadow = {label: {c: shadow_verdict([r["shadow"] for r in recs])
                      for c, recs in res[(kind, "frozen_shadow")].items()}
              for kind, label in ARMS}
    for kind, label in ARMS:
        shadow[label]["pooled"] = shadow_verdict(
            [r["shadow"] for r in _pooled(res[(kind, "frozen_shadow")])])
    ea = shadow_verdict([r["shadow"] for kind, _ in ARMS
                         for r in _pooled(res[(kind, "frozen_shadow")])])
    eb = {label: closed_loop_verdict(_pooled(res[(kind, "frozen_shadow")]),
                                     _pooled(res[(kind, "fast")]))
          for kind, label in ARMS}
    summ = {label: {mode: summaries(res[(kind, mode)], f"{label} {mode}") for mode in MODES}
            for kind, label in ARMS}
    return {"E-a": ea, "E-a_by_arm": shadow, "E-b": eb, "summaries": summ,
            "decision": training_solver(ea, eb)}


def reproduces_hd0(res):
    """Do the E-a frozen episodes reproduce the committed HD0 HD_DEV records? {label: [diffs]}."""
    keys = ("outcome", "steps", "path_length", "min_clearance", "n_recovery_events")
    out = {}
    for kind, label in ARMS:
        ref = {(r["cond"], r["seed"]): r["rec"]
               for r in HB._read_jsonl(HD0 / f"{label}_HD_DEV.jsonl")[1:]}
        diffs = []
        for c, recs in res[(kind, "frozen_shadow")].items():
            for r in recs:
                h = ref.get((c, r["seed"]))
                if h is None or any(h[k] != r[k] for k in keys):
                    diffs.append([c, r["seed"]])
        if len(ref) != len(_pooled(res[(kind, "frozen_shadow")])):
            diffs.append(["count", len(ref)])
        out[label] = diffs
    return out


# --------------------------------------------------------------------------- report
def _yes(b):
    return "**PASS**" if b else "**FAIL**"


def _ea_table(by_arm, ea):
    lines = ["| arm | condition | steps | category agreement | both optimal | ‖Δu‖∞ ≤ 1e−3 | share | max ‖Δu‖∞ |",
             "|---|---|---|---|---|---|---|---|"]
    rows = [(a, c, by_arm[a][c]) for _, a in ARMS
            for c in ("fixed", "randomized", "pooled")] + [("**both**", "**pooled**", ea)]
    for a, c, v in rows:
        lines.append(f"| {a} | {c} | {v['steps']} | {v['agree']}/{v['steps']} "
                     f"({v['agreement']:.4%}) | {v['both_optimal']} | {v['du_le_tol']} | "
                     f"{v['du_share']:.4%} | {v['du_max']:.3e} |")
    return "\n".join(lines)


def _hist(h):
    order = sorted(h, key=lambda k: (k.startswith(">"), float(k.lstrip("<=>"))))
    return "\n".join([f"| {'bin':>8} | count |", "|---|---|"] +
                     [f"| {k} | {h[k]} |" for k in order])


def _eb_table(eb):
    lines = ["| arm | n pairs | S_frozen | S_fast | fast-only / frozen-only | McNemar p | diff (fast − frozen) | 95% CI | McNemar | CI | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, a in ARMS:
        v = eb[a]
        lines.append(f"| {a} | {v['n']} | {v['S_frozen']:.4f} | {v['S_fast']:.4f} | "
                     f"{v['only_fast']} / {v['only_frozen']} | {v['p']:.4g} | {v['diff']:+.4f} | "
                     f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                     f"{'ok' if v['pass_mcnemar'] else 'fail'} | "
                     f"{'ok' if v['pass_ci'] else 'fail'} | {_yes(v['pass'])} |")
    return "\n".join(lines)


ROWS = [("success", "success_rate"), ("collision", "collision_rate"),
        ("timeout", "timeout_rate"), ("infeasible-step rate", "infeasible_step_rate"),
        ("Random events / ep", "infeasible_events_per_episode"),
        ("QP ms (mean)", "qp_time_ms_mean"),
        ("step ms (mean; frozen_shadow includes the shadow solve)", "step_time_ms_mean_ep")]


def _desc_table(summ):
    cols = [(a, m) for _, a in ARMS for m in MODES]
    lines = ["| metric (pooled) | " + " | ".join(f"{a} {m}" for a, m in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, key in ROWS:
        lines.append(f"| {name} | " + " | ".join(_fmt(summ[a][m]["pooled"].get(key))
                                                 for a, m in cols) + " |")
    return "\n".join(lines)


def report(analysis, prov, trace_sha, hd0=None):
    ea, eb, d = analysis["E-a"], analysis["E-b"], analysis["decision"]
    tuned = json.loads((REPO / FROZEN_PARAM_FILES[0]).read_text())["params"]
    spec = json.loads((REPO / FROZEN_PARAM_FILES[1]).read_text())["spec"]
    n = analysis["summaries"]["floor"]["fast"]["fixed"]["episodes"]
    verdict = ("**PASS → PPO training uses the fast solver** (`FastClfCbfDrccpController`, "
               "`reduced_osqp`, unmodified)." if d["solver"] == "fast" else
               "**FAIL → PPO training uses the frozen SCS controller.** The fast solver is not "
               "modified.")
    if hd0 is None:
        repro = "not checked (smoke run)"
    elif not any(hd0.values()):
        repro = ("yes: every E-a frozen episode reproduces the committed HD0 HD_DEV record "
                 "(outcome, steps, path length, min clearance, Random events)")
    else:
        repro = f"**NO**: mismatches {hd0}"
    return f"""# HD Experiment 1 — phase 3: fast-solver training-suitability check

Pre-registration: [[HD_DESIGN]] §8. Generated by `experiments/highdim/hd1_solver_check.py`.

- Commit `{prov['commit'][:7]}`; code tree clean: {"yes" if not prov['code_changes'] else "NO"}
- Frozen-file manifest: modified {prov['manifest']['modified']}, missing {prov['manifest']['missing']}
- Frozen parameters: tuned α {tuned['alpha']}, r_W {tuned['wasserstein_r']}, ε {tuned['epsilon']}, k_v {tuned['k_v']}, {tuned['k_scans']} scans, τ_eff {tuned['tau_eff']}; Random K {spec['K']}, H {spec['H']}, speeds {spec['speeds']}
- Fast solver: `FastClfCbfDrccpController(variant="reduced_osqp")`, unmodified, built with the frozen controller's own parameters. Its preconditions hold there (ε·n_keep = 0.5 < 1; max_v = 1 ≤ max(1, α)), giving τ = r_W·max(1, α)/ε = 0.12
- Corpus: HD_DEV ({n} seeds × 2 conditions), floor (goal-only + Random) and ceiling (A* + Random)
- E-a frozen episodes reproduce HD0: {repro}

## Verdict

| test | pass condition | result |
|---|---|---|
| E-a category | feasibility category agrees on 100 % of steps | {ea['agree']}/{ea['steps']} → {_yes(ea['pass_category'])} |
| E-a action | ‖Δu‖∞ ≤ 1e−3 on ≥ 99 % of both-optimal steps | {ea['du_le_tol']}/{ea['both_optimal']} = {ea['du_share']:.4%}, max {ea['du_max']:.3e} → {_yes(ea['pass_du'])} |
| E-b floor | McNemar p > 0.05 and CI ⊂ ±0.03 | {_yes(eb['floor']['pass'])} |
| E-b ceiling | McNemar p > 0.05 and CI ⊂ ±0.03 | {_yes(eb['ceiling']['pass'])} |

{verdict}

**Scope.** This check answers one question: is the fast solver faithful enough to the frozen SCS
controller to speed up PPO training? It selects the *training* solver only. Evaluation, and every
reported number of HD Experiment 1, use the frozen SCS controller. The fast solver is an
acceleration mechanism, not a formally equivalent replacement.

**Background: Phase 7 Stage 1** (`MD_files/week5/phase7/STAGE1_FINAL_REPORT.md`) found feasible-set
equality, guarantee-tier agreement and closed-loop equivalence passed, and the strict KKT
certificate failed (maximum 5.429e−5 against 1e−6, three attempts; stopping rule applied). That
verdict stands: Stage 1 is not reopened,
amended or reinterpreted by this check.

## E-a — shadow, per step

The frozen solver drives the loop. At every step the fast solver solves the identical inputs
(p, γ, ξ, u_prev), where u_prev is the frozen controller's previous action as Random recovery
left it. Categories follow what the policy does with a status: *optimal* (action used),
*infeasible* (Random recovery), *other* (frozen u = 0).

Interpretations fixed in code before the run (disclosed here, not deviations from a rule):
- §8 says "feasibility category (optimal vs infeasible)". Statuses that are neither (e.g.
  `optimal_inaccurate`, `solver_error`) form a third category, *other*, because the policy treats
  them differently from both. This is stricter than a two-way split: it can only move the
  decision towards frozen SCS.
- E-a is decided on the corpus pooled over both arms and both conditions (§8: "driving arm 1
  and arm 2"); per-arm and per-condition rows are descriptive.
- Steps with h_crit < 0 are delegated by the fast solver to the frozen controller and agree by
  construction: {ea['delegated']} such steps in the corpus.

{_ea_table(analysis['E-a_by_arm'], ea)}

Category disagreements (frozen → fast): {ea['crosstab'] or "none"}

Distribution of ‖u_fast − u_frozen‖∞ on both-optimal steps (all arms and conditions):

{_hist(ea['du_hist'])}

## E-b — closed loop, fast vs frozen, paired by seed, pooled over conditions

The frozen side of each pair is the E-a episode itself: the shadow never changes the frozen
episode (`test_shadow_leaves_the_frozen_episode_bit_identical`), so a separate frozen run would
be identical.

{_eb_table(eb)}

## Descriptive (pooled over conditions)

{_desc_table(analysis['summaries'])}

## Data

- Records: `results/highdim/HD1/{{floor,ceiling}}_{{frozen_shadow,fast}}.jsonl`
- Per-step traces (with the per-step shadow comparison) are gitignored; each episode re-runs
  deterministically from its seed. sha256:
{chr(10).join(f"  - `{k}`: `{h}`" for k, h in sorted(trace_sha.items()))}
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--smoke", action="store_true", help="2 HD_DEV seeds, 60 steps, into --out")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    prov = provenance()
    if a.smoke:
        if a.out is None:
            raise SystemExit("--smoke needs --out DIR")
        analysis = analyse(run(a.out, workers=a.workers, n=2, max_steps=60, progress=0))
        print(json.dumps(analysis["decision"]))
        return analysis

    def clean(p):
        return not (p["code_changes"] or p["manifest"]["modified"] or p["manifest"]["missing"])
    if not clean(prov):
        raise SystemExit(f"refusing to run: code changes or manifest failure\n{prov}")
    res = run(RESULTS, workers=a.workers, tag=prov["commit"])
    analysis = analyse(res)
    hd0 = reproduces_hd0(res)
    analysis["reproduces_HD0"] = hd0
    after = provenance()
    if not clean(after) or after["commit"] != prov["commit"]:
        raise SystemExit(f"code changed during the run; results not written\n{after}")
    trace_sha = {p.name: PM._sha(p) for p in sorted(RESULTS.glob("*.traces.jsonl.gz"))}
    write_all_new({
        RESULTS / "provenance.json": json.dumps(prov, indent=1),
        RESULTS / "analysis.json": json.dumps(analysis, indent=1, default=float),
        RESULTS / "traces.sha256": "".join(f"{h}  results/highdim/HD1/{k}\n"
                                           for k, h in sorted(trace_sha.items())),
        REPORT: report(analysis, prov, trace_sha, hd0)})
    print(json.dumps(analysis["decision"], indent=1))
    return analysis


if __name__ == "__main__":
    main()
