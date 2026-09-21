"""HD Experiment 1, phase 5: the pilot verdict (HD_DESIGN.md section 9.1).

The pilot itself is a training run (experiments.highdim.hd_train, run 0 / seed 0 / 1M control
steps; hold 1, or hold 5 for the single re-pilot). This entry point then:

    1. evaluates the pilot's 1M checkpoint (`ckpt_1000000`) deterministically on HD_DEV,
       100 seeds x {fixed, randomized}, frozen SCS (highdim.harness), with per-step traces;
    2. reads the floor and ceiling HD_DEV blocks of phases 1-2 back from results/highdim/HD0
       (deterministic given the seed, so they are not re-run; the stored fingerprints and trace
       hashes are checked);
    3. applies section 9.1: implementation sanity from the trainer's health log
       (highdim.gates.pilot_sanity) and the learning rule, pilot success >= floor success on the
       same 200 episodes (highdim.gates.pilot_learning);
    4. writes results/highdim/HD2/pilot_h<k>_*; once the verdict is final (viable, or NO-GO
       after the hold-5 re-pilot) it writes MD_files/highdim/HD2_PILOT_REPORT.md.

HD_FINAL is never opened here.

    python -m experiments.highdim.hd2_pilot --hold 1 --run-dir models/highdim/pilot_s0_h1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuation.stats import summaries
from experiments.highdim import hd0_baselines as HD0
from experiments.highdim import protect_manifest as PM
from highdim import blocks as HB
from highdim.gates import pilot_learning, pilot_sanity, pilot_verdict
from highdim.seeds import seed_block
from highdim.train import training_solver

REPO = PM.REPO
RESULTS = REPO / "results/highdim/HD2"
BASELINES = REPO / "results/highdim/HD0"
REPORT = REPO / "MD_files/highdim/HD2_PILOT_REPORT.md"
PILOT_MARK = 1_000_000
FLOOR_DEV_SUCCESSES = 124          # of 200: HD0_BASELINES_REPORT.md, HD_DEV pooled 0.62
#: |u_exec - u_nom| above which a step counts as a filter intervention. Descriptive only (the
#: pre-registration defines no threshold); u_nom is saturated at max_v = 1 m/s.
INTERVENTION = 0.1
GOAL_ATOL = 1e-5                   # m; as tests/test_highdim.py's floor gamma = goal check
GAMMA_BINS = (0.25, 0.5, 0.75, 0.9, 0.99, 1.0 + 1e-9)
ARMS = {"ppo": "ppo_random", "floor": "goal_random", "ceiling": "astar_random"}


def provenance():
    return HD0.provenance()


# --------------------------------------------------------------------------- per-step statistics
def _bin(d):
    lo = 0.0
    for hi in GAMMA_BINS:
        if d < hi:
            return f"[{lo:g}, {min(hi, 1.0):g}{')' if hi < 1 else ']'}"
        lo = hi
    return ">1"


def step_stats(rows, recs):
    """Filter and subgoal behaviour pooled over steps, from trace rows and their records (goal).

    gamma distance is |gamma - p| with p the position the controller saw at that step;
    a goal switch is a step whose gamma is the goal (the goal inside L). The policy's goal is
    rebuilt from the float32 goal offset, so it matches the env's to GOAL_ATOL, not exactly."""
    dist, switch, dev, infeas, other, events = [], [], [], [], [], []
    for row, rec in zip(rows, recs, strict=True):
        goal = np.asarray(rec["goal"], float)
        for t, s in enumerate(row["trace"]):
            g = np.asarray(s["gamma"], float)
            dist.append(float(np.linalg.norm(g - np.asarray(row["trajectory"][t], float))))
            switch.append(bool(np.allclose(g, goal, rtol=0, atol=GOAL_ATOL)))
            if s["u_nom"] is not None:
                dev.append(float(np.linalg.norm(np.subtract(s["u_exec"], s["u_nom"]))))
            st = str(s["qp_status"])
            infeas.append("infeasible" in st)
            other.append(st != "optimal" and "infeasible" not in st)
            events.append(bool(s["random_event"]))
    dist, switch, dev = np.array(dist), np.array(switch), np.array(dev)
    hist = {}
    for b in map(_bin, dist):
        hist[b] = hist.get(b, 0) + 1
    off = dist[~switch]
    return {"steps": int(len(dist)), "goal_switch_rate": float(switch.mean()),
            "gamma_dist_mean": float(dist.mean()),
            "gamma_dist_mean_off_goal": float(off.mean()) if len(off) else float("nan"),
            "gamma_dist_median_off_goal": float(np.median(off)) if len(off) else float("nan"),
            "gamma_dist_hist": hist,
            "u_dev_mean": float(dev.mean()) if len(dev) else float("nan"),
            "u_dev_median": float(np.median(dev)) if len(dev) else float("nan"),
            "intervention_rate": float((dev > INTERVENTION).mean()) if len(dev) else float("nan"),
            "u_nom_steps": int(len(dev)), "infeasible_rate": float(np.mean(infeas)),
            "other_status_rate": float(np.mean(other)),
            "random_event_rate": float(np.mean(events))}


# --------------------------------------------------------------------------- evaluation
def _pooled(res):
    return [r for c in ("fixed", "randomized") for r in res[c]]


def _rows(res, traces):
    return [traces[(c, r["seed"])] for c in ("fixed", "randomized") for r in res[c]]


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def baseline_blocks(seeds, workers, max_steps, out_dir, source):
    """{label: (records, traces)} for floor and ceiling on `seeds`: read from `source` (the HD0
    results, fingerprints and trace hashes checked) or, with source=None, run into out_dir."""
    out = {}
    if source is not None:
        source = Path(source)
        tag = json.loads((source / "provenance.json").read_text())["commit"]
        want = {}
        for ln in (source / "traces.sha256").read_text().splitlines():
            h, rel = ln.split()
            want[Path(rel).name] = h
    for label in ("floor", "ceiling"):
        kind, stem = ARMS[label], f"{label}_HD_DEV"
        if source is None:
            kw = dict(checkpoint=out_dir / f"{stem}.jsonl",
                      traces=out_dir / f"{stem}.traces.jsonl.gz")
            HB.run_block(kind, seeds, workers=workers, max_steps=max_steps, **kw)
            out[label] = HB.load_block(kind, seeds, tag="", max_steps=max_steps, **kw)
            continue
        tr = source / f"{stem}.traces.jsonl.gz"
        if PM._sha(tr) != want[tr.name]:
            raise SystemExit(f"{tr} does not match {source / 'traces.sha256'}")
        out[label] = HB.load_block(kind, seeds, checkpoint=source / f"{stem}.jsonl",
                                   traces=tr, tag=tag)
    return out


def evaluate_pilot(run_dir, out_dir, *, hold, mark=PILOT_MARK, n=None, workers=8,
                   max_steps=None, baselines=None, tag=""):
    """Section 9.1 for one pilot: its `mark` checkpoint on HD_DEV against the floor.

    `baselines` is the HD0 results dir; None (tests, smoke) runs floor and ceiling afresh."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    meta = json.loads((run_dir / f"ckpt_{mark}.json").read_text())
    final = json.loads((run_dir / "final.json").read_text())
    if meta["hold"] != hold or final["hold"] != hold:
        raise SystemExit(f"{run_dir} was trained with hold {meta['hold']}, not {hold}")
    log = _jsonl(run_dir / "train_log.jsonl")
    if final["steps"] != meta["steps"] or not log or log[-1]["ppo_steps"] != meta["steps"]:
        raise SystemExit(f"{run_dir}: the run did not end at its {mark} checkpoint, so the "
                         "health log is not the checkpoint's")
    seeds = seed_block("HD_DEV", n)
    stem = out_dir / f"pilot_h{hold}_HD_DEV"
    ckpt = run_dir / f"ckpt_{mark}.zip"
    kw = dict(checkpoint=f"{stem}.jsonl", traces=f"{stem}.traces.jsonl.gz", model=str(ckpt),
              hold=hold, max_steps=max_steps, tag=tag)
    HB.run_block("ppo_random", seeds, workers=workers, progress=50, **kw)
    blocks = {"ppo": HB.load_block("ppo_random", seeds, **kw),
              **baseline_blocks(seeds, workers, max_steps, out_dir, baselines)}
    for label in ("floor", "ceiling"):
        HB.check_paired(blocks["ppo"][0], blocks[label][0])

    summ, stats = {}, {}
    for label, (res, traces) in blocks.items():
        summ[label] = HD0._with_stuck(summaries(res, label), res)
        stats[label] = step_stats(_rows(res, traces), _pooled(res))
    health = final["health"]
    return {"hold": hold, "mark": mark, "run_dir": str(run_dir), "seeds": [seeds[0], seeds[-1]],
            "checkpoint_sha256": PM._sha(ckpt), "checkpoint_meta": meta,
            "train": {k: final.get(k) for k in ("commit", "solver", "seed", "run", "steps",
                                                "control_steps", "wall_s", "eval_s",
                                                "steps_per_s")},
            "health": health,
            "sanity": pilot_sanity(health, log),
            "learning": pilot_learning(_pooled(blocks["ppo"][0]),
                                       _pooled(blocks["floor"][0])),
            "learning_curve": _jsonl(run_dir / "evals.jsonl"),
            "train_log_summary": _log_summary(log),
            "summaries": summ, "step_stats": stats, "records": blocks["ppo"][0]}


def _log_summary(log):
    if not log:
        return {}
    ev = [r["explained_variance"] for r in log]
    kl = np.array([r["approx_kl"] for r in log])
    return {"updates": len(log), "explained_variance_first": ev[0],
            "explained_variance_last": ev[-1],
            "approx_kl_p50": float(np.percentile(kl, 50)),
            "approx_kl_p90": float(np.percentile(kl, 90)), "approx_kl_max": float(kl.max()),
            "epochs_mean": float(np.mean([r["epochs"] for r in log])),
            "std_first": log[0]["std"], "std_last": log[-1]["std"]}


# --------------------------------------------------------------------------- report
ARM_ROWS = [("success", "success_rate"), ("collision", "collision_rate"),
            ("  dynamic", "dynamic_collision_rate"), ("  static", "static_collision_rate"),
            ("  wall", "wall_collision_rate"), ("timeout", "timeout_rate"),
            ("stuck", "stuck_rate"), ("SPL", "mean_spl"),
            ("min clearance (mean)", "min_clearance_mean"),
            ("Random events / episode", "infeasible_events_per_episode"),
            ("mean steps", "mean_steps")]
STEP_ROWS = [(f"filter intervention (‖u_exec − u_nom‖ > {INTERVENTION})", "intervention_rate"),
             ("‖u_exec − u_nom‖ mean", "u_dev_mean"), ("‖u_exec − u_nom‖ median", "u_dev_median"),
             ("QP infeasible (step rate)", "infeasible_rate"),
             ("QP other non-optimal (step rate)", "other_status_rate"),
             ("Random event (step rate)", "random_event_rate")]
VERDICT_TEXT = {
    "viable": "**VIABLE.** Phase 6 (main training, ticket 09) proceeds with hold length {hold}.",
    "NO-GO": "**NO-GO.** Both the hold-1 pilot and the hold-5 re-pilot failed the learning rule. "
             "Phases 6–7 are cancelled (tickets 09–10 `wontfix`) and HD_FINAL is never opened "
             "for PPO.",
    "re-pilot": "**RE-PILOT.** The hold-1 pilot failed the learning rule; the single "
                "pre-registered re-pilot runs with hold 5.",
    "fix and re-run": "**IMPLEMENTATION SANITY FAILED** (hold {hold}). This is a bug: fix it and "
                      "re-run the pilot at the same hold; logged as a deviation, and it does not "
                      "use up the hold-5 re-pilot."}


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "NO"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.{nd}f}"


def _arm_table(a):
    cols = [(lab, c) for lab in ("ppo", "floor", "ceiling") for c in ("fixed", "randomized",
                                                                       "pooled")]
    head = "| metric | " + " | ".join(f"{lab} {c}" for lab, c in cols) + " |"
    lines = [head, "|---" * (len(cols) + 1) + "|"]
    for name, key in ARM_ROWS:
        lines.append(f"| {name} | " + " | ".join(_f(a["summaries"][lab][c].get(key))
                                                 for lab, c in cols) + " |")
    return "\n".join(lines)


def _step_table(a):
    lines = ["| per control step, pooled | PPO | floor | ceiling |", "|---|---|---|---|"]
    for name, key in STEP_ROWS:
        lines.append(f"| {name} | " + " | ".join(_f(a["step_stats"][lab][key])
                                                 for lab in ("ppo", "floor", "ceiling")) + " |")
    lines.append("| control steps | " + " | ".join(_f(a["step_stats"][lab]["steps"])
                                                   for lab in ("ppo", "floor", "ceiling")) + " |")
    return "\n".join(lines)


def _subgoal(a):
    s = a["step_stats"]["ppo"]
    n = s["steps"]
    hist = "\n".join(f"| {k} | {v} | {v / n:.3f} |" for k, v in sorted(s["gamma_dist_hist"].items()))
    return f"""- goal-switch rate (γ = goal, goal inside L = 1 m): {_f(s['goal_switch_rate'])} of steps
- γ distance ‖γ − p‖: mean {_f(s['gamma_dist_mean'])}; off the goal switch mean \
{_f(s['gamma_dist_mean_off_goal'])}, median {_f(s['gamma_dist_median_off_goal'])} m

| γ distance (m) | steps | share |
|---|---|---|
{hist}"""


def _curve(a):
    lines = ["| mark | PPO steps | success | fixed | randomized | collision | timeout |",
             "|---|---|---|---|---|---|---|"]
    for e in a["learning_curve"]:
        lines.append(f"| {_f(e['steps'])} | {_f(e.get('ppo_steps'))} | {_f(e['success'])} | "
                     f"{_f(e.get('success_fixed'))} | {_f(e.get('success_randomized'))} | "
                     f"{_f(e['collision'])} | {_f(e['timeout'])} |")
    return "\n".join(lines)


def _sanity(a):
    s, h, lg, t = a["sanity"], a["health"], a["train_log_summary"], a["train"]
    return f"""| check (gated) | result |
|---|---|
| no non-finite rollout (action, value, log-prob, reward, return, advantage) | {_f(s['finite_rollouts'])} ({h['nonfinite_rollouts']} of {h['rollouts']} rollouts) |
| no non-finite update (losses, parameters) | {_f(s['finite_updates'])} ({h['nonfinite_updates']} of {h['updates']} updates) |
| every diagnostic key present, finite γ and actions | {_f(s['diagnostics_complete'])} ({h['diag_incomplete']} incomplete of {_f(h['diag_steps'])} steps) |
| explained variance > 0 at the last update | {_f(s['explained_variance_positive'])} ({_f(s['explained_variance'])}) |
| no exception | yes (training and evaluation completed) |

Reported, not gated (approx_kl is SB3's per-update mean over the last epoch's minibatches, \
compared with the target; SB3's own stop fires on one minibatch above 1.5 × target): approx_kl median {_f(lg.get('approx_kl_p50'), 4)}, p90 \
{_f(lg.get('approx_kl_p90'), 4)}, max {_f(lg.get('approx_kl_max'), 4)}; \
{_f(s['kl_below_target_frac'])} of updates end below the target {s['target_kl']}; \
target-KL stop before the last epoch in {_f(s['early_stop_frac'])} of updates (mean epochs \
{_f(lg.get('epochs_mean'), 2)}); policy std {_f(lg.get('std_first'))} → {_f(lg.get('std_last'))}. \
Throughput {_f(t['steps_per_s'], 1)} control steps/s ({_f(t['control_steps'])} control steps, \
{_f(t['wall_s'] / 3600, 2)} h wall of which {_f(t['eval_s'] / 60, 1)} min in-training evaluation), \
solver `{t['solver']}`."""


def _ablation_note(a):
    """Ticket 08: unusually high intervention is noted as a candidate for a later, separately
    labelled ablation, never added to this experiment. "Unusually high" is read as above the
    floor's, the arm with no planner at all."""
    st = a["step_stats"]
    if st["ppo"]["intervention_rate"] > st["floor"]["intervention_rate"]:
        return ("**PPO relies on the filter more than even the no-planner floor: a candidate "
                "for a separately labelled intervention-penalty ablation later, not added to "
                "this experiment.**")
    return ("PPO's intervention rate is not above the no-planner floor's, so no "
            "intervention-penalty ablation is flagged.")


def _pilot_section(a):
    L = a["learning"]
    ppo_rate = a["step_stats"]["ppo"]["intervention_rate"]
    ceil_rate = a["step_stats"]["ceiling"]["intervention_rate"]
    ratio = f"{ppo_rate / ceil_rate:.2f}×" if ceil_rate else "— (ceiling rate 0)"
    return f"""## Pilot, hold {a['hold']}

Checkpoint `{Path(a['run_dir']).name}/ckpt_{a['mark']}.zip` (saved at \
{_f(a['checkpoint_meta']['steps'])} PPO steps, {_f(a['checkpoint_meta']['control_steps'])} \
control steps; sha256 `{a['checkpoint_sha256'][:16]}…`), trained at commit \
`{a['train']['commit'][:7]}`, run {a['train']['run']}, seed {a['train']['seed']}.

### Implementation sanity

{_sanity(a)}

### Learning rule (HD_DEV, {L['n']} paired episodes, deterministic)

| S_PPO | S_floor | successes PPO / floor | rule | result |
|---|---|---|---|---|
| {L['S_ppo']:.3f} | {L['S_floor']:.3f} | {L['k_ppo']} / {L['k_floor']} | S_PPO ≥ S_floor | {"pass" if L['pass'] else "FAIL"} |

### Learning curve (in-training evaluation, 50 HD_DEV seeds × 2 conditions)

{_curve(a)}

### Outcomes on HD_DEV (collision breakdown)

{_arm_table(a)}

### Filter reliance

{_step_table(a)}

PPO's intervention rate is {ratio} the ceiling arm's. The threshold {INTERVENTION} m/s is \
descriptive; the pre-registration defines none. u_nom is always saturated at 1 m/s, so the \
QP's smoothing term alone moves u_exec off u_nom. {_ablation_note(a)}

### Subgoal behaviour

{_subgoal(a)}
"""


def _verdict(pilots):
    return pilot_verdict({h: {"sanity": a["sanity"], "learning": a["learning"]}
                          for h, a in pilots.items()})


def _saved_pilots():
    return {h: json.loads((RESULTS / f"pilot_h{h}_analysis.json").read_text())
            for h in (1, 5) if (RESULTS / f"pilot_h{h}_analysis.json").exists()}


def report(pilots, prov):
    v = _verdict(pilots)
    sections = "\n".join(_pilot_section(pilots[h]) for h in sorted(pilots))
    return f"""# HD Experiment 1 — phase 5: pilot verdict

Pre-registration: [[HD_DESIGN]] §9.1 (clarifications §13). Generated by \
`experiments/highdim/hd2_pilot.py`.

- Evaluation commit `{prov['commit'][:7]}`; code tree clean: {"yes" if not prov['code_changes'] else "NO"}
- Frozen-file manifest: modified {prov['manifest']['modified']}, missing {prov['manifest']['missing']}
- Evaluation: frozen SCS, deterministic (mean) actions, HD_DEV 100 seeds × {{fixed, randomized}}; \
HD_FINAL not opened
- Floor and ceiling: the phase 1–2 HD_DEV episodes (`results/highdim/HD0`), read back with \
their fingerprints and trace hashes checked, not re-run
- Pilot training: 8 environments (4 fixed + 4 randomized), SB3 PPO with the `config.json` \
hyperparameters, MLP [256, 256], 1M control steps

## Verdict

{VERDICT_TEXT[v['verdict']].format(hold=v['hold'])}

Rule (§9.1): implementation sanity must hold, and the 1M checkpoint's pooled HD_DEV success \
must be ≥ the floor's on the same 200 episodes (a tie passes). One hold-5 re-pilot is allowed \
after a learning failure; a second failure is NO-GO.

{sections}"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, required=True, choices=(1, 5))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(argv)

    def clean(p):
        return not (p["code_changes"] or p["manifest"]["modified"] or p["manifest"]["missing"])
    prov = provenance()
    if not clean(prov):
        raise SystemExit(f"refusing to run: code changes or manifest failure\n{prov}")
    out = RESULTS / f"pilot_h{a.hold}_analysis.json"
    if out.exists():
        raise SystemExit(f"{out} exists; HD results are never overwritten. A sanity-failure "
                         "re-run (a logged deviation) first moves the failed pilot's files aside")
    earlier = _saved_pilots()
    if a.hold == 5 and (1 not in earlier or _verdict(earlier)["verdict"] != "re-pilot"):
        raise SystemExit("the hold-5 re-pilot runs only after a hold-1 learning failure")
    if a.hold == 1 and earlier:
        raise SystemExit(f"pilot analyses already exist: {sorted(earlier)}")
    an = evaluate_pilot(a.run_dir, RESULTS, hold=a.hold, workers=a.workers,
                        baselines=BASELINES, tag=prov["commit"])
    after = provenance()
    if not clean(after) or after["commit"] != prov["commit"]:
        raise SystemExit(f"code changed during the run; results not written\n{after}")
    t = an["train"]
    if (t["seed"], t["run"], t["solver"]) != (0, 0, training_solver()):
        raise SystemExit(f"the pilot is run 0 / seed 0 on the selected solver, got {t}")
    if an["learning"]["k_floor"] != FLOOR_DEV_SUCCESSES:
        raise SystemExit(f"floor dev successes {an['learning']['k_floor']} are not the reported "
                         f"{FLOOR_DEV_SUCCESSES} / 200")
    an.pop("records")
    stem = f"pilot_h{a.hold}_HD_DEV.traces.jsonl.gz"
    HD0.write_all_new({
        RESULTS / f"pilot_h{a.hold}_provenance.json": json.dumps(prov, indent=1),
        out: json.dumps(an, indent=1, default=float),
        RESULTS / f"pilot_h{a.hold}_traces.sha256":
            f"{PM._sha(RESULTS / stem)}  results/highdim/HD2/{stem}\n"})

    pilots = _saved_pilots()
    v = _verdict(pilots)
    if v["verdict"] in ("viable", "NO-GO"):
        HD0.write_all_new({REPORT: report(pilots, prov)})
    print(json.dumps({"learning": an["learning"], "sanity": an["sanity"]["pass"], **v},
                     indent=1, default=float))
    return an


if __name__ == "__main__":
    main()
