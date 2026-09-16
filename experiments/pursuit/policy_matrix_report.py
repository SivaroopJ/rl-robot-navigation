"""Generate the data tables for MD_files/pursuit/MPC_CD_REPORT.md from saved artifacts only.

    python -m experiments.pursuit.policy_matrix_report --out MD_files/pursuit/MPC_CD_REPORT_DATA.md

Reads: historical config A (results/pursuit_hunter_variants/main, both maps), Phase I
(results/pursuit_policy_matrix/main_B), Phase III (main_C, if present), the frozen thresholds,
and the single-worker latency runs (latency_A/B/C, if present). Nothing is simulated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuation.congestion.detector import DetectorConfig, alarm_stats, replay
from continuation.pursuit.hunter_policies import ALIASES
from experiments.pursuit.cd_calibrate import load_series
from experiments.pursuit.p3_main import boot_ci, mcnemar, wilson

ROOT = Path("results/pursuit_policy_matrix")
HIST = {"dynamic": "results/pursuit_hunter_variants/main/HV_main_P-Random_hunter1.json",
        "static": "results/pursuit_hunter_variants/main/HV_main_P-Random_hunter1_static.json"}
MAPS = ("static", "dynamic")
POL = ("1", "1.5", "2", "2.5", "3", "3.5")
WITHIN = (("3", "1"), ("3.5", "1.5"), ("3", "2"), ("3.5", "2.5"), ("1.5", "1"), ("2", "1"))
TESTED = (("capture", lambda r: r["outcome"] == "capture"),
          ("Protag collision", lambda r: r["outcome"] == "protagonist_collision"),
          ("goal", lambda r: r["outcome"] == "goal"),
          ("not escaped", lambda r: r["outcome"] in ("capture", "protagonist_collision")),
          ("hunter collision", lambda r: bool(r["hunter_collision"])))


# --------------------------------------------------------------------------- loading
def load_config(name):
    if name == "A":
        out = {}
        for mp in MAPS:
            recs = json.loads(Path(HIST[mp]).read_text())["records"]
            out[mp] = {ALIASES[k]: v for k, v in recs.items()}
        return out
    #: "C2" = Phase III rerun with the post-hoc fixed detector (deviations, D1 follow-up 2)
    p = ROOT / ("main_C_fix4" if name == "C2" else f"main_{name}") / "records.json"
    return json.loads(p.read_text())["records"] if p.exists() else None


def fnum(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------- metrics
def core(recs):
    n = len(recs)
    steps = sum(r["steps"] for r in recs)
    o = {}
    for key, f in TESTED:
        c = sum(1 for r in recs if f(r))
        o[key] = (c / n, wilson(c, n))
    o["timeout"] = (sum(r["outcome"] == "timeout" for r in recs) / n, None)
    m = lambda k, cond=lambda r: True: (float(np.mean([r[k] for r in recs
                                                        if cond(r) and r[k] is not None]))
                                        if any(cond(r) and r[k] is not None for r in recs)
                                        else float("nan"))
    o["time_to_capture"] = m("time_to_capture_s")
    o["time_to_goal"] = m("time_to_goal_s")
    o["min_hunter_distance"] = m("min_hunter_distance")
    o["min_clearance"] = m("min_clearance")
    o["spl"] = m("spl")
    o["path_length"] = m("path_length")
    o["prot_infeasible"] = sum(r["prot_n_infeasible"] for r in recs) / steps
    o["hunter_infeasible"] = sum(r["hunter_n_infeasible"] for r in recs) / steps
    o["prot_recovery_per_ep"] = float(np.mean([r["prot_n_recovery_events"] for r in recs]))
    o["hunter_recovery_per_ep"] = float(np.mean([r.get("hunter_n_recovery_events", 0)
                                                 for r in recs]))
    # config A has no recovery on the hunter: every non-optimal solve is a u = 0 step
    o["hunter_u0"] = sum(r.get("hunter_n_u0_steps", r["hunter_n_fallback"]) for r in recs) / steps
    o["prot_u0"] = (sum(r["prot_n_u0_steps"] for r in recs) / steps
                    if "prot_n_u0_steps" in recs[0] else None)
    o["prot_ms"] = float(np.mean([r["prot_ms_mean"] for r in recs]))
    o["hunter_ms"] = float(np.mean([r["hunter_ms_mean"] for r in recs]))
    return o


def paired_rows(x, y):
    assert [r["seed"] for r in x] == [r["seed"] for r in y]
    assert all((a["start"], a["goal_xy"], a["hunter_start"]) == (b["start"], b["goal_xy"],
                                                               b["hunter_start"])
               for a, b in zip(x, y)), "unpaired layouts"
    out = {}
    for key, f in TESTED:
        a = [int(f(r)) for r in x]
        b = [int(f(r)) for r in y]
        d = np.asarray(a, float) - np.asarray(b, float)
        out[key] = {**mcnemar(a, b), "diff": float(d.mean()), "ci95": boot_ci(d)}
    return out


# --------------------------------------------------------------------------- tables
def outcome_table(configs, mp):
    head = ("| config | policy | capture | Protag collision | not escaped | goal | timeout | "
            "hunter collision | t capture (s) | t goal (s) | min H–P dist | min clearance | SPL | "
            "Protag inf/step | hunter inf/step | hunter u=0/step | hunter rec/ep | Protag rec/ep |")
    lines = [head, "|" + "---|" * 18]
    for name, data in configs.items():
        if data is None or mp not in data:
            continue
        for pid in POL:
            if pid not in data[mp]:
                continue
            c = core(data[mp][pid])
            cell = lambda k: f"{c[k][0]:.3f} [{c[k][1][0]:.2f},{c[k][1][1]:.2f}]"
            lines.append(
                f"| {name} | {pid} | {cell('capture')} | {cell('Protag collision')} | "
                f"{cell('not escaped')} | {cell('goal')} | {c['timeout'][0]:.3f} | "
                f"{cell('hunter collision')} | {fnum(c['time_to_capture'], 1)} | "
                f"{fnum(c['time_to_goal'], 1)} | {fnum(c['min_hunter_distance'], 2)} | "
                f"{fnum(c['min_clearance'], 3)} | {fnum(c['spl'], 3)} | "
                f"{c['prot_infeasible']:.4f} | {c['hunter_infeasible']:.4f} | "
                f"{c['hunter_u0']:.4f} | {c['hunter_recovery_per_ep']:.2f} | "
                f"{c['prot_recovery_per_ep']:.2f} |")
    return "\n".join(lines)


def paired_table(pairs, family_size, title):
    alpha = 0.05 / family_size
    lines = [f"{title} — Bonferroni α = 0.05/{family_size} = {alpha:.4f} (✱ = significant after correction)",
             "", "| comparison | " + " | ".join(k for k, _ in TESTED) + " |",
             "|" + "---|" * (len(TESTED) + 1)]
    for label, res in pairs:
        cells = []
        for key, _ in TESTED:
            v = res[key]
            star = " ✱" if v["p"] < alpha else ""
            cells.append(f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
                         f"{v['only_a']}/{v['only_b']} p={v['p']:.3g}{star}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mpc_table(configs):
    lines = ["| config | map | policy | candidates | horizon (steps) | MPC ms mean | MPC ms max | "
             "fallback rate | mean rejected | predicted-capture rate | pred. err 1 s (m) | "
             "terminal capture |", "|" + "---|" * 12]
    for name, data in configs.items():
        if data is None:
            continue
        for mp in MAPS:
            for pid in ("3", "3.5"):
                recs = data.get(mp, {}).get(pid)
                if not recs:
                    continue
                ms = sum(r["mpc_n_steps"] for r in recs)
                pe = [r["mpc_pred_err_1s_mean"] for r in recs if r["mpc_pred_err_1s_mean"]]
                lines.append(
                    f"| {name} | {mp} | {pid} | {recs[0]['mpc_config']['n_candidates']} | "
                    f"{recs[0]['mpc_config']['horizon_steps']} | "
                    f"{np.nanmean([r['mpc_ms']['mean'] for r in recs]):.2f} | "
                    f"{np.nanmax([r['mpc_ms']['max'] for r in recs]):.2f} | "
                    f"{sum(r['mpc_n_fallback'] for r in recs) / ms:.4f} | "
                    f"{np.nanmean([r['mpc_mean_rejected'] for r in recs]):.1f} | "
                    f"{sum(r['mpc_n_pred_capture'] for r in recs) / ms:.4f} | "
                    f"{np.mean(pe):.3f} | "
                    f"{np.mean([r['outcome'] == 'capture' for r in recs]):.3f} |")
    return "\n".join(lines)


def _alarm_row(label, al_list, leads):
    on = sum(a["infeasible_onsets"] for a in al_list)
    miss = sum(a["missed"] for a in al_list)
    wo = sum(a["warning_onsets"] for a in al_list)
    ta = sum(a["true_alarms"] for a in al_list)
    ws = sum(a["warning_steps"] for a in al_list)
    act = sum(a["active_steps"] for a in al_list)
    sc = [sum(a["state_counts"][i] for a in al_list) for i in range(3)]
    tot = max(sum(sc), 1)
    return (f"| {label} | {act} | {sc[0] / tot:.3f} / {sc[1] / tot:.3f} / {sc[2] / tot:.3f} | "
            f"{ws / max(act, 1):.3f} | {wo} | {ta} | {wo - ta} | "
            f"{fnum(ta / wo if wo else None)} | {on} | {miss} | "
            f"{fnum(1 - miss / on if on else None)} | "
            f"{fnum(float(np.mean(leads)) if leads else None, 2)} |")


ALARM_HEAD = ("| cell | active steps | state share N / C / Cr | warning fraction | warning onsets | "
              "true alarms | false alarms | precision | infeasibility onsets | missed | coverage | "
              "mean lead (steps) |")


def shadow_alarm_table(run_dir, det):
    series, _ = load_series(run_dir, require_dev=False)
    cells = {}
    for agent, mp, pid, seed, m, c, inf, act in series:
        st = replay(det, m, [5.0 if x is None else x for x in c])
        al = alarm_stats(st, inf, act)
        cells.setdefault((agent, mp, pid), []).append(al)
    lines = [ALARM_HEAD, "|" + "---|" * 12]
    for (agent, mp, pid), als in sorted(cells.items()):
        leads = [x for a in als for x in a["lead_steps"]]
        lines.append(_alarm_row(f"{agent} / {mp} / {pid}", als, leads))
    return "\n".join(lines)


def active_alarm_table(data):
    lines = [ALARM_HEAD, "|" + "---|" * 12]
    extra = ["| cell | mean speed scale | direction changes / ep | executed speed by state N / C / Cr (m/s) |",
             "|---|---|---|---|"]
    for mp in MAPS:
        for pid in POL:
            recs = data.get(mp, {}).get(pid)
            if not recs:
                continue
            for agent in ("prot", "hunter"):
                als = [r[f"{agent}_cd_alarms"] for r in recs]
                leads = [x for r in recs for x in r[f"{agent}_cd_lead_steps"]]
                lines.append(_alarm_row(f"{agent} / {mp} / {pid}", als, leads))
                sp = np.array([[np.nan if v is None else v for v in r[f"{agent}_cd_speed_by_state"]]
                               for r in recs], float)
                extra.append(
                    f"| {agent} / {mp} / {pid} | "
                    f"{np.mean([r[f'{agent}_cd_mean_scale'] for r in recs]):.3f} | "
                    f"{np.mean([r[f'{agent}_cd_n_direction_changes'] for r in recs]):.1f} | "
                    + " / ".join(fnum(float(np.nanmean(sp[:, i])) if np.isfinite(sp[:, i]).any()
                                      else None, 2) for i in range(3)) + " |")
    return "\n".join(lines) + "\n\n" + "\n".join(extra)


def latency_table():
    lines = ["| config | map | policy | agent | total mean | total p95 | total max | QP mean | "
             "recovery mean | CD mean | MPC mean | steps > 100 ms |", "|" + "---|" * 12]
    found = False
    for name in ("A", "B", "C"):
        p = ROOT / f"latency_{name}" / "summary.json"
        if not p.exists():
            continue
        found = True
        s = json.loads(p.read_text())["summary"]
        for mp in MAPS:
            for pid in POL:
                cell = s.get(mp, {}).get(pid)
                if not cell:
                    continue
                for agent in ("prot", "hunter"):
                    L = cell[f"{agent}_latency_ms"]
                    g = lambda comp, k="mean_of_episode_means": (fnum(L[comp][k], 1)
                                                                  if comp in L else "—")
                    lines.append(
                        f"| {name} | {mp} | {pid} | {agent} | {g('total')} | "
                        f"{g('total', 'mean_of_episode_p95')} | {g('total', 'max')} | "
                        f"{g('qp')} | {g('recovery')} | {g('cd')} | {g('mpc')} | "
                        f"{cell[f'{agent}_frac_steps_over_budget_pooled']:.4f} |")
    return "\n".join(lines) if found else "_latency runs not found_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; write to a new file")
    configs = {"A": load_config("A"), "B": load_config("B"), "C": load_config("C"),
               "C2": load_config("C2")}
    doc = ["# Policy Matrix — Generated Data Tables", "",
           "Generated by `python -m experiments.pursuit.policy_matrix_report` from saved records only. "
           "Rates are terminal outcomes with 95% Wilson intervals unless stated; hunter collision is an event flag. "
           "Config A = historical records (seeds 14 100 000–199), B = Phase I, C = Phase III, "
           "C2 = Phase III rerun with the post-hoc fixed detector (exploratory, see MPC_CD_DEVIATIONS.md); "
           "all paired on those seeds.", ""]
    for mp in MAPS:
        doc += [f"## Outcomes — {mp}", "", outcome_table(configs, mp), ""]
    for name in ("B", "C", "C2"):
        data = configs[name]
        if data is None:
            continue
        for mp in MAPS:
            pairs = [(f"{x}−{y}", paired_rows(data[mp][x], data[mp][y])) for x, y in WITHIN]
            doc += [f"## Within config {name} — {mp}", "",
                    paired_table(pairs, len(WITHIN) * len(TESTED),
                                 f"Paired (first − second), config {name}, {mp}"), ""]
    for label, (hi, lo), pols in (("B − A", ("B", "A"), ("1", "1.5", "2", "2.5")),
                                  ("C − B", ("C", "B"), POL),
                                  ("C2 − C (fixed detector vs pre-registered)", ("C2", "C"), POL),
                                  ("C2 − B (fixed detector vs no detector)", ("C2", "B"), POL)):
        if configs[hi] is None or configs[lo] is None:
            continue
        for mp in MAPS:
            pairs = [(f"{pid}: {hi}−{lo}", paired_rows(configs[hi][mp][pid], configs[lo][mp][pid]))
                     for pid in pols]
            doc += [f"## Across configurations {label} — {mp}", "",
                    paired_table(pairs, len(pols) * len(TESTED), f"Paired {label}, {mp}"), ""]
    doc += ["## MPC diagnostics", "", mpc_table({k: v for k, v in configs.items() if k != "A"}), ""]
    thr = ROOT / "calibration" / "cd_thresholds.json"
    if thr.exists():
        t = json.loads(thr.read_text())
        det = DetectorConfig(**t["detector"])
        doc += ["## Congestion calibration (development seeds)", "",
                f"Frozen thresholds: c1 = {det.c1}, c2 = {det.c2} — {t['selected_by']}. "
                f"Achieved on development data: coverage {fnum(t['achieved']['coverage'])}, "
                f"warning fraction {t['achieved']['warning_fraction']:.3f}, "
                f"{t['achieved']['onsets']} infeasibility onsets, {t['achieved']['missed']} missed.", ""]
        if (ROOT / "main_B" / "steps.jsonl.gz").exists():
            doc += ["## Congestion warnings, Phase I shadow mode (evaluation seeds, frozen thresholds, reporting only)",
                    "", shadow_alarm_table(ROOT / "main_B", det), ""]
    if configs["C"] is not None:
        doc += ["## Congestion warnings, Phase III active mode", "", active_alarm_table(configs["C"]), ""]
    if configs["C2"] is not None:
        doc += ["## Congestion warnings, Phase III rerun with the fixed detector (C2)", "",
                active_alarm_table(configs["C2"]), ""]
    doc += ["## Latency — single-worker pass (smoke seeds, 5 per cell), ms per step", "",
            latency_table(), ""]
    out.write_text("\n".join(doc))
    print("wrote", out)


if __name__ == "__main__":
    main()
