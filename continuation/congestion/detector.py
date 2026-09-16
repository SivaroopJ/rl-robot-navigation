"""Three-state congestion detector with hysteresis, and the alarm statistics (decisions 10, 12).

STATES   0 Normal, 1 Cautious, 2 Critical; inputs margin = M_look - tau_eff and clr = predicted
         clearance (both from the monitor).
ENTER    Critical if margin < 0 or clr < 0.05; Cautious if margin < c1 or clr < c2. Escalation is
         immediate.
LEAVE    from state s, the exit condition must hold for `hold_steps` consecutive steps, then the
         state drops ONE level:  Critical: margin >= 0 + h and clr >= 0.05 + h;
         Cautious: margin >= c1 + h and clr >= c2 + h;  h = 0.05 (margin units / metres).

ALARMS (per agent, over steps where the agent is active)
    warning onset      state switches from 0 to >= 1
    true alarm         an infeasible QP step within t+1 .. t+W of the onset (W = 10)
    lead time          first such step - t
    infeasibility onset  infeasible at t and not at t-1
    missed             infeasibility onset with state < 1 on every step t-W .. t-1
    coverage           1 - missed / onsets;  warning fraction = share of steps with state >= 1
The detector is a deterministic function of the (margin, clr) series, so shadow-mode series can be
replayed offline with any thresholds (calibration).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

NORMAL, CAUTIOUS, CRITICAL = 0, 1, 2


@dataclass(frozen=True)
class DetectorConfig:
    c1: float = 0.15
    c2: float = 0.30
    crit_margin: float = 0.0
    crit_clear: float = 0.05
    hysteresis: float = 0.05
    hold_steps: int = 3

    def __post_init__(self):
        if self.c1 < self.crit_margin or self.c2 < self.crit_clear:
            raise ValueError("Cautious thresholds must not be stricter than Critical ones")

    def as_dict(self):
        return asdict(self)


class CongestionDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = NORMAL
        self._ok = 0

    def raw_level(self, margin, clr):
        c = self.cfg
        if margin < c.crit_margin or clr < c.crit_clear:
            return CRITICAL
        if margin < c.c1 or clr < c.c2:
            return CAUTIOUS
        return NORMAL

    def exit_ok(self, state, margin, clr):
        c = self.cfg
        if state == CRITICAL:
            return margin >= c.crit_margin + c.hysteresis and clr >= c.crit_clear + c.hysteresis
        if state == CAUTIOUS:
            return margin >= c.c1 + c.hysteresis and clr >= c.c2 + c.hysteresis
        return True

    def update(self, margin, clr):
        raw = self.raw_level(margin, clr)
        if raw > self.state:
            self.state, self._ok = raw, 0
        elif self.state > NORMAL and self.exit_ok(self.state, margin, clr):
            self._ok += 1
            if self._ok >= self.cfg.hold_steps:
                self.state, self._ok = self.state - 1, 0
        else:
            self._ok = 0
        return self.state


def replay(cfg: DetectorConfig, margins, clrs):
    det = CongestionDetector(cfg)
    return np.array([det.update(m, c) for m, c in zip(margins, clrs)], dtype=int)


def alarm_stats(states, infeasible, active=None, *, window=10):
    s = np.asarray(states, int)
    inf = np.asarray(infeasible, bool)
    act = np.ones(len(s), bool) if active is None else np.asarray(active, bool)
    n = len(s)
    warn = s >= CAUTIOUS
    onsets = [t for t in range(n) if act[t] and warn[t] and (t == 0 or not warn[t - 1])]
    true_alarms, leads = 0, []
    for t in onsets:
        hits = [k for k in range(t + 1, min(n, t + window + 1)) if inf[k]]
        if hits:
            true_alarms += 1
            leads.append(hits[0] - t)
    inf_onsets = [t for t in range(n) if act[t] and inf[t] and (t == 0 or not inf[t - 1])]
    missed = sum(1 for t in inf_onsets if not warn[max(0, t - window):t].any())
    n_act = int(act.sum())
    return {"warning_onsets": len(onsets), "true_alarms": true_alarms,
            "false_alarms": len(onsets) - true_alarms, "lead_steps": leads,
            "infeasible_onsets": len(inf_onsets), "missed": missed,
            "warning_steps": int((warn & act).sum()), "active_steps": n_act,
            "state_counts": [int(((s == k) & act).sum()) for k in (0, 1, 2)]}
