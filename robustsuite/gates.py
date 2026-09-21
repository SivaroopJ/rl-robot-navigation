"""The pre-registered GO gate of phase 6 (RS_DESIGN.md section 9), as pure functions.

Pure: it takes the finished per-episode records of the baseline and the winner and returns the
verdict. Nothing here runs, loads or writes an episode, and the gate never sees an arm's name.

An arm is `{cell name: [records]}` over the 20 generated cells; the records of a cell must be
paired with the other arm's by (cell, motion, seed), which `_paired` enforces. Pooled rates weight
all 20 cells equally (section 9, spec story 50) and are exact Fractions, as `robustsuite.selection`
keeps them. Significance is the exact two-sided McNemar test on the episode-level paired outcomes
(`continuation.stats.paired_binary`, which `highdim.gates` uses too); the bootstrap CI it returns
is reported but is not part of the gate.

    1. Success  S_win > S_base with McNemar p < 0.05.
    2. Safety   fails if C_win > C_base with McNemar p < 0.05.
    3. Anchor   fails if S_win(M0) < S_base(M0) with McNemar p < 0.05.

GO when 1 holds and neither 2 nor 3 fails. Per-cell differences are reported with their own
McNemar p-values and are NOT gated: 20 cell-level tests produce false positives, and the report
says so next to them.
"""
from __future__ import annotations

from robustsuite.selection import POOLED_CELLS, pooled

#: Section 9: a difference counts as significant below this exact two-sided McNemar p.
ALPHA = 0.05


def _key(recs):
    """The pairing key. Every field is required: a record without one is not a paired episode."""
    return [(str(r["cell"]), r["motion"], r["seed"]) for r in recs]


def _paired(base, winner, *, cells=POOLED_CELLS):
    """The two arms' records flattened in one shared order, or ValueError if they are not
    paired episode for episode by (cell, motion, seed)."""
    for name, arm in (("baseline", base), ("winner", winner)):
        if set(arm) != set(cells):
            missing = sorted(set(cells) - set(arm))
            raise ValueError(f"the {name} is missing the cells {missing}")
    b = [r for c in cells for r in base[c]]
    w = [r for c in cells for r in winner[c]]
    if _key(b) != _key(w):
        raise ValueError("the baseline and the winner are not paired by (cell, motion, seed)")
    return b, w


def _counts(recs):
    return {"success": sum(int(r["success"]) for r in recs),
            "collision": sum(int(r["collision"]) for r in recs),
            "episodes": len(recs)}


def _rates(arm, cells):
    return pooled({c: _counts(arm[c]) for c in cells}, cells=cells)


def _test(x, y, key):
    """The exact McNemar test of x against y on a binary key, plus its bootstrap CI."""
    from continuation.stats import paired_binary
    return paired_binary(x, y, key)


def condition(base, winner, key, *, cells=POOLED_CELLS):
    """One arm-level comparison: the two pooled rates, the paired test, and the direction."""
    b, w = _paired(base, winner, cells=cells)
    t = _test(w, b, key)
    rb, rw = _rates(base, cells), _rates(winner, cells)
    return {"base": rb[key], "winner": rw[key], "diff": float(rw[key] - rb[key]),
            "only_winner": t["only_a"], "only_base": t["only_b"], "p": t["p"],
            "ci95": t["ci95"], "n": len(b)}


def per_cell(base, winner, *, cells=POOLED_CELLS):
    """Each cell's paired differences and their own McNemar p, on success and on collision
    (section 9: "per-cell regressions", spec story 51). Reported, never gated."""
    _paired(base, winner, cells=cells)
    out = {}
    for c in cells:
        n = len(base[c])
        row = {"n": n}
        for key in ("success", "collision"):
            t = _test(winner[c], base[c], key)
            kb = sum(int(r[key]) for r in base[c])
            kw = sum(int(r[key]) for r in winner[c])
            row[key] = {"base": kb / n, "winner": kw / n, "diff": (kw - kb) / n, "p": t["p"],
                        "only_winner": t["only_a"], "only_base": t["only_b"]}
        # the success view, flat, is what the report's per-cell table reads
        row.update({k: row["success"][k] for k in ("base", "winner", "p")},
                   d_success=row["success"]["diff"])
        out[c] = row
    return out


def anchor_condition(base_recs, winner_recs):
    """Section 9.3 on the M0 anchor's paired episodes: a fail needs a lower AND significant
    success rate. The anchor is outside the pooled rates."""
    if _key(base_recs) != _key(winner_recs):
        raise ValueError("the anchor's records are not paired by (cell, motion, seed)")
    t = _test(winner_recs, base_recs, "success")
    n = len(base_recs)
    kb = sum(int(r["success"]) for r in base_recs)
    kw = sum(int(r["success"]) for r in winner_recs)
    lower = kw < kb
    return {"base": kb / n, "winner": kw / n, "k_base": kb, "k_winner": kw, "n": n,
            "only_winner": t["only_a"], "only_base": t["only_b"], "p": t["p"],
            "ci95": t["ci95"], "lower": lower, "fail": bool(lower and t["p"] < ALPHA)}


def go_gate(base, winner, *, anchor=None, cells=POOLED_CELLS):
    """Section 9 on RS_FINAL: the winner against the baseline, paired.

    `base` and `winner` are {cell name: [records]}; `anchor` is the M0 anchor's
    (baseline records, winner records) pair, or None when the run has no anchor.
    """
    cells = tuple(cells)
    s = condition(base, winner, "success", cells=cells)
    c = condition(base, winner, "collision", cells=cells)
    s.update({"higher": s["winner"] > s["base"],
              "pass": bool(s["winner"] > s["base"] and s["p"] < ALPHA)})
    c.update({"higher": c["winner"] > c["base"],
              "fail": bool(c["winner"] > c["base"] and c["p"] < ALPHA)})
    a = None if anchor is None else anchor_condition(*anchor)
    ok = s["pass"] and not c["fail"] and not (a is not None and a["fail"])
    return {"success": s, "safety": c, "anchor": a, "alpha": ALPHA,
            "per_cell": per_cell(base, winner, cells=cells),
            "verdict": "GO" if ok else "NO-GO"}
