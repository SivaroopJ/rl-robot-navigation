# 12: Tuning-block selection

**What to build:** The researcher gets the pre-registered selection outcome: a single winner, or "no improvement found", decided on the tuning block by the fixed rule.

**Blocked by:** 11a, 11b, 11c

**Status:** ready-for-human

- [x] The selection rule as a pure function: qualify at ≥ +2 pp pooled success and ≤ +1 pp pooled collision against the baseline; pooled rates weight all 20 cells equally over both motion conditions; exact arithmetic
- [x] The winner is the highest-success qualifier; ties go to the candidate with fewer changed components; no qualifier means no winner. Tested at the thresholds
- [x] The baseline and every candidate run on the tuning block (20 × 2 × 50), paired
- [x] A candidate whose p95 step time on the tuning block exceeds 100 ms is disqualified before the rule applies
- [x] The selection report gives the per-candidate table, the rule's application and the verdict. The vault is updated

## Comments

**2026-09-20, done. No improvement found.**

The rules are `robustsuite/selection.py` (pure functions, exact rational arithmetic, tested at
their thresholds in `tests/test_robustsuite_selection.py`); the run is
`experiments/robustsuite/rs5_selection.py`, in two stages. Readings of open wording are in
`RS_DESIGN.md`, "Readings before the phase 5 run (2026-09-20, ticket 12)". Report:
`MD_files/robustsuite/RS5_SELECTION_REPORT.md`.

The §7.3 search (19 blocks, the first 20 RS_DIAG seeds) chose `detour` K8_L1.0_stall_off,
`yield` H2_D0.9_L1.0 and their pair. Every `stall on` configuration went over the search's
collision limit.

On RS_TUNE (20 cells x 2 motions x 50 seeds per arm, paired), with every arm inside the §7.2
p95 limit (46.1-49.6 ms):

| arm | success | collision |
|---|---|---|
| `astar_random` | 0.700 | 0.102 |
| `detour` | 0.710 (+1.00 pp) | 0.117 (+1.50 pp) |
| `yield` | 0.584 (-11.65 pp) | 0.101 |
| `detour_yield` | 0.579 (-12.15 pp) | 0.101 |

No candidate qualifies (§8 needs >= +2 pp success and <= +1 pp collision), so there is no winner
and phase 6 (ticket 13) runs the baseline alone on RS_FINAL as the suite's characterisation.

`detour` clears blocks more often -- the five trigger_block cells go 0.132 -> 0.200 mean success
-- but meets the pedestrians that walk past the block while doing it (0.154 -> 0.208 collision
in those same cells, essentially its whole +1.50 pp). `yield` behaves as the phase 4 checks
predicted: it steps aside for tracks the frozen tracker builds out of static geometry and loses
most in cells holding no pedestrian at all (aisles_static 1.00 -> 0.72).

**For the researcher.** Ticket 13 needs no decision from this one: "no improvement found" is a
pre-registered outcome (§8, spec story 47). The RS_FINAL block is untouched.
