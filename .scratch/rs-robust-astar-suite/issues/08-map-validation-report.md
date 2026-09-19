# 08: Map validation report

**What to build:** The researcher gets a map validation report for the whole suite and approves it (or asks for changes) before any baseline data is produced.

**Blocked by:** 03, 04, 05, 06, 07

**Status:** done (approved by the researcher, 2026-09-19)

- [x] An entry point generates specs for all 20 cells × 2 motion conditions on the diagnostic seed indices, plus the anchor, without running episodes
- [x] The report shows, per cell: acceptance and resampling rates; clearance, passage-width and route-length distributions; start/goal distance bins; pedestrian counts; trigger fractions and block and spawn distances
- [x] Rendered samples of every cell (layout, start/goal, pedestrian routes, trigger region, block or spawn), plus short rollout renders showing that pedestrians respect walls
- [x] The report is written to the reports folder and linked in the vault note
- [x] **The researcher approves the report.** The approval is recorded in this ticket and the design document before ticket 09 starts

## Notes from ticket 06 (2026-09-19)

- **Open-clutter blocks.** A block can span from one box to the wall or to another box, so the robot detours round the box instead of through a doorway-like gap. The researcher asked for the report to show how often this happens: the rate per family, and rendered samples of such blocks.
- Trigger placement statistics are in `spec.trigger` (fraction, arc) and `spec.block` (offset, axis, passage width), and each spec's `event_redraws` counts the failed event draws.

## Notes from ticket 07 (2026-09-19)

- **Spawn statistics:** `spec.spawn` has the start, variant, target, scripted entry and full route. Report the spawn distance and the variant split. The variant is a fair coin per seed (RS_DESIGN §14.3 item 3).
- **Zero-length crossing walks:** a crossing spawn with no room past the route stops at its target, so it never crosses (§14.3 item 6 allows this). Report how often it happens per family.
- **Render each spawn's scripted entry.** It ignores the block zone (§14.3 item 1), so in `trigger_spawn` renders the block should not be drawn.

## Comments

- 2026-09-19. Built. The generator is `2d0d624` and the report was generated at that commit: `MD_files/robustsuite/RS1_MAP_VALIDATION_REPORT.md`, figures in `MD_files/robustsuite/figures/rs1/`, and `results/robustsuite/RS1/stats.json`. It takes 1m48s on 14 workers. No robot episode is run, and a test checks that.
  - **Per family, not per cell.** The four cells of a family are views of one draw (§4.3), so the report gives the draw statistics once per family. A consistency check regenerates all 4 conditions × 2 motions on the first seed of each family: 40 checked, 0 mismatches.
  - **What it shows:**
    - every clearance floor holds;
    - the pedestrian-only rollouts meet the floors (0.45 m fixed, 0.3 m randomized) and the 0.0675 m step;
    - at most 17 layout draws for any one seed;
    - no zero-length crossing walks;
    - variants split 136 crossing / 114 head-on.
  - **For the researcher to judge:**
    - accepted trigger fractions lean early: 108 / 62 / 46 / 34 of 250 per 0.1 bin;
    - blocks with a detour under 1 m: dense_clutter 14 / 50, open_clutter 5 / 50, 0–1 elsewhere;
    - aisles never produce the 8.5–10.5 m distance bin.
- **Two-axis review fixes:**
  - a report-writer crash (a name read before it was assigned);
  - duplicated constants now come from the generator;
  - resampling as rates; five-number summaries; loop lengths;
  - an M0 rollout render;
  - the summary states only measured facts.
- 2026-09-19. **Approved by the researcher.** Recorded in RS_DESIGN ("Approval: map validation report"). Two properties are recorded as known, not defects: the early-leaning trigger fractions and the short-detour blocks (dense_clutter 14 / 50, open_clutter 5 / 50). Per-family reporting is accepted. Ticket 09 may start.
