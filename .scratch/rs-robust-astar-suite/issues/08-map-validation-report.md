# 08: Map validation report

**What to build:** The researcher gets a map validation report for the whole suite and approves it (or asks for changes) before any baseline data is produced.

**Blocked by:** 03, 04, 05, 06, 07

**Status:** ready-for-agent

- [ ] An entry point generates specs for all 20 cells × 2 motion conditions on the diagnostic seed indices, plus the anchor, without running episodes
- [ ] The report shows, per cell: acceptance and resampling rates; clearance, passage-width and route-length distributions; start/goal distance bins; pedestrian counts; trigger fractions and block and spawn distances
- [ ] Rendered samples of every cell (layout, start/goal, pedestrian routes, trigger region, block or spawn), plus short rollout renders showing that pedestrians respect walls
- [ ] The report is written to the reports folder and linked in the vault note
- [ ] **The researcher approves the report.** The approval is recorded in this ticket and the design document before ticket 09 starts

## Notes from ticket 06 (2026-09-19)

- **Open-clutter blocks.** A block can span from one box to the wall or to another box, so the robot detours round the box instead of through a doorway-like gap. The researcher asked for the report to show how often this happens: the rate per family, and rendered samples of such blocks.
- Trigger placement statistics are in `spec.trigger` (fraction, arc) and `spec.block` (offset, axis, passage width), and each spec's `event_redraws` counts the failed event draws.
