# 09: Baseline diagnostic

**What to build:** The researcher gets the frozen A* + Random baseline's failure breakdown on the diagnostic block of the new suite, the evidence for choosing improvement candidates.

**Blocked by:** 08 (researcher approval)

**Status:** done

- [x] The baseline runs on the diagnostic block: 20 cells × 2 motion conditions × 50 seeds, plus the M0 anchor, frozen SCS, provenance recorded (commit, manifest, parameter hashes, generator version)
- [x] Per cell, per family and pooled: success, collision (dynamic/static/wall), timeout, stuck, SPL, minimum clearance, infeasible-step rate, Random event rate and recovery margins
- [x] Triggered cells are reported overall and conditional on the trigger having fired, with collision timing and location relative to the trigger
- [x] Rendered trajectories of the most common failure in each cell
- [x] The report (reports folder) ranks the failure modes by frequency and names the likely mechanism of each
- [x] The M0 anchor matches the known F-D numbers within sampling noise

## Comments

**2026-09-20 — done.** Run at `4c01512` (clean tree, manifest verified, `rs-gen-0.4`, RS_DIAG, 2 100 episodes, about 35 min on 14 workers); report regenerated at `6a69dd0` with the evidence columns and the written reading → `MD_files/robustsuite/RS2_BASELINE_DIAGNOSTIC_REPORT.md`, results in `results/robustsuite/RS2/` (trace sidecars not committed; hashes in `traces.sha256`).

- Pooled (20 cells, equal weights): success 0.719, collision 0.102, timeout 0.178.
- Ranked modes: `timeout_at_block` 321 (a feasible CLF–CBF standstill 0.45 m from the block face; A* never replans), `collision_dynamic` 183 (half in trigger_block cells, while waiting at the block), `timeout_stuck` 36, `collision_spawned` 11, `collision_static` 7, `collision_block` 4.
- DCPError freezes: 2 of 2 000 episodes, both 0.56 m from the outer wall. Measured: the env's LiDAR reads the outer wall 0.3 m short and rectangles at their true distance, so h < 0 happens within 0.6 m of the outer wall only. RS_DESIGN §14.1 item 1 words it as "an obstacle"; not edited (design file), flagged for the researcher.
- M0 anchor 87/100, consistent with HD0's 361/400 (Fisher p 0.36).
- Reading choices, recorded in `robustsuite/diagnostic.py`: one mode per failure in a fixed order; freeze = ≥ 10 consecutive DCPError steps; "at the block" = fired block within 1.0 m at the timeout; anchor compared unpaired with Fisher's exact test.
