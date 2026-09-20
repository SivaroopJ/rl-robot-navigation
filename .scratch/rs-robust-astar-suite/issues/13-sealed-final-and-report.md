# 13: Sealed final evaluation and report

**What to build:** The researcher gets the sealed-block result: the baseline's characterisation of the suite and, if there is a winner, the pre-registered GO/NO-GO verdict against the baseline, plus the final report.

**Blocked by:** 12

**Status:** ready-for-human

- [x] The GO gate as a pure function: pooled success higher with exact McNemar p < 0.05; fail if pooled collisions are higher with p < 0.05; fail if M0 success is lower with p < 0.05. Tested at the thresholds
- [x] The final-evaluation entry point is the only one that opens the sealed block. It runs the baseline and the winner (or the baseline only) once: 20 × 2 × 200, plus the anchor
- [x] The report gives per-cell arm tables, pooled and per-family results, worst-cell success, triggered-event analyses (overall and conditional on firing), per-cell regressions with a multiple-comparison note, the gate, and a dated Deviations section
- [x] Provenance, trace hashes and the manifest are recorded
- [x] Vault: the experiment note verdict, Timeline, Open Threads, Reports Index and Home are updated, and the vault is synced

## Comments

**2026-09-21, done. RS Experiment 1 is complete.**

The gate is `robustsuite/gates.py` (pure, 17 tests placing the discordant pairs either side of
p = 0.05); the run is `experiments/robustsuite/rs6_final.py`, the only entry point that opens
RS_FINAL. Report: `MD_files/robustsuite/RS6_FINAL_REPORT.md`.

The baseline ran alone (no winner, §8), once, at `e0af5ab`: 20 cells x 2 motions x 200 seeds plus
the M0 anchor, 8 400 episodes in 2 h 06. §9 is recorded as not applicable; the arm was fixed
before the sealed block was opened. The two-arm path was exercised on a smoke run so the gate is
not dead code.

**Pooled: 0.699 success, 0.110 collision, 0.190 timeout, SPL 0.611.** RS_DIAG 0.719, RS_TUNE
0.700, RS_FINAL 0.699 — three disjoint blocks agree, so this is the stack and the suite, not a
draw.

- Static cells 0.97-1.00 at SPL 0.92-0.98: every family is solvable and the route is good, so the
  whole loss is pedestrians and events.
- Conditional on the block appearing, the trigger_block cells reach corridors 0.003, rooms 0.020,
  aisles 0.032, open clutter 0.075, dense clutter 0.192. `timeout_at_block` is 1315 of 8000.
- 334 of the 756 pedestrian collisions are in trigger_block cells, where the robot is standing
  still in front of the block. The block and the pedestrians are one problem.
- M0 anchor 361 / 400, exactly HD0's count on its own seeds. 38.0 ms mean step time.

**For the researcher.** The experiment is closed; the vault note carries the verdict. The open
question is the next axis: phase 5 showed that leaving the standstill without deciding how to move
among people trades timeouts for collisions, and Random recovery (14% feasible directions) is
still untouched.
