# 13: Sealed final evaluation and report

**What to build:** The researcher gets the sealed-block result: the baseline's characterisation of the suite and, if there is a winner, the pre-registered GO/NO-GO verdict against the baseline, plus the final report.

**Blocked by:** 12

**Status:** ready-for-agent

- [ ] The GO gate as a pure function: pooled success higher with exact McNemar p < 0.05; fail if pooled collisions are higher with p < 0.05; fail if M0 success is lower with p < 0.05. Tested at the thresholds
- [ ] The final-evaluation entry point is the only one that opens the sealed block. It runs the baseline and the winner (or the baseline only) once: 20 × 2 × 200, plus the anchor
- [ ] The report gives per-cell arm tables, pooled and per-family results, worst-cell success, triggered-event analyses (overall and conditional on firing), per-cell regressions with a multiple-comparison note, the gate, and a dated Deviations section
- [ ] Provenance, trace hashes and the manifest are recorded
- [ ] Vault: the experiment note verdict, Timeline, Open Threads, Reports Index and Home are updated, and the vault is synced
