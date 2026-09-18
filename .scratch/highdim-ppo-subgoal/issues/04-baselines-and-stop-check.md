# 04: Run floor then ceiling on HD_DEV and HD_FINAL, then apply the stop check

**What to build:** The researcher gets paired floor and ceiling results on the fresh M0 blocks, plus a written, pre-registered stop verdict saying whether A* contributes enough on M0 to justify training PPO.

**Blocked by:** 01 (Pre-registration), 03 (Ceiling arm)

**Status:** done (`da0bcb4` code, `9134abc` results). Verdict: PROCEED

- [x] Run order, as pre-registered, using the frozen SCS controller and 2 conditions throughout:
  1. Floor on HD_DEV (100 × 2)
  2. Floor on HD_FINAL (200 × 2)
  3. Ceiling on the same HD_FINAL seeds
  4. Ceiling on HD_DEV
- [x] Results files record the commit, manifest hash and frozen-parameter hashes
- [x] The report gives success, collision (dynamic/static/wall), timeout, stuck rate, min clearance, SPL and infeasible-step rate for each arm and condition
- [x] The ceiling vs floor comparison is paired (McNemar and bootstrap CI via the existing statistics functions)
- [x] A written stop verdict: pooled S_A* − S_floor < 0.05 → STOP, otherwise proceed
- [x] The historical F-D numbers are cited as context only
- [x] The vault experiment note is updated with the baseline numbers and the verdict
- [x] ~~If the verdict is STOP, mark tickets 05–10 `wontfix`~~: not triggered, the verdict is PROCEED

## Comments

- 2026-09-18. Code `da0bcb4`, results `9134abc`. Report: `MD_files/highdim/HD0_BASELINES_REPORT.md`.
- **Stop check (HD_FINAL, pooled, 400 paired per arm):** S_floor 0.6075, S_ceiling 0.9025, gap +0.295 ≥ 0.05 → **PROCEED**.
- **Main findings:**
  - Floor failures are mostly *dynamic collisions* (0.295), not timeouts (0.080). Goal-only + Random gets stuck behind rectangles (stuck 0.343 vs 0.175) and is then hit while pinned.
  - Ceiling success 0.9025 is consistent with the historical F-D 0.910.
  - Floor success 0.6075 is below Phase 5's standalone goal-direct 0.72 (a different harness and parameters). This is why the floor was re-measured.
- **Pilot bar (HD_DESIGN §9.1):** floor dev success = **0.62** (HD_DEV, 200 pooled).
- **Review follow-ups before the run:**
  - trace sidecar made kill-safe and deterministic (partial JSONL, then a gz with mtime 0);
  - commit included in the checkpoint fingerprint;
  - provenance refuses untracked or modified code before and after the run;
  - outputs are all-or-nothing;
  - numpy-safe JSON;
  - the `MAX_STEPS` leak fixed;
  - the stuck metric reproduces exp5 exactly.
- **Disclosure** (also in the report): one pre-review smoke run used 2 HD_FINAL seeds for 60 steps, on the frozen arms only. The output was discarded.
