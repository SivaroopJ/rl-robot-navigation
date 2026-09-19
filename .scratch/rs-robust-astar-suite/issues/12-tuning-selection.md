# 12: Tuning-block selection

**What to build:** The researcher gets the pre-registered selection outcome: a single winner, or "no improvement found", decided on the tuning block by the fixed rule.

**Blocked by:** 11a, 11b, 11c

**Status:** ready-for-agent

- [ ] The selection rule as a pure function: qualify at ≥ +2 pp pooled success and ≤ +1 pp pooled collision against the baseline; pooled rates weight all 20 cells equally over both motion conditions; exact arithmetic
- [ ] The winner is the highest-success qualifier; ties go to the candidate with fewer changed components; no qualifier means no winner. Tested at the thresholds
- [ ] The baseline and every candidate run on the tuning block (20 × 2 × 50), paired
- [ ] A candidate whose p95 step time on the tuning block exceeds 100 ms is disqualified before the rule applies
- [ ] The selection report gives the per-candidate table, the rule's application and the verdict. The vault is updated
