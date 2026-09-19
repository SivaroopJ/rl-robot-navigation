# 10: Candidate list

**What to build:** The researcher gets at most three written improvement candidates, grounded in the diagnostic, and approves them before any candidate is built.

**Blocked by:** 09

**Status:** ready-for-human (awaiting the researcher's approval)

- [x] Each candidate is written down: mechanism, changed components, information used (onboard only), expected effect on the named failure modes, parameter-search budget, and an expected compute cost
- [x] At most three candidates; a combination of mechanisms may be one of them
- [x] No candidate changes the CBF/DR formulation or the dynamics, or uses ground-truth or trigger/spawn information
- [x] The list is recorded in the design document before any tuning run
- [ ] **The researcher approves the list.** Ticket 11 is then replaced by one implementation ticket per approved candidate

## Comments

**2026-09-20 — candidate list written, awaiting approval.** Appended to `MD_files/robustsuite/RS_DESIGN.md` as "Candidate list (section 7.4), 2026-09-20". Built from the RS2 diagnostic and a new collision probe (`experiments/robustsuite/rs2_collision_probe.py`, `results/robustsuite/RS2/collision_probe.json`; replays of all 206 pedestrian collisions, no new episode).

1. `replan` (1 component): a LiDAR map of new static obstacles plus A* replanning on a blocked path or a stall. Targets `timeout_at_block` (321), about 45 collisions while waiting at the block, and `timeout_stuck` (36). Expected +10 to +19 pp.
2. `yield` (1 component): a behaviour layer that sets γ to a side point when a tracked pedestrian's closest approach within H s is under D. Targets pedestrian collisions (194). Expected +2 to +5 pp.
3. `replan_yield` (2 components): the combination. Expected +12 to +22 pp.

The two-axis review caught that m = 0.15 in candidate 1's default would inflate the parked robot's own cell, so A* would never replan at the block. Fixed: the default is m = 0.0 and A* starts from the nearest free cell. It also sharpened the K argument, the known-map source (reset-time snapshot), read-only tracker access, and the probe wording.

On approval: record the approval in RS_DESIGN and replace ticket 11 with one implementation ticket per approved candidate.
