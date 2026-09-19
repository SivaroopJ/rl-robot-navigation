# 06: Triggered static block

**What to build:** The researcher can run the "+triggered static block" condition in all five families. When the robot enters a trigger region on its planned route, a block appears across the passage ahead. The block is visible to LiDAR and to collision checks, never enters A*'s map, and always leaves a detour.

**Blocked by:** 03, 04

**Status:** done (see Comments for the commit)

- [x] The trigger is placed on the static-layout A* route (arm-independent) at a fraction drawn between 0.3 and 0.7 of the route length
- [x] The block appears 1.5–2.5 m further along the route, spanning the passage (a doorway, aisle or corridor), with pre-registered sizes
- [x] Seam 2: a clear path still exists with the block added, on a large seed sample per family
- [x] The env fires the block when the robot enters the trigger region, adding it to the static obstacles, and logs the step and robot position
- [x] Seam 1: from the firing step on, the block is visible in LiDAR and in collision checks; the A* map never contains it
- [x] Seam 1: a policy's actions are unchanged when the spec's trigger and block fields are perturbed before firing (the information boundary)
- [x] Records say whether and when the trigger fired

## Notes from tickets 02–05 (2026-09-19)

- **Draw order.** Events go into the family draw after start and goal and before the pedestrians (§4.7). Bump `GENERATOR_VERSION`.
- **Reserved block zone.** It must also go into `PedestrianMotion`'s layout in every condition, not only into route drawing. Randomized pedestrians may come within 0.3 m of obstacles, which a zone reserved at only 0.45 m does not cover.
- **The fired block must reach the env as a new `static_obstacles` list**, not an in-place append, so the policy's map (taken at build time) never contains it.
- **`EpisodeSpec.route`** is the arm-independent start→goal route at r_c, the one to place the trigger on.

## Comments

- 2026-09-19. Built. Generator version `rs-gen-0.3`: every family draw now carries the trigger and the block, so the accepted draws of every cell changed.
  - `robustsuite/events.py`: the trigger disc and the block (passage measurement, placement, feasibility).
  - `robustsuite/geometry.py`: `with_rect` adds one rectangle to an oracle's occupancy without the oracle's cell-by-cell rebuild (tested equal to a rebuilt oracle); `same_component`.
  - `robustsuite/scenario.py`: events drawn after start and goal, before the pedestrians; pedestrians routed on the layout plus the block in every condition; `trigger`, `block`, `event_redraws`, `active_block` and `pedestrian_layout` in the spec.
  - `robustsuite/scenario_env.py`: the trigger is checked through a wrapper on the pedestrian model, because the frozen `step()` calls the model after the robot moves and before the collision check and observation. The block arrives as a new `static_obstacles` list. Events: `trigger_fired` (step, position) and `block_added`.
  - `robustsuite/harness.py`: the record has `trigger_fired` and `trigger_step`.
- **Readings, recorded in RS_DESIGN §14.2 before any run.** The one that tightens a rule, which the researcher approved: the block keeps ≥ 0.8 m (disc radius + robot radius) from the trigger centre, not just outside the disc. Otherwise it could appear on top of a robot already inside the disc.
- **Review notes, left open by decision:**
  - In open clutter, a block may span from one box to the wall or to another box, so the detour goes round the box rather than through a doorway. This meets §4.6 as written. The researcher chose to check how often it happens in the map validation report (ticket 08).
  - Randomized motion perturbs the trigger fields only, because the block rectangle is also the pedestrians' reserved zone.
  - The scenario event log is empty before firing and never reaches a policy. It is not perturbed separately.
- **Tests:**
  - generator, re-derived from the rectangles on 50 draws per family:
    - the trigger is at a fraction of 0.3–0.7 on the route, and the start is outside the disc;
    - the block's offset, axis, thickness and length; both passage edges touch an obstacle or the outer wall, and the passage between them is free;
    - its clearances from start, goal and trigger;
    - a clear path on a freshly built frozen oracle with the block added (seam 2);
    - a failed event draw redraws the layout, and the cap still raises;
  - env: the block fires on the exact step the robot's centre enters the disc, once; never in `static` or `dynamic`; LiDAR and collisions see it from the firing step on, and not before; pedestrians are identical in `dynamic` and `trigger_block`, fired or not;
  - harness: actions and traces before firing are identical when the trigger and block fields are perturbed (the information boundary). The A* map and the planner's occupancy never contain the block. The record gives the firing step and position.
- **Measured** (20 draws per family): the event draw failed on 1–19 layouts per family out of 26–43 layout draws. Passage widths had a median of 1.4–2.1 m. Generation takes 0.9–2.8 s per spec, mostly pedestrian route simplification, as before. In short baseline runs (150 steps), the robot fired the trigger in 11 of 12 episodes, at steps 16–97.
