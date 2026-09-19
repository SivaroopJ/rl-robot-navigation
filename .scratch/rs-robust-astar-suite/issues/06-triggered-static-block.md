# 06: Triggered static block

**What to build:** The researcher can run the "+triggered static block" condition in all five families. When the robot enters a trigger region on its planned route, a block appears across the passage ahead. The block is visible to LiDAR and to collision checks, never enters A*'s map, and always leaves a detour.

**Blocked by:** 03, 04

**Status:** ready-for-agent

- [ ] The trigger is placed on the static-layout A* route (arm-independent) at a fraction drawn between 0.3 and 0.7 of the route length
- [ ] The block appears 1.5–2.5 m further along the route, spanning the passage (a doorway, aisle or corridor), with pre-registered sizes
- [ ] Seam 2: a clear path still exists with the block added, on a large seed sample per family
- [ ] The env fires the block when the robot enters the trigger region, adding it to the static obstacles, and logs the step and robot position
- [ ] Seam 1: from the firing step on, the block is visible in LiDAR and in collision checks; the A* map never contains it
- [ ] Seam 1: a policy's actions are unchanged when the spec's trigger and block fields are perturbed before firing (the information boundary)
- [ ] Records say whether and when the trigger fired

## Notes from tickets 02–05 (2026-09-19)

- **Draw order.** Events go into the family draw after start and goal and before the pedestrians (§4.7). Bump `GENERATOR_VERSION`.
- **Reserved block zone.** It must also go into `PedestrianMotion`'s layout in every condition, not only into route drawing. Randomized pedestrians may come within 0.3 m of obstacles, which a zone reserved at only 0.45 m does not cover.
- **The fired block must reach the env as a new `static_obstacles` list**, not an in-place append, so the policy's map (taken at build time) never contains it.
- **`EpisodeSpec.route`** is the arm-independent start→goal route at r_c, the one to place the trigger on.
