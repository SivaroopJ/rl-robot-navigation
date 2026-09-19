# 07: Triggered dynamic spawn

**What to build:** The researcher can run the "+triggered dynamic spawn" condition in all five families. When the robot enters the trigger region, a pedestrian appears from behind the nearest occluder and either crosses the planned route or comes at the robot head-on, then continues as a normal pedestrian.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] The spawn start is behind the nearest occluding obstacle, 1.5–3 m from the trigger
- [ ] Two entry variants: crossing (straight across the planned route ahead) and head-on (along the route towards the robot); the variant is drawn from the seed and recorded
- [ ] Seam 2 feasibility, on a large seed sample: the spawn is at least 1.5 m from any robot position that can fire the trigger; it never lands on the start or goal; its speed is at most the robot's maximum
- [ ] After its scripted entry, the spawned pedestrian continues as a normal pedestrian and never enters walls
- [ ] Seam 1: spawn events are logged; paired arms see the same spawn; the information-boundary test covers the spawn fields

## Notes from tickets 02–05 (2026-09-19)

- **The env sizes the pedestrian arrays and `n_dynamic_obstacles` at reset.** A spawn needs a pre-allocated slot that stays out of LiDAR, collision checks and the observation until the trigger fires. `PedestrianMotion` has no inactive state yet.
- **Per-route speed:** `Route.speed` drives each pedestrian.

## Notes from ticket 06 (2026-09-19)

- **The trigger disc is already in every spec** (`spec.trigger`), drawn in `robustsuite.events.draw_trigger_block`. The spawn has to share that disc. Each of the 50 draws of f must succeed for the block and the spawn together, so the spawn draw goes inside the f loop, and generator version `rs-gen-0.4` follows.
- **The block zone is already reserved.** `spec.pedestrian_layout` is the layout plus the block, and the scenario env runs `PedestrianMotion` on it in every condition. The spawn's entry paths should use the same map.
- **The firing hook exists.** `RSScenarioEnv._check_trigger` runs through the `_Triggered` wrapper on the pedestrian model, after the robot moves and before the pedestrians move. The spawn can activate its slot there, so the pedestrian moves from the firing step on. It stays one event per episode.
- **The information-boundary test** (`test_actions_before_firing_do_not_see_the_trigger_or_block_fields`) perturbs the trigger and block fields. Extend it to the spawn fields.
