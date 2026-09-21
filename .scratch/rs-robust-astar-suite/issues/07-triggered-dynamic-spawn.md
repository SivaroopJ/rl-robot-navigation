# 07: Triggered dynamic spawn

**What to build:** The researcher can run the "+triggered dynamic spawn" condition in all five families. When the robot enters the trigger region, a pedestrian appears from behind the nearest occluder and either crosses the planned route or comes at the robot head-on, then continues as a normal pedestrian.

**Blocked by:** 05, 06

**Status:** done (see Comments for the commit)

- [x] The spawn start is behind the nearest occluding obstacle, 1.5–3 m from the trigger
- [x] Two entry variants: crossing (straight across the planned route ahead) and head-on (along the route towards the robot); the variant is drawn from the seed and recorded
- [x] Seam 2 feasibility, on a large seed sample: the spawn is at least 1.5 m from any robot position that can fire the trigger; it never lands on the start or goal; its speed is at most the robot's maximum
- [x] After its scripted entry, the spawned pedestrian continues as a normal pedestrian and never enters walls
- [x] Seam 1: spawn events are logged; paired arms see the same spawn; the information-boundary test covers the spawn fields

## Notes from tickets 02–05 (2026-09-19)

- **The env sizes the pedestrian arrays and `n_dynamic_obstacles` at reset.** A spawn needs a pre-allocated slot that stays out of LiDAR, collision checks and the observation until the trigger fires. `PedestrianMotion` has no inactive state yet.
- **Per-route speed:** `Route.speed` drives each pedestrian.

## Notes from ticket 06 (2026-09-19)

- **The trigger disc is already in every spec** (`spec.trigger`), drawn in `robustsuite.events.draw_trigger_block`. The spawn has to share that disc. Each of the 50 draws of f must succeed for the block and the spawn together, so the spawn draw goes inside the f loop, and generator version `rs-gen-0.4` follows.
- **The block zone is already reserved.** `spec.pedestrian_layout` is the layout plus the block, and the scenario env runs `PedestrianMotion` on it in every condition. The spawn's entry paths should use the same map.
- **The firing hook exists.** `RSScenarioEnv._check_trigger` runs through the `_Triggered` wrapper on the pedestrian model, after the robot moves and before the pedestrians move. The spawn can activate its slot there, so the pedestrian moves from the firing step on. It stays one event per episode.
- **The information-boundary test** (`test_actions_before_firing_do_not_see_the_trigger_or_block_fields`) perturbs the trigger and block fields. Extend it to the spawn fields.

## Comments

- 2026-09-19. Built. Generator version `rs-gen-0.4`: every family draw now also carries the spawn, so the accepted draws of every cell changed again.
  - `robustsuite/events.py`:
    - the spawn start (2–3 m from the trigger, occluded, clear of the start and goal);
    - the crossing and head-on entries;
    - the walk into the spawned pedestrian's own waypoint cycle.
    The block and the spawn must both succeed for the same draw of f. The variant is the seed's first draw.
  - `robustsuite/scenario_env.py`: on firing, the spawned pedestrian becomes the last obstacle slot and takes its first step in the same env step. It has its own motion model on the static layout and its own generator (episode seed, 1). The event is `spawned` (step, start, variant).
  - `robustsuite/pedestrians.py`: `draw_cycle` split out of `draw_route`. It fails at once from a start that is not on a free grid cell, and it stops routing legs after the first one that fails. The spawn search was 5–13 s per spec before this and 1.4–2.2 s after.
  - `robustsuite/geometry.py`: `segment_crosses`, `route_slice`.
- **Researcher decisions before any run (RS_DESIGN §14.3):**
  - **The spawn ignores the block zone.** The head-on target was always within 0.3 m of the block, so no head-on entry could ever be drawn.
  - **The spawned pedestrian has its own noise stream.** Otherwise the regular pedestrians' noise after firing would depend on the robot.
  - Readings: the variant is a fair coin per seed; the occluder rule; attempt counts; the crossing and head-on geometry.
- **Occluder rule.** The ticket says "behind the nearest occluding obstacle". §4.6, which governs, requires only that the segment from the trigger to the start crosses a static rectangle. "Nearest" is not enforced.
- **Tests:** 49 new, re-derived from the rectangles and the route on 50 draws per family:
  - generator:
    - the spawn distance, clearance, occlusion and the start/goal margins;
    - the variant split;
    - the crossing geometry (target 1–2 m ahead, axis, direction, stop 0.45 m short);
    - head-on (target 2 m ahead, 3 m back along the route);
    - entry and cycle clearance; the slots;
  - motion: 1500-step rollouts of the spawned pedestrian, both motions;
  - env:
    - it fires on the exact entry step, once, and only in `trigger_spawn`;
    - before firing it is absent from every array; from the firing step it is in the LiDAR, the collisions and the observation;
    - the regular pedestrians equal the dynamic cell's through a spawn;
    - the spawned walk is identical whatever step it fires on;
  - harness: actions before firing are unchanged when the spawn fields are perturbed.
- **Two-axis review fixes:** the triplicated drive loop and the LiDAR helpers in the tests; a named state for the fired spawn; constant and docstring conventions; a guard for the six obstacle slots; the version note (rs-gen-0.4 also changed the regular pedestrians' draws).
- **Full suite:** 834 passed, plus the 4 known legacy failures and 1 new one. The new one, `test_resampling_skips_infeasible_draws_and_counts_them`, assumed the first real layout would be accepted, which is not true under rs-gen-0.4. It now asserts the invariant (every skipped draw is counted) and passes when re-run on its own.
