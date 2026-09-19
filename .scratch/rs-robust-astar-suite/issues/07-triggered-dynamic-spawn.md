# 07: Triggered dynamic spawn

**What to build:** The researcher can run the "+triggered dynamic spawn" condition in all five families. When the robot enters the trigger region, a pedestrian appears from behind the nearest occluder and either crosses the planned route or comes at the robot head-on, then continues as a normal pedestrian.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] The spawn start is behind the nearest occluding obstacle, 1.5–3 m from the trigger
- [ ] Two entry variants: crossing (straight across the planned route ahead) and head-on (along the route towards the robot); the variant is drawn from the seed and recorded
- [ ] Seam 2 feasibility, on a large seed sample: the spawn is at least 1.5 m from any robot position that can fire the trigger; it never lands on the start or goal; its speed is at most the robot's maximum
- [ ] After its scripted entry, the spawned pedestrian continues as a normal pedestrian and never enters walls
- [ ] Seam 1: spawn events are logged; paired arms see the same spawn; the information-boundary test covers the spawn fields
