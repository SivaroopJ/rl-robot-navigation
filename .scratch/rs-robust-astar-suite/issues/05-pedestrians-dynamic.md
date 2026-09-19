# 05: Pedestrian motion and the +dynamic condition

**What to build:** The researcher can run the "+dynamic" obstacle condition, with pedestrians that walk routes through free space and never pass through walls. Both motion conditions are available: fixed (exact paths) and randomized (OU heading noise).

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] A pedestrian motion model with the same interface as the frozen motion models: routes through free space between waypoints at 0.675 m/s; the next waypoint is drawn from the env's RNG
- [ ] Fixed follows paths exactly. Randomized adds OU heading noise with steering back towards the path
- [ ] Pedestrian counts follow family density (pre-registered, between 3 and 6); radius and speed are M0's
- [ ] The episode spec carries the pedestrian routes, identical across motion conditions and arms
- [ ] Tests: pedestrians never enter a static obstacle (over long rollouts in every family built so far); they never exceed the robot's maximum speed; they never start on the start or goal
- [ ] Seam 1: paired arms see identical pedestrian trajectories for the same seed and motion condition
- [ ] The M0 anchor keeps its original motion (the bit-identical test still passes)
