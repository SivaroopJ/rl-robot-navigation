# 05: Pedestrian motion and the +dynamic condition

**What to build:** The researcher can run the "+dynamic" obstacle condition, with pedestrians that walk routes through free space and never pass through walls. Both motion conditions are available: fixed (exact paths) and randomized (OU heading noise).

**Blocked by:** 02

**Status:** done (see Comments for the commit)

- [x] A pedestrian motion model with the same interface as the frozen motion models: routes through free space between waypoints at 0.675 m/s; the next waypoint is drawn from the env's RNG
- [x] Fixed follows paths exactly. Randomized adds OU heading noise with steering back towards the path
- [x] Pedestrian counts follow family density (pre-registered, between 3 and 6); radius and speed are M0's
- [x] The episode spec carries the pedestrian routes, identical across motion conditions and arms
- [x] Tests: pedestrians never enter a static obstacle (over long rollouts in every family built so far); they never exceed the robot's maximum speed; they never start on the start or goal
- [x] Seam 1: paired arms see identical pedestrian trajectories for the same seed and motion condition
- [x] The M0 anchor keeps its original motion (the bit-identical test still passes)

## Comments

- 2026-09-19. Built together with tickets 03, 04 in one commit. Generator version `rs-gen-0.2`.
  - `robustsuite/families.py`: the five families and their structural start/goal rules.
  - `robustsuite/geometry.py`: exact segment clearance; grid paths on the frozen oracle's own occupancy and moves (the lengths equal `ShortestPathOracle.path_length`); the corridor wall complement.
  - `robustsuite/pedestrians.py`: routes and the motion model.
  - `robustsuite/scenario.py`: the generator, with the start→goal route in the spec.
- **Researcher decisions, recorded in RS_DESIGN §14.1 before any run:**
  - Start and goal clearance is 0.6 m. Closer than that, the frozen controller raises DCPError every step (h < 0) and never moves; at 0.45 m, 20–44% of starts did.
  - Randomized pedestrians: the turn-rate clip bounds the OU noise only; steering is at 4.0 /s towards 1 m ahead.
  - The other readings (exact leg step, corridor construction, aisle outer strips, turn rule) are listed there too.
- **Candidate idea for phase 3, not a decision:** run Random recovery on DCPError steps as well as infeasible ones. The diagnostic should count DCPError freezes per cell, at the start and mid-route.
- **Tests:**
  - routes: counts 5/3/4/3/4; starts ≥ 1.5 m from the robot start and ≥ 1.0 m from the goal; 8 waypoints walked in order; exact clearance ≥ 0.45 along every leg;
  - rollouts of 1500 steps in every family, both motions: fixed pedestrians keep ≥ 0.45 m and randomized ones ≥ 0.3 m; steps ≤ 0.0675 m; nobody parks;
  - fixed pedestrians draw nothing from the RNG;
  - pedestrians are identical under different robot actions, in every family and both motions (the pairing guarantee);
  - static cells have none; the M0 anchor is still bit-identical.
- **Bugs found and fixed:**
  - the canonical reset ran the previous episode's pedestrian model (an IndexError, and RNG use that depended on history). An empty model is now installed first;
  - randomized pedestrians deadlocked at walls until progress became the pedestrian's projection onto its route and steering left the clip (§14.1).
- **Measured** (3 draws × 5 families, 1500 steps): randomized pedestrians sit a median of about 5 cm from their route (95th percentile about 18 cm). Fallback steps take 0.5–0.9% of steps; holds are 0.
