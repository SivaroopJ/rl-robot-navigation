# Trajectory examples — evidence that the randomization is smooth

Figure: `results/week4/trajectory_examples/obstacle_trajectories.png`

## What the figure shows

Four columns. The first is one obstacle under the **original** deterministic motion; the
remaining three are the **randomized** model under three different seeds. The top row is the
path through the arena; the bottom row is the angular velocity `omega` over the same 400
steps, with the `±omega_max` bound drawn, and each panel titled with the measured maximum
per-step heading change against its theoretical bound.

The `omega` trace is on the same figure deliberately. A path plot alone cannot establish
smoothness — a curve that *looks* smooth could have been produced by a bounded but
uncorrelated process that merely takes tiny independent steps and wanders nowhere. The
distinguishing evidence is that `omega` is itself continuous and correlated, which is what
the lower row shows.

The original column has no `omega` trace because it has no angular velocity at all: its
heading is constant between walls and flips discontinuously at each reflection. That
contrast is the point of the figure.

## Measured properties

Over 380,000 obstacle-steps, 25 seeds × {3, 6, 10} obstacles × 800 steps, with obstacles
spawned **flush against the walls** (the worst case for boundary handling — an obstacle
starting inside its own turning radius of a wall):

| property | measured | requirement |
|---|---|---|
| max per-step heading change | **0.0700 rad (4.01°)** | `omega_max * dt = 0.0700` |
| `\|omega\|` bound | never exceeded | `<= omega_max = 0.7` |
| speed | constant to 1e-6 | held per episode |
| position validity | always inside the arena after every update | required |
| wall contact | **0.74%** of obstacle-steps, projection depth ≤ one step of travel | graze, not bounce |
| `omega` lag-1 autocorrelation | **0.98** | ≫ 0, i.e. not white noise |
| arena coverage | **0.86** of grid cells visited | comparable to the original billiard motion |

Asserted by `tests/test_week4_env.py`:
`test_angular_velocity_is_bounded`, `test_no_sudden_direction_changes`,
`test_obstacle_speed_stays_in_range`, `test_obstacles_stay_inside_the_valid_region`,
`test_wall_contact_is_a_graze_and_not_a_bounce`,
`test_angular_velocity_is_temporally_correlated`.

## Reproducibility

`test_same_seed_reproduces_the_whole_episode` asserts that a given seed reproduces the
start, goal, obstacle initial states and the entire trajectory;
`test_different_seeds_give_different_trajectories` asserts that different seeds diverge.
Both matter beyond ordinary reproducibility: Sibling Rivalry's pairing depends on two
sibling environments being byte-identical, and if that ever broke, `rho` would silently
begin measuring environment variance rather than policy variance.

## A note on the boundary, and a correction

The first implementation used a steer proportional to heading error across the whole
margin. It acted far too late — near zero at margin entry — so the *effective* margin was
much smaller than the configured one. Compensating by enlarging the margin to 3.0 left only
a 4-unit interior in a 10-unit arena, and obstacles then spent their lives turning: they
looped tightly and covered only **55%** of the arena against essentially 100% for the
original motion. That would have been a far larger behavioural change than the
"deterministic trajectory → stochastic trajectory" the brief sanctions.

The steer now saturates on both axes — beyond 45° of heading error, and 35% of the way into
the margin — which permits a smaller margin (2.2, still above the 1.43-unit turning radius)
and restores traversal to 0.86 coverage.

Separately, the *assertion* was corrected. Demanding that the position projection never fire
was optimising a proxy. The brief prohibits instant velocity reversal and sudden trajectory
discontinuity; the projection is per-axis, so an obstacle reaching a wall keeps its
tangential motion and **slides**, with its heading untouched. It is a graze, not a bounce.
What the tests now assert is that contact is rare, shallow (at most one step of travel), and
— via `test_no_sudden_direction_changes` over the same traces — provably introduces no
heading discontinuity.

## Start/goal coverage

Figures: `results/week4/start_goal_coverage/region_coverage.png` and
`distance_distribution.png`; numbers in `coverage_summary.json`.

These support the separate claim that the stratified sampler covers the map and spans the
distance range, plotted against the original uniform scheme because the claim is
comparative. Measured over 6000 draws: mean separation 5.21 → **6.87**, minimum 2.00 →
**4.00**, fraction under 4.0 units **31.7% → 0%**, band shares 25.7 / 25.4 / 24.4 / 24.6%,
fallback rate **0.00%**, and every one of the 15 usable regions drawn as both start and goal
at above half its uniform share.

```bash
python -m visualization.week4_plots --what coverage obstacles
```
