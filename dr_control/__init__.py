"""Phase 1 of the CLF-DR-CBF port: controller mathematics, no sensor.

Adapted from DR_Safe_Navigation (https://github.com/ExistentialRobotics/DR_Safe_Navigation),
the reference implementation of

    K. Long, Y. Yi, Z. Dai, S. Herbert, J. Cortes, N. Atanasov,
    "Sensor-Based Distributionally Robust Control for Safe Robot Navigation
     in Dynamic Environments", arXiv:2405.18251.

Every deliberate divergence from the paper or from the reference implementation is marked
inline with one of three tags:

    [adaptation]                        forced by this environment (single integrator, 10 Hz)
    [reference implementation correction]  a bug or portability issue in DR_Safe_Navigation
    [extension]                         beyond the paper

BARRIER CONVENTION USED THROUGHOUT THIS PACKAGE  [adaptation]
-------------------------------------------------------------
The reference carries a RAW SENSOR DISTANCE `d` in `h_samples` and subtracts the robot radius
inside the controller, so its barrier term reads `alpha * (d - r_robot)`.

Here `h` is ALWAYS the robot-body clearance, with the radius already folded in at the source:

    circle     h = ||p - o|| - r_obs - r_robot
    rectangle  h = (closest-point distance) - r_robot
    LiDAR      h = range - r_robot        (Phase 2; not used in Phase 1)

so the barrier term is `alpha * h` and the radius is NEVER subtracted twice:

    CBC = grad_h . u + alpha * h + dh_dt

This is a change of where the radius is applied, not a change to the method: `d - r_robot`
and `h` are the same number. The DRCCP reformulation structure is untouched.
"""
