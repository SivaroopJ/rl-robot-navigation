# Track A — A-P1 perception results: the pre-declared gate FAILED, large run not launched

**Status: Track A stopped at its own falsification gate, as the design note required.** No pursuit
batch was run with perception. The audited pursuit results are untouched.

Artifacts: `results/week6_tracka/A_P1/a_p1_perception.json`,
code `continuation/perception/hunter_detector.py`, runner
`experiments/week6_tracka/a_p1_perception.py`, dev seeds 13 000 000–13 000 002.

## What was implemented (VERIFIED FROM CODE)

- The hunter became a LiDAR-visible disc, in **our** `PursuitEnv` subclass only
  (`hunter_visible_to_lidar`, on only when `info_model="lidar_estimate"`). The known-state baseline
  is byte-unchanged: all 36 pursuit tests still pass.
- `HunterDetector`: a dedicated `LidarVelocityTracker` plus a pursuit-signature score
  (`velocity alignment toward the robot` × `speed / v_ref`), averaged over a window, with
  enter/exit hysteresis and stale-track handling.
- The detector imports nothing from `robot_env` and never receives the env; a test enforces it.

## Measured results (MEASURED IN AN EXPERIMENT, dev scenarios, association tolerance 0.6 m)

| scenario | detection rate | identification precision | false-positive rate | pos. error | max gap |
|---|---|---|---|---|---|
| S1 hunter stationary, visible | 0.00 | — | 0.00 | — | 35 |
| S2 hunter approaching, no obstacles | 0.71 | **1.00** | 0.00 | 0.001 m | 10 |
| S3 occluded by a rectangle, then emerging | 0.16 | **1.00** | 0.00 | 0.072 m | 46 |
| **S4 approaching, six moving obstacles** | 0.86 | **0.37** | **0.54** | 0.000 m | 3 |
| **S5 all moving (closest to the real loop)** | 0.87 | **0.33** | **0.58** | 0.052 m | 2 |

Parameter sweep on the dev scenarios (window ∈ {5, 10, 15, 20} × threshold ∈ {0.45, 0.6, 0.75} ×
min-consecutive ∈ {3, 5}, 24 configurations): precision never exceeded **0.52** in S4/S5, and the
score threshold had **no effect at all**.

## Why it fails (VERIFIED FROM CODE + MEASURED)

1. **The sensor is geometry-only and the bodies are identical.** `cast_rays` returns 24 bare
   ranges; the hunter and all six dynamic obstacles are discs of radius 0.3. Nothing in a scan
   distinguishes them.
2. **The motion cue is not discriminative in this arena.** The obstacles travel at 0.675 m/s under
   `SmoothStochasticMotion` and frequently head straight at the robot, giving alignment ≈ 1 — the
   same signature as a pursuer. Because a bouncing obstacle's alignment saturates the score, the
   threshold is inert, which is exactly what the sweep shows.
3. **Where it does work, it works perfectly** (S2/S3 precision 1.00): with no decoys, the pursuit
   signature is unambiguous. The failure is decoy confusion, not tracking quality — position error
   is ≤ 0.07 m whenever the right track is chosen.
4. S1 confirms the detector is conservative by design: a stationary hunter has no pursuit signature
   and is never claimed.

## Consequence

Running a perception-vs-oracle pursuit comparison now would compare the oracle against a protagonist
that is **wrong about which body is the hunter roughly 60 % of the time**. Any difference would
measure detector confusion, not perception-limited pursuit. Per the design note's escalation rule
and the prompt's instruction, the large batch was **not** launched.

## Recommended narrower experiment (HYPOTHESIS / RECOMMENDATION, not run)

**A-i′ — "unlabelled threat" pursuit.** Drop identification entirely. Make the hunter LiDAR-visible
and let the protagonist treat *every* confirmed track as an obstacle, which the frozen barrier
source already does. The comparison becomes:

- **oracle row** (audited baseline) vs **no hunter row, hunter merely visible**.

This measures exactly what the privileged information is worth — *knowing which disc is the
adversary* — needs no detector, has no identification error to confound it, and is implementable
with the code already written (`hunter_visible_to_lidar=True` plus `clear_hunter()`). It is the
honest version of the question A-i asked.

Secondary options, in order: (a) make the hunter physically distinguishable (a different radius),
which would make identification well-posed but changes the environment and so needs approval;
(b) A-ii, interception instead of pure pursuit, which needs no perception at all.

## Caveat recorded for any future A-i′ run

When the hunter is LiDAR-visible **and** an oracle row is also appended, the hunter occupies two
rows of the sample set. The DR constraint is a min over rows so the value is unchanged, but the
frozen `n_keep = 5` truncation means the duplicate can displace another obstacle row. Any future
comparison must state which row structure each arm uses.
