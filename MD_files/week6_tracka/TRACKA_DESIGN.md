# Track A — pursuit–evasion: literature memo, sensor audit, and design note

Independent of Track B. Builds on the audited pursuit baseline
(`MD_files/pursuit/PURSUIT_RESULTS_v2.md`), which is **not** rerun or modified.

---

## A-P0.1 Sensor and code audit (VERIFIED FROM CODE)

| question | finding |
|---|---|
| How are LiDAR rays generated? | `robot_env/lidar_core.cast_rays(origin, n_rays, lidar_range, world_size, agent_radius, rects, circles, circle_radius)`; 24 bearings evenly spaced from +x, **geometry only**: one float distance per ray. No identity, no semantics, no return intensity. |
| Is the hunter visible to the protagonist's LiDAR? | **No.** `RobotNavEnv._cast_lidar_rays` passes `circles=self.obstacle_positions`, which excludes the hunter. In the audited baseline the hunter reaches the protagonist **only** through the oracle row in `HunterAugmentedSource`. |
| Occlusion behaviour | Handled implicitly and correctly: each ray takes the **nearest** hit among walls, rectangles and circles, so a nearer body hides a farther one. There is no see-through. |
| Existing tracking utilities | `LidarVelocityTracker`: segments the scan, rejects non-compact (wall-like) segments, reconstructs disc centres assuming radius `r_nominal = 0.3`, associates by gated nearest-neighbour and runs a constant-velocity Kalman filter per track. Exposes `.tracks` with `id`, `position`, `velocity`, `confirmed`, `age`, `hits`, `misses`, and coasts a track for `n_coast = 5` missed frames. |
| How does hunter state enter the barrier today? | `HunterAugmentedSource.push_hunter(pos, vel)` → one appended row `h = ‖p−p_h‖ − 0.6`, `grad_h` unit, `dh_dt = −grad_h·v_h`. Privileged. |
| Hunter crash/disablement | `PursuitEnv`: on wall/rect/obstacle contact the hunter is frozen in place, `hunter_disabled = True`, cannot capture; episode continues; crash type and step are recorded. |
| Reusable without touching frozen code | `cast_rays`, `LidarVelocityTracker`, `EstimatedLidarBarrierSource` subclassing, controller wrapping by assignment. `PursuitEnv` is ours, so making the hunter LiDAR-visible needs **no frozen edit**. |

### The decisive finding

**The hunter and the dynamic obstacles are geometrically identical to this sensor**: both are discs
of radius 0.3, and the scan returns only ranges. Therefore a LiDAR-perception experiment **cannot**
identify the hunter from geometry alone, and using the environment's object ordering would be
hidden ground truth. Any honest detector must separate the hunter from six decoys **by motion
signature**: the hunter is the only disc whose velocity persistently points at the protagonist.
This is the scientific content of Track A, and it is measurable (A-P1) before any pursuit claim.

---

## A-P0.2 Literature memo

Sources consulted (searched 2026-09-12). Each line states the specific idea relevant to us.

| source | relevance |
|---|---|
| Zhou, Shaikh, Chaubey, Haggerty, Koga, Panagou, Atanasov, *Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility and Safety Barrier Functions*, arXiv:2411.01321 (Nov 2024, rev. Mar 2025) | Pursuit under **occlusion** with two CBFs: a *visibility* CBF from the signed distance of the field of view, and a *safety* CBF from obstacle SDFs. Confirms that "keep the target in view" can be posed as a barrier constraint alongside safety — directly analogous to adding rows to our QP. LITERATURE-SUPPORTED. |
| *Active Adversarial Evader Tracking with a Probabilistic Pursuer under the Pursuit-Evasion Game Framework*, arXiv:1904.09307 | Pursuer with a **limited-visibility depth sensor** actively tracking an evader among obstacles; motivates treating detection/tracking as part of the control loop rather than assuming state. |
| *Mobile Target Search with Imperfect Perception: A Partially Observable Stochastic Game Theoretical Approach*, arXiv:2606.20232 | Practical sensing models with **false alarms and missed detections**; supports our requirement to measure false-positive identification, not only detection rate. |
| Wabersich & Zeilinger, *Predictive control barrier functions*, arXiv:2105.10241 (2021/22) | (Cited for context across tracks.) Safety-filter **infeasibility** is a known failure mode addressed by constraint tightening plus a terminal CBF. |

**Novelty check.** Visibility-aware pursuit, LiDAR target tracking and CBF safety filtering all
exist in the literature. What is *not* claimed as novel here: the tracker, the CBF, the pursuit law.
What is specific to this project: measuring how a **frozen, previously audited stochastic-recovery
navigation controller** degrades when its adversary's state comes from its own geometry-only sensor
rather than an oracle, with six same-radius decoys present. HYPOTHESIS/RECOMMENDATION, not a novelty
claim.

---

## A-P0.3 Candidate avenues

### A-i (RECOMMENDED) — protagonist perceives the hunter by LiDAR
- **Question:** does the audited advantage of Random-CLF-DR-CBF survive when the hunter's state is
  estimated from the protagonist's own 24-ray LiDAR instead of given?
- **Changes:** hunter becomes a LiDAR-visible disc in `PursuitEnv`; a new detector picks the
  hunter track by motion signature; `HunterAugmentedSource` is fed the *estimate* instead of truth.
  Frozen controller untouched.
- **Known vs estimated:** protagonist estimates hunter position/velocity; hunter keeps true
  protagonist position (staged, per the plan).
- **Benefit:** removes the single biggest limitation of the audited result.
- **Failure modes:** misidentification among six same-radius decoys; track loss behind rectangles;
  stale coasting; late detection at range.
- **Effort:** moderate, all in new files. Runtime: one extra tracker pass, negligible.
- **Falsifying diagnostic (A-P1):** if identification precision/recall is poor even with the hunter
  visible and unoccluded, the pursuit claim cannot be supported and we stop and report.
- **Escalation rule:** proceed to a paired evaluation only if A-P1 shows usable identification.

### A-ii — interception instead of pure pursuit
- **Question:** is the low equal-speed capture rate a property of pursuit–evasion or of the *pure
  pursuit* law? A constant-velocity intercept point would test it.
- **Changes:** hunter nominal action only. Cheap.
- **Why not first:** it improves the *hunter*, leaving the privileged-information limitation of the
  protagonist untouched, and changes the adversary rather than answering a perception question.

### A-iii — visibility-aware evasion (occlusion exploitation)
- **Question:** can the evader exploit rectangles to break line of sight?
- **Why not first:** requires a visibility term in the protagonist's objective, i.e. a change to the
  frozen controller's objective or an extra planner layer — the largest scope increase of the three,
  and it presupposes the perception layer of A-i to be meaningful.

**Recommendation: A-i first**, because it converts the audited result's main limitation into a
measured quantity, needs no frozen change, and has a clean falsification test before any large run.

---

## A-P0.4 Design of the A-i implementation

**Detector (`continuation/perception/hunter_detector.py`, new).** Inputs per step: the protagonist's
own LiDAR ranges and ego pose — nothing else.
1. Feed the scan to a dedicated `LidarVelocityTracker` instance (the frozen class, separate from the
   barrier source's own tracker so no frozen behaviour changes).
2. For each confirmed track compute a **pursuit score** over a sliding window:
   `s = mean_t [ v̂_track · û_(ego − track) ]` — the cosine alignment of the track's velocity with
   the direction from the track to the robot, i.e. "is it steering at me", averaged over the window,
   combined with the closing-speed magnitude.
3. The hunter estimate is the highest-scoring track whose score exceeds a threshold held for a
   minimum number of consecutive frames (hysteresis).
4. **Not detected / below threshold / stale:** no hunter row is appended at all, so the protagonist
   behaves exactly as the frozen navigation controller. A coasting track (tracker `misses > 0`) is
   used for up to `n_coast` frames and flagged as stale.
5. Everything logged: track id, score, age, misses, whether the estimate is fresh or coasted, and
   the estimate's error against ground truth **for offline evaluation only**.

**Leak prevention.** The detector module imports nothing from `robot_env` and never receives the
env; a test parses its AST to enforce this, mirroring the existing information-ledger tests.

**Metrics (A-P1)** — measured against ground truth offline: identification precision/recall,
detection rate vs range and occlusion, position and velocity error, track-loss duration,
reacquisition time, and the false-positive rate (a dynamic obstacle labelled as the hunter).
