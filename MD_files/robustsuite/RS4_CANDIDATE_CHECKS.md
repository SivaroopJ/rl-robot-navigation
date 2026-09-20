# RS Experiment 1: candidate checks (phase 4)

Tickets 11a-11c. Produced by `experiments/robustsuite/rs4_candidate_checks.py`. No tuning run: the phantom counts are shadows over replayed baseline episodes, and the step times are one seed per cell.

- Commit: `31437c1e1f9cb2238787c00e346e52b7b017c8a1`  
- Manifest: `3f4d5938a060ea0a...`  
- Generator: `rs-gen-0.4`  
- Seeds: RS_DIAG[0:20] (phantoms), RS_DIAG[0] (step times)

## 1. Phantom new cells and phantom yields

Cells in which no new static obstacle ever appears, so every new cell is a phantom and every detour trigger (a) it would fire is needless. `static` has no pedestrian at all, so its yields come entirely from tracks the frozen tracker builds out of static geometry (reading 14). A detour counted below is one the layer would have started on the baseline's own trajectory; where a row shows no phantom cell at all, trigger (a) cannot have fired, so every one of them is the stall trigger (b).

| condition | episodes | eps with a phantom cell (K=8) | mean cells (K=8) | eps with a detour (K=8) | eps with a phantom cell (K=14) | mean cells (K=14) | eps with a detour (K=14) | yield steps / episode | yield rate | yields to a pedestrian |
|---|---|---|---|---|---|---|---|---|---|---|
| static | 200 | 0 / 200 | 0.00 | 8 / 200 | 0 / 200 | 0.00 | 8 / 200 | 38.6 | 0.345 | 0 / 7726 |
| dynamic | 200 | 11 / 200 | 0.11 | 36 / 200 | 0 / 200 | 0.00 | 35 / 200 | 67.4 | 0.486 | 5883 / 13473 |
| trigger_spawn | 200 | 17 / 200 | 0.16 | 54 / 200 | 0 / 200 | 0.00 | 52 / 200 | 77.6 | 0.522 | 8142 / 15512 |
| anchor | 40 | 3 / 40 | 0.07 | 5 / 40 | 0 / 40 | 0.00 | 4 / 40 | 66.8 | 0.643 | 1275 / 2671 |

## 2. Step time (RS_DESIGN 7.2: p95 must be at most 100 ms)

| arm | episodes | control steps | mean (ms) | p50 (ms) | p95 (ms) | max (ms) | 7.2 |
|---|---|---|---|---|---|---|---|
| `astar_random` | 42 | 9382 | 38.2 | 37.0 | 46.1 | 119.5 | pass |
| `detour` | 42 | 8830 | 39.2 | 38.2 | 46.5 | 242.5 | pass |
| `detour_yield` | 42 | 12857 | 39.3 | 38.3 | 46.4 | 91.8 | pass |
| `yield` | 42 | 12470 | 39.0 | 38.0 | 46.6 | 122.5 | pass |

Default configurations: `detour` K14_L1.0_stall_on, `yield` H3_D1.2_L1.0, `detour_yield` K14_L1.0_stall_on+H3_D1.2_L1.0.

## 3. What the numbers say

- **Pedestrians leave no phantom cells at K = 14**: 0 of 640 replayed episodes end with one. At K = 8, 31 do, up to 7 cells in one episode. The default K = 14 is on the safe side of that.
- **The stall trigger is what fires here**: with no new cell anywhere at K = 14, trigger (a) cannot fire, so all 99 detours of 640 episodes are stalls -- the robot standing still for 3 s while a pedestrian passes. `stall` is a grid switch, so the 7.3 search decides whether that helps.
- **yield is active, and often on nothing**: it yields on 34% of steps in the `static` cells, which hold no pedestrian at all. Over every cell 24082 of 39382 yields are to a track farther than 0.5 m from any pedestrian, i.e. to the tracker's own wall artefacts (reading 14).
- **Step time**: every arm is inside the 7.2 limit on this sample. The rule is applied for real on RS_TUNE in ticket 12.
