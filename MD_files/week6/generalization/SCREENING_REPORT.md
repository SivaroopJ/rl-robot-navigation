# Week 6 / Generalization Suite — SCREENING REPORT

**118 of 118 screening cells complete.** 50 paired episodes per cell, both arms, seeds `SCREEN_SEED_BASE = 8_000_000 … +49`.

**Screening is descriptive.** It characterises maps; it selects nothing. No number here may be used to tune, reorder or reject any part of the preregistered design. At n = 50 a paired test resolves only ~0.15 absolute, so cell-level p-values are reported for information and are almost never conclusive.

**X1–X7 are feasibility characterisation** and are excluded from every benchmark aggregate below. No success rate is reported for them.

## 1. Results by family (benchmark families only)

Episode-weighted means over the cells of each family. Δ = B − A.

| family | cells | A succ | B succ | Δ succ | A coll | B coll | Δ coll | A inf | B inf | B recov |
|---|---|---|---|---|---|---|---|---|---|---|
| F1' | 16 | 0.775 | 0.886 | +0.111 | 0.211 | 0.099 | -0.112 | 0.0368 | 0.0215 | 2223 |
| F2a | 8 | 0.818 | 0.910 | +0.092 | 0.175 | 0.083 | -0.092 | 0.0356 | 0.0196 | 945 |
| F2b | 8 | 0.745 | 0.815 | +0.070 | 0.242 | 0.163 | -0.080 | 0.0338 | 0.0205 | 1066 |
| F3 | 6 | 0.757 | 0.847 | +0.090 | 0.213 | 0.120 | -0.093 | 0.0338 | 0.0197 | 572 |
| F4 | 8 | 0.760 | 0.860 | +0.100 | 0.240 | 0.133 | -0.108 | 0.0428 | 0.0255 | 1205 |
| F5 | 8 | 0.603 | 0.767 | +0.165 | 0.392 | 0.230 | -0.162 | 0.0464 | 0.0353 | 1857 |
| F6 | 10 | 0.864 | 0.936 | +0.072 | 0.132 | 0.060 | -0.072 | 0.0359 | 0.0144 | 868 |
| F8 | 10 | 0.800 | 0.902 | +0.102 | 0.200 | 0.098 | -0.102 | 0.0451 | 0.0199 | 1324 |
| F9 | 10 | 0.742 | 0.866 | +0.124 | 0.258 | 0.134 | -0.124 | 0.0230 | 0.0191 | 1597 |
| F10 | 10 | 0.328 | 0.384 | +0.056 | 0.604 | 0.522 | -0.082 | 0.0378 | 0.0307 | 2771 |
| F12 | 6 | 0.600 | 0.643 | +0.043 | 0.380 | 0.333 | -0.047 | 0.0327 | 0.0158 | 557 |
| F7 | 8 | 0.378 | 0.548 | +0.170 | 0.623 | 0.453 | -0.170 | 0.0518 | 0.0503 | 2524 |

## 2. Results by configuration

| id | model | A succ | B succ | Δ | A coll | B coll | Δ | p(coll) | class |
|---|---|---|---|---|---|---|---|---|---|
| F1_rot15_obs0 | deter | 0.76 | 0.88 | +0.12 | 0.24 | 0.12 | -0.12 | 0.146 | interface |
| F1_rot15_obs0 | stoch | 0.84 | 0.92 | +0.08 | 0.16 | 0.08 | -0.08 | 0.289 | interface |
| F1_rot30_all | deter | 0.86 | 0.98 | +0.12 | 0.14 | 0.02 | -0.12 | **0.031** | interface |
| F1_rot30_all | stoch | 0.74 | 0.94 | +0.20 | 0.26 | 0.06 | -0.20 | **0.002** | interface |
| F1_rot30_obs0 | deter | 0.78 | 0.90 | +0.12 | 0.22 | 0.10 | -0.12 | 0.146 | interface |
| F1_rot30_obs0 | stoch | 0.84 | 0.92 | +0.08 | 0.16 | 0.08 | -0.08 | 0.219 | interface |
| F1_rot45_all | deter | 0.86 | 0.92 | +0.06 | 0.14 | 0.08 | -0.06 | 0.453 | interface |
| F1_rot45_all | stoch | 0.78 | 0.90 | +0.12 | 0.22 | 0.08 | -0.14 | 0.065 | interface |
| F1_rot45_obs0 | deter | 0.76 | 0.90 | +0.14 | 0.24 | 0.10 | -0.14 | 0.065 | interface |
| F1_rot45_obs0 | stoch | 0.84 | 0.96 | +0.12 | 0.16 | 0.04 | -0.12 | 0.070 | interface |
| F1_rot60_obs0 | deter | 0.76 | 0.88 | +0.12 | 0.24 | 0.12 | -0.12 | 0.109 | interface |
| F1_rot60_obs0 | stoch | 0.82 | 0.96 | +0.14 | 0.18 | 0.04 | -0.14 | **0.039** | interface |
| F1_rot90_all | deter | 0.60 | 0.64 | +0.04 | 0.30 | 0.28 | -0.02 | 1.000 | interface |
| F1_rot90_all | stoch | 0.50 | 0.64 | +0.14 | 0.38 | 0.22 | -0.16 | **0.008** | interface |
| F1_rot90_obs0 | deter | 0.78 | 0.90 | +0.12 | 0.22 | 0.10 | -0.12 | 0.109 | interface |
| F1_rot90_obs0 | stoch | 0.88 | 0.94 | +0.06 | 0.12 | 0.06 | -0.06 | 0.250 | interface |
| F2a_scale0.5 | deter | 0.92 | 0.98 | +0.06 | 0.08 | 0.02 | -0.06 | 0.250 | supported |
| F2a_scale0.5 | stoch | 0.90 | 0.94 | +0.04 | 0.10 | 0.06 | -0.04 | 0.625 | supported |
| F2a_scale0.75 | deter | 0.76 | 0.90 | +0.14 | 0.22 | 0.08 | -0.14 | **0.016** | supported |
| F2a_scale0.75 | stoch | 0.74 | 0.90 | +0.16 | 0.22 | 0.06 | -0.16 | **0.008** | supported |
| F2a_scale1.25 | deter | 0.72 | 0.82 | +0.10 | 0.28 | 0.18 | -0.10 | 0.062 | supported |
| F2a_scale1.25 | stoch | 0.74 | 0.90 | +0.16 | 0.26 | 0.10 | -0.16 | **0.008** | supported |
| F2a_scale1.5 | deter | 0.88 | 0.90 | +0.02 | 0.12 | 0.10 | -0.02 | 1.000 | supported |
| F2a_scale1.5 | stoch | 0.88 | 0.94 | +0.06 | 0.12 | 0.06 | -0.06 | 0.375 | supported |
| F2b_n2 | deter | 0.92 | 0.92 | +0.00 | 0.08 | 0.06 | -0.02 | 1.000 | supported |
| F2b_n2 | stoch | 0.86 | 0.98 | +0.12 | 0.14 | 0.02 | -0.12 | 0.070 | supported |
| F2b_n3 | deter | 0.90 | 0.92 | +0.02 | 0.10 | 0.08 | -0.02 | 1.000 | supported |
| F2b_n3 | stoch | 0.80 | 0.90 | +0.10 | 0.20 | 0.10 | -0.10 | 0.180 | supported |
| F2b_n7 | deter | 0.74 | 0.88 | +0.14 | 0.26 | 0.12 | -0.14 | **0.016** | supported |
| F2b_n7 | stoch | 0.64 | 0.84 | +0.20 | 0.36 | 0.16 | -0.20 | **0.006** | supported |
| F2b_n9 | deter | 0.54 | 0.54 | +0.00 | 0.40 | 0.36 | -0.04 | 0.688 | supported |
| F2b_n9 | stoch | 0.56 | 0.54 | -0.02 | 0.40 | 0.40 | +0.00 | 1.000 | supported |
| F3_world15 | deter | 0.86 | 0.88 | +0.02 | 0.10 | 0.08 | -0.02 | 1.000 | supported |
| F3_world15 | stoch | 0.76 | 0.80 | +0.04 | 0.16 | 0.10 | -0.06 | 0.453 | supported |
| F3_world20 | deter | 0.90 | 0.92 | +0.02 | 0.10 | 0.08 | -0.02 | 1.000 | supported |
| F3_world20 | stoch | 0.88 | 0.94 | +0.06 | 0.06 | 0.00 | -0.06 | 0.250 | supported |
| F3_world7 | deter | 0.62 | 0.82 | +0.20 | 0.38 | 0.18 | -0.20 | **0.013** | supported |
| F3_world7 | stoch | 0.52 | 0.72 | +0.20 | 0.48 | 0.28 | -0.20 | **0.002** | supported |
| F4_r0.15 | deter | 0.94 | 0.94 | +0.00 | 0.06 | 0.06 | +0.00 | 1.000 | supported |
| F4_r0.15 | stoch | 0.76 | 0.92 | +0.16 | 0.24 | 0.04 | -0.20 | **0.006** | supported |
| F4_r0.2 | deter | 0.86 | 0.92 | +0.06 | 0.14 | 0.08 | -0.06 | 0.375 | supported |
| F4_r0.2 | stoch | 0.78 | 0.94 | +0.16 | 0.22 | 0.04 | -0.18 | **0.012** | supported |
| F4_r0.45 | deter | 0.84 | 0.86 | +0.02 | 0.16 | 0.14 | -0.02 | 1.000 | interface |
| F4_r0.45 | stoch | 0.72 | 0.76 | +0.04 | 0.28 | 0.24 | -0.04 | 0.688 | interface |
| F4_r0.6 | deter | 0.66 | 0.86 | +0.20 | 0.34 | 0.14 | -0.20 | **0.013** | interface |
| F4_r0.6 | stoch | 0.52 | 0.68 | +0.16 | 0.48 | 0.32 | -0.16 | **0.008** | interface |
| F5_n10 | deter | 0.38 | 0.62 | +0.24 | 0.62 | 0.38 | -0.24 | **0.002** | supported |
| F5_n10 | stoch | 0.22 | 0.40 | +0.18 | 0.78 | 0.60 | -0.18 | **0.004** | supported |
| F5_n2 | deter | 0.96 | 1.00 | +0.04 | 0.02 | 0.00 | -0.02 | 1.000 | supported |
| F5_n2 | stoch | 0.84 | 0.94 | +0.10 | 0.16 | 0.06 | -0.10 | 0.062 | supported |
| F5_n4 | deter | 0.82 | 0.94 | +0.12 | 0.18 | 0.06 | -0.12 | 0.109 | supported |
| F5_n4 | stoch | 0.64 | 0.84 | +0.20 | 0.34 | 0.14 | -0.20 | **0.006** | supported |
| F5_n8 | deter | 0.58 | 0.72 | +0.14 | 0.42 | 0.28 | -0.14 | **0.039** | supported |
| F5_n8 | stoch | 0.38 | 0.68 | +0.30 | 0.62 | 0.32 | -0.30 | **0.000** | supported |
| F6_v0.3 | deter | 0.98 | 0.98 | +0.00 | 0.00 | 0.00 | +0.00 | 1.000 | supported |
| F6_v0.3 | stoch | 0.98 | 1.00 | +0.02 | 0.02 | 0.00 | -0.02 | 1.000 | supported |
| F6_v0.45 | deter | 0.98 | 0.98 | +0.00 | 0.00 | 0.00 | +0.00 | 1.000 | supported |
| F6_v0.45 | stoch | 0.96 | 0.98 | +0.02 | 0.04 | 0.02 | -0.02 | 1.000 | supported |
| F6_v0.85 | deter | 0.90 | 0.94 | +0.04 | 0.10 | 0.06 | -0.04 | 0.500 | supported |
| F6_v0.85 | stoch | 0.80 | 0.98 | +0.18 | 0.20 | 0.02 | -0.18 | **0.004** | supported |
| F6_v0.96 | deter | 0.92 | 0.90 | -0.02 | 0.08 | 0.10 | +0.02 | 1.000 | supported |
| F6_v0.96 | stoch | 0.80 | 0.94 | +0.14 | 0.20 | 0.06 | -0.14 | **0.039** | supported |
| F6_v1.1 | deter | 0.70 | 0.86 | +0.16 | 0.30 | 0.14 | -0.16 | **0.021** | interface |
| F6_v1.1 | stoch | 0.62 | 0.80 | +0.18 | 0.38 | 0.20 | -0.18 | **0.012** | interface |
| F8a_open_goal | deter | 0.92 | 0.96 | +0.04 | 0.08 | 0.04 | -0.04 | 0.625 | supported |
| F8a_open_goal | stoch | 0.80 | 0.96 | +0.16 | 0.20 | 0.04 | -0.16 | **0.021** | supported |
| F8b_goal_behind_rect | deter | 0.74 | 0.90 | +0.16 | 0.26 | 0.10 | -0.16 | **0.039** | supported |
| F8b_goal_behind_rect | stoch | 0.70 | 0.92 | +0.22 | 0.30 | 0.08 | -0.22 | **0.003** | supported |
| F8c_goal_confined_slot | deter | 0.84 | 0.92 | +0.08 | 0.16 | 0.08 | -0.08 | 0.125 | supported |
| F8c_goal_confined_slot | stoch | 0.72 | 0.82 | +0.10 | 0.28 | 0.18 | -0.10 | 0.125 | supported |
| F8d_goal_in_traffic | deter | 0.90 | 0.94 | +0.04 | 0.10 | 0.06 | -0.04 | 0.500 | supported |
| F8d_goal_in_traffic | stoch | 0.96 | 1.00 | +0.04 | 0.04 | 0.00 | -0.04 | 0.500 | supported |
| F8e_long_diagonal | deter | 0.76 | 0.80 | +0.04 | 0.24 | 0.20 | -0.04 | 0.625 | supported |
| F8e_long_diagonal | stoch | 0.66 | 0.80 | +0.14 | 0.34 | 0.20 | -0.14 | 0.065 | supported |
| F9_w0.75 | deter | 0.88 | 0.86 | -0.02 | 0.12 | 0.14 | +0.02 | 1.000 | supported |
| F9_w0.75 | stoch | 0.36 | 0.48 | +0.12 | 0.64 | 0.52 | -0.12 | 0.146 | supported |
| F9_w0.85 | deter | 0.88 | 0.94 | +0.06 | 0.12 | 0.06 | -0.06 | 0.375 | supported |
| F9_w0.85 | stoch | 0.46 | 0.74 | +0.28 | 0.54 | 0.26 | -0.28 | **0.001** | supported |
| F9_w1.0 | deter | 0.88 | 0.94 | +0.06 | 0.12 | 0.06 | -0.06 | 0.250 | supported |
| F9_w1.0 | stoch | 0.58 | 0.92 | +0.34 | 0.42 | 0.08 | -0.34 | **0.000** | supported |
| F9_w1.2 | deter | 0.92 | 0.98 | +0.06 | 0.08 | 0.02 | -0.06 | 0.250 | supported |
| F9_w1.2 | stoch | 0.76 | 0.90 | +0.14 | 0.24 | 0.10 | -0.14 | 0.092 | supported |
| F9_w1.6 | deter | 0.94 | 0.98 | +0.04 | 0.06 | 0.02 | -0.04 | 0.625 | supported |
| F9_w1.6 | stoch | 0.76 | 0.92 | +0.16 | 0.24 | 0.08 | -0.16 | **0.008** | supported |
| F10a_behind | deter | 0.84 | 0.92 | +0.08 | 0.16 | 0.08 | -0.08 | 0.289 | interface |
| F10a_behind | stoch | 0.76 | 0.86 | +0.10 | 0.24 | 0.14 | -0.10 | 0.180 | interface |
| F10b_beside | deter | 0.84 | 0.94 | +0.10 | 0.16 | 0.06 | -0.10 | 0.062 | interface |
| F10b_beside | stoch | 0.70 | 0.82 | +0.12 | 0.30 | 0.18 | -0.12 | 0.109 | interface |
| F10c_blocks_route | deter | 0.00 | 0.06 | +0.06 | 0.80 | 0.64 | -0.16 | **0.008** | interface |
| F10c_blocks_route | stoch | 0.04 | 0.04 | +0.00 | 0.94 | 0.90 | -0.04 | 0.500 | interface |
| F10d_at_waypoint | deter | 0.02 | 0.06 | +0.04 | 0.82 | 0.72 | -0.10 | 0.062 | interface |
| F10d_at_waypoint | stoch | 0.06 | 0.12 | +0.06 | 0.94 | 0.88 | -0.06 | 0.250 | interface |
| F10e_before_arrival | deter | 0.00 | 0.02 | +0.02 | 0.76 | 0.74 | -0.02 | 1.000 | interface |
| F10e_before_arrival | stoch | 0.02 | 0.00 | -0.02 | 0.92 | 0.88 | -0.04 | 0.625 | interface |
| F12a_crossing | scrip | 0.66 | 0.66 | +0.00 | 0.32 | 0.32 | +0.00 | 1.000 | supported |
| F12b_head_on | scrip | 0.86 | 0.88 | +0.02 | 0.10 | 0.08 | -0.02 | 1.000 | supported |
| F12c_parallel | scrip | 0.98 | 1.00 | +0.02 | 0.02 | 0.00 | -0.02 | 1.000 | supported |
| F12d_converging | scrip | 0.36 | 0.46 | +0.10 | 0.64 | 0.54 | -0.10 | 0.062 | supported |
| F12e_circling_goal | scrip | 0.22 | 0.44 | +0.22 | 0.78 | 0.56 | -0.22 | **0.003** | supported |
| F12f_corridor_traffic | scrip | 0.52 | 0.42 | -0.10 | 0.42 | 0.50 | +0.08 | 0.219 | supported |
| F7a_r45_v96 | deter | 0.82 | 0.84 | +0.02 | 0.18 | 0.16 | -0.02 | 1.000 | interface |
| F7a_r45_v96 | stoch | 0.66 | 0.80 | +0.14 | 0.34 | 0.20 | -0.14 | 0.065 | interface |
| F7b_n10_v96 | deter | 0.38 | 0.68 | +0.30 | 0.62 | 0.32 | -0.30 | **0.001** | supported |
| F7b_n10_v96 | stoch | 0.22 | 0.46 | +0.24 | 0.78 | 0.54 | -0.24 | **0.000** | supported |
| F7c_r45_n10 | deter | 0.26 | 0.50 | +0.24 | 0.74 | 0.50 | -0.24 | **0.000** | interface |
| F7c_r45_n10 | stoch | 0.20 | 0.30 | +0.10 | 0.80 | 0.70 | -0.10 | 0.180 | interface |
| F7d_r45_n10_v96 | deter | 0.30 | 0.52 | +0.22 | 0.70 | 0.48 | -0.22 | **0.003** | interface |
| F7d_r45_n10_v96 | stoch | 0.18 | 0.28 | +0.10 | 0.82 | 0.72 | -0.10 | 0.125 | interface |

## 3. Collision split, timeout and infeasibility

| family | A dyn/stat/wall | B dyn/stat/wall | A timeout | B timeout | A clr | B clr | A worst | B worst |
|---|---|---|---|---|---|---|---|---|
| F1' | 169/0/0 | 79/0/0 | 0.014 | 0.015 | 0.245 | 0.281 | -0.0988 | -0.0813 |
| F2a | 70/0/0 | 33/0/0 | 0.007 | 0.007 | 0.265 | 0.299 | -0.0872 | -0.0826 |
| F2b | 95/2/0 | 61/4/0 | 0.013 | 0.022 | 0.252 | 0.280 | -0.0948 | -0.1322 |
| F3 | 64/0/0 | 34/2/0 | 0.030 | 0.033 | 0.278 | 0.315 | -0.0901 | -0.0874 |
| F4 | 96/0/0 | 53/0/0 | 0.000 | 0.007 | 0.242 | 0.282 | -0.1288 | -0.1362 |
| F5 | 157/0/0 | 90/2/0 | 0.005 | 0.003 | 0.165 | 0.211 | -0.0975 | -0.1234 |
| F6 | 66/0/0 | 30/0/0 | 0.004 | 0.004 | 0.294 | 0.321 | -0.1020 | -0.0580 |
| F8 | 100/0/0 | 48/1/0 | 0.000 | 0.000 | 0.227 | 0.264 | -0.1092 | -0.0740 |
| F9 | 128/1/0 | 65/2/0 | 0.000 | 0.000 | 0.131 | 0.149 | -0.0958 | -0.1450 |
| F10 | 301/1/0 | 258/3/0 | 0.068 | 0.094 | 0.067 | 0.101 | -0.0930 | -0.0922 |
| F12 | 114/0/0 | 100/0/0 | 0.020 | 0.023 | 0.138 | 0.156 | -0.4877 | -0.4877 |
| F7 | 249/0/0 | 179/2/0 | 0.000 | 0.000 | 0.082 | 0.121 | -0.1421 | -0.1506 |

## 4. Recovery behaviour (arm B) and guarantee tiers

A recovered action is never described as safe: the tier and margin accompany it. T0 = DR guarantee intact, T1 = nominal CBF only, T2 = `m < 0`, forward invariance lost.

| family | infeasible steps | recovered | T0 | T1 | T2 | rungs |
|---|---|---|---|---|---|---|
| F1' | 2223 | 2223 | 0 | 57 | 2166 | {'R2': 1672, 'R1': 499, 'R2_lp': 52} |
| F2a | 945 | 945 | 0 | 13 | 932 | {'R1': 186, 'R2': 725, 'R2_lp': 34} |
| F2b | 1066 | 1066 | 0 | 28 | 1038 | {'R2': 839, 'R1': 199, 'R2_lp': 28} |
| F3 | 572 | 572 | 0 | 13 | 559 | {'R2': 472, 'R1': 81, 'R2_lp': 19} |
| F4 | 1205 | 1205 | 0 | 25 | 1180 | {'R2': 906, 'R1': 256, 'R2_lp': 43} |
| F5 | 1857 | 1857 | 0 | 28 | 1829 | {'R2': 1477, 'R2_lp': 56, 'R1': 324} |
| F6 | 868 | 868 | 0 | 22 | 846 | {'R2': 665, 'R1': 183, 'R2_lp': 20} |
| F8 | 1324 | 1324 | 0 | 20 | 1304 | {'R2': 1028, 'R1': 251, 'R2_lp': 45} |
| F9 | 1597 | 1597 | 0 | 21 | 1576 | {'R2': 1200, 'R1': 350, 'R2_lp': 47} |
| F10 | 2771 | 2771 | 0 | 38 | 2733 | {'R2': 2044, 'R1': 657, 'R2_lp': 70} |
| F12 | 557 | 557 | 0 | 11 | 546 | {'R2': 434, 'R1': 102, 'R2_lp': 21} |
| F7 | 2524 | 2524 | 0 | 31 | 2493 | {'R2': 2076, 'R1': 400, 'R2_lp': 48} |

## 5. F10 — reactive walls

No replanning for either arm. The wall is visible only through LiDAR.

| id | model | trigger | rect | A succ | B succ | A coll | B coll |
|---|---|---|---|---|---|---|---|
| F10a_behind | deter | [5.0, 5.0] | [3.6, 5.0, 0.5, 1.2] | 0.84 | 0.92 | 0.16 | 0.08 |
| F10a_behind | stoch | [5.0, 5.0] | [3.6, 5.0, 0.5, 1.2] | 0.76 | 0.86 | 0.24 | 0.14 |
| F10b_beside | deter | [5.0, 5.0] | [5.0, 6.4, 1.2, 0.4] | 0.84 | 0.94 | 0.16 | 0.06 |
| F10b_beside | stoch | [5.0, 5.0] | [5.0, 6.4, 1.2, 0.4] | 0.70 | 0.82 | 0.30 | 0.18 |
| F10c_blocks_route | deter | [4.0, 5.0] | [6.2, 5.0, 0.4, 1.4] | 0.00 | 0.06 | 0.80 | 0.64 |
| F10c_blocks_route | stoch | [4.0, 5.0] | [6.2, 5.0, 0.4, 1.4] | 0.04 | 0.04 | 0.94 | 0.90 |
| F10d_at_waypoint | deter | [3.0, 5.0] | [5.0, 5.0, 0.5, 1.0] | 0.02 | 0.06 | 0.82 | 0.72 |
| F10d_at_waypoint | stoch | [3.0, 5.0] | [5.0, 5.0, 0.5, 1.0] | 0.06 | 0.12 | 0.94 | 0.88 |
| F10e_before_arrival | deter | [7.0, 5.0] | [8.4, 5.0, 0.4, 1.2] | 0.00 | 0.02 | 0.76 | 0.74 |
| F10e_before_arrival | stoch | [7.0, 5.0] | [8.4, 5.0, 0.4, 1.2] | 0.02 | 0.00 | 0.92 | 0.88 |

## 6. F1′ — staircase fidelity

Interface/generalization test, **not** literal rotated-rectangle support. Every static obstacle remains an axis-aligned 4-tuple.

| id | degrees | area ratio | Hausdorff | cells | rects |
|---|---|---|---|---|---|
| F1_rot15_obs0 | 15 | 1.0033 | 0.0860 | 301 | 36 |
| F1_rot30_all | 30 | 0.9992 | 0.1041 | 1299 | 125 |
| F1_rot30_obs0 | 30 | 1.0000 | 0.1041 | 300 | 35 |
| F1_rot45_all | 45 | 1.0231 | 0.1063 | 1330 | 126 |
| F1_rot45_obs0 | 45 | 1.0033 | 0.0925 | 301 | 32 |
| F1_rot60_obs0 | 60 | 1.0000 | 0.0844 | 300 | 27 |
| F1_rot90_all | 90 | 1.0000 | 0.0000 | 1300 | 80 |
| F1_rot90_obs0 | 90 | 1.0000 | 0.0000 | 300 | 14 |

Bound: `cell x sqrt(2)` = 0.1414 m. Max observed Hausdorff 0.1063 m.

## 7. X1–X7 — feasibility characterisation (NOT part of any benchmark aggregate)

**No success rate is reported.** The endpoints are what happens when safe action is scarce: infeasibility, recovery, achieved margin, tier, and collision.

| id | model | A coll | B coll | A timeout | B timeout | A inf | B inf | B recov | B tiers | A worst clr | B worst clr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X1_corridor_blocked | scrip | 0.00 | 0.00 | 1.00 | 1.00 | 0 | 0 | 0 | {} | +0.0835 | +0.0835 |
| X2_passage_closes | scrip | 0.00 | 0.00 | 1.00 | 1.00 | 0 | 0 | 0 | {} | +0.0976 | +0.0976 |
| X3_wall_plus_obstacle | scrip | 0.00 | 0.00 | 1.00 | 1.00 | 0 | 0 | 0 | {} | +0.1000 | +0.1000 |
| X4_goal_enclosed | scrip | 0.76 | 0.74 | 0.00 | 0.00 | 17 | 9 | 9 | {'2': 8, '1': 1} | -0.3721 | -0.3721 |
| X_v1.2 | deter | 0.28 | 0.14 | 0.00 | 0.00 | 251 | 113 | 113 | {'2': 112, '1': 1} | -0.1042 | -0.0581 |
| X_v1.2 | stoch | 0.42 | 0.14 | 0.00 | 0.00 | 245 | 131 | 131 | {'2': 130, '1': 1} | -0.1103 | -0.0493 |
| X_v1.35 | deter | 0.46 | 0.20 | 0.00 | 0.00 | 251 | 104 | 104 | {'2': 98, '1': 6} | -0.1212 | -0.0916 |
| X_v1.35 | stoch | 0.60 | 0.34 | 0.00 | 0.00 | 259 | 175 | 175 | {'2': 175} | -0.1199 | -0.0684 |
| X_v1.5 | deter | 0.56 | 0.24 | 0.00 | 0.00 | 290 | 140 | 140 | {'2': 132, '1': 8} | -0.1326 | -0.0645 |
| X_v1.5 | stoch | 0.56 | 0.46 | 0.00 | 0.00 | 260 | 158 | 158 | {'2': 156, '1': 2} | -0.1430 | -0.0837 |

## 8. What is statistically meaningful, and what is not

At n = 50 and alpha = 0.05, **36 of 108** benchmark cells reach significance on collision and **35** on success under exact McNemar.

| id | model | endpoint | A | B | n10 | n01 | p |
|---|---|---|---|---|---|---|---|
| F10c_blocks_route | deter | collision | 0.80 | 0.64 | 8 | 0 | 0.0078 |
| F12e_circling_goal | scrip | collision | 0.78 | 0.56 | 12 | 1 | 0.0034 |
| F12e_circling_goal | scrip | success | 0.22 | 0.44 | 1 | 12 | 0.0034 |
| F1_rot30_all | deter | collision | 0.14 | 0.02 | 6 | 0 | 0.0312 |
| F1_rot30_all | deter | success | 0.86 | 0.98 | 0 | 6 | 0.0312 |
| F1_rot30_all | stoch | collision | 0.26 | 0.06 | 10 | 0 | 0.0020 |
| F1_rot30_all | stoch | success | 0.74 | 0.94 | 0 | 10 | 0.0020 |
| F1_rot60_obs0 | stoch | collision | 0.18 | 0.04 | 8 | 1 | 0.0391 |
| F1_rot60_obs0 | stoch | success | 0.82 | 0.96 | 1 | 8 | 0.0391 |
| F1_rot90_all | stoch | collision | 0.38 | 0.22 | 8 | 0 | 0.0078 |
| F1_rot90_all | stoch | success | 0.50 | 0.64 | 0 | 7 | 0.0156 |
| F2a_scale0.75 | deter | collision | 0.22 | 0.08 | 7 | 0 | 0.0156 |
| F2a_scale0.75 | deter | success | 0.76 | 0.90 | 0 | 7 | 0.0156 |
| F2a_scale0.75 | stoch | collision | 0.22 | 0.06 | 8 | 0 | 0.0078 |
| F2a_scale0.75 | stoch | success | 0.74 | 0.90 | 0 | 8 | 0.0078 |
| F2a_scale1.25 | stoch | collision | 0.26 | 0.10 | 8 | 0 | 0.0078 |
| F2a_scale1.25 | stoch | success | 0.74 | 0.90 | 0 | 8 | 0.0078 |
| F2b_n7 | deter | collision | 0.26 | 0.12 | 7 | 0 | 0.0156 |
| F2b_n7 | deter | success | 0.74 | 0.88 | 0 | 7 | 0.0156 |
| F2b_n7 | stoch | collision | 0.36 | 0.16 | 11 | 1 | 0.0063 |
| F2b_n7 | stoch | success | 0.64 | 0.84 | 1 | 11 | 0.0063 |
| F3_world7 | deter | collision | 0.38 | 0.18 | 12 | 2 | 0.0129 |
| F3_world7 | deter | success | 0.62 | 0.82 | 2 | 12 | 0.0129 |
| F3_world7 | stoch | collision | 0.48 | 0.28 | 10 | 0 | 0.0020 |
| F3_world7 | stoch | success | 0.52 | 0.72 | 0 | 10 | 0.0020 |
| F4_r0.15 | stoch | collision | 0.24 | 0.04 | 11 | 1 | 0.0063 |
| F4_r0.15 | stoch | success | 0.76 | 0.92 | 2 | 10 | 0.0386 |
| F4_r0.2 | stoch | collision | 0.22 | 0.04 | 10 | 1 | 0.0117 |
| F4_r0.2 | stoch | success | 0.78 | 0.94 | 2 | 10 | 0.0386 |
| F4_r0.6 | deter | collision | 0.34 | 0.14 | 12 | 2 | 0.0129 |
| F4_r0.6 | deter | success | 0.66 | 0.86 | 2 | 12 | 0.0129 |
| F4_r0.6 | stoch | collision | 0.48 | 0.32 | 8 | 0 | 0.0078 |
| F4_r0.6 | stoch | success | 0.52 | 0.68 | 0 | 8 | 0.0078 |
| F5_n10 | deter | collision | 0.62 | 0.38 | 13 | 1 | 0.0018 |
| F5_n10 | deter | success | 0.38 | 0.62 | 1 | 13 | 0.0018 |
| F5_n10 | stoch | collision | 0.78 | 0.60 | 9 | 0 | 0.0039 |
| F5_n10 | stoch | success | 0.22 | 0.40 | 0 | 9 | 0.0039 |
| F5_n4 | stoch | collision | 0.34 | 0.14 | 11 | 1 | 0.0063 |
| F5_n4 | stoch | success | 0.64 | 0.84 | 1 | 11 | 0.0063 |
| F5_n8 | deter | collision | 0.42 | 0.28 | 8 | 1 | 0.0391 |
| F5_n8 | deter | success | 0.58 | 0.72 | 1 | 8 | 0.0391 |
| F5_n8 | stoch | collision | 0.62 | 0.32 | 16 | 1 | 0.0003 |
| F5_n8 | stoch | success | 0.38 | 0.68 | 1 | 16 | 0.0003 |
| F6_v0.85 | stoch | collision | 0.20 | 0.02 | 9 | 0 | 0.0039 |
| F6_v0.85 | stoch | success | 0.80 | 0.98 | 0 | 9 | 0.0039 |
| F6_v0.96 | stoch | collision | 0.20 | 0.06 | 8 | 1 | 0.0391 |
| F6_v0.96 | stoch | success | 0.80 | 0.94 | 1 | 8 | 0.0391 |
| F6_v1.1 | deter | collision | 0.30 | 0.14 | 9 | 1 | 0.0215 |
| F6_v1.1 | deter | success | 0.70 | 0.86 | 1 | 9 | 0.0215 |
| F6_v1.1 | stoch | collision | 0.38 | 0.20 | 10 | 1 | 0.0117 |
| F6_v1.1 | stoch | success | 0.62 | 0.80 | 1 | 10 | 0.0117 |
| F7b_n10_v96 | deter | collision | 0.62 | 0.32 | 18 | 3 | 0.0015 |
| F7b_n10_v96 | deter | success | 0.38 | 0.68 | 3 | 18 | 0.0015 |
| F7b_n10_v96 | stoch | collision | 0.78 | 0.54 | 12 | 0 | 0.0005 |
| F7b_n10_v96 | stoch | success | 0.22 | 0.46 | 0 | 12 | 0.0005 |
| F7c_r45_n10 | deter | collision | 0.74 | 0.50 | 12 | 0 | 0.0005 |
| F7c_r45_n10 | deter | success | 0.26 | 0.50 | 0 | 12 | 0.0005 |
| F7d_r45_n10_v96 | deter | collision | 0.70 | 0.48 | 12 | 1 | 0.0034 |
| F7d_r45_n10_v96 | deter | success | 0.30 | 0.52 | 1 | 12 | 0.0034 |
| F8a_open_goal | stoch | collision | 0.20 | 0.04 | 9 | 1 | 0.0215 |
| F8a_open_goal | stoch | success | 0.80 | 0.96 | 1 | 9 | 0.0215 |
| F8b_goal_behind_rect | deter | collision | 0.26 | 0.10 | 10 | 2 | 0.0386 |
| F8b_goal_behind_rect | deter | success | 0.74 | 0.90 | 2 | 10 | 0.0386 |
| F8b_goal_behind_rect | stoch | collision | 0.30 | 0.08 | 12 | 1 | 0.0034 |
| F8b_goal_behind_rect | stoch | success | 0.70 | 0.92 | 1 | 12 | 0.0034 |
| F9_w0.85 | stoch | collision | 0.54 | 0.26 | 16 | 2 | 0.0013 |
| F9_w0.85 | stoch | success | 0.46 | 0.74 | 2 | 16 | 0.0013 |
| F9_w1.0 | stoch | collision | 0.42 | 0.08 | 18 | 1 | 0.0001 |
| F9_w1.0 | stoch | success | 0.58 | 0.92 | 1 | 18 | 0.0001 |
| F9_w1.6 | stoch | collision | 0.24 | 0.08 | 8 | 0 | 0.0078 |
| F9_w1.6 | stoch | success | 0.76 | 0.92 | 0 | 8 | 0.0078 |

**Everything else in this report is descriptive.** Family-level rows average over cells and carry no test; they are for characterising the suite, not for concluding anything about the controllers. The final tier at 200 episodes is what the design reserves for inference.

## 9. Runtime

Measured **5.44 s/episode** over 118 completed cells (17.8 h of cumulative compute).

| tier | episodes | at the measured rate | at 8-way |
|---|---|---|---|
| screening (this run) | 11800 | 17.8 h | 2.2 h |
| final (not started) | 41400 | 62.6 h | 7.8 h |

Slowest cells: `X1_corridor_blocked` 19.3 s/ep, `X3_wall_plus_obstacle` 17.3 s/ep, `X2_passage_closes` 17.2 s/ep, `F10c_blocks_route` 10.7 s/ep, `F10e_before_arrival` 10.3 s/ep

## 10. A* feasibility rule, rejections and regenerations

The preregistered rule is **≥ 90 % A\* feasibility for the applicable *generated* maps**, with
rejection handled by regenerating from the next declared generation seed.

**No map was rejected and no map was regenerated. `GEN_SEED_BASE = 7_000_000` was never
consumed.** Every layout in the suite is a *declared constant* — the canonical rectangles, their
scalings, the staircase raster, the declared `EXTRA_RECTS`, the corridor construction, the F8
pairs — so no family samples a layout and the regeneration procedure has nothing to advance to.

Feasibility measured over all 11 800 screening episodes, using the planner's own
`planner_failed` flag:

| family | A*-feasible start/goal pairs | | family | A*-feasible | |
|---|---|---|---|---|---|
| F1′ | 800/800 = 100 % | ✅ | F8 | 500/500 = 100 % | ✅ |
| F2a | 400/400 = 100 % | ✅ | F9 | 500/500 = 100 % | ✅ |
| **F2b** | **368/400 = 92.0 %** | ✅ family | F10 | 500/500 = 100 % | ✅ |
| F3 | 300/300 = 100 % | ✅ | F12 | 300/300 = 100 % | ✅ |
| F4 | 400/400 = 100 % | ✅ | F7 | 400/400 = 100 % | ✅ |
| F5 | 400/400 = 100 % | ✅ | X | 500/500 = 100 % | ✅ |
| F6 | 500/500 = 100 % | ✅ | | | |

### 10.1 One configuration is below the criterion — reported, not acted on

**`F2b_n9` (the D+ level, 9 static rectangles) is at 68 % per cell** — 16 of 50 start/goal pairs
have no A\* path — in **both** motion models. The family passes at 92 % only because the other
three F2b levels are at 100 %.

Facts, before any interpretation:

* The **same 16 seeds** fail for **both arms**, in both motion models. The fairness invariant
  holds: this is a property of the map, not of a controller.
* On a planner failure both arms degrade identically — `follower = None`, so the CLF tracks the
  goal directly with no waypoints. Neither arm is advantaged.
* Outcomes on those 16: arm A 13 collisions / 3 timeouts (deterministic), arm B 11 / 5. On the
  34 planned episodes the two arms are nearly identical (27 successes each, deterministic).

**Why the preregistered remedy does not apply:** the rule covers *generated* maps and repairs
them by advancing a generation seed. `F2b_n9` is a declared constant — canonical five rectangles
plus four declared `EXTRA_RECTS` — so there is no seed to advance and nothing to regenerate. The
rule is silent on this case rather than satisfied by it.

### 10.2 RESOLVED — Amendment 1 (2026-09-09): excluded from the final tier, not replaced

Reviewed and decided **after** screening completed and **before** any final-tier episode ran.
Recorded in `MAP_SUITE_SPEC.md` §4.7 and implemented as `suite.FINAL_EXCLUDED`.

**`F2b_n9` is excluded from the benchmark and the final tier. It is NOT replaced.** It remains in
the manifest, in the 118-cell screening suite, and in this report.

Retained as a **feasibility-boundary observation**, stated explicitly:

* **68 % A\* feasibility** — 34 of 50 declared screening start/goal pairs have a path, per cell,
  against the preregistered ≥ 90 % criterion.
* **The same 16 seeds and layouts fail for both arms**, in both motion models.
* **Behaviour where infeasible is identical**: both arms lose the waypoint follower and track the
  goal directly, so neither is advantaged.
* **No controller advantage or disadvantage was created** by the exclusion — on the 34 planned
  episodes the arms are near-identical (27 successes each, deterministic).
* **Excluded before final evaluation** because it does not satisfy the intended navigable-map
  assumption — a structural property of the layout, not a controller result.

**No replacement map was invented.** The preregistered "advance the generation seed" remedy is
inapplicable to a fixed declared layout, and constructing a new one after seeing screening
results is what the preregistration forbids.

**Derived counts:** final cells 111 → **109**, benchmark cells 101 → **99**, final episodes
41 400 → **40 600**, total 53 200 → **52 400**. Screening stays at **118 cells / 11 800
episodes**.

## 11. Fairness and assertion status

Every one of the **11 800 episodes** ran under the paired fairness assertion: identical map,
seed, start, goal, initial obstacle states, horizon, world size and static-geometry count, with
only the controller differing. A violation raises rather than warns.

* **118/118 cells complete**, 50 paired episodes each, **zero malformed cells**, and the seed
  lists of arm A and arm B match element-for-element in every cell.
* **One assertion fired during the run**, and it caught a real bug. See §12.
* No arm read `obs[28:52]` or any ground-truth obstacle state; the leakage test covering both
  arms passes.
* `v_cap = 0.96`, LiDAR range 5.0 m, `max_speed = 1.0`, 24 rays and `agent_radius = 0.3` were
  identical across all 64 configurations, asserted by test.

## 12. Implementation anomalies

### 12.1 The F10 fairness failure — a genuine bug, caught by the assertion

The first F10 worker aborted at `F10a_behind seed 8000046: obstacles differ`.

**Cause.** `ScenarioEnv.reset()` restored the base static obstacles *after* calling
`super().reset()`. `RobotNavEnv.reset()` samples obstacle positions with
`_random_free_position`, which rejects candidates against `self.static_obstacles` — so with a
wall still standing from the previous episode it consumed a **different number of RNG draws**,
and the next episode's obstacle layout changed. Arms A and B fire walls at different steps, so
the two arms desynchronised.

**Fix.** Restore the base geometry and clear wall state *before* `super().reset()`. A seed-swept
regression test was added and **verified to fail on the old code and pass on the new** — the
first version of that test passed on the buggy code, because whether the stale rectangle forces a
rejection depends on where the sampler lands, so a single seed is not enough.

**Scope.** Only the 10 F10 cells were affected and only those were re-run; the other 108 cells
were untouched. No family definition, parameter, seed or selection rule changed. The pre-fix
partial F10 output was never written, since the worker aborted before its first cell completed.

### 12.2 X1–X3 are bit-identical between the two arms

`X1_corridor_blocked`, `X2_passage_closes` and `X3_wall_plus_obstacle` produced **50/50
identical episodes** for arms A and B — identical outcomes, step counts and clearances to
machine precision — with **zero infeasible steps** in either arm.

This is not an error. In those three scenarios the robot stops short of the blockage and times
out; the DR problem never becomes infeasible, so the Stage-4 ladder never activates, and the
estimated closing rates never exceed 0.96, so the T1 cap never binds. **With neither of arm B's
two changes active, arm B *is* arm A.** The scenarios therefore discriminate nothing between the
controllers, which is itself the recorded characterisation result.

`X1`'s previously-reported timeout-without-infeasibility was left unchanged at your instruction,
and the same behaviour now appears in X2 and X3.

### 12.3 No other anomalies

No solver crashes, no memory events, no overwritten outputs, no missing cells.

## 13. Runtime recalibration

The runner recomputed its projection after 5, 10 and 20 completed cells in each of the 8
workers. Selected lines, spanning the observed range:

| cells | measured | screening projection | at 8-way |
|---|---|---|---|
| 5 (F2b/F8 worker) | 4.10 s/ep | 13.5 h | 1.7 h |
| 5 | 4.35 s/ep | 14.3 h | 1.8 h |
| 10 | 4.36 s/ep | 14.3 h | 1.8 h |
| 18 | 4.48 s/ep | 14.7 h | 1.8 h |
| 16 (F1′ worker) | 5.60 s/ep | 18.4 h | 2.3 h |
| 16 | 6.07 s/ep | 19.9 h | 2.5 h |
| 10 (F9 worker) | 7.74 s/ep | 25.4 h | 3.2 h |
| **5 (X worker)** | **11.81 s/ep** | 38.7 h | 4.8 h |

**Final measured rate across all 118 cells: 5.44 s/episode**, against the proposal's assumed
5.65 and the smoke test's 4.68. The spread is real and family-dependent: the X scenarios run
full 500-step timeouts (11.8 s/ep) and F9's narrow corridors are slow, while open maps run at
~4.1 s/ep. The smoke estimate was optimistic because it sampled one cheap cell per family.

**Revised projection for the final tier at 5.44 s/ep: 41 400 episodes = 62.6 h single-process,
≈ 7.8 h at 8-way.** Actual screening wall-clock was ≈ 3 h at 8-way plus ≈ 1 h for the F10 re-run.

**This is diagnostic only.** No episode allocation, family definition or selection rule was
changed because of it.
