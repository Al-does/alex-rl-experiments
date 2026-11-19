# RockSample r7 nuisance-controlled steering (v3)

Frozen r7 update-29 sampled policies, 7 current-Good marginals, fair-prior action/observation Bayesian targets; independent own-policy fit/test streams, all valid rows including reset, no warmup exclusion. Separate affine least squares (rcond 1e-6). Compare v2 calibrated encoder = solve(EW,E) with nuisance-controlled encoder E_nc from joint activation regression on [beliefs,nuisance], calibrated as solve(E_nc W,E_nc). For each calibrated variant choose the smallest tuning λ with pooled Good→Bad pickup ≤0.10 AND Bad→Good pickup ≥0.5, else λ=1. Choose displayed variant on tuning only by (Bad→Good pickup minus Good→Bad pickup minus maximum absolute nontarget conditional-rate drift from intact across directions/qualities); ties favor v2 calibrated encoder. Freeze across agents/sites. Matched environment/action RNG across conditions; Bad→Good stops at first target pickup, Good→Bad continues every decision step. Cached K/V propagation preserved.

Nuisance design: 25-cell rover-position one-hot (24 reference-coded columns), 100-step decision-index one-hot (99 columns; reset reference), previous-action one-hot (12 columns plus reset reference), current sensor-symbol one-hot (2 columns plus NONE reference), and 7 already-sampled indicators computed from preceding executed actions and observed positions. Total 144 nuisance columns, plus 7 beliefs and an intercept. No reward, hidden quality, or future event enters the regression. No additional nuisance features; fixed layout is constant. One-hot reference coding avoids dummy/intercept redundancy. Counterfactual sensor-pair replay skipped to keep the final CPU iteration focused.

Displayed variant: **nuisance_controlled_encoder**, chosen using tuning only.

Fit/test: 256/256 episodes per agent; tuning: 128; confirmation: 1,024 per
rock/direction/condition/variant/site (smoke uses 8). Fresh confirmation seed 101429.
Endpoints are measured at the intervened site, before downstream nonlinearities.
Sites 0–2 = post-block pre-final-LN; site 3 = post-final-LN.

## Tuning evidence

| Variant | λ | Good→Bad pickup | Bad→Good pickup |
|---|---:|---:|---:|
| calibrated_encoder | 0.5 | 0.6607 | 0.1105 |
| calibrated_encoder | 1 | 0.2712 | 0.2422 |
| calibrated_encoder | 1.5 | 0.1451 | 0.3002 |
| calibrated_encoder | 2 | 0.1205 | 0.2277 |
| calibrated_encoder | 3 | 0.0692 | 0.1864 |
Selected calibrated_encoder: λ=1.0; threshold met: False; tuning score=-0.4334.

| Variant | λ | Good→Bad pickup | Bad→Good pickup |
|---|---:|---:|---:|
| nuisance_controlled_encoder | 0.5 | 0.7065 | 0.1183 |
| nuisance_controlled_encoder | 1 | 0.2366 | 0.3460 |
| nuisance_controlled_encoder | 1.5 | 0.0547 | 0.2734 |
| nuisance_controlled_encoder | 2 | 0.0201 | 0.1674 |
| nuisance_controlled_encoder | 3 | 0.0056 | 0.1049 |
Selected nuisance_controlled_encoder: λ=1.0; threshold met: False; tuning score=-0.2415.

## Primary-agent variant comparison (site 2)

| Variant | λ | Direction | Target pickup | Other Good | Other Bad | Return | Decoded after |
|---|---:|---|---:|---:|---:|---:|---:|
| calibrated_encoder | 1.0 | Good→Bad | 0.2944 | 0.5838 | 0.0046 | 25.19 | 0.000000 |
| calibrated_encoder | 1.0 | Bad→Good | 0.2648 | 0.6512 | 0.0855 | -24.87 | 1.000000 |
| nuisance_controlled_encoder | 1.0 | Good→Bad | 0.2681 | 0.6468 | 0.0043 | 27.02 | -0.000000 |
| nuisance_controlled_encoder | 1.0 | Bad→Good | 0.3277 | 0.7142 | 0.0886 | -14.36 | 1.000000 |

## Primary-agent selectivity, return and endpoint sanity (site 2)

| Variant | Direction | Condition | Other Good | Other Bad | Return | Decoded before→after | Max endpoint error | Max other-coordinate change |
|---|---|---|---:|---:|---:|---|---:|---:|
| calibrated_encoder | Good→Bad | intact | 0.9798 | 0.0125 | 48.32 | 0.35063→0.35063 | 1.47 | 0 |
| calibrated_encoder | Good→Bad | steered | 0.5838 | 0.0046 | 25.19 | 0.61327→0.00000 | 4.57e-06 | 1.07e-06 |
| calibrated_encoder | Good→Bad | control | 0.6343 | 0.0485 | 24.03 | 0.51890→0.51890 | 1.68 | 3.64e-07 |
| calibrated_encoder | Bad→Good | intact | 0.9807 | 0.0127 | 38.47 | 0.17726→0.17726 | 1.49 | 0 |
| calibrated_encoder | Bad→Good | steered | 0.6512 | 0.0855 | -24.87 | 0.16595→1.00000 | 1.79e-06 | 6.56e-07 |
| calibrated_encoder | Bad→Good | control | 0.4117 | 0.0556 | 11.44 | 0.18344→0.18344 | 1.39 | 3.58e-07 |
| nuisance_controlled_encoder | Good→Bad | intact | 0.9798 | 0.0125 | 48.32 | 0.35063→0.35063 | 1.47 | 0 |
| nuisance_controlled_encoder | Good→Bad | steered | 0.6468 | 0.0043 | 27.02 | 0.62295→-0.00000 | 6.39e-06 | 7.75e-07 |
| nuisance_controlled_encoder | Good→Bad | control | 0.6893 | 0.0435 | 28.18 | 0.50094→0.50094 | 1.73 | 3.87e-07 |
| nuisance_controlled_encoder | Bad→Good | intact | 0.9807 | 0.0127 | 38.47 | 0.17726→0.17726 | 1.49 | 0 |
| nuisance_controlled_encoder | Bad→Good | steered | 0.7142 | 0.0886 | -14.36 | 0.15339→1.00000 | 1.67e-06 | 7.15e-07 |
| nuisance_controlled_encoder | Bad→Good | control | 0.4817 | 0.0501 | 15.13 | 0.16707→0.16707 | 1.39 | 3.58e-07 |

## Best variant: 20261002T191712Z-7f1dc1ef seed 3

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.982 [0.972, 0.989] | 0.184 [0.161, 0.208] | 0.767 [0.740, 0.791] | 0.017 [0.010, 0.026] | 0.638 [0.608, 0.667] | 0.012 [0.007, 0.020] |
| 1 | 0.988 [0.980, 0.993] | 0.568 [0.538, 0.598] | 0.963 [0.949, 0.973] | 0.013 [0.007, 0.022] | 0.128 [0.109, 0.150] | 0.013 [0.007, 0.022] |
| 2 | 0.965 [0.952, 0.974] | 0.657 [0.628, 0.686] | 0.643 [0.613, 0.671] | 0.015 [0.009, 0.024] | 0.304 [0.276, 0.333] | 0.047 [0.036, 0.062] |
| 3 | 0.975 [0.963, 0.983] | 0.413 [0.383, 0.444] | 0.045 [0.034, 0.059] | 0.008 [0.004, 0.015] | 0.199 [0.176, 0.225] | 0.006 [0.003, 0.013] |
| 4 | 0.987 [0.978, 0.993] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.011 [0.006, 0.019] | 0.009 [0.005, 0.017] | 0.000 [0.000, 0.004] |
| 5 | 0.974 [0.962, 0.982] | 0.020 [0.013, 0.030] | 0.493 [0.463, 0.524] | 0.015 [0.009, 0.024] | 0.872 [0.850, 0.891] | 0.016 [0.010, 0.025] |
| 6 | 0.991 [0.983, 0.995] | 0.035 [0.026, 0.048] | 0.927 [0.909, 0.941] | 0.005 [0.002, 0.011] | 0.145 [0.124, 0.167] | 0.058 [0.045, 0.074] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| calibrated_encoder | 0 | 0.491 | 0.347 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| calibrated_encoder | 1 | 0.581 | 0.247 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| calibrated_encoder | 2 | 0.294 | 0.265 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| calibrated_encoder | 3 | 0.731 | 0.065 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |
| nuisance_controlled_encoder | 0 | 0.141 | 0.702 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| nuisance_controlled_encoder | 1 | 0.348 | 0.540 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| nuisance_controlled_encoder | 2 | 0.268 | 0.328 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| nuisance_controlled_encoder | 3 | 0.576 | 0.272 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |

## Best variant: 20261002T182024Z-ad0c3a34 seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.990 [0.982, 0.995] | 0.278 [0.252, 0.307] | 0.970 [0.957, 0.979] | 0.013 [0.007, 0.022] | 0.472 [0.441, 0.502] | 0.010 [0.005, 0.018] |
| 1 | 0.993 [0.986, 0.997] | 0.901 [0.882, 0.918] | 0.795 [0.769, 0.819] | 0.007 [0.003, 0.014] | 0.122 [0.103, 0.144] | 0.009 [0.005, 0.017] |
| 2 | 0.970 [0.957, 0.979] | 0.623 [0.593, 0.652] | 0.950 [0.935, 0.962] | 0.012 [0.007, 0.020] | 0.122 [0.103, 0.144] | 0.017 [0.010, 0.026] |
| 3 | 0.987 [0.978, 0.993] | 0.528 [0.498, 0.559] | 0.572 [0.542, 0.602] | 0.013 [0.007, 0.022] | 0.179 [0.156, 0.203] | 0.018 [0.011, 0.028] |
| 4 | 0.984 [0.975, 0.990] | 0.003 [0.001, 0.009] | 0.053 [0.041, 0.068] | 0.012 [0.007, 0.020] | 0.022 [0.015, 0.033] | 0.015 [0.009, 0.024] |
| 5 | 0.964 [0.951, 0.974] | 0.118 [0.100, 0.139] | 0.906 [0.887, 0.923] | 0.016 [0.010, 0.025] | 0.242 [0.217, 0.269] | 0.118 [0.100, 0.139] |
| 6 | 0.989 [0.981, 0.994] | 0.055 [0.042, 0.070] | 0.004 [0.002, 0.010] | 0.006 [0.003, 0.013] | 0.131 [0.112, 0.153] | 0.000 [0.000, 0.004] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| calibrated_encoder | 0 | 0.462 | 0.146 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| calibrated_encoder | 1 | 0.556 | 0.117 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| calibrated_encoder | 2 | 0.421 | 0.202 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| calibrated_encoder | 3 | 0.726 | 0.063 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |
| nuisance_controlled_encoder | 0 | 0.152 | 0.471 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| nuisance_controlled_encoder | 1 | 0.390 | 0.329 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| nuisance_controlled_encoder | 2 | 0.358 | 0.184 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| nuisance_controlled_encoder | 3 | 0.551 | 0.177 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |

## Best variant: 20261002T191712Z-7f1dc1ef seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.970 [0.957, 0.979] | 0.061 [0.048, 0.077] | 0.945 [0.930, 0.958] | 0.011 [0.006, 0.019] | 0.576 [0.546, 0.606] | 0.018 [0.011, 0.028] |
| 1 | 0.986 [0.977, 0.992] | 0.711 [0.682, 0.738] | 0.982 [0.972, 0.989] | 0.010 [0.005, 0.018] | 0.128 [0.109, 0.150] | 0.016 [0.010, 0.025] |
| 2 | 0.972 [0.960, 0.980] | 0.815 [0.791, 0.838] | 0.908 [0.889, 0.924] | 0.014 [0.008, 0.023] | 0.107 [0.090, 0.128] | 0.019 [0.012, 0.029] |
| 3 | 0.982 [0.972, 0.989] | 0.780 [0.754, 0.805] | 0.675 [0.646, 0.703] | 0.005 [0.002, 0.011] | 0.070 [0.056, 0.088] | 0.015 [0.009, 0.024] |
| 4 | 0.979 [0.969, 0.987] | 0.012 [0.007, 0.020] | 0.007 [0.003, 0.014] | 0.011 [0.006, 0.019] | 0.009 [0.005, 0.017] | 0.006 [0.003, 0.013] |
| 5 | 0.966 [0.953, 0.975] | 0.026 [0.018, 0.038] | 0.811 [0.785, 0.833] | 0.019 [0.012, 0.029] | 0.741 [0.714, 0.767] | 0.104 [0.086, 0.124] |
| 6 | 0.989 [0.981, 0.994] | 0.008 [0.004, 0.015] | 0.026 [0.018, 0.038] | 0.013 [0.007, 0.022] | 0.171 [0.149, 0.195] | 0.008 [0.004, 0.015] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| calibrated_encoder | 0 | 0.468 | 0.166 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| calibrated_encoder | 1 | 0.513 | 0.137 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| calibrated_encoder | 2 | 0.413 | 0.190 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| calibrated_encoder | 3 | 0.732 | 0.052 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |
| nuisance_controlled_encoder | 0 | 0.124 | 0.705 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| nuisance_controlled_encoder | 1 | 0.416 | 0.412 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| nuisance_controlled_encoder | 2 | 0.345 | 0.258 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| nuisance_controlled_encoder | 3 | 0.568 | 0.256 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |

## Resistant-rock rollout evidence: primary agent, best variant, site 2

Full confirmation rates below; examples are fixed episodes 0–3, not selected for outcomes.
Raw position/action traces stay in ignored artifacts.

- Rock 0, Good→Bad: pickup 0.184, visit 0.425, checked 0.991, target checks 7.52/episode; visited without pickup 0.241.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [2, 1], [2, 2], [3, 2], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4], [4, 3], [3, 3], [3, 2], [2, 2], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1], [2, 0], [1, 0], [2, 0], [3, 0], [4, 0], [3, 0], [3, 1], [3, 0], [3, 1]]; check counts by rock [11, 11, 6, 6, 10, 7, 11]; target pickup False; final action 10.
  Episode 1: route [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [4, 3], [3, 3], [3, 2], [2, 2], [1, 2], [1, 1], [2, 1], [1, 1], [2, 1], [3, 1], [4, 1], [3, 1], [2, 1], [2, 0], [3, 0], [4, 0], [3, 0], [2, 0], [1, 0], [2, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [4, 2]]; check counts by rock [5, 5, 3, 6, 4, 5, 6]; target pickup True; final action 2.
- Rock 1, Good→Bad: pickup 0.568, visit 0.954, checked 0.999, target checks 7.50/episode; visited without pickup 0.386.
  Episode 0: route [[0, 2], [0, 1], [1, 1], [2, 1], [2, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2]]; check counts by rock [13, 6, 5, 4, 7, 12, 9]; target pickup True; final action 2.
  Episode 1: route [[0, 2], [1, 2], [2, 2], [2, 1], [2, 2], [3, 2], [2, 2], [2, 1], [2, 0], [1, 0], [1, 1], [2, 1], [2, 2], [2, 1], [2, 2], [3, 2], [2, 2], [2, 1], [3, 1], [4, 1], [4, 2]]; check counts by rock [4, 5, 3, 5, 3, 3, 4]; target pickup True; final action 2.
- Rock 1, Bad→Good: pickup 0.128, visit 0.738, checked 1.000, target checks 5.25/episode; visited without pickup 0.610.
  Episode 0: route [[0, 2], [0, 1], [1, 1], [1, 0], [1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4], [2, 4], [2, 3], [3, 3], [3, 4], [3, 3], [3, 2], [4, 2]]; check counts by rock [9, 7, 3, 3, 6, 2, 6]; target pickup False; final action 2.
  Episode 1: route [[0, 2], [0, 1], [1, 1], [1, 0], [1, 1], [2, 1], [2, 2], [2, 1], [1, 1], [2, 1], [2, 2], [3, 2], [4, 2]]; check counts by rock [4, 5, 4, 3, 1, 2, 3]; target pickup False; final action 2.
- Rock 2, Good→Bad: pickup 0.657, visit 0.987, checked 0.987, target checks 4.24/episode; visited without pickup 0.330.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [2, 1], [1, 1], [2, 1], [2, 2], [2, 3], [2, 4], [3, 4], [4, 4]]; check counts by rock [3, 2, 3, 3, 5, 1, 2]; target pickup True; final action 2.
  Episode 1: route [[0, 2], [1, 2], [2, 2], [2, 1], [1, 1], [2, 1], [1, 1], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [4, 2]]; check counts by rock [3, 2, 3, 1, 3, 1, 3]; target pickup True; final action 2.
- Rock 2, Bad→Good: pickup 0.304, visit 0.793, checked 1.000, target checks 12.82/episode; visited without pickup 0.489.
  Episode 0: route [[0, 2], [0, 1], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]; check counts by rock [12, 10, 17, 8, 7, 12, 8]; target pickup False; final action 10.
  Episode 1: route [[0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 2], [4, 1], [3, 1], [3, 0], [4, 0]]; check counts by rock [4, 4, 4, 4, 3, 1, 3]; target pickup True; final action 2.
- Rock 3, Good→Bad: pickup 0.413, visit 0.889, checked 0.994, target checks 7.30/episode; visited without pickup 0.476.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [2, 2], [2, 3], [2, 4], [3, 4], [3, 3], [4, 3]]; check counts by rock [3, 4, 3, 8, 7, 8, 10]; target pickup True; final action 2.
  Episode 1: route [[0, 2], [1, 2], [2, 2], [2, 1], [1, 1], [1, 0], [0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [1, 2], [2, 2], [3, 2], [4, 2]]; check counts by rock [5, 5, 3, 6, 8, 3, 9]; target pickup True; final action 2.
- Rock 3, Bad→Good: pickup 0.199, visit 0.330, checked 1.000, target checks 6.72/episode; visited without pickup 0.131.
  Episode 0: route [[0, 2], [0, 1], [0, 0], [1, 0], [1, 1]]; check counts by rock [7, 6, 6, 5, 4, 9, 6]; target pickup False; final action 4.
  Episode 1: route [[0, 2], [0, 1], [0, 0]]; check counts by rock [5, 4, 4, 5, 2, 5, 1]; target pickup False; final action 4.
- Rock 4, Bad→Good: pickup 0.009, visit 0.045, checked 0.895, target checks 1.77/episode; visited without pickup 0.036.
  Episode 0: route [[0, 2], [0, 1]]; check counts by rock [0, 1, 0, 1, 2, 1, 0]; target pickup False; final action 4.
  Episode 1: route [[0, 2], [1, 2], [1, 1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [4, 4]]; check counts by rock [7, 4, 4, 4, 5, 4, 2]; target pickup False; final action 2.
- Rock 6, Bad→Good: pickup 0.145, visit 0.255, checked 0.999, target checks 5.39/episode; visited without pickup 0.110.
  Episode 0: route [[0, 2], [0, 1], [1, 1], [1, 0], [1, 1], [2, 1], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4]]; check counts by rock [3, 2, 1, 3, 3, 1, 1]; target pickup False; final action 2.
  Episode 1: route [[0, 2], [1, 2], [1, 1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [3, 3], [3, 2], [4, 2]]; check counts by rock [4, 4, 4, 5, 3, 4, 3]; target pickup True; final action 2.

Nontarget conditional rates pool the six nontarget rocks by initial quality.
Wilson binomial 95% intervals do not represent training-seed uncertainty.
Calibrated maps are solved in float64 and executed in float32; identity errors,
condition numbers and row norms are recorded in results.json. Rank deficiency raises.
Controls are decoder-null random directions with the counterfactual shift norm on
their own current history, not a copied norm from a diverged steering trajectory.
Intact episodes are reused across variants/sites because they are identical.
Steering can still be off-manifold; endpoint calibration alone does not ensure selectivity.

## Interpretation of resistant rocks

At the primary site, rocks 1 and 2 resist Bad→Good despite visits in 73.8% and
79.3% of episodes and checks in every episode: pickup is only 12.8% and 30.4%.
Their failure is therefore not simply failure to reach or sense the rock. Rock 2
also remains sampled under Good→Bad in 65.7% of episodes, with 98.7% visitation.
For rock 3 Bad→Good, visitation is only 33.0%; both fixed examples remain in the
western grid and finish with a sample action rather than reaching the target.
Rock 4 Bad→Good visits only 4.5%; example 0 stays at positions [0,2]/[0,1], whereas
example 1 travels north and east without target pickup. Rock 6 Bad→Good visits
25.5%; the two examples traverse the grid but differ in whether they pick it up.
These observations separate arrival failures from arrival-without-pickup, but do
not identify an internal mechanism. Rock 0 Good→Bad is a partial suppression
(98.2% to 18.4%), not a complete flip; rocks 1 and 3 remain sampled in 56.8% and
41.3% despite visits in 95.4% and 88.9%.

## Secondary-site selectivity: primary agent, nuisance-controlled encoder, λ=1

These sites were part of the predeclared evaluation, but were not used to select
the variant or λ. They do not replace the primary-site results or figure.

| Site | Direction | Condition | Target pickup | Other Good | Other Bad | Return |
|---|---|---|---:|---:|---:|---:|
| 0 | Good→Bad | steered | 0.1410 | 0.9592 | 0.0115 | 38.85 |
| 0 | Good→Bad | control | 0.9191 | 0.9701 | 0.0132 | 47.12 |
| 0 | Bad→Good | steered | 0.7020 | 0.9105 | 0.0349 | 23.41 |
| 0 | Bad→Good | control | 0.0173 | 0.9524 | 0.0142 | 35.64 |
| 1 | Good→Bad | steered | 0.3484 | 0.9747 | 0.0100 | 41.12 |
| 1 | Good→Bad | control | 0.9515 | 0.9775 | 0.0171 | 47.25 |
| 1 | Bad→Good | steered | 0.5399 | 0.9510 | 0.0272 | 28.53 |
| 1 | Bad→Good | control | 0.0128 | 0.9706 | 0.0152 | 37.38 |

Intact is identical across sites: Good→Bad target/other-Good/other-Bad/return =
0.9803/0.9798/0.0125/48.32; Bad→Good = 0.0117/0.9807/0.0127/38.47.
At site 0, rock-specific Good→Bad pickup is 0.097/0.234/0.118/0.202/0.033/0.006/0.297
and Bad→Good pickup is 0.645/0.425/0.947/0.698/0.926/0.991/0.282 (rocks 0–6).
Thus rocks 0, 4 and 5 meet both numerical target thresholds at this secondary
site, unlike the primary site; these per-rock observations are descriptive, not
new tuning criteria. Site-0 pooled Good→Bad still exceeds 0.10. Across the two
replicates, site-0 Good→Bad is 0.152/0.124 and Bad→Good is 0.471/0.705; nontarget
Good pickup is 0.944/0.906 and 0.813/0.846 respectively, so selectivity and
bidirectional strength are not uniform across agents.

## Overall conclusion across v1–v3

v1 established that activation shifts can change pickup, but overshot decoded
endpoints and broadly disrupted behavior. v2 fixed those endpoints; min-norm
shifts largely preserved nontarget behavior but barely changed target pickup,
whereas calibrated encoders produced larger, nonselective effects. v3 nuisance
adjustment modestly improves the primary agent's primary-site pooled pickup
(Good→Bad 0.294→0.268; Bad→Good 0.265→0.328). At that site, rock 5 shows strong
two-way effects (Good pickup 0.974→0.020; Bad pickup 0.015→0.872), rock 0 shows
substantial but incomplete two-way effects, and Good→Bad strongly suppresses
rocks 4 and 6. Nevertheless nontarget Good pickup falls from about 0.980 to
0.647/0.714, and Bad→Good return is negative; matched-null controls also disrupt
behavior. Primary-site calibration and nuisance adjustment do not establish a
selective seven-rock belief flip.

Earlier residual sites are more encouraging: at site 0, primary-agent pooled
Good→Bad/Bad→Good pickup is 0.141/0.702 versus intact 0.980/0.012 and control
0.919/0.017, while nontarget Good pickup is 0.959/0.910 versus about 0.980 intact.
Rocks 0, 4 and 5 meet both target thresholds there; site 1 preserves nontarget
Good pickup still better (0.975/0.951) but has weaker Good→Bad suppression.
These are site-dependent causal behavioral effects with improved, not perfect,
selectivity, and weaker replication on one agent. No variant met the predeclared
primary-site tuning criterion. Statistical nuisance adjustment is not an
identification of nuisance-free causal belief coordinates, and these results
do not establish an internal Bayesian model or isolate why navigation changes.
