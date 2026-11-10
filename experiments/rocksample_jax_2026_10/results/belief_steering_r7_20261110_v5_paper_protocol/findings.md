# RockSample r7 paper-protocol steering (v5)

Paper-faithful Joint Reward port with RockSample adaptations: separate plain affine D (activation→belief) and E (belief→activation), no calibration and no nuisance regressors. Independent own-policy sampled-action fit/test streams, 256 episodes each; exclude decision indices 0–7 from fitting and probe scoring only. Evaluation includes every decision. Marginal targets replace one of seven decoded Good probabilities. Joint D/E fit coordinates 0–126, omit all-Good configuration 127 and reconstruct it as 1 minus the sum. Set the target rock to a point mass while preserving the decoded joint over the other six rocks, then shift on 127 coordinates. Plain residual-times-encoder shifts at one site at a time, including KV-cache propagation. Tune on 128 episodes per rock/direction at site 2 using new seed 371839 and grid {0,0.5,0.75,0.8,0.85,0.9,0.95,1}. Independently select each representation/direction: Good→Bad smallest λ with pooled target pickup ≤ measured uniform-random Good pickup; Bad→Good smallest λ with pooled pickup ≥0.5. If no value qualifies, use λ=1. Freeze across all sites and agents. Confirm on seed 471841 with 1,024 episodes per rock/direction/condition. Matched environment/action RNG; stochastic actions; intact/steered/decoder-null control/uniform-random conditions. Bad→Good releases after first target pickup; Good→Bad continues every step. Smoke uses 8 episodes per stream and the primary agent only.

## Tuning at the paper's site (site 2, after block 3 pre-LN)

Uniform-random Good pickup threshold: 0.1931.

| Representation | Direction | λ | Target pickup | Other Good | Other Bad | Return |
|---|---|---:|---:|---:|---:|---:|
| marginal | Good→Bad | 0.0 | 0.9788 | 0.9761 | 0.0128 | 48.87 |
| marginal | Good→Bad | 0.5 | 0.6875 | 0.7918 | 0.0087 | 37.75 |
| marginal | Good→Bad | 0.75 | 0.5268 | 0.6637 | 0.0060 | 30.88 |
| marginal | Good→Bad | 0.8 | 0.5033 | 0.6398 | 0.0045 | 29.70 |
| marginal | Good→Bad | 0.85 | 0.4699 | 0.6118 | 0.0045 | 28.26 |
| marginal | Good→Bad | 0.9 | 0.4163 | 0.5949 | 0.0045 | 27.03 |
| marginal | Good→Bad | 0.95 | 0.3895 | 0.5773 | 0.0045 | 26.04 |
| marginal | Good→Bad | 1.0 | 0.3616 | 0.5548 | 0.0026 | 24.92 |
| marginal | Bad→Good | 0.0 | 0.0112 | 0.9831 | 0.0135 | 39.26 |
| marginal | Bad→Good | 0.5 | 0.0525 | 0.9500 | 0.0677 | 25.74 |
| marginal | Bad→Good | 0.75 | 0.1507 | 0.7675 | 0.1035 | -12.11 |
| marginal | Bad→Good | 0.8 | 0.1752 | 0.7347 | 0.1031 | -17.97 |
| marginal | Bad→Good | 0.85 | 0.2165 | 0.7270 | 0.1035 | -23.36 |
| marginal | Bad→Good | 0.9 | 0.2288 | 0.6862 | 0.1023 | -31.15 |
| marginal | Bad→Good | 0.95 | 0.2299 | 0.6641 | 0.1042 | -31.33 |
| marginal | Bad→Good | 1.0 | 0.2533 | 0.6446 | 0.0963 | -37.44 |
| joint | Good→Bad | 0.0 | 0.9788 | 0.9761 | 0.0128 | 48.87 |
| joint | Good→Bad | 0.5 | 0.6016 | 0.8694 | 0.0188 | 38.65 |
| joint | Good→Bad | 0.75 | 0.4754 | 0.7634 | 0.0226 | 29.48 |
| joint | Good→Bad | 0.8 | 0.4397 | 0.7476 | 0.0263 | 28.37 |
| joint | Good→Bad | 0.85 | 0.4208 | 0.7344 | 0.0241 | 26.26 |
| joint | Good→Bad | 0.9 | 0.4096 | 0.7163 | 0.0275 | 23.81 |
| joint | Good→Bad | 0.95 | 0.3806 | 0.6895 | 0.0305 | 22.11 |
| joint | Good→Bad | 1.0 | 0.3460 | 0.6678 | 0.0309 | 18.86 |
| joint | Bad→Good | 0.0 | 0.0112 | 0.9831 | 0.0135 | 39.26 |
| joint | Bad→Good | 0.5 | 0.0960 | 0.6015 | 0.0767 | 3.58 |
| joint | Bad→Good | 0.75 | 0.1496 | 0.5081 | 0.1076 | -15.94 |
| joint | Bad→Good | 0.8 | 0.1696 | 0.4834 | 0.1046 | -21.88 |
| joint | Bad→Good | 0.85 | 0.1663 | 0.4823 | 0.1072 | -25.51 |
| joint | Bad→Good | 0.9 | 0.1674 | 0.4639 | 0.1110 | -30.97 |
| joint | Bad→Good | 0.95 | 0.1819 | 0.4452 | 0.1151 | -34.23 |
| joint | Bad→Good | 1.0 | 0.1696 | 0.4157 | 0.1151 | -39.27 |

| Representation | Direction | Selected λ | Criterion met |
|---|---|---:|---|
| marginal | Good→Bad | 1.0 | False |
| marginal | Bad→Good | 1.0 | False |
| joint | Good→Bad | 1.0 | False |
| joint | Bad→Good | 1.0 | False |

## 20261002T191712Z-7f1dc1ef seed 3

### Pooled confirmation: all sites and conditions

| Site | Representation | Direction | Condition | Target pickup | Other Good | Other Bad | Return |
|---|---|---|---|---:|---:|---:|---:|
| 0 | marginal | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 0 | marginal | Good→Bad | steered | 0.9056 | 0.9797 | 0.0112 | 47.99 |
| 0 | marginal | Good→Bad | control | 0.9750 | 0.9750 | 0.0137 | 48.61 |
| 0 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | marginal | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 0 | marginal | Bad→Good | steered | 0.0615 | 0.9778 | 0.0195 | 37.19 |
| 0 | marginal | Bad→Good | control | 0.0135 | 0.9764 | 0.0130 | 38.77 |
| 0 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 0 | joint | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 0 | joint | Good→Bad | steered | 0.7439 | 0.8321 | 0.0282 | -14.80 |
| 0 | joint | Good→Bad | control | 0.8072 | 0.8064 | 0.0134 | 41.15 |
| 0 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | joint | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 0 | joint | Bad→Good | steered | 0.1737 | 0.5134 | 0.0844 | -5.34 |
| 0 | joint | Bad→Good | control | 0.0674 | 0.3691 | 0.0128 | 14.65 |
| 0 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | marginal | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 1 | marginal | Good→Bad | steered | 0.8691 | 0.9822 | 0.0109 | 47.54 |
| 1 | marginal | Good→Bad | control | 0.9761 | 0.9785 | 0.0143 | 48.60 |
| 1 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | marginal | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 1 | marginal | Bad→Good | steered | 0.0728 | 0.9643 | 0.0214 | 35.70 |
| 1 | marginal | Bad→Good | control | 0.0149 | 0.9774 | 0.0147 | 38.66 |
| 1 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | joint | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 1 | joint | Good→Bad | steered | 0.6053 | 0.8913 | 0.0220 | 38.79 |
| 1 | joint | Good→Bad | control | 0.8235 | 0.9134 | 0.0311 | 41.92 |
| 1 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | joint | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 1 | joint | Bad→Good | steered | 0.1702 | 0.4378 | 0.0671 | 6.32 |
| 1 | joint | Bad→Good | control | 0.0621 | 0.3646 | 0.0385 | 9.97 |
| 1 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | marginal | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 2 | marginal | Good→Bad | steered | 0.3472 | 0.5610 | 0.0036 | 25.00 |
| 2 | marginal | Good→Bad | control | 0.5198 | 0.6450 | 0.0464 | 25.94 |
| 2 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | marginal | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 2 | marginal | Bad→Good | steered | 0.2610 | 0.6386 | 0.0985 | -34.36 |
| 2 | marginal | Bad→Good | control | 0.0222 | 0.4145 | 0.0549 | 11.90 |
| 2 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | joint | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 2 | joint | Good→Bad | steered | 0.3446 | 0.6652 | 0.0289 | 19.07 |
| 2 | joint | Good→Bad | control | 0.4802 | 0.5807 | 0.0299 | 12.32 |
| 2 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | joint | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 2 | joint | Bad→Good | steered | 0.1892 | 0.4226 | 0.1156 | -38.74 |
| 2 | joint | Bad→Good | control | 0.0188 | 0.2245 | 0.0078 | 11.25 |
| 2 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | marginal | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 3 | marginal | Good→Bad | steered | 0.8472 | 0.9688 | 0.0111 | 46.35 |
| 3 | marginal | Good→Bad | control | 0.7875 | 0.9196 | 0.0171 | 42.62 |
| 3 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | marginal | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 3 | marginal | Bad→Good | steered | 0.0427 | 0.8721 | 0.0231 | 30.41 |
| 3 | marginal | Bad→Good | control | 0.0379 | 0.8068 | 0.0174 | 28.00 |
| 3 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | joint | Good→Bad | intact | 0.9798 | 0.9790 | 0.0147 | 48.77 |
| 3 | joint | Good→Bad | steered | 0.4768 | 0.7598 | 0.0164 | 31.01 |
| 3 | joint | Good→Bad | control | 0.3739 | 0.4892 | 0.0238 | 17.57 |
| 3 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | joint | Bad→Good | intact | 0.0133 | 0.9803 | 0.0137 | 38.94 |
| 3 | joint | Bad→Good | steered | 0.1176 | 0.4091 | 0.0524 | 8.12 |
| 3 | joint | Bad→Good | control | 0.0188 | 0.1078 | 0.0248 | 1.20 |
| 3 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |

### Held-out decoder R² (first eight decisions excluded)

| Site | Marginal R², rocks 0–6 | Joint overall R² |
|---|---|---:|
| 0 | 0.533, 0.484, 0.253, 0.404, 0.564, -0.002, 0.724 | 0.3209 |
| 1 | 0.706, 0.619, 0.382, 0.786, 0.664, 0.182, 0.853 | 0.4183 |
| 2 | 0.909, 0.936, 0.584, 0.930, 0.932, 0.160, 0.945 | 0.6017 |
| 3 | 0.921, 0.943, 0.607, 0.941, 0.940, 0.217, 0.954 | 0.6168 |

### Per-rock steered pickup: marginal, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.317 [0.290, 0.347] | 0.642 [0.612, 0.670] |
| 1 | 0.266 [0.239, 0.294] | 0.341 [0.312, 0.370] |
| 2 | 0.854 [0.831, 0.874] | 0.065 [0.052, 0.082] |
| 3 | 0.167 [0.145, 0.191] | 0.062 [0.049, 0.079] |
| 4 | 0.000 [0.000, 0.004] | 0.010 [0.005, 0.018] |
| 5 | 0.814 [0.789, 0.837] | 0.627 [0.597, 0.656] |
| 6 | 0.013 [0.007, 0.022] | 0.080 [0.065, 0.098] |

### Per-rock steered pickup: joint, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.191 [0.168, 0.217] | 0.423 [0.393, 0.453] |
| 1 | 0.302 [0.274, 0.331] | 0.197 [0.174, 0.223] |
| 2 | 0.731 [0.703, 0.758] | 0.001 [0.000, 0.006] |
| 3 | 0.425 [0.395, 0.455] | 0.361 [0.332, 0.391] |
| 4 | 0.032 [0.023, 0.045] | 0.039 [0.029, 0.053] |
| 5 | 0.621 [0.591, 0.650] | 0.240 [0.215, 0.267] |
| 6 | 0.109 [0.092, 0.130] | 0.062 [0.049, 0.079] |

### Per-rock steered pickup: marginal, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.947 [0.932, 0.959] | 0.030 [0.021, 0.043] |
| 1 | 0.950 [0.935, 0.962] | 0.030 [0.021, 0.043] |
| 2 | 0.948 [0.933, 0.960] | 0.013 [0.007, 0.022] |
| 3 | 0.951 [0.936, 0.963] | 0.021 [0.014, 0.032] |
| 4 | 0.713 [0.684, 0.740] | 0.277 [0.251, 0.306] |
| 5 | 0.960 [0.946, 0.970] | 0.022 [0.015, 0.033] |
| 6 | 0.869 [0.847, 0.888] | 0.036 [0.026, 0.049] |

### Per-rock steered pickup: joint, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.756 [0.729, 0.781] | 0.225 [0.200, 0.251] |
| 1 | 0.911 [0.892, 0.927] | 0.030 [0.021, 0.043] |
| 2 | 0.915 [0.896, 0.931] | 0.373 [0.344, 0.403] |
| 3 | 0.931 [0.913, 0.945] | 0.233 [0.209, 0.260] |
| 4 | 0.083 [0.068, 0.102] | 0.217 [0.193, 0.243] |
| 5 | 0.883 [0.862, 0.901] | 0.028 [0.020, 0.040] |
| 6 | 0.729 [0.700, 0.755] | 0.109 [0.092, 0.130] |

## 20261002T182024Z-ad0c3a34 seed 2

### Pooled confirmation: all sites and conditions

| Site | Representation | Direction | Condition | Target pickup | Other Good | Other Bad | Return |
|---|---|---|---|---:|---:|---:|---:|
| 0 | marginal | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 0 | marginal | Good→Bad | steered | 0.9009 | 0.9682 | 0.0085 | 47.27 |
| 0 | marginal | Good→Bad | control | 0.9760 | 0.9745 | 0.0105 | 48.54 |
| 0 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | marginal | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 0 | marginal | Bad→Good | steered | 0.0198 | 0.9776 | 0.0145 | 38.19 |
| 0 | marginal | Bad→Good | control | 0.0123 | 0.9784 | 0.0093 | 38.85 |
| 0 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 0 | joint | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 0 | joint | Good→Bad | steered | 0.6364 | 0.7216 | 0.0915 | 16.21 |
| 0 | joint | Good→Bad | control | 0.4993 | 0.5161 | 0.0504 | -28.84 |
| 0 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | joint | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 0 | joint | Bad→Good | steered | 0.0882 | 0.4588 | 0.0607 | -6.47 |
| 0 | joint | Bad→Good | control | 0.0315 | 0.4054 | 0.0227 | 15.94 |
| 0 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | marginal | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 1 | marginal | Good→Bad | steered | 0.8200 | 0.9713 | 0.0089 | 46.42 |
| 1 | marginal | Good→Bad | control | 0.9608 | 0.9785 | 0.0158 | 47.78 |
| 1 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | marginal | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 1 | marginal | Bad→Good | steered | 0.0449 | 0.9686 | 0.0202 | 35.61 |
| 1 | marginal | Bad→Good | control | 0.0123 | 0.9767 | 0.0103 | 38.26 |
| 1 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | joint | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 1 | joint | Good→Bad | steered | 0.5226 | 0.5947 | 0.0545 | 19.63 |
| 1 | joint | Good→Bad | control | 0.5113 | 0.5020 | 0.0265 | 22.66 |
| 1 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | joint | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 1 | joint | Bad→Good | steered | 0.2146 | 0.5542 | 0.0617 | -12.63 |
| 1 | joint | Bad→Good | control | 0.0077 | 0.3918 | 0.0076 | 16.13 |
| 1 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | marginal | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 2 | marginal | Good→Bad | steered | 0.4574 | 0.6570 | 0.0053 | 29.28 |
| 2 | marginal | Good→Bad | control | 0.5607 | 0.7045 | 0.0476 | 25.39 |
| 2 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | marginal | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 2 | marginal | Bad→Good | steered | 0.2229 | 0.8153 | 0.1078 | 2.52 |
| 2 | marginal | Bad→Good | control | 0.0120 | 0.6472 | 0.0516 | 19.39 |
| 2 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | joint | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 2 | joint | Good→Bad | steered | 0.3100 | 0.5179 | 0.0528 | 12.07 |
| 2 | joint | Good→Bad | control | 0.3245 | 0.4101 | 0.0242 | 18.58 |
| 2 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | joint | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 2 | joint | Bad→Good | steered | 0.2393 | 0.4055 | 0.0784 | -0.68 |
| 2 | joint | Bad→Good | control | 0.0038 | 0.2124 | 0.0505 | -4.35 |
| 2 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | marginal | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 3 | marginal | Good→Bad | steered | 0.8168 | 0.9645 | 0.0081 | 45.53 |
| 3 | marginal | Good→Bad | control | 0.6526 | 0.8310 | 0.0323 | 35.81 |
| 3 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | marginal | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 3 | marginal | Bad→Good | steered | 0.0638 | 0.8855 | 0.0210 | 31.27 |
| 3 | marginal | Bad→Good | control | 0.0165 | 0.7823 | 0.0253 | 28.41 |
| 3 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | joint | Good→Bad | intact | 0.9802 | 0.9808 | 0.0095 | 48.86 |
| 3 | joint | Good→Bad | steered | 0.3919 | 0.5394 | 0.0589 | 0.43 |
| 3 | joint | Good→Bad | control | 0.2998 | 0.3722 | 0.0033 | 19.49 |
| 3 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | joint | Bad→Good | intact | 0.0092 | 0.9801 | 0.0081 | 39.06 |
| 3 | joint | Bad→Good | steered | 0.1327 | 0.3453 | 0.0330 | 5.38 |
| 3 | joint | Bad→Good | control | 0.0008 | 0.0787 | 0.0043 | 7.28 |
| 3 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |

### Held-out decoder R² (first eight decisions excluded)

| Site | Marginal R², rocks 0–6 | Joint overall R² |
|---|---|---:|
| 0 | 0.525, 0.482, 0.340, 0.517, 0.624, -0.206, 0.540 | 0.3142 |
| 1 | 0.729, 0.859, 0.622, 0.657, 0.738, -0.228, 0.742 | 0.4193 |
| 2 | 0.911, 0.930, 0.719, 0.908, 0.945, -0.372, 0.941 | 0.5926 |
| 3 | 0.919, 0.939, 0.733, 0.916, 0.948, -0.365, 0.948 | 0.6044 |

### Per-rock steered pickup: marginal, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.355 [0.327, 0.385] | 0.755 [0.728, 0.780] |
| 1 | 0.684 [0.654, 0.711] | 0.226 [0.201, 0.252] |
| 2 | 0.938 [0.921, 0.951] | 0.044 [0.033, 0.058] |
| 3 | 0.249 [0.224, 0.276] | 0.384 [0.354, 0.414] |
| 4 | 0.005 [0.002, 0.011] | 0.009 [0.005, 0.017] |
| 5 | 0.957 [0.943, 0.968] | 0.024 [0.017, 0.036] |
| 6 | 0.015 [0.009, 0.024] | 0.119 [0.101, 0.140] |

### Per-rock steered pickup: joint, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.127 [0.108, 0.149] | 0.782 [0.756, 0.806] |
| 1 | 0.009 [0.005, 0.017] | 0.319 [0.292, 0.349] |
| 2 | 0.630 [0.600, 0.659] | 0.064 [0.051, 0.081] |
| 3 | 0.362 [0.333, 0.392] | 0.214 [0.190, 0.240] |
| 4 | 0.040 [0.030, 0.054] | 0.040 [0.030, 0.054] |
| 5 | 0.954 [0.939, 0.965] | 0.250 [0.224, 0.277] |
| 6 | 0.048 [0.036, 0.063] | 0.005 [0.002, 0.011] |

### Per-rock steered pickup: marginal, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.948 [0.933, 0.960] | 0.013 [0.007, 0.022] |
| 1 | 0.952 [0.937, 0.964] | 0.020 [0.013, 0.030] |
| 2 | 0.929 [0.911, 0.943] | 0.026 [0.018, 0.038] |
| 3 | 0.726 [0.697, 0.752] | 0.021 [0.014, 0.032] |
| 4 | 0.842 [0.818, 0.863] | 0.026 [0.018, 0.038] |
| 5 | 0.956 [0.942, 0.967] | 0.022 [0.015, 0.033] |
| 6 | 0.954 [0.939, 0.965] | 0.010 [0.005, 0.018] |

### Per-rock steered pickup: joint, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.906 [0.887, 0.923] | 0.069 [0.055, 0.087] |
| 1 | 0.169 [0.147, 0.193] | 0.181 [0.158, 0.205] |
| 2 | 0.902 [0.883, 0.919] | 0.213 [0.189, 0.239] |
| 3 | 0.595 [0.564, 0.624] | 0.021 [0.013, 0.031] |
| 4 | 0.268 [0.241, 0.296] | 0.000 [0.000, 0.004] |
| 5 | 0.956 [0.942, 0.967] | 0.133 [0.113, 0.155] |
| 6 | 0.659 [0.630, 0.688] | 0.001 [0.000, 0.006] |

## 20261002T191712Z-7f1dc1ef seed 2

### Pooled confirmation: all sites and conditions

| Site | Representation | Direction | Condition | Target pickup | Other Good | Other Bad | Return |
|---|---|---|---|---:|---:|---:|---:|
| 0 | marginal | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 0 | marginal | Good→Bad | steered | 0.8934 | 0.9550 | 0.0137 | 46.72 |
| 0 | marginal | Good→Bad | control | 0.9616 | 0.9503 | 0.0144 | 47.29 |
| 0 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | marginal | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 0 | marginal | Bad→Good | steered | 0.0550 | 0.9556 | 0.0188 | 36.17 |
| 0 | marginal | Bad→Good | control | 0.0179 | 0.9637 | 0.0127 | 37.97 |
| 0 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 0 | joint | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 0 | joint | Good→Bad | steered | 0.5960 | 0.7773 | 0.0647 | 29.96 |
| 0 | joint | Good→Bad | control | 0.6197 | 0.6739 | 0.0121 | 31.60 |
| 0 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 0 | joint | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 0 | joint | Bad→Good | steered | 0.0724 | 0.3572 | 0.0540 | 6.72 |
| 0 | joint | Bad→Good | control | 0.1444 | 0.3852 | 0.0141 | 10.07 |
| 0 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | marginal | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 1 | marginal | Good→Bad | steered | 0.7722 | 0.9399 | 0.0110 | 44.98 |
| 1 | marginal | Good→Bad | control | 0.9272 | 0.9627 | 0.0247 | 46.10 |
| 1 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | marginal | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 1 | marginal | Bad→Good | steered | 0.0831 | 0.9401 | 0.0281 | 33.97 |
| 1 | marginal | Bad→Good | control | 0.0169 | 0.9554 | 0.0225 | 35.97 |
| 1 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 1 | joint | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 1 | joint | Good→Bad | steered | 0.5149 | 0.6518 | 0.0368 | 31.50 |
| 1 | joint | Good→Bad | control | 0.6549 | 0.6633 | 0.0448 | 23.61 |
| 1 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 1 | joint | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 1 | joint | Bad→Good | steered | 0.0543 | 0.2123 | 0.0503 | 2.91 |
| 1 | joint | Bad→Good | control | 0.0011 | 0.2216 | 0.0519 | 0.27 |
| 1 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | marginal | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 2 | marginal | Good→Bad | steered | 0.4909 | 0.6226 | 0.0060 | 30.22 |
| 2 | marginal | Good→Bad | control | 0.6099 | 0.7090 | 0.0434 | 26.86 |
| 2 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | marginal | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 2 | marginal | Bad→Good | steered | 0.1773 | 0.6243 | 0.1299 | -46.05 |
| 2 | marginal | Bad→Good | control | 0.0204 | 0.6043 | 0.0485 | 13.53 |
| 2 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 2 | joint | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 2 | joint | Good→Bad | steered | 0.4307 | 0.6562 | 0.0433 | 29.29 |
| 2 | joint | Good→Bad | control | 0.4692 | 0.4566 | 0.0112 | 17.58 |
| 2 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 2 | joint | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 2 | joint | Bad→Good | steered | 0.1011 | 0.2699 | 0.1312 | -24.67 |
| 2 | joint | Bad→Good | control | 0.0357 | 0.0558 | 0.0121 | -17.39 |
| 2 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | marginal | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 3 | marginal | Good→Bad | steered | 0.8305 | 0.9630 | 0.0107 | 46.29 |
| 3 | marginal | Good→Bad | control | 0.9537 | 0.9554 | 0.0209 | 47.38 |
| 3 | marginal | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | marginal | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 3 | marginal | Bad→Good | steered | 0.0600 | 0.8178 | 0.0248 | 27.71 |
| 3 | marginal | Bad→Good | control | 0.0162 | 0.9053 | 0.0210 | 35.95 |
| 3 | marginal | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |
| 3 | joint | Good→Bad | intact | 0.9750 | 0.9739 | 0.0134 | 48.57 |
| 3 | joint | Good→Bad | steered | 0.4789 | 0.6167 | 0.0374 | 28.06 |
| 3 | joint | Good→Bad | control | 0.5871 | 0.6056 | 0.0183 | 30.34 |
| 3 | joint | Good→Bad | random | 0.1967 | 0.1959 | 0.1975 | -2.35 |
| 3 | joint | Bad→Good | intact | 0.0138 | 0.9739 | 0.0131 | 38.73 |
| 3 | joint | Bad→Good | steered | 0.1159 | 0.1752 | 0.0648 | -108.06 |
| 3 | joint | Bad→Good | control | 0.0031 | 0.1773 | 0.0308 | 7.85 |
| 3 | joint | Bad→Good | random | 0.1967 | 0.1959 | 0.1975 | -6.29 |

### Held-out decoder R² (first eight decisions excluded)

| Site | Marginal R², rocks 0–6 | Joint overall R² |
|---|---|---:|
| 0 | 0.538, 0.482, 0.382, 0.459, 0.626, -0.039, 0.593 | 0.3741 |
| 1 | 0.652, 0.645, 0.475, 0.802, 0.820, -0.016, 0.893 | 0.5215 |
| 2 | 0.855, 0.942, 0.559, 0.903, 0.931, -0.032, 0.933 | 0.6218 |
| 3 | 0.868, 0.951, 0.583, 0.919, 0.949, -0.025, 0.953 | 0.6433 |

### Per-rock steered pickup: marginal, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.462 [0.432, 0.493] | 0.727 [0.698, 0.753] |
| 1 | 0.422 [0.392, 0.452] | 0.216 [0.192, 0.242] |
| 2 | 0.941 [0.925, 0.954] | 0.030 [0.021, 0.043] |
| 3 | 0.643 [0.613, 0.671] | 0.094 [0.077, 0.113] |
| 4 | 0.005 [0.002, 0.011] | 0.003 [0.001, 0.009] |
| 5 | 0.964 [0.951, 0.974] | 0.111 [0.093, 0.132] |
| 6 | 0.000 [0.000, 0.004] | 0.061 [0.048, 0.077] |

### Per-rock steered pickup: joint, site 2

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.162 [0.141, 0.186] | 0.017 [0.010, 0.026] |
| 1 | 0.447 [0.417, 0.478] | 0.150 [0.130, 0.174] |
| 2 | 0.674 [0.645, 0.702] | 0.028 [0.020, 0.040] |
| 3 | 0.687 [0.657, 0.714] | 0.188 [0.166, 0.214] |
| 4 | 0.028 [0.020, 0.040] | 0.186 [0.163, 0.211] |
| 5 | 0.970 [0.957, 0.979] | 0.026 [0.018, 0.038] |
| 6 | 0.047 [0.036, 0.062] | 0.112 [0.094, 0.133] |

### Per-rock steered pickup: marginal, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.955 [0.941, 0.966] | 0.017 [0.010, 0.026] |
| 1 | 0.956 [0.942, 0.967] | 0.023 [0.016, 0.035] |
| 2 | 0.938 [0.922, 0.952] | 0.015 [0.009, 0.024] |
| 3 | 0.916 [0.897, 0.931] | 0.049 [0.037, 0.064] |
| 4 | 0.724 [0.695, 0.750] | 0.212 [0.188, 0.238] |
| 5 | 0.969 [0.956, 0.978] | 0.022 [0.015, 0.033] |
| 6 | 0.796 [0.770, 0.819] | 0.047 [0.036, 0.062] |

### Per-rock steered pickup: joint, site 0

| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |
|---|---:|---:|
| 0 | 0.256 [0.230, 0.283] | 0.097 [0.080, 0.116] |
| 1 | 0.938 [0.921, 0.951] | 0.019 [0.012, 0.029] |
| 2 | 0.856 [0.834, 0.877] | 0.116 [0.098, 0.137] |
| 3 | 0.874 [0.852, 0.893] | 0.111 [0.093, 0.132] |
| 4 | 0.190 [0.168, 0.216] | 0.108 [0.091, 0.129] |
| 5 | 0.969 [0.956, 0.978] | 0.003 [0.001, 0.009] |
| 6 | 0.089 [0.073, 0.108] | 0.053 [0.041, 0.068] |

## Interpretation and reporting limits

Sites 0/1/2 are after blocks 1/2/3 before final LayerNorm; site 3 is after LayerNorm.
Pickup intervals are Wilson binomial 95% intervals for each target rock, not training-seed uncertainty.
Nontarget rates pool the other six rocks by their initial quality, not by the forced target quality.
Joint overall R² is 1 minus summed squared error over all reconstructed 128 coordinates divided
by summed held-out coordinate-wise variance. It is not the mean of coordinate-wise R² values.
Decoded joint vectors and targets can be signed: no clipping, simplex projection or renormalization
is added beyond reconstructing the omitted all-Good coordinate. Targets sum to one algebraically.
Controls project random activation directions into each fitted decoder's numerical null space
(relative SVD threshold 1e-6) and match the shift norm on their own current histories.
Null dimensions and float32 orthogonality errors are recorded in results.json; the joint decoder
may leave very few null dimensions. Endpoint calibration and nuisance regression are absent.
Intact and random-action baselines are reused across representations/sites because their policies
do not depend on the intervention. Random actions are uniform over all 12 actions, not a random
trained policy. Evaluation includes reset and the first eight decisions. Raw arrays stay in artifacts.

## Comparison with nuisance-controlled v3/v4

Primary agent (run 2, seed 3). Each entry is Good→Bad / Bad→Good; other-Good/Bad
rates condition the six nontarget rocks on their initial quality. Historical rows
come from the committed v3/v4 findings, not a new matched-seed rerun.

| Protocol | Site | Target pickup | Other Good | Other Bad | Return |
|---|---:|---:|---:|---:|---:|
| v3 nuisance-controlled, calibrated | 2 | 0.2681 / 0.3277 | 0.6468 / 0.7142 | 0.0043 / 0.0886 | 27.02 / -14.36 |
| v4 nuisance-controlled, calibrated | 0 | 0.1452 / 0.6860 | 0.9633 / 0.9072 | 0.0100 / 0.0327 | 39.08 / 23.36 |
| v5 plain marginal | 2 | 0.3472 / 0.2610 | 0.5610 / 0.6386 | 0.0036 / 0.0985 | 25.00 / -34.36 |
| v5 plain joint | 2 | 0.3446 / 0.1892 | 0.6652 / 0.4226 | 0.0289 / 0.1156 | 19.07 / -38.74 |
| v5 plain marginal | 0 | 0.9056 / 0.0615 | 0.9797 / 0.9778 | 0.0112 / 0.0195 | 47.99 / 37.19 |
| v5 plain joint | 0 | 0.7439 / 0.1737 | 0.8321 / 0.5134 | 0.0282 / 0.0844 | -14.80 / -5.34 |

V4 was the held-out test of the site-0 choice made after inspecting v3. V5 does
not reproduce that strong site-0 pattern with the plain paper recipe: marginal
steering is comparatively mild, while joint steering has larger target effects
but much poorer selectivity and return. This comparison changes calibration,
nuisance adjustment, warmup fitting/scoring, and evaluation seed together; it
does not identify which change explains the difference. All v5 sites and both
representations are reported; confirmation outcomes did not select a new λ.

At the paper's site, all three agents miss both pooled target criteria for both
representations at their frozen strengths. Marginal Good→Bad pickup is
0.3472/0.4574/0.4909 and joint is 0.3446/0.3100/0.4307 (agent order as above),
all above the confirmation random baseline 0.1967. Marginal Bad→Good is
0.2610/0.2229/0.1773 and joint is 0.1892/0.2393/0.1011, all below 0.5.
Joint has lower Good→Bad pickup, but the primary-agent difference is only
0.0026 and is not evidence of a reliable advantage. Joint Bad→Good target
pickup is worse on two agents and slightly higher on one; nontarget Good pickup
is worse on every agent (0.4226/0.4055/0.2699 versus marginal
0.6386/0.8153/0.6243). Matched-null controls also often disrupt pickup, so a
change from intact alone is not sufficient evidence of selective steering.

Decoder accessibility is heterogeneous: primary site-2 marginal R² ranges from
0.160 to 0.945, and rock-index 5 has negative marginal R² on both replicate
agents. Joint overall site-2 R² is 0.602/0.593/0.622. These held-out scores
describe linear prediction, not causal sufficiency, and do not establish the
cause of steering failures. The joint decoder's numerical null space has only
3–14 dimensions across agents/sites, unlike the much wider marginal null
space; matched-null controls are consequently not identical perturbation
families across representations. Decoded signed joint vectors need not be
probability distributions despite their algebraic normalization.

### Overall conclusion

The paper's plain least-squares recipe causes directional changes in RockSample
pickup, but this port does not achieve its predeclared pooled two-way target
criteria or a clean selective seven-rock belief flip. Some effects are large:
at the paper's site, marginal Bad→Good raises rock-index 0 pickup to
0.642/0.755/0.727 across the three agents, while marginal Good→Bad nearly
eliminates pickup of rock-indices 4 and 6. Those effects coexist with substantial
nontarget disruption and sometimes strongly negative return. Joint does not
consistently beat marginal: it suppresses Good pickup more on some agents/sites,
but fails the pooled criteria, often impairs selectivity, and is not reliably
better for Bad→Good. The more selective held-out site-0 result remains the
nuisance-controlled, calibrated v4 protocol, not this paper-faithful port.
Neither linear decodability nor these interventions prove an internal Bayesian
model or isolate a nuisance-free causal belief representation.
