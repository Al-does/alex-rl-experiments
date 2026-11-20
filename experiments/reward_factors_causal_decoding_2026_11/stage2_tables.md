# Stage 2: seed-42 local CPU pilot

Ladder and moving: eight episodes each for fit, held-out diagnosis, tuning and confirmation.
Other tests: two episodes per split. All episodes: 1,024 steps. Gradient writes: 60 steps.
Ladder cells: rho-flip occupancy / delta-mirror actions changed. Others: reward/step.

| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |
|---|---:|---:|---:|---:|---:|
| eq4 | 0.1282 / 0.0105 | 0.9600 | 0.6406 | 0.5518 | 0.1406 / 0.5596 |
| eq4_joint | 0.1282 / 0.0105 | 1.0144 | 0.5947 | 0.6924 | 0.1406 / 0.5596 |
| policy | 0.1687 / 0.0002 | 1.0188 | 0.5439 | 0.6602 | 0.2695 / 0.2148 |
| das | 0.1338 / 0.1095 | 0.9645 | 0.5508 | 0.7021 | 0.2153 / 0.5693 |
| band | 0.4480 / 0.0001 | 0.8120 | 0.6313 | 0.5034 | 0.2285 / 0.0034 |
| mlp | 0.4480 / 0.0001 | 0.8284 | 0.5913 | 0.5034 | 0.2407 / 0.0112 |
| null | 0.1058 / 0.0105 | 0.9905 | 0.7104 | 0.6064 | 0.1406 / 0.5596 |
| norm_null | 0.4480 / 0.0002 | 0.8521 | 0.5659 | 0.4702 | 0.2690 / 0.0054 |
| intact | 0.4480 / 0.0000 | 0.8284 | 0.5947 | 0.5034 | 0.2285 / 0.0000 |
| map | 0.4480 / 0.0000 | 1.0793 | 0.5366 | 0.7129 | 0.2285 / 0.0000 |
| context_exact | 0.4480 / 0.0000 | 0.8953 | 0.5947 | 0.5034 | 0.2285 / 0.0000 |
| context_policy | 0.4480 / 0.0000 | 0.8284 | 0.5947 | 0.5034 | 0.2285 / 0.0000 |
| context_exact_full | 0.1263 / 0.0193 | 0.9653 | 0.5674 | 0.6753 | 0.1255 / 0.5674 |
| context_policy_full | 0.4010 / 0.0070 | 1.0347 | 0.5044 | 0.5034 | 0.2285 / 0.4839 |

## Moving transfer

Gap uses paired intact and exact-filter map references.

| Method | Reward | Gap closed | Factor-1 reward | Factor-1 actions changed | Donor coverage |
|---|---:|---:|---:|---:|---:|
| eq4 | 0.9600 | 0.524 | 0.3055 | 0.1313 | 0.000 |
| eq4_joint | 1.0144 | 0.741 | 0.3353 | 0.0747 | 0.000 |
| policy | 1.0188 | 0.759 | 0.3619 | 0.0258 | 0.000 |
| das | 0.9645 | 0.542 | 0.3531 | 0.0592 | 0.000 |
| band | 0.8120 | -0.065 | 0.2574 | 0.0004 | 0.000 |
| mlp | 0.8284 | 0.000 | 0.2672 | 0.0000 | 0.000 |
| null | 0.9905 | 0.646 | 0.3317 | 0.0906 | 0.000 |
| norm_null | 0.8521 | 0.094 | 0.3596 | 0.0283 | 0.000 |
| intact | 0.8284 | 0.000 | 0.2672 | 0.0000 | 0.000 |
| map | 1.0793 | 1.000 | 0.4171 | 0.0769 | 0.000 |
| context_exact | 0.8953 | 0.267 | 0.3187 | 0.0020 | 1.000 |
| context_policy | 0.8284 | 0.000 | 0.2672 | 0.0000 | 0.698 |
| context_exact_full | 0.9653 | 0.546 | 0.3531 | 0.1177 | 1.000 |
| context_policy_full | 1.0347 | 0.822 | 0.3866 | 0.0641 | 0.688 |

## Policy-map calibration

KL direction is actual agent || map(Bayes), temperature 1.5.
Mean-policy KL uses a constant fitted on training policies; inverse KL uses map(decoded belief).

| Test | Map train KL | Map test KL | Map test TV | Mean-policy test KL | Inverse policy KL | Inverse–Bayes MAE/factor | rho MAE/factor | signed delta MAE/factor |
|---|---:|---:|---:|---:|---:|---|---|---|
| ladder | 0.09783 | 0.10180 | 0.08767 | 0.58982 | 0.00606 | 0.1788 | 0.1305 | 0.3468 |
| moving | 0.32328 | 0.29738 | 0.25054 | 1.22713 | 0.05126 | 0.1280, 0.1432 | 0.1499, 0.1633 | 0.1755, 0.2165 |
| xor_eps02 | 0.30746 | 0.35028 | 0.27706 | 1.36788 | 0.05201 | 0.1214, 0.1508 | 0.1342, 0.1624 | 0.1713, 0.2366 |
| xor_v2 | 0.27268 | 0.20055 | 0.18699 | 1.68213 | 0.03298 | 0.1154, 0.1242 | 0.1302, 0.1423 | 0.1571, 0.1661 |
| ladder_full | 0.06506 | 0.07390 | 0.07335 | 0.77768 | 0.00613 | 0.1060 | 0.1299 | 0.1149 |

## Passive decoders and DAS

| Test | Affine R² | Band R² | MLP R² | DAS rank | Block-4 held-out fidelity |
|---|---:|---:|---:|---:|---:|
| ladder | 0.877 | 0.927 | 0.846 | 8 | 0.969 |
| moving | 0.902 | 0.916 | 0.940 | 8 | 0.820 |
| xor_eps02 | 0.833 | 0.824 | 0.885 | 8 | 0.825 |
| xor_v2 | 0.909 | 0.915 | 0.940 | 8 | 0.869 |
| ladder_full | 0.975 | 0.973 | 0.976 | 8 | 0.947 |

## Wall seconds per method (two allocated CPU cores per worker)

Includes checkpoint loading, fitting, diagnostics, lambda selection, and confirmation.
Jobs share eight CPU cores; these are not isolated performance benchmarks.

| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |
|---|---:|---:|---:|---:|---:|
| eq4 | 14.2 | 11.0 | 2.4 | 5.2 | 6.8 |
| eq4_joint | 14.4 | 10.9 | 2.6 | 3.4 | 4.5 |
| policy | 639.5 | 977.0 | 709.6 | 722.7 | 304.0 |
| das | 88.7 | 64.5 | 49.3 | 53.3 | 38.0 |
| band | 26.9 | 30.1 | 14.2 | 15.3 | 14.4 |
| mlp | 30.2 | 26.9 | 15.4 | 19.1 | 19.1 |
| null | 26.1 | 18.7 | 6.5 | 6.6 | 12.2 |
| norm_null | 10.5 | 6.4 | 2.6 | 2.5 | 6.4 |
| intact | 10.2 | 5.3 | 2.3 | 2.5 | 6.3 |
| map | 9.1 | 5.0 | 2.4 | 2.5 | 6.4 |
| context_exact | 44.6 | 36.1 | 8.9 | 8.4 | 14.5 |
| context_policy | 51.4 | 46.1 | 13.3 | 12.2 | 20.5 |
| context_exact_full | 40.3 | 36.1 | 8.4 | 8.8 | 14.2 |
| context_policy_full | 48.0 | 46.3 | 11.7 | 12.4 | 16.2 |

Total allocated core-hours: 2.546 (excludes preliminary diagnostic runs).
