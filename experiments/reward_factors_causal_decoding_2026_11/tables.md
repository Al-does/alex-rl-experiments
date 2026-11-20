# Seed-42 CPU pilot

Two fit, tuning and confirmation episodes; 1,024 decisions each.
No uncertainty or population inference is justified. Eq. (4) retains archived strengths,
but its pilot decoder fits use fewer rows than the archive.

## Closed loop

Ladder cells: rho-flip occupancy / delta-mirror actions changed. Other cells: reward per step.

| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State control |
|---|---:|---:|---:|---:|---:|
| intact | 0.3130 / 0.0000 | 0.8867 | 0.5947 | 0.5034 | 0.2285 / 0.0000 |
| eq4 | 0.0869 / 0.0117 | 1.1816 | 0.6406 | 0.5518 | 0.1406 / 0.5596 |
| eq4_joint | 0.0869 / 0.0117 | 1.2217 | 0.5947 | 0.6924 | 0.1406 / 0.5596 |
| policy | 0.2490 / 0.0010 | 0.9658 | 0.6299 | 0.6030 | 0.2979 / 0.0259 |
| das | 0.1665 / 0.1655 | 0.9478 | 0.5508 | 0.7021 | 0.2153 / 0.5693 |
| band | 0.3130 / 0.0000 | 0.8950 | 0.6313 | 0.5034 | 0.2285 / 0.0024 |
| mlp | 0.3130 / 0.0000 | 0.8452 | 0.6191 | 0.5034 | 0.2285 / 0.0083 |
| null | 0.0869 / 0.0117 | 1.0879 | 0.7104 | 0.6064 | 0.1406 / 0.5596 |
| norm_null | 0.3130 / 0.0000 | 0.8784 | 0.5659 | 0.4702 | 0.2690 / 0.0054 |
| map | — | 1.2358 | 0.5366 | 0.7129 | — |

## Moving transfer

Gap closed uses the *paired pilot* exact-filter map and intact controls.
| Method | Reward | Gap closed | Factor-1 reward | Factor-1 actions changed |
|---|---:|---:|---:|---:|
| intact | 0.8867 | 0.000 | 0.2983 | 0.0000 |
| eq4 | 1.1816 | 0.845 | 0.4731 | 0.1680 |
| eq4_joint | 1.2217 | 0.959 | 0.4683 | 0.0762 |
| policy | 0.9658 | 0.227 | 0.4766 | 0.0044 |
| das | 0.9478 | 0.175 | 0.3447 | 0.0312 |
| band | 0.8950 | 0.024 | 0.2783 | 0.0063 |
| mlp | 0.8452 | -0.119 | 0.2578 | 0.0010 |
| null | 1.0879 | 0.576 | 0.4429 | 0.0811 |
| norm_null | 0.8784 | -0.024 | 0.2769 | 0.0400 |
| map | 1.2358 | 1.000 | 0.4800 | 0.0649 |

## Held-out decoders

R² is pooled across belief probabilities; policy inverse entries are per-factor probability MAE.
| Test | Affine R² | Band R² | MLP R² | Policy-implied MAE | DAS rank | DAS block-4 fidelity / reference | Block-2 fidelity / reference (same rank) |
|---|---:|---:|---:|---|---:|---:|---:|
| ladder | 0.849 | 0.900 | 0.841 | 0.180 | 8 | 0.972 / 1.000 | 0.226 / 0.376 |
| moving | 0.833 | 0.824 | 0.885 | 0.121, 0.151 | 8 | 0.797 / 1.000 | 0.801 / 0.913 |
| xor_eps02 | 0.833 | 0.824 | 0.885 | 0.121, 0.151 | 8 | 0.825 / 1.000 | 0.834 / 0.905 |
| xor_v2 | 0.909 | 0.915 | 0.940 | 0.115, 0.124 | 8 | 0.869 / 1.000 | 0.782 / 0.773 |
| ladder_full | 0.975 | 0.973 | 0.976 | 0.106 | 8 | 0.947 / 1.000 | 0.664 / 0.611 |

Recorded final-job CPU allocation × wall time: 1.10 core-hours (not process CPU time; excludes discarded pilots).
