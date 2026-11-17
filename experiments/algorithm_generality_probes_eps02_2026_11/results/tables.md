# Algorithm generality: eps02 ladder probes

CPU-only, frozen final actors at step 2,012,217,344. Five-seed cohort: 42/44/46/49/52.

## Global normalized probe error (1−R²)

| Algorithm | Condition | b | ρ | δ | c[B] / merged target |
|---|---|---:|---:|---:|---:|
| REINFORCE (15) | Trivial Quotient | 0.1390 | 0.1125 | 0.2510 | — |
| REINFORCE (15) | Reward-State Quotient | 0.1019 | 0.0173 | 0.2405 | 0.0158 |
| REINFORCE (15) | Full-State Quotient | 0.0316 | 0.0275 | 0.0386 | 0.0453 |
| REINFORCE (5) | Trivial Quotient | 0.1513 | 0.1247 | 0.2642 | — |
| REINFORCE (5) | Reward-State Quotient | 0.1024 | 0.0166 | 0.2441 | 0.0150 |
| REINFORCE (5) | Full-State Quotient | 0.0297 | 0.0268 | 0.0348 | 0.0460 |
| PPO (5) | Trivial Quotient | 0.0362 | 0.0050 | 0.1679 | — |
| PPO (5) | Reward-State Quotient | 0.1080 | 0.0036 | 0.2820 | 0.0026 |
| PPO (5) | Full-State Quotient | 0.0140 | 0.0030 | 0.0327 | 0.0107 |
| SAC auto-alpha (5) | Trivial Quotient | 0.1114 | 0.0748 | 0.2664 | — |
| SAC auto-alpha (5) | Reward-State Quotient | 0.0522 | 0.0115 | 0.1194 | 0.0117 |
| SAC auto-alpha (5) | Full-State Quotient | 0.0418 | 0.0350 | 0.0530 | 0.0494 |

The last column is an exact quotient only for Reward-State; for Full-State it is the merged-token filter. Each target has its own variance denominator.

## Within-action-class R²: matched init → trained

| Algorithm | Condition | b | ρ | δ |
|---|---|---:|---:|---:|
| REINFORCE (15) | Trivial Quotient | 0.8065 → 0.8610 | 0.8323 → 0.8875 | 0.6966 → 0.7490 |
| REINFORCE (15) | Reward-State Quotient | 0.8854 → 0.8646 | 0.9021 → 0.9684 | 0.8748 → 0.7977 |
| REINFORCE (15) | Full-State Quotient | 0.8397 → 0.9512 | 0.8599 → 0.9674 | 0.7958 → 0.9158 |
| REINFORCE (5) | Trivial Quotient | 0.8029 → 0.8487 | 0.8284 → 0.8753 | 0.6916 → 0.7358 |
| REINFORCE (5) | Reward-State Quotient | 0.8863 → 0.8618 | 0.9025 → 0.9695 | 0.8770 → 0.7954 |
| REINFORCE (5) | Full-State Quotient | 0.8502 → 0.9549 | 0.8700 → 0.9702 | 0.8070 → 0.9213 |
| PPO (5) | Trivial Quotient | 0.8029 → 0.9638 | 0.8284 → 0.9950 | 0.6916 → 0.8321 |
| PPO (5) | Reward-State Quotient | 0.8845 → 0.8660 | 0.8988 → 0.9921 | 0.8741 → 0.7749 |
| PPO (5) | Full-State Quotient | 0.8489 → 0.9758 | 0.8662 → 0.9943 | 0.8110 → 0.9353 |
| SAC auto-alpha (5) | Trivial Quotient | -1.1465 → 0.8928 | 0.1929 → 0.9277 | -7.2924 → 0.7455 |
| SAC auto-alpha (5) | Reward-State Quotient | 0.8769 → 0.9380 | 0.8870 → 0.9866 | 0.8682 → 0.8969 |
| SAC auto-alpha (5) | Full-State Quotient | 0.8307 → 0.9487 | 0.8505 → 0.9670 | 0.7900 → 0.9112 |

Trivial Quotient is descriptive, not evidence for task-required action regions: most policies use one class. SAC seed 44 has one held-out rare action absent from fit (1/80,000 rows excluded); seed 46 has 4 fit / 11 test rare actions. All Reward-State and Full-State within-class scores have complete coverage.

## Per-seed results

Each triple is global error b/ρ/δ; within-class cells are matched-init → trained R² for b/ρ/δ.

| Algorithm | Condition | Seed | Global errors | Coarse error | Within-class R² |
|---|---|---:|---|---:|---|
| REINFORCE (5) | Trivial Quotient | 42 | 0.1330 / 0.1112 / 0.2240 | — | 0.819→0.867 / 0.847→0.889 / 0.704→0.776 |
| REINFORCE (5) | Trivial Quotient | 44 | 0.1340 / 0.0968 / 0.3047 | — | 0.753→0.866 / 0.813→0.903 / 0.474→0.695 |
| REINFORCE (5) | Trivial Quotient | 46 | 0.1254 / 0.1037 / 0.2096 | — | 0.833→0.875 / 0.845→0.896 / 0.787→0.790 |
| REINFORCE (5) | Trivial Quotient | 49 | 0.1401 / 0.1200 / 0.2148 | — | 0.804→0.860 / 0.815→0.880 / 0.764→0.785 |
| REINFORCE (5) | Trivial Quotient | 52 | 0.2239 / 0.1919 / 0.3677 | — | 0.806→0.776 / 0.823→0.808 / 0.729→0.632 |
| REINFORCE (5) | Reward-State Quotient | 42 | 0.1262 / 0.0180 / 0.3131 | 0.0164 | 0.889→0.839 / 0.908→0.966 / 0.877→0.753 |
| REINFORCE (5) | Reward-State Quotient | 44 | 0.1029 / 0.0187 / 0.2504 | 0.0170 | 0.890→0.857 / 0.909→0.966 / 0.877→0.787 |
| REINFORCE (5) | Reward-State Quotient | 46 | 0.1088 / 0.0178 / 0.2629 | 0.0159 | 0.897→0.844 / 0.905→0.967 / 0.893→0.766 |
| REINFORCE (5) | Reward-State Quotient | 49 | 0.1009 / 0.0103 / 0.2259 | 0.0094 | 0.894→0.870 / 0.927→0.980 / 0.878→0.818 |
| REINFORCE (5) | Reward-State Quotient | 52 | 0.0732 / 0.0180 / 0.1685 | 0.0163 | 0.862→0.899 / 0.864→0.969 / 0.860→0.853 |
| REINFORCE (5) | Full-State Quotient | 42 | 0.0280 / 0.0235 / 0.0358 | 0.0408 | 0.850→0.952 / 0.873→0.972 / 0.801→0.910 |
| REINFORCE (5) | Full-State Quotient | 44 | 0.0309 / 0.0280 / 0.0361 | 0.0462 | 0.842→0.953 / 0.858→0.966 / 0.806→0.923 |
| REINFORCE (5) | Full-State Quotient | 46 | 0.0305 / 0.0274 / 0.0359 | 0.0492 | 0.861→0.956 / 0.875→0.971 / 0.829→0.921 |
| REINFORCE (5) | Full-State Quotient | 49 | 0.0318 / 0.0300 / 0.0348 | 0.0481 | 0.870→0.952 / 0.902→0.968 / 0.802→0.918 |
| REINFORCE (5) | Full-State Quotient | 52 | 0.0274 / 0.0251 / 0.0313 | 0.0458 | 0.828→0.962 / 0.842→0.974 / 0.797→0.934 |
| PPO (5) | Trivial Quotient | 42 | 0.0346 / 0.0036 / 0.1641 | — | 0.819→0.965 / 0.847→0.996 / 0.704→0.836 |
| PPO (5) | Trivial Quotient | 44 | 0.0391 / 0.0052 / 0.1946 | — | 0.753→0.961 / 0.813→0.995 / 0.474→0.805 |
| PPO (5) | Trivial Quotient | 46 | 0.0362 / 0.0042 / 0.1605 | — | 0.833→0.964 / 0.845→0.996 / 0.787→0.840 |
| PPO (5) | Trivial Quotient | 49 | 0.0307 / 0.0083 / 0.1140 | — | 0.804→0.969 / 0.815→0.992 / 0.764→0.886 |
| PPO (5) | Trivial Quotient | 52 | 0.0403 / 0.0035 / 0.2061 | — | 0.806→0.960 / 0.823→0.997 / 0.729→0.794 |
| PPO (5) | Reward-State Quotient | 42 | 0.0849 / 0.0031 / 0.2240 | 0.0031 | 0.886→0.893 / 0.902→0.993 / 0.873→0.814 |
| PPO (5) | Reward-State Quotient | 44 | 0.1523 / 0.0039 / 0.4225 | 0.0020 | 0.889→0.793 / 0.912→0.991 / 0.873→0.644 |
| PPO (5) | Reward-State Quotient | 46 | 0.0863 / 0.0026 / 0.2292 | 0.0021 | 0.893→0.902 / 0.900→0.994 / 0.887→0.835 |
| PPO (5) | Reward-State Quotient | 49 | 0.0865 / 0.0031 / 0.2291 | 0.0022 | 0.882→0.889 / 0.896→0.993 / 0.872→0.811 |
| PPO (5) | Reward-State Quotient | 52 | 0.1300 / 0.0053 / 0.3051 | 0.0037 | 0.872→0.854 / 0.885→0.989 / 0.865→0.771 |
| PPO (5) | Full-State Quotient | 42 | 0.0166 / 0.0037 / 0.0378 | 0.0113 | 0.854→0.971 / 0.868→0.993 / 0.822→0.924 |
| PPO (5) | Full-State Quotient | 44 | 0.0149 / 0.0031 / 0.0353 | 0.0107 | 0.852→0.976 / 0.864→0.994 / 0.826→0.935 |
| PPO (5) | Full-State Quotient | 46 | 0.0115 / 0.0021 / 0.0276 | 0.0090 | 0.853→0.981 / 0.866→0.996 / 0.823→0.948 |
| PPO (5) | Full-State Quotient | 49 | 0.0128 / 0.0028 / 0.0297 | 0.0103 | 0.869→0.978 / 0.892→0.995 / 0.819→0.941 |
| PPO (5) | Full-State Quotient | 52 | 0.0144 / 0.0035 / 0.0332 | 0.0121 | 0.817→0.973 / 0.840→0.993 / 0.765→0.928 |
| SAC auto-alpha (5) | Trivial Quotient | 42 | 0.1052 / 0.0716 / 0.2457 | — | 0.819→0.895 / 0.847→0.928 / 0.704→0.754 |
| SAC auto-alpha (5) | Trivial Quotient | 44 | 0.1548 / 0.0968 / 0.4212 | — | -8.980→0.866 / -2.353→0.917 / -39.420→0.634 |
| SAC auto-alpha (5) | Trivial Quotient | 46 | 0.0861 / 0.0577 / 0.1966 | — | 0.819→0.914 / 0.833→0.941 / 0.762→0.808 |
| SAC auto-alpha (5) | Trivial Quotient | 49 | 0.1178 / 0.0854 / 0.2386 | — | 0.804→0.882 / 0.815→0.915 / 0.764→0.761 |
| SAC auto-alpha (5) | Trivial Quotient | 52 | 0.0931 / 0.0627 / 0.2298 | — | 0.806→0.907 / 0.823→0.937 / 0.729→0.770 |
| SAC auto-alpha (5) | Reward-State Quotient | 42 | 0.0596 / 0.0118 / 0.1356 | 0.0127 | 0.882→0.929 / 0.895→0.985 / 0.871→0.882 |
| SAC auto-alpha (5) | Reward-State Quotient | 44 | 0.0563 / 0.0086 / 0.1366 | 0.0087 | 0.876→0.934 / 0.894→0.990 / 0.860→0.884 |
| SAC auto-alpha (5) | Reward-State Quotient | 46 | 0.0523 / 0.0123 / 0.1186 | 0.0126 | 0.885→0.936 / 0.891→0.985 / 0.881→0.898 |
| SAC auto-alpha (5) | Reward-State Quotient | 49 | 0.0447 / 0.0128 / 0.0979 | 0.0119 | 0.885→0.948 / 0.896→0.988 / 0.876→0.914 |
| SAC auto-alpha (5) | Reward-State Quotient | 52 | 0.0482 / 0.0119 / 0.1084 | 0.0125 | 0.855→0.942 / 0.859→0.986 / 0.853→0.907 |
| SAC auto-alpha (5) | Full-State Quotient | 42 | 0.0442 / 0.0346 / 0.0595 | 0.0491 | 0.835→0.944 / 0.856→0.967 / 0.797→0.900 |
| SAC auto-alpha (5) | Full-State Quotient | 44 | 0.0446 / 0.0379 / 0.0559 | 0.0533 | 0.830→0.953 / 0.839→0.971 / 0.810→0.915 |
| SAC auto-alpha (5) | Full-State Quotient | 46 | 0.0405 / 0.0325 / 0.0543 | 0.0466 | 0.834→0.950 / 0.855→0.969 / 0.790→0.912 |
| SAC auto-alpha (5) | Full-State Quotient | 49 | 0.0354 / 0.0267 / 0.0499 | 0.0394 | 0.845→0.951 / 0.873→0.971 / 0.787→0.911 |
| SAC auto-alpha (5) | Full-State Quotient | 52 | 0.0443 / 0.0435 / 0.0455 | 0.0585 | 0.810→0.946 / 0.830→0.958 / 0.766→0.919 |

Head-blind and logit-only controls, all per-class scores and excluded-row counts are retained in archived agent JSONs (details_manifest.json); group means are in summary.json.

Global constrained-replacement events: 0. Raw global fits are retained in every agent JSON.
