# Steering algorithm generality

## Ladder: full strength after block 4

Reward-state occupancy; parentheses are action-change fractions. Intact/random are controls.

| Algorithm / condition | Intact | Random | delta point | delta mirror | rho flip | rho point |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE (15): Trivial Quotient | 0.9428 | 0.1880 | 0.9428 (0.000) | 0.9428 (0.000) | 0.6879 (0.270) | 0.8100 (0.140) |
| REINFORCE (15): Reward-State Quotient | 0.4358 | 0.1057 | 0.4369 (0.007) | 0.4362 (0.005) | 0.0866 (0.496) | 0.0725 (0.664) |
| REINFORCE (15): Full-State Quotient | 0.4126 | 0.0961 | 0.1857 (0.538) | 0.1919 (0.455) | 0.0851 (0.441) | 0.0595 (0.546) |
| REINFORCE (5): Trivial Quotient | 0.9428 | 0.1880 | 0.9428 (0.000) | 0.9428 (0.000) | 0.9350 (0.007) | 0.9350 (0.007) |
| REINFORCE (5): Reward-State Quotient | 0.4360 | 0.1057 | 0.4328 (0.006) | 0.4323 (0.005) | 0.0948 (0.498) | 0.0717 (0.693) |
| REINFORCE (5): Full-State Quotient | 0.4152 | 0.0961 | 0.1792 (0.556) | 0.1857 (0.477) | 0.0842 (0.472) | 0.0574 (0.586) |
| PPO (5): Trivial Quotient | 0.9428 | 0.1880 | 0.9428 (0.000) | 0.9428 (0.000) | 0.9363 (0.002) | 0.9369 (0.002) |
| PPO (5): Reward-State Quotient | 0.4395 | 0.1057 | 0.4365 (0.003) | 0.4385 (0.002) | 0.0788 (0.499) | 0.0661 (0.585) |
| PPO (5): Full-State Quotient | 0.4100 | 0.0961 | 0.1870 (0.540) | 0.1973 (0.441) | 0.1119 (0.410) | 0.0739 (0.471) |
| SAC auto-alpha (5): Trivial Quotient | 0.9428 | 0.1880 | 0.9428 (0.000) | 0.9428 (0.000) | 0.5794 (0.396) | 0.8924 (0.032) |
| SAC auto-alpha (5): Reward-State Quotient | 0.4464 | 0.1057 | 0.4490 (0.008) | 0.4476 (0.006) | 0.1060 (0.731) | 0.0772 (0.812) |
| SAC auto-alpha (5): Full-State Quotient | 0.4071 | 0.0961 | 0.1974 (0.480) | 0.1840 (0.394) | 0.1138 (0.665) | 0.0760 (0.741) |

## Numerical exact-filter ladder reference (256 episodes)

| Condition | Intact | Random | delta point | delta mirror | rho flip | rho point |
|---|---:|---:|---:|---:|---:|---:|
| Trivial Quotient | 0.9358 | 0.1849 | 0.9358 | 0.9358 | 0.9358 | 0.9358 |
| Reward-State Quotient | 0.4356 | 0.1055 | 0.4152 | 0.4356 | 0.1242 | 0.0772 |
| Full-State Quotient | 0.4068 | 0.1046 | 0.1030 | 0.0713 | 0.1734 | 0.0780 |

## Ladder: tuning-selected steering and decoder-orthogonal controls

| Algorithm / condition | Mode | Block 4 | Null block 4 | Final norm | Null final norm |
|---|---|---:|---:|---:|---:|
| REINFORCE (15): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (15): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (15): Trivial Quotient | rho_flip | 0.7007 | 0.9278 | 0.7808 | 0.9385 |
| REINFORCE (15): Trivial Quotient | rho_point | 0.8144 | 0.9392 | 0.8278 | 0.9405 |
| REINFORCE (15): Reward-State Quotient | point | 0.4367 | 0.4356 | 0.4364 | 0.4372 |
| REINFORCE (15): Reward-State Quotient | mirror | 0.4377 | 0.4353 | 0.4368 | 0.4366 |
| REINFORCE (15): Reward-State Quotient | rho_flip | 0.0930 | 0.4342 | 0.0912 | 0.4376 |
| REINFORCE (15): Reward-State Quotient | rho_point | 0.0826 | 0.4354 | 0.0853 | 0.4373 |
| REINFORCE (15): Full-State Quotient | point | 0.1867 | 0.3983 | 0.1861 | 0.4035 |
| REINFORCE (15): Full-State Quotient | mirror | 0.1952 | 0.4052 | 0.1959 | 0.4100 |
| REINFORCE (15): Full-State Quotient | rho_flip | 0.0859 | 0.3973 | 0.0866 | 0.4053 |
| REINFORCE (15): Full-State Quotient | rho_point | 0.0636 | 0.3963 | 0.0653 | 0.4056 |
| REINFORCE (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | rho_flip | 0.9366 | 0.9329 | 0.9424 | 0.9309 |
| REINFORCE (5): Trivial Quotient | rho_point | 0.9366 | 0.9428 | 0.9424 | 0.9369 |
| REINFORCE (5): Reward-State Quotient | point | 0.4328 | 0.4352 | 0.4325 | 0.4358 |
| REINFORCE (5): Reward-State Quotient | mirror | 0.4339 | 0.4352 | 0.4341 | 0.4365 |
| REINFORCE (5): Reward-State Quotient | rho_flip | 0.0975 | 0.4362 | 0.0883 | 0.4331 |
| REINFORCE (5): Reward-State Quotient | rho_point | 0.0811 | 0.4422 | 0.0779 | 0.4330 |
| REINFORCE (5): Full-State Quotient | point | 0.1796 | 0.4019 | 0.1760 | 0.4111 |
| REINFORCE (5): Full-State Quotient | mirror | 0.1835 | 0.4064 | 0.1852 | 0.4109 |
| REINFORCE (5): Full-State Quotient | rho_flip | 0.0865 | 0.4001 | 0.0857 | 0.4096 |
| REINFORCE (5): Full-State Quotient | rho_point | 0.0624 | 0.4002 | 0.0609 | 0.4110 |
| PPO (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Trivial Quotient | rho_flip | 0.9397 | 0.9386 | 0.9422 | 0.9407 |
| PPO (5): Trivial Quotient | rho_point | 0.9396 | 0.9386 | 0.9422 | 0.9407 |
| PPO (5): Reward-State Quotient | point | 0.4389 | 0.4397 | 0.4390 | 0.4399 |
| PPO (5): Reward-State Quotient | mirror | 0.4396 | 0.4395 | 0.4396 | 0.4395 |
| PPO (5): Reward-State Quotient | rho_flip | 0.0876 | 0.4368 | 0.0914 | 0.4388 |
| PPO (5): Reward-State Quotient | rho_point | 0.0760 | 0.4377 | 0.1012 | 0.4391 |
| PPO (5): Full-State Quotient | point | 0.1861 | 0.4036 | 0.1956 | 0.4041 |
| PPO (5): Full-State Quotient | mirror | 0.1973 | 0.4065 | 0.1978 | 0.4065 |
| PPO (5): Full-State Quotient | rho_flip | 0.1135 | 0.4036 | 0.1099 | 0.4087 |
| PPO (5): Full-State Quotient | rho_point | 0.0797 | 0.4016 | 0.0949 | 0.4074 |
| SAC auto-alpha (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| SAC auto-alpha (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| SAC auto-alpha (5): Trivial Quotient | rho_flip | 0.5794 | 0.8631 | 0.7134 | 0.9117 |
| SAC auto-alpha (5): Trivial Quotient | rho_point | 0.9064 | 0.9413 | 0.9321 | 0.9380 |
| SAC auto-alpha (5): Reward-State Quotient | point | 0.4487 | 0.4448 | 0.4472 | 0.4462 |
| SAC auto-alpha (5): Reward-State Quotient | mirror | 0.4442 | 0.4474 | 0.4451 | 0.4462 |
| SAC auto-alpha (5): Reward-State Quotient | rho_flip | 0.1154 | 0.4365 | 0.1281 | 0.4508 |
| SAC auto-alpha (5): Reward-State Quotient | rho_point | 0.0806 | 0.4353 | 0.0730 | 0.4514 |
| SAC auto-alpha (5): Full-State Quotient | point | 0.2010 | 0.3976 | 0.2151 | 0.4076 |
| SAC auto-alpha (5): Full-State Quotient | mirror | 0.1872 | 0.4070 | 0.1919 | 0.4033 |
| SAC auto-alpha (5): Full-State Quotient | rho_flip | 0.1138 | 0.3862 | 0.1179 | 0.4036 |
| SAC auto-alpha (5): Full-State Quotient | rho_point | 0.0982 | 0.3883 | 0.0994 | 0.4071 |

## Ladder sites: tuning-selected occupancy

| Algorithm / condition | Mode | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |
|---|---|---:|---:|---:|---:|---:|
| REINFORCE (15): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (15): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (15): Trivial Quotient | rho_flip | 0.6907 | 0.5990 | 0.6406 | 0.7007 | 0.7808 |
| REINFORCE (15): Trivial Quotient | rho_point | 0.6904 | 0.5862 | 0.7533 | 0.8144 | 0.8278 |
| REINFORCE (15): Reward-State Quotient | point | 0.4357 | 0.4367 | 0.4368 | 0.4367 | 0.4364 |
| REINFORCE (15): Reward-State Quotient | mirror | 0.4073 | 0.4363 | 0.4364 | 0.4377 | 0.4368 |
| REINFORCE (15): Reward-State Quotient | rho_flip | 0.1805 | 0.1186 | 0.1065 | 0.0930 | 0.0912 |
| REINFORCE (15): Reward-State Quotient | rho_point | 0.2062 | 0.1112 | 0.0940 | 0.0826 | 0.0853 |
| REINFORCE (15): Full-State Quotient | point | 0.2951 | 0.2298 | 0.1991 | 0.1867 | 0.1861 |
| REINFORCE (15): Full-State Quotient | mirror | 0.3445 | 0.2672 | 0.2090 | 0.1952 | 0.1959 |
| REINFORCE (15): Full-State Quotient | rho_flip | 0.1606 | 0.0953 | 0.0895 | 0.0859 | 0.0866 |
| REINFORCE (15): Full-State Quotient | rho_point | 0.1385 | 0.0724 | 0.0654 | 0.0636 | 0.0653 |
| REINFORCE (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | rho_flip | 0.7504 | 0.5629 | 0.7586 | 0.9366 | 0.9424 |
| REINFORCE (5): Trivial Quotient | rho_point | 0.7504 | 0.5629 | 0.9364 | 0.9366 | 0.9424 |
| REINFORCE (5): Reward-State Quotient | point | 0.4327 | 0.4341 | 0.4333 | 0.4328 | 0.4325 |
| REINFORCE (5): Reward-State Quotient | mirror | 0.4321 | 0.4329 | 0.4318 | 0.4339 | 0.4341 |
| REINFORCE (5): Reward-State Quotient | rho_flip | 0.1856 | 0.1206 | 0.0936 | 0.0975 | 0.0883 |
| REINFORCE (5): Reward-State Quotient | rho_point | 0.2144 | 0.1145 | 0.0829 | 0.0811 | 0.0779 |
| REINFORCE (5): Full-State Quotient | point | 0.2804 | 0.2234 | 0.1931 | 0.1796 | 0.1760 |
| REINFORCE (5): Full-State Quotient | mirror | 0.3227 | 0.2335 | 0.1974 | 0.1835 | 0.1852 |
| REINFORCE (5): Full-State Quotient | rho_flip | 0.1825 | 0.1138 | 0.0935 | 0.0865 | 0.0857 |
| REINFORCE (5): Full-State Quotient | rho_point | 0.1436 | 0.0813 | 0.0642 | 0.0624 | 0.0609 |
| PPO (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Trivial Quotient | rho_flip | 0.9424 | 0.9428 | 0.9407 | 0.9397 | 0.9422 |
| PPO (5): Trivial Quotient | rho_point | 0.9428 | 0.9428 | 0.9410 | 0.9396 | 0.9422 |
| PPO (5): Reward-State Quotient | point | 0.4385 | 0.4374 | 0.4379 | 0.4389 | 0.4390 |
| PPO (5): Reward-State Quotient | mirror | 0.4395 | 0.4397 | 0.4398 | 0.4396 | 0.4396 |
| PPO (5): Reward-State Quotient | rho_flip | 0.2008 | 0.1252 | 0.1037 | 0.0876 | 0.0914 |
| PPO (5): Reward-State Quotient | rho_point | 0.2010 | 0.1141 | 0.1002 | 0.0760 | 0.1012 |
| PPO (5): Full-State Quotient | point | 0.2486 | 0.2141 | 0.1942 | 0.1861 | 0.1956 |
| PPO (5): Full-State Quotient | mirror | 0.2921 | 0.2125 | 0.1926 | 0.1973 | 0.1978 |
| PPO (5): Full-State Quotient | rho_flip | 0.1503 | 0.1213 | 0.1152 | 0.1135 | 0.1099 |
| PPO (5): Full-State Quotient | rho_point | 0.1407 | 0.0912 | 0.0982 | 0.0797 | 0.0949 |
| SAC auto-alpha (5): Trivial Quotient | point | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| SAC auto-alpha (5): Trivial Quotient | mirror | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| SAC auto-alpha (5): Trivial Quotient | rho_flip | 0.7559 | 0.5535 | 0.7632 | 0.5794 | 0.7134 |
| SAC auto-alpha (5): Trivial Quotient | rho_point | 0.7501 | 0.6256 | 0.9264 | 0.9064 | 0.9321 |
| SAC auto-alpha (5): Reward-State Quotient | point | 0.4483 | 0.4420 | 0.4464 | 0.4487 | 0.4472 |
| SAC auto-alpha (5): Reward-State Quotient | mirror | 0.4462 | 0.4454 | 0.4433 | 0.4442 | 0.4451 |
| SAC auto-alpha (5): Reward-State Quotient | rho_flip | 0.1620 | 0.1686 | 0.1359 | 0.1154 | 0.1281 |
| SAC auto-alpha (5): Reward-State Quotient | rho_point | 0.1710 | 0.1513 | 0.1011 | 0.0806 | 0.0730 |
| SAC auto-alpha (5): Full-State Quotient | point | 0.2451 | 0.2305 | 0.2085 | 0.2010 | 0.2151 |
| SAC auto-alpha (5): Full-State Quotient | mirror | 0.2778 | 0.2323 | 0.2154 | 0.1872 | 0.1919 |
| SAC auto-alpha (5): Full-State Quotient | rho_flip | 0.1622 | 0.1458 | 0.1099 | 0.1138 | 0.1179 |
| SAC auto-alpha (5): Full-State Quotient | rho_point | 0.1632 | 0.1313 | 0.1039 | 0.0982 | 0.0994 |

## Ladder: norm-matched delta after block 4

Confirmation occupancy and achieved mean shift norms relative to full-strength rho flip. Calibration uses tuning data only.

| Algorithm / condition | delta point | delta mirror | rho flip | Point/rho norm | Mirror/rho norm |
|---|---:|---:|---:|---:|---:|
| REINFORCE (15): Trivial Quotient | 0.8854 | 0.7714 | 0.6879 | 0.891 | 7.802 |
| REINFORCE (15): Reward-State Quotient | 0.3252 | 0.3367 | 0.0866 | 1.037 | 1.063 |
| REINFORCE (15): Full-State Quotient | 0.1846 | 0.1855 | 0.0851 | 0.976 | 0.997 |
| REINFORCE (5): Trivial Quotient | 0.9375 | 0.7536 | 0.9350 | 1.871 | 50.776 |
| REINFORCE (5): Reward-State Quotient | 0.3244 | 0.3325 | 0.0948 | 0.987 | 1.024 |
| REINFORCE (5): Full-State Quotient | 0.1824 | 0.1869 | 0.0842 | 0.975 | 0.984 |
| PPO (5): Trivial Quotient | 0.9137 | 0.9161 | 0.9363 | 1.060 | 1.165 |
| PPO (5): Reward-State Quotient | 0.3220 | 0.3507 | 0.0788 | 1.002 | 1.017 |
| PPO (5): Full-State Quotient | 0.1794 | 0.1874 | 0.1119 | 1.006 | 1.034 |
| SAC auto-alpha (5): Trivial Quotient | 0.7317 | 0.6713 | 0.5794 | 1.136 | 1.223 |
| SAC auto-alpha (5): Reward-State Quotient | 0.3733 | 0.3710 | 0.1060 | 1.005 | 1.057 |
| SAC auto-alpha (5): Full-State Quotient | 0.1904 | 0.1983 | 0.1138 | 1.010 | 1.015 |

## Single-write pulses after block 1

Mean absolute decoded-coordinate response at lags 0/1, maximum last nonzero lag across training seeds, and the full-filter response at lag 128. Mirror measures delta; flip measures rho. Blocks 2/3 are retained in agent reports.

| Algorithm / condition | Pulse | Lag 0 | Lag 1 | Last nonzero (max) | Full filter lag 128 |
|---|---|---:|---:|---:|---:|
| REINFORCE (15): Trivial Quotient | mirror | 0.022141 | 0.009814 | 96 | 0.000025 |
| REINFORCE (15): Trivial Quotient | rho_flip | 1.221041 | 0.066695 | 96 | 0.000084 |
| REINFORCE (15): Reward-State Quotient | mirror | 0.133581 | 0.023086 | 96 | 0.001723 |
| REINFORCE (15): Reward-State Quotient | rho_flip | 0.288827 | 0.063050 | 96 | 0.000160 |
| REINFORCE (15): Full-State Quotient | mirror | 0.184100 | 0.014824 | 96 | 0.005912 |
| REINFORCE (15): Full-State Quotient | rho_flip | 0.236194 | 0.038629 | 96 | 0.000475 |
| REINFORCE (5): Trivial Quotient | mirror | 0.027403 | 0.010860 | 96 | 0.000025 |
| REINFORCE (5): Trivial Quotient | rho_flip | 0.345845 | 0.084266 | 96 | 0.000084 |
| REINFORCE (5): Reward-State Quotient | mirror | 0.124801 | 0.016631 | 96 | 0.001529 |
| REINFORCE (5): Reward-State Quotient | rho_flip | 0.283399 | 0.043945 | 96 | 0.000166 |
| REINFORCE (5): Full-State Quotient | mirror | 0.192689 | 0.018219 | 96 | 0.005770 |
| REINFORCE (5): Full-State Quotient | rho_flip | 0.244262 | 0.053928 | 96 | 0.000429 |
| PPO (5): Trivial Quotient | mirror | 0.008520 | 0.000075 | 96 | 0.000025 |
| PPO (5): Trivial Quotient | rho_flip | 0.339664 | 0.008171 | 96 | 0.000084 |
| PPO (5): Reward-State Quotient | mirror | 0.226657 | 0.021692 | 96 | 0.002264 |
| PPO (5): Reward-State Quotient | rho_flip | 0.258225 | 0.017055 | 96 | 0.000273 |
| PPO (5): Full-State Quotient | mirror | 0.238568 | 0.009598 | 96 | 0.005414 |
| PPO (5): Full-State Quotient | rho_flip | 0.269968 | 0.013091 | 96 | 0.000446 |
| SAC auto-alpha (5): Trivial Quotient | mirror | 0.007474 | 0.002052 | 96 | 0.000025 |
| SAC auto-alpha (5): Trivial Quotient | rho_flip | 0.286787 | 0.040104 | 96 | 0.000084 |
| SAC auto-alpha (5): Reward-State Quotient | mirror | 0.107755 | 0.011086 | 96 | 0.002555 |
| SAC auto-alpha (5): Reward-State Quotient | rho_flip | 0.235243 | 0.023033 | 96 | 0.000438 |
| SAC auto-alpha (5): Full-State Quotient | mirror | 0.125241 | 0.018661 | 96 | 0.007238 |
| SAC auto-alpha (5): Full-State Quotient | rho_flip | 0.133825 | 0.014596 | 96 | 0.000605 |

## Moving reward: block 4, discounted map

| Algorithm | Intact | Random | Matched null (joint) | Factored | Joint | Exact-belief map |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE (15) | 0.9009 | 0.6021 | 0.8987 | 0.9985 | 1.0420 | 1.0919 |
| REINFORCE (5) | 0.8968 | 0.6021 | 0.9006 | 0.9924 | 1.0334 | 1.0919 |
| PPO (5) | 0.9003 | 0.6021 | 0.9013 | 1.0017 | 1.0321 | 1.0919 |
| SAC auto-alpha (5) | 0.9084 | 0.6021 | 0.9040 | 0.9619 | 1.0092 | 1.0919 |

## Moving reward by intervention site

| Algorithm | Target | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |
|---|---|---:|---:|---:|---:|---:|
| REINFORCE (15) | Factored | 0.9984 | 1.0260 | 1.0183 | 0.9985 | 1.0190 |
| REINFORCE (15) | Joint | 0.9996 | 1.0373 | 1.0444 | 1.0420 | 1.0428 |
| REINFORCE (5) | Factored | 1.0000 | 1.0212 | 1.0127 | 0.9924 | 1.0125 |
| REINFORCE (5) | Joint | 0.9910 | 1.0243 | 1.0372 | 1.0334 | 1.0364 |
| PPO (5) | Factored | 0.9837 | 1.0071 | 1.0091 | 1.0017 | 1.0053 |
| PPO (5) | Joint | 1.0063 | 1.0250 | 1.0360 | 1.0321 | 1.0339 |
| SAC auto-alpha (5) | Factored | 0.9503 | 0.9882 | 0.9841 | 0.9619 | 0.9926 |
| SAC auto-alpha (5) | Joint | 0.9408 | 0.9990 | 1.0120 | 1.0092 | 1.0125 |

## Per-seed ladder: block 4, full strength

| Algorithm / condition | Seed | Intact | Random | delta point | delta mirror | rho flip | rho point |
|---|---:|---:|---:|---:|---:|---:|---:|
| REINFORCE (5): Trivial Quotient | 42 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | 44 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9183 | 0.9183 |
| REINFORCE (5): Trivial Quotient | 46 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| REINFORCE (5): Trivial Quotient | 49 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9300 | 0.9300 |
| REINFORCE (5): Trivial Quotient | 52 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9409 | 0.9409 |
| REINFORCE (5): Reward-State Quotient | 42 | 0.4398 | 0.1057 | 0.4299 | 0.4332 | 0.1158 | 0.0870 |
| REINFORCE (5): Reward-State Quotient | 44 | 0.4311 | 0.1057 | 0.4432 | 0.4397 | 0.0912 | 0.0748 |
| REINFORCE (5): Reward-State Quotient | 46 | 0.4614 | 0.1057 | 0.4561 | 0.4543 | 0.0932 | 0.0609 |
| REINFORCE (5): Reward-State Quotient | 49 | 0.3908 | 0.1057 | 0.3850 | 0.3846 | 0.0951 | 0.0731 |
| REINFORCE (5): Reward-State Quotient | 52 | 0.4571 | 0.1057 | 0.4497 | 0.4497 | 0.0789 | 0.0628 |
| REINFORCE (5): Full-State Quotient | 42 | 0.4097 | 0.0961 | 0.1679 | 0.1792 | 0.0744 | 0.0573 |
| REINFORCE (5): Full-State Quotient | 44 | 0.4184 | 0.0961 | 0.1694 | 0.1573 | 0.0815 | 0.0562 |
| REINFORCE (5): Full-State Quotient | 46 | 0.4242 | 0.0961 | 0.2078 | 0.1915 | 0.0894 | 0.0581 |
| REINFORCE (5): Full-State Quotient | 49 | 0.4127 | 0.0961 | 0.1584 | 0.2023 | 0.0989 | 0.0640 |
| REINFORCE (5): Full-State Quotient | 52 | 0.4110 | 0.0961 | 0.1925 | 0.1984 | 0.0765 | 0.0514 |
| PPO (5): Trivial Quotient | 42 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9285 | 0.9285 |
| PPO (5): Trivial Quotient | 44 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Trivial Quotient | 46 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9418 | 0.9418 |
| PPO (5): Trivial Quotient | 49 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9258 | 0.9289 |
| PPO (5): Trivial Quotient | 52 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| PPO (5): Reward-State Quotient | 42 | 0.4540 | 0.1057 | 0.4495 | 0.4495 | 0.0854 | 0.0700 |
| PPO (5): Reward-State Quotient | 44 | 0.4331 | 0.1057 | 0.4264 | 0.4313 | 0.0836 | 0.0685 |
| PPO (5): Reward-State Quotient | 46 | 0.4580 | 0.1057 | 0.4609 | 0.4582 | 0.0759 | 0.0676 |
| PPO (5): Reward-State Quotient | 49 | 0.4411 | 0.1057 | 0.4344 | 0.4422 | 0.0799 | 0.0616 |
| PPO (5): Reward-State Quotient | 52 | 0.4114 | 0.1057 | 0.4114 | 0.4114 | 0.0693 | 0.0627 |
| PPO (5): Full-State Quotient | 42 | 0.4229 | 0.0961 | 0.2091 | 0.1695 | 0.1001 | 0.0687 |
| PPO (5): Full-State Quotient | 44 | 0.4226 | 0.0961 | 0.1703 | 0.2004 | 0.1235 | 0.0796 |
| PPO (5): Full-State Quotient | 46 | 0.4160 | 0.0961 | 0.2057 | 0.2069 | 0.1127 | 0.0750 |
| PPO (5): Full-State Quotient | 49 | 0.4089 | 0.0961 | 0.1911 | 0.2139 | 0.1132 | 0.0793 |
| PPO (5): Full-State Quotient | 52 | 0.3799 | 0.0961 | 0.1585 | 0.1961 | 0.1097 | 0.0667 |
| SAC auto-alpha (5): Trivial Quotient | 42 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.0749 | 0.9204 |
| SAC auto-alpha (5): Trivial Quotient | 44 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9146 | 0.9181 |
| SAC auto-alpha (5): Trivial Quotient | 46 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.9338 | 0.9426 |
| SAC auto-alpha (5): Trivial Quotient | 49 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.8147 | 0.8079 |
| SAC auto-alpha (5): Trivial Quotient | 52 | 0.9428 | 0.1880 | 0.9428 | 0.9428 | 0.1589 | 0.8732 |
| SAC auto-alpha (5): Reward-State Quotient | 42 | 0.4488 | 0.1057 | 0.4456 | 0.4501 | 0.1022 | 0.0743 |
| SAC auto-alpha (5): Reward-State Quotient | 44 | 0.4512 | 0.1057 | 0.4469 | 0.4498 | 0.1248 | 0.0699 |
| SAC auto-alpha (5): Reward-State Quotient | 46 | 0.4412 | 0.1057 | 0.4565 | 0.4484 | 0.0806 | 0.0665 |
| SAC auto-alpha (5): Reward-State Quotient | 49 | 0.4457 | 0.1057 | 0.4426 | 0.4439 | 0.1075 | 0.0899 |
| SAC auto-alpha (5): Reward-State Quotient | 52 | 0.4451 | 0.1057 | 0.4532 | 0.4457 | 0.1147 | 0.0856 |
| SAC auto-alpha (5): Full-State Quotient | 42 | 0.4104 | 0.0961 | 0.1976 | 0.1974 | 0.1169 | 0.0666 |
| SAC auto-alpha (5): Full-State Quotient | 44 | 0.4110 | 0.0961 | 0.2100 | 0.1750 | 0.1219 | 0.0917 |
| SAC auto-alpha (5): Full-State Quotient | 46 | 0.4028 | 0.0961 | 0.1800 | 0.1803 | 0.1097 | 0.0690 |
| SAC auto-alpha (5): Full-State Quotient | 49 | 0.4096 | 0.0961 | 0.1881 | 0.1796 | 0.0929 | 0.0718 |
| SAC auto-alpha (5): Full-State Quotient | 52 | 0.4017 | 0.0961 | 0.2114 | 0.1876 | 0.1276 | 0.0807 |

## Per-seed moving reward: block 4, discounted map

| Algorithm | Seed | Intact | Random | Null (joint) | Factored | Joint | Exact-belief map |
|---|---:|---:|---:|---:|---:|---:|---:|
| REINFORCE (5) | 42 | 0.8830 | 0.6021 | 0.8986 | 0.9887 | 1.0349 | 1.0919 |
| REINFORCE (5) | 44 | 0.9012 | 0.6021 | 0.9221 | 0.9917 | 1.0313 | 1.0919 |
| REINFORCE (5) | 46 | 0.9037 | 0.6021 | 0.8871 | 1.0004 | 1.0310 | 1.0919 |
| REINFORCE (5) | 49 | 0.9038 | 0.6021 | 0.9179 | 0.9896 | 1.0419 | 1.0919 |
| REINFORCE (5) | 52 | 0.8921 | 0.6021 | 0.8773 | 0.9917 | 1.0277 | 1.0919 |
| PPO (5) | 42 | 0.9254 | 0.6021 | 0.9387 | 1.0489 | 1.0728 | 1.0919 |
| PPO (5) | 44 | 0.8891 | 0.6021 | 0.8747 | 0.8977 | 0.9509 | 1.0919 |
| PPO (5) | 46 | 0.8952 | 0.6021 | 0.8904 | 1.0121 | 1.0502 | 1.0919 |
| PPO (5) | 49 | 0.9031 | 0.6021 | 0.9097 | 1.0258 | 1.0416 | 1.0919 |
| PPO (5) | 52 | 0.8884 | 0.6021 | 0.8931 | 1.0241 | 1.0451 | 1.0919 |
| SAC auto-alpha (5) | 42 | 0.9049 | 0.6021 | 0.9128 | 0.9913 | 1.0351 | 1.0919 |
| SAC auto-alpha (5) | 44 | 0.8939 | 0.6021 | 0.8937 | 0.9259 | 0.9565 | 1.0919 |
| SAC auto-alpha (5) | 46 | 0.9006 | 0.6021 | 0.8979 | 0.9705 | 1.0026 | 1.0919 |
| SAC auto-alpha (5) | 49 | 0.9195 | 0.6021 | 0.8961 | 0.9527 | 1.0266 | 1.0919 |
| SAC auto-alpha (5) | 52 | 0.9231 | 0.6021 | 0.9198 | 0.9690 | 1.0250 | 1.0919 |

Aggregate occupancy/reward/control and pulse metrics are in summary.json. Full per-seed/site metrics, myopic-map results, phase/component rewards and agent reports are in the verified B2 archive (../details_manifest.json). Numerical references are not certified optima.
