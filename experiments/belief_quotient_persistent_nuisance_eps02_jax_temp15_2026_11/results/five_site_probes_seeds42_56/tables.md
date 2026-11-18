# Fifteen-seed five-site probes

Cells: mean normalized error (1−R²) [95% seed-bootstrap percentile CI].
100,000 resamples of 15 training seeds with replacement; RNG seed 42.
The same seed-index resamples are used across sites, targets, conditions and checkpoints.
Full-State Quotient c is the merged-token control, not an exact quotient.

## Initialization

| Condition | Site | b | c | ρ | δ |
|---|---|---:|---:|---:|---:|
| Trivial Quotient | block_1 | 45.9360 [0.1830, 137.4218] | 3.0110 [0.1967, 8.5671] | 3.6938 [0.1995, 10.6090] | 45.9693 [0.2174, 137.4519] |
| Trivial Quotient | block_2 | 0.9845 [0.1517, 2.6393] | 4.2880 [0.1758, 12.4425] | 4.3867 [0.1785, 12.7329] | 1.0185 [0.1696, 2.6801] |
| Trivial Quotient | block_3 | 0.2517 [0.1344, 0.4738] | 0.5407 [0.1690, 1.2201] | 0.5545 [0.1720, 1.2550] | 0.2904 [0.1495, 0.5245] |
| Trivial Quotient | block_4 | 0.1423 [0.1290, 0.1567] | 0.3413 [0.1726, 0.6143] | 0.3373 [0.1756, 0.5961] | 0.1657 [0.1378, 0.1965] |
| Trivial Quotient | post_final_norm | 0.1403 [0.1282, 0.1538] | 0.3900 [0.1767, 0.7522] | 0.3865 [0.1799, 0.7347] | 0.1642 [0.1352, 0.1975] |
| Reward-State Quotient | block_1 | 45.9490 [0.2006, 137.4310] | 3.1230 [0.3137, 8.6624] | 3.8052 [0.3160, 10.7042] | 45.9431 [0.1904, 137.4290] |
| Reward-State Quotient | block_2 | 0.9903 [0.1551, 2.6457] | 4.3751 [0.2663, 12.5168] | 4.4734 [0.2689, 12.8067] | 0.9825 [0.1487, 2.6378] |
| Reward-State Quotient | block_3 | 0.2499 [0.1330, 0.4717] | 0.6085 [0.2422, 1.2755] | 0.6213 [0.2444, 1.3100] | 0.2414 [0.1256, 0.4631] |
| Reward-State Quotient | block_4 | 0.1400 [0.1273, 0.1532] | 0.3995 [0.2373, 0.6617] | 0.3948 [0.2398, 0.6428] | 0.1316 [0.1199, 0.1441] |
| Reward-State Quotient | post_final_norm | 0.1376 [0.1265, 0.1491] | 0.4492 [0.2428, 0.8001] | 0.4449 [0.2456, 0.7819] | 0.1286 [0.1191, 0.1375] |
| Full-State Quotient | block_1 | 0.2030 [0.1909, 0.2150] | 0.3723 [0.3003, 0.4388] | 0.3823 [0.3118, 0.4449] | 0.1973 [0.1828, 0.2111] |
| Full-State Quotient | block_2 | 0.1614 [0.1493, 0.1747] | 0.3357 [0.2682, 0.3997] | 0.3157 [0.2573, 0.3706] | 0.1536 [0.1421, 0.1656] |
| Full-State Quotient | block_3 | 0.1404 [0.1315, 0.1499] | 0.2939 [0.2422, 0.3419] | 0.2978 [0.2472, 0.3440] | 0.1305 [0.1224, 0.1385] |
| Full-State Quotient | block_4 | 0.1381 [0.1275, 0.1489] | 0.2854 [0.2387, 0.3296] | 0.2954 [0.2465, 0.3415] | 0.1275 [0.1178, 0.1368] |
| Full-State Quotient | post_final_norm | 0.1399 [0.1288, 0.1513] | 0.2919 [0.2451, 0.3362] | 0.3019 [0.2527, 0.3483] | 0.1286 [0.1185, 0.1383] |

## Final

| Condition | Site | b | c | ρ | δ |
|---|---|---:|---:|---:|---:|
| Trivial Quotient | block_1 | 0.2514 [0.2003, 0.3091] | 0.2180 [0.1727, 0.2698] | 0.2236 [0.1758, 0.2780] | 0.3672 [0.3021, 0.4398] |
| Trivial Quotient | block_2 | 0.1470 [0.1167, 0.1864] | 0.1269 [0.0982, 0.1647] | 0.1288 [0.0994, 0.1677] | 0.2232 [0.1857, 0.2665] |
| Trivial Quotient | block_3 | 0.1249 [0.1078, 0.1471] | 0.1002 [0.0845, 0.1207] | 0.1034 [0.0870, 0.1248] | 0.2150 [0.1931, 0.2413] |
| Trivial Quotient | block_4 | 0.1320 [0.1178, 0.1507] | 0.1033 [0.0896, 0.1210] | 0.1079 [0.0937, 0.1262] | 0.2328 [0.2137, 0.2553] |
| Trivial Quotient | post_final_norm | 0.1345 [0.1202, 0.1531] | 0.1050 [0.0912, 0.1228] | 0.1096 [0.0953, 0.1279] | 0.2381 [0.2180, 0.2613] |
| Reward-State Quotient | block_1 | 0.1865 [0.1777, 0.1948] | 0.0383 [0.0330, 0.0433] | 0.0398 [0.0344, 0.0448] | 0.4306 [0.4084, 0.4524] |
| Reward-State Quotient | block_2 | 0.1216 [0.1083, 0.1355] | 0.0225 [0.0198, 0.0253] | 0.0238 [0.0209, 0.0266] | 0.2837 [0.2487, 0.3208] |
| Reward-State Quotient | block_3 | 0.0997 [0.0880, 0.1139] | 0.0169 [0.0143, 0.0197] | 0.0182 [0.0155, 0.0210] | 0.2344 [0.2055, 0.2679] |
| Reward-State Quotient | block_4 | 0.0950 [0.0860, 0.1053] | 0.0155 [0.0133, 0.0180] | 0.0168 [0.0145, 0.0193] | 0.2240 [0.2038, 0.2460] |
| Reward-State Quotient | post_final_norm | 0.1010 [0.0919, 0.1112] | 0.0157 [0.0136, 0.0180] | 0.0170 [0.0148, 0.0194] | 0.2398 [0.2190, 0.2618] |
| Full-State Quotient | block_1 | 0.0806 [0.0685, 0.0924] | 0.1052 [0.0877, 0.1223] | 0.0701 [0.0579, 0.0823] | 0.0982 [0.0861, 0.1106] |
| Full-State Quotient | block_2 | 0.0417 [0.0390, 0.0446] | 0.0504 [0.0449, 0.0557] | 0.0330 [0.0298, 0.0364] | 0.0563 [0.0532, 0.0595] |
| Full-State Quotient | block_3 | 0.0324 [0.0304, 0.0346] | 0.0390 [0.0357, 0.0425] | 0.0241 [0.0218, 0.0266] | 0.0464 [0.0440, 0.0491] |
| Full-State Quotient | block_4 | 0.0292 [0.0274, 0.0311] | 0.0354 [0.0326, 0.0385] | 0.0215 [0.0196, 0.0235] | 0.0423 [0.0398, 0.0449] |
| Full-State Quotient | post_final_norm | 0.0327 [0.0307, 0.0349] | 0.0458 [0.0422, 0.0494] | 0.0279 [0.0257, 0.0302] | 0.0408 [0.0381, 0.0436] |

## Reward-State Quotient: is final b error below initialization?

Strict inequality, computed separately within each training seed and site.

| Seed | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |
|---:|---|---|---|---|---|
| 42 | Yes | No | No | Yes | Yes |
| 43 | No | No | No | No | No |
| 44 | Yes | Yes | Yes | Yes | Yes |
| 45 | Yes | Yes | Yes | Yes | Yes |
| 46 | No | No | Yes | Yes | Yes |
| 47 | Yes | Yes | Yes | Yes | Yes |
| 48 | No | Yes | Yes | Yes | Yes |
| 49 | Yes | Yes | Yes | Yes | Yes |
| 50 | Yes | Yes | Yes | Yes | Yes |
| 51 | Yes | Yes | Yes | Yes | Yes |
| 52 | Yes | Yes | Yes | Yes | Yes |
| 53 | Yes | Yes | Yes | Yes | Yes |
| 54 | No | Yes | Yes | No | No |
| 55 | Yes | Yes | Yes | Yes | Yes |
| 56 | Yes | Yes | Yes | Yes | Yes |
| **Below-init count / 15** | **11/15** | **12/15** | **13/15** | **13/15** | **13/15** |
