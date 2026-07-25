# Audit of the committed multi-seed token-guess results

Belief-probe R² is reported against the range it can move through: 0% is an affine probe on the raw observations (0.9668) and 100% is the supervised next-token replication (0.9989).

## `mess_3_kelly_cycle_2`

| condition | R² | 95% CI | usable range | above floor |
|---|---:|---|---:|---|
| `correctness_iqn` | 0.9857 | [0.9841, 0.9872] | +59% | yes |
| `correctness_mean` | 0.9834 | [0.9790, 0.9879] | +52% | yes |
| `decoupled_kelly_iqn` | 0.9572 | [0.9467, 0.9678] | -30% | no |
| `conditional_decoupled_kelly_mean` | 0.9559 | [0.9330, 0.9787] | -34% | no |
| `decoupled_kelly_mean` | 0.9529 | [0.9413, 0.9646] | -43% | no |
| `conditional_decoupled_kelly_iqn` | 0.9491 | [0.9289, 0.9693] | -55% | no |
| `coupled_kelly_iqn` | 0.9467 | [0.9283, 0.9651] | -63% | no |
| `coupled_kelly_mean` | 0.9375 | [0.9232, 0.9518] | -91% | no |

Of 28 pairwise orderings, 0 survive a Holm correction across the family.

| comparison | difference | 95% CI | Holm p | seeds for 80% power |
|---|---:|---|---:|---:|
| coupled_kelly_mean vs decoupled_kelly_iqn | -0.0197 | [-0.0235, -0.0159] | 0.056 | 5 |
| correctness_iqn vs coupled_kelly_mean | +0.0481 | [+0.0346, +0.0617] | 0.114 | 3 |
| correctness_iqn vs decoupled_kelly_iqn | +0.0284 | [+0.0187, +0.0382] | 0.164 | 3 |
| correctness_iqn vs decoupled_kelly_mean | +0.0327 | [+0.0211, +0.0444] | 0.170 | 4 |
| correctness_mean vs coupled_kelly_mean | +0.0459 | [+0.0293, +0.0625] | 0.170 | 3 |
| correctness_mean vs decoupled_kelly_mean | +0.0305 | [+0.0186, +0.0424] | 0.186 | 3 |
| correctness_iqn vs coupled_kelly_iqn | +0.0390 | [+0.0221, +0.0559] | 0.220 | 4 |
| coupled_kelly_mean vs decoupled_kelly_mean | -0.0154 | [-0.0229, -0.0079] | 0.263 | 3 |
| correctness_mean vs decoupled_kelly_iqn | +0.0262 | [+0.0131, +0.0392] | 0.263 | 3 |
| conditional_decoupled_kelly_mean vs coupled_kelly_mean | +0.0183 | [+0.0091, +0.0276] | 0.263 | 4 |
| conditional_decoupled_kelly_iqn vs correctness_iqn | -0.0366 | [-0.0563, -0.0168] | 0.277 | 3 |
| correctness_mean vs coupled_kelly_iqn | +0.0368 | [+0.0139, +0.0596] | 0.344 | 3 |
| conditional_decoupled_kelly_iqn vs correctness_mean | -0.0344 | [-0.0571, -0.0116] | 0.366 | 3 |
| conditional_decoupled_kelly_mean vs correctness_iqn | -0.0298 | [-0.0521, -0.0075] | 0.434 | 3 |
| conditional_decoupled_kelly_mean vs correctness_mean | -0.0276 | [-0.0518, -0.0034] | 0.548 | 4 |
| decoupled_kelly_iqn vs decoupled_kelly_mean | +0.0043 | [-0.0020, +0.0106] | 1.000 | 5 |
| coupled_kelly_iqn vs decoupled_kelly_iqn | -0.0106 | [-0.0274, +0.0062] | 1.000 | 6 |
| coupled_kelly_iqn vs coupled_kelly_mean | +0.0091 | [-0.0088, +0.0271] | 1.000 | 8 |
| correctness_iqn vs correctness_mean | +0.0022 | [-0.0037, +0.0082] | 1.000 | 12 |
| conditional_decoupled_kelly_iqn vs coupled_kelly_mean | +0.0116 | [-0.0194, +0.0425] | 1.000 | 12 |
| conditional_decoupled_kelly_mean vs coupled_kelly_iqn | +0.0092 | [-0.0166, +0.0350] | 1.000 | 13 |
| conditional_decoupled_kelly_iqn vs decoupled_kelly_iqn | -0.0082 | [-0.0359, +0.0195] | 1.000 | 17 |
| coupled_kelly_iqn vs decoupled_kelly_mean | -0.0063 | [-0.0292, +0.0166] | 1.000 | 20 |
| conditional_decoupled_kelly_mean vs decoupled_kelly_mean | +0.0029 | [-0.0096, +0.0155] | 1.000 | 26 |
| conditional_decoupled_kelly_iqn vs conditional_decoupled_kelly_mean | -0.0068 | [-0.0470, +0.0334] | 1.000 | 47 |
| conditional_decoupled_kelly_iqn vs decoupled_kelly_mean | -0.0038 | [-0.0352, +0.0275] | 1.000 | 87 |
| conditional_decoupled_kelly_iqn vs coupled_kelly_iqn | +0.0024 | [-0.0174, +0.0222] | 1.000 | 88 |
| conditional_decoupled_kelly_mean vs decoupled_kelly_iqn | -0.0014 | [-0.0141, +0.0114] | 1.000 | 109 |

## `mess_3_kelly_cycle_3`

| condition | R² | 95% CI | usable range | above floor |
|---|---:|---|---:|---|
| `conditional_decoupled_kelly_iqn` | 0.9824 | [0.9788, 0.9859] | +49% | yes |
| `conditional_decoupled_kelly_mean` | 0.9717 | [0.9596, 0.9839] | +15% | no |
| `iqn` | 0.9688 | [0.9413, 0.9963] | +6% | no |
| `ppo` | 0.8742 | [0.7428, 1.0057] | -289% | no |

Of 6 pairwise orderings, 0 survive a Holm correction across the family.

| comparison | difference | 95% CI | Holm p | seeds for 80% power |
|---|---:|---|---:|---:|
| conditional_decoupled_kelly_iqn vs conditional_decoupled_kelly_mean | +0.0106 | [-0.0009, +0.0222] | 0.351 | 4 |
| iqn vs ppo | +0.0946 | [-0.0167, +0.2059] | 0.351 | 5 |
| conditional_decoupled_kelly_iqn vs ppo | +0.1081 | [-0.0198, +0.2361] | 0.351 | 5 |
| conditional_decoupled_kelly_mean vs ppo | +0.0975 | [-0.0302, +0.2253] | 0.351 | 5 |
| conditional_decoupled_kelly_iqn vs iqn | +0.0136 | [-0.0111, +0.0383] | 0.351 | 7 |
| conditional_decoupled_kelly_mean vs iqn | +0.0029 | [-0.0304, +0.0363] | 0.740 | 165 |
