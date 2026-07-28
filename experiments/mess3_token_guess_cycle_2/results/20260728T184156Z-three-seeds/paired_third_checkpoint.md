# Paired same-seed summary (through third checkpoint)

Window: checkpoint indices `0..2` only (init, ~0.33M, **~0.66M**).
Later checkpoints excluded because PPO-family MSE rises after ~0.66M.

Design: for each arm pair, compute same-seed differences on held-out affine probe MSE at the third checkpoint, then report the **mean paired ΔMSE** (A − B; negative ⇒ A better). Paired *t* / Cohen *d_z* accompany that number; with n=3 they are descriptive only.

## Arm means at third checkpoint (~0.66M)

| condition | MSE | belief R² | token acc |
|---|---:|---:|---:|
| a2c | 0.003649 ± 0.000224 | 0.9695 ± 0.0019 | 0.6735 ± 0.0031 |
| ppo | 0.000382 ± 0.000104 | 0.9968 ± 0.0009 | 0.6828 ± 0.0036 |
| predictive_loss | 0.000328 ± 0.000096 | 0.9973 ± 0.0008 | 0.6853 ± 0.0046 |
| decoupled_kelly | 0.000306 ± 0.000060 | 0.9974 ± 0.0005 | 0.6844 ± 0.0033 |
| iqn | 0.000414 ± 0.000015 | 0.9965 ± 0.0001 | 0.6851 ± 0.0026 |

## Single arm score: mean paired MSE advantage

Average over all other arms and seeds of `(other_mse − arm_mse)` at the third checkpoint. **Higher = better.**

| rank | condition | mean paired MSE advantage |
|---:|---|---:|
| 1 | decoupled_kelly | +8.874894e-04 |
| 2 | predictive_loss | +8.593947e-04 |
| 3 | ppo | +7.922827e-04 |
| 4 | iqn | +7.519060e-04 |
| 5 | a2c | -3.291073e-03 |

## Pairwise mean paired ΔMSE (A − B)

| A | B | mean ΔMSE | paired t | p | Cohen d_z |
|---|---|---:|---:|---:|---:|
| decoupled_kelly | iqn | -1.0847e-04 | -3.92 | 0.059 | -2.27 |
| predictive_loss | iqn | -8.5991e-05 | -1.80 | 0.214 | -1.04 |
| ppo | iqn | -3.2301e-05 | -0.60 | 0.610 | -0.35 |
| predictive_loss | decoupled_kelly | +2.2476e-05 | +0.47 | 0.682 | +0.27 |
| ppo | predictive_loss | +5.3690e-05 | +0.82 | 0.500 | +0.47 |
| ppo | decoupled_kelly | +7.6165e-05 | +2.84 | 0.105 | +1.64 |
| a2c | iqn | +3.2344e-03 | +26.67 | 0.001 | +15.40 |
| a2c | ppo | +3.2667e-03 | +40.25 | 0.001 | +23.24 |
| a2c | predictive_loss | +3.3204e-03 | +34.64 | 0.001 | +20.00 |
| a2c | decoupled_kelly | +3.3428e-03 | +34.53 | 0.001 | +19.94 |

Machine-readable copy: [`paired_third_checkpoint.json`](paired_third_checkpoint.json).
