# MESS3 token-guess cycle 2 — 0.66M / 15-seed campaign

Truncated runs through the third checkpoint (~0.66M steps), seeds 45–59 (n=15), run suffix `-0p66m`.

| condition | seeds | belief R² | token accuracy | mse |
|---|---:|---:|---:|---:|
| ppo | 15 | 0.9966 ± 0.0005 | 0.6824 ± 0.0037 | 0.000408 ± 0.000064 |
| predictive_loss | 15 | 0.9966 ± 0.0008 | 0.6862 ± 0.0029 | 0.000411 ± 0.000101 |
| decoupled_kelly | 15 | 0.9976 ± 0.0004 | 0.6853 ± 0.0035 | 0.000286 ± 0.000048 |

## Focused paired t-tests (same seed)

- **decoupled_kelly vs predictive_loss** at checkpoint 2 (~659,185 steps): mean ΔMSE=-1.402720e-04, t=-5.734, p=5.171e-05 (df=14).
- **decoupled_kelly vs ppo** at checkpoint 2 (~659,185 steps): mean ΔMSE=-1.070534e-04, t=-6.052, p=2.975e-05 (df=14).
