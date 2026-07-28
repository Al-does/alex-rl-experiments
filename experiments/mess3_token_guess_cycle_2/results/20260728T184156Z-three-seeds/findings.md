# MESS3 token-guess cycle 2

Full runs (2.5M steps), seeds 42–44, LR=1e-4, predictive coeff=1.0, kelly coeff=1.0, gamma=0.

Launched on RunPod Flash (`7gqwl8g0r1qb6p`, 3 parallel workers); all 15 jobs reported `workload_success=true` with B2 + `results` publication.

| condition | seeds | belief R² | token accuracy | mse |
|---|---:|---:|---:|---:|
| a2c | 3 | 0.9696 ± 0.0009 | 0.6735 ± 0.0025 | 0.003643 ± 0.000106 |
| ppo | 3 | 0.9948 ± 0.0012 | 0.6746 ± 0.0094 | 0.000622 ± 0.000146 |
| predictive_loss | 3 | 0.9896 ± 0.0042 | 0.6863 ± 0.0021 | 0.001246 ± 0.000510 |
| decoupled_kelly | 3 | 0.9953 ± 0.0011 | 0.6853 ± 0.0032 | 0.000559 ± 0.000130 |
| iqn | 3 | 0.9941 ± 0.0007 | 0.6840 ± 0.0008 | 0.000703 ± 0.000081 |

## Ranking

By held-out belief R²: **decoupled_kelly ≈ ppo ≈ iqn > predictive_loss > a2c**.

By greedy token accuracy: **predictive_loss ≈ decoupled_kelly ≈ iqn > ppo ≈ a2c** (bayesian optimal context-10 accuracy is ~0.686–0.69 in this study).

Decoupled Kelly has the best joint profile (top belief geometry and near-oracle token accuracy with lowest MSE). Predictive loss wins token accuracy slightly but with weaker / noisier belief R². A2C lags on both axes.
