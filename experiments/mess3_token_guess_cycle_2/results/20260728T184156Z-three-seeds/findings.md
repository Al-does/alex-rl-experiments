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

## MSE over training

Paper-style checkpoint bars are under [`mse_over_training/`](mse_over_training/):
15 per-run charts, one all-runs figure, one condition-level figure, and the
compact plotted values.

All four PPO-family conditions reach their best mean MSE near 0.66M environment
steps, then drift upward to varying degrees. Decoupled Kelly has the lowest
checkpoint mean (0.000306) and the best final mean (0.000559). Predictive loss
has a similarly low early minimum (0.000328), but degrades most by the final
checkpoint (0.001246), driven especially by seed 43. A2C does not show the same
representation transition: its final mean MSE improves only 2.3% from true
initialization, versus 66.6–85.0% for the PPO-family conditions.

Every per-run error bar is the existing 95% interval from 1,000 bootstrap
resamples clustered by environment episode. The condition-level figure instead
shows individual trained-model seeds and mean ± SD. See
[`bootstrap_assessment.md`](mse_over_training/bootstrap_assessment.md) for why
no additional three-seed bootstrap is used.

## Paired same-seed summary (through third checkpoint)

Because PPO-family MSE rises after ~0.66M steps, arm comparisons below use only
the third checkpoint (index 2, ~0.66M). For each arm pair, the primary number is
the **mean paired ΔMSE** on the same three seeds (A − B; negative ⇒ A better).

Single arm score = mean over opponents and seeds of `(other_mse − arm_mse)`:

| rank | condition | mean paired MSE advantage |
|---:|---|---:|
| 1 | decoupled_kelly | +8.87e-04 |
| 2 | predictive_loss | +8.59e-04 |
| 3 | ppo | +7.92e-04 |
| 4 | iqn | +7.52e-04 |
| 5 | a2c | −3.29e-03 |

At this checkpoint, decoupled Kelly edges predictive loss (mean paired
ΔMSE = +2.25e-05 for predictive − kelly; not significant at n=3). All
PPO-family arms beat A2C by ~3.2–3.3e-03 MSE (paired |t| > 26, p ≤ 0.001).

Full pairwise table: [`paired_third_checkpoint.md`](paired_third_checkpoint.md).
