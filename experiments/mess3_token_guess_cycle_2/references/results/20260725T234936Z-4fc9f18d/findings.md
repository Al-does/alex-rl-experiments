# Token-guess metric reference points

## Belief-probe R²

An affine probe reading the one-hot encoded last 64 observations,
with no network and no training, already scores R² = 0.9668.
The supervised next-token replication reaches 0.9989.
Belief-probe R² therefore moves through a usable range of only 0.0321.

| observations visible to the probe | R² |
|---:|---:|
| 1 | 0.8043 |
| 2 | 0.9302 |
| 3 | 0.9581 |
| 4 | 0.9648 |
| 6 | 0.9667 |
| 8 | 0.9669 |
| 16 | 0.9669 |
| 64 | 0.9668 |

A randomly initialised copy of the study transformer scores R² = 0.8733 with greedy accuracy 0.3412.

Bootstrap resampling of the probe's test set puts its own sampling noise at [0.9664, 0.9673].

## Where the cycle-1 scores sit

| condition | reported R² | fraction of the floor-to-ceiling range |
|---|---:|---:|
| `comparison/reward_only` | 0.8552 | -347.6% |
| `comparison/predictive_loss` | 0.9319 | -108.6% |
| `comparison/max_entropy` | 0.8558 | -345.7% |
| `iqn_value` | 0.9760 | +28.7% |
| `kelly_cycle_2/correctness_iqn` | 0.9857 | +58.9% |
| `kelly_cycle_3/conditional_decoupled_kelly_iqn` | 0.9824 | +48.7% |

## Greedy token accuracy

| observations visible to an exact Bayesian filter | accuracy |
|---:|---:|
| 1 | 0.6732 |
| 2 | 0.6732 |
| 3 | 0.6859 |
| 4 | 0.6860 |
| 6 | 0.6879 |
| 8 | 0.6884 |
| 16 | 0.6883 |
| 64 | 0.6883 |

One observation reproduces the trivial repeat-the-previous-token rule at 0.6732; the filter saturates at 0.6883. Greedy token accuracy therefore moves through a usable range of only 0.0151.
