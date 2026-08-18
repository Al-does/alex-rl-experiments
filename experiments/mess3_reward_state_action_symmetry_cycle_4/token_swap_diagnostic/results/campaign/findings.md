# Cycle 4 variant 2 token-swap versus real-target probes

Token-swap diagnostics replay fixed greedy histories through frozen final checkpoints with token channels 0 and 1 exchanged. Full-probe and symmetry-probe rows come from existing published summaries on the same seeds.

## Aggregate comparison

- Token-swap factual MSE: 0.003499 ± 0.000467
- Token-swap ΔMSE (counterfactual − factual): 0.000013 ± 0.000023
- Token-swap equivariance MSE: 0.001490 ± 0.000866
- Token-swap antisymmetric sign-reversal RMSE: 0.083927 ± 0.028482
- Full 3D belief probe final MSE: 0.003495 ± 0.000471
- Symmetry-probe antisymmetric target MSE: 0.019147 ± 0.002501
- Symmetry-probe symmetric-b2 target MSE: 0.000640 ± 0.000102

## Per-seed rows

| seed | swap factual MSE | swap ΔMSE | equivariance MSE | sign-reversal RMSE | full-probe MSE | antisym probe MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.003057 | 0.000030 | 0.001950 | 0.101477 | 0.003081 | 0.016831 |
| 43 | 0.003356 | -0.000008 | 0.001076 | 0.073448 | 0.003408 | 0.018392 |
| 44 | 0.003194 | 0.000041 | 0.000385 | 0.043976 | 0.003143 | 0.017688 |
| 45 | 0.004232 | 0.000016 | 0.002665 | 0.118945 | 0.004253 | 0.023241 |
| 46 | 0.003657 | -0.000012 | 0.001376 | 0.081788 | 0.003591 | 0.019582 |

Interpretation: low equivariance MSE and sign-reversal RMSE indicate the decoded representation exchanges b0/b1 under the exact token swap. A large ΔMSE rejects the hypothesis that counterfactual decoding stays as accurate as factual decoding.
