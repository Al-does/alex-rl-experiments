# Cycle 5 variant 2 token-swap versus real-target probes

Token-swap diagnostics replay fixed greedy histories through frozen final checkpoints with token channels 0 and 1 exchanged. Full-probe and symmetry-probe rows come from existing published summaries on the same seeds.

## Aggregate comparison

- Token-swap factual MSE: 0.002453 ± 0.000749
- Token-swap ΔMSE (counterfactual − factual): -0.000033 ± 0.000078
- Token-swap equivariance MSE: 0.000676 ± 0.000595
- Token-swap antisymmetric sign-reversal RMSE: 0.056411 ± 0.025517
- Full 3D belief probe final MSE: 0.002430 ± 0.000728

## Per-seed rows

| seed | swap factual MSE | swap ΔMSE | equivariance MSE | sign-reversal RMSE | full-probe MSE | antisym probe MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.001184 | 0.000016 | 0.001689 | 0.096075 | 0.001183 | n/a |
| 43 | 0.002509 | -0.000005 | 0.000461 | 0.049813 | 0.002523 | n/a |
| 44 | 0.002610 | 0.000000 | 0.000112 | 0.025043 | 0.002582 | n/a |
| 45 | 0.002819 | -0.000007 | 0.000539 | 0.054487 | 0.002809 | n/a |
| 46 | 0.003140 | -0.000171 | 0.000577 | 0.056635 | 0.003053 | n/a |

Interpretation: low equivariance MSE and sign-reversal RMSE indicate the decoded representation exchanges b0/b1 under the exact token swap. A large ΔMSE rejects the hypothesis that counterfactual decoding stays as accurate as factual decoding.
