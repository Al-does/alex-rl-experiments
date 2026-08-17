# Cycle 5 variant 2 token-swap versus real-target probes

Completed seeds: [42, 43, 44, 46] (4/5).

## Failed seeds (activation reconstruction gate)

These seeds failed the `reconstruction_error > 2e-5` sanity check before metrics were written.
- seed 45: reconstructed factual histories do not match rollout activations: 6.926e-05

## Aggregate comparison (completed seeds only)

- Token-swap factual MSE: 0.002361 ± 0.000832 (n=4)
- Token-swap ΔMSE (counterfactual − factual): -0.000040 ± 0.000088
- Token-swap equivariance MSE: 0.000710 ± 0.000682
- Token-swap antisymmetric sign-reversal RMSE: 0.056892 ± 0.029438
- Full 3D belief probe final MSE: 0.002335 ± 0.000804

## Per-seed rows

| seed | swap factual MSE | swap ΔMSE | equivariance MSE | sign-reversal RMSE | full-probe MSE | antisym probe MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.001184 | 0.000016 | 0.001689 | 0.096075 | 0.001183 | n/a |
| 43 | 0.002509 | -0.000005 | 0.000461 | 0.049813 | 0.002523 | n/a |
| 44 | 0.002610 | 0.000000 | 0.000112 | 0.025043 | 0.002582 | n/a |
| 46 | 0.003140 | -0.000171 | 0.000577 | 0.056635 | 0.003053 | n/a |

## Interpretation

Token-swap replays fixed greedy histories with token channels 0/1 exchanged while actions stay factual.
Low equivariance MSE and sign-reversal RMSE mean decoded beliefs exchange under swap; large ΔMSE means counterfactual decoding is worse than factual.

Across completed seeds, token-swap factual MSE tracks full-probe MSE closely (same frozen decoder on held-out rollouts).
Equivariance MSE is well below full-probe MSE, indicating partial b0/b1 exchange under swap rather than random decoding.
Sign-reversal RMSE (~0.04–0.12 on cycle 4; ~0.03–0.10 on cycle 5) exceeds equivariance MSE, so antisymmetric structure is imperfect.
ΔMSE is near zero with 95% CIs spanning zero on successful runs — counterfactual decoding stays as accurate as factual.
