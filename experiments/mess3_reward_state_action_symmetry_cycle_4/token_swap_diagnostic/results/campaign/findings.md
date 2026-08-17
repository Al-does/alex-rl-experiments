# Cycle 4 variant 2 token-swap versus real-target probes

Completed seeds: [44, 45, 46] (3/5).

## Failed seeds (activation reconstruction gate)

These seeds failed the `reconstruction_error > 2e-5` sanity check before metrics were written.
- seed 42: reconstructed factual histories do not match rollout activations: 4.244e-05
- seed 43: reconstructed factual histories do not match rollout activations: 2.209e-03

## Aggregate comparison (completed seeds only)

- Token-swap factual MSE: 0.003694 ± 0.000520 (n=3)
- Token-swap ΔMSE (counterfactual − factual): 0.000015 ± 0.000027
- Token-swap equivariance MSE: 0.001475 ± 0.001143
- Token-swap antisymmetric sign-reversal RMSE: 0.081570 ± 0.037485
- Full 3D belief probe final MSE: 0.003662 ± 0.000559
- Symmetry-probe antisymmetric target MSE: 0.020170 ± 0.002823
- Symmetry-probe symmetric-b2 target MSE: 0.000670 ± 0.000125

## Per-seed rows

| seed | swap factual MSE | swap ΔMSE | equivariance MSE | sign-reversal RMSE | full-probe MSE | antisym probe MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 44 | 0.003194 | 0.000041 | 0.000385 | 0.043976 | 0.003143 | 0.017688 |
| 45 | 0.004232 | 0.000016 | 0.002665 | 0.118945 | 0.004253 | 0.023241 |
| 46 | 0.003657 | -0.000012 | 0.001376 | 0.081788 | 0.003591 | 0.019582 |

## Interpretation

Token-swap replays fixed greedy histories with token channels 0/1 exchanged while actions stay factual.
Low equivariance MSE and sign-reversal RMSE mean decoded beliefs exchange under swap; large ΔMSE means counterfactual decoding is worse than factual.

Across completed seeds, token-swap factual MSE tracks full-probe MSE closely (same frozen decoder on held-out rollouts).
Equivariance MSE is well below full-probe MSE, indicating partial b0/b1 exchange under swap rather than random decoding.
Sign-reversal RMSE (~0.04–0.12 on cycle 4; ~0.03–0.10 on cycle 5) exceeds equivariance MSE, so antisymmetric structure is imperfect.
ΔMSE is near zero with 95% CIs spanning zero on successful runs — counterfactual decoding stays as accurate as factual.
Symmetry-probe antisymmetric MSE (~0.017–0.023) is higher than token-swap equivariance MSE (~0.0004–0.0027), consistent with the swap test being a stricter paired intervention than the published antisymmetric target alone.
