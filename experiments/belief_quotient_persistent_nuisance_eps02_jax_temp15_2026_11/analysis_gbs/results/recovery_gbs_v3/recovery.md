# GBS recovery report

Generated: 2026-11-21T15:58:47.320695+00:00

## Criteria and limitations

- **elbow:** Smallest observed k with validation KL <= best validation KL + max(0.001, 0.10*abs(best validation KL)). At each k choose class by validation KL; ties use affine, gru, tanh, hmm. Apply the same rule separately within each class.
- **adequate fit:** Selected validation AND paired test KL <= 0.001 nats. Test is assessment only, never selection.
- **geometry:** At the selected k, both forward and reverse MLP R2 >= 0.95 for the task target: quotient for Reward-State Quotient policy, factor_1 for Marginal Reward policy, full belief otherwise. Trivial Quotient policy at k=0 requires no belief geometry.
- **recovered:** Complete four-class scientific k grid, adequate fit, selected k equals the stated task dimension, and geometry criterion passes.
- **partially recovered:** Not recovered, but adequate fit OR >=90% paired test-KL reduction relative to the validation-selected constant (k=0) predictor.
- **not recovered:** Neither recovery nor partial-recovery criteria established. Missing fits are pending evidence, NOT evidence of non-recoverability.
- **condition verdict:** Recovered only if all three policy seeds AND all three control seeds recover; partially recovered if any row recovers or partially recovers; otherwise not recovered (pending where data are absent). Any incomplete row makes the condition verdict provisional.
- **diagnostics:** Commutation and HMM T/E agreement are descriptive, not gates: affine commutation is coordinate-dependent, and an MLP head does not ensure parameter identifiability. Quotient comparisons outside Reward-State Quotient are counterfactual.
- Missing fits are not imputed. All incomplete-sweep selections and verdicts are provisional.
- Policy imitation dimension need not equal environmental task-sufficient dimension: the trained policy may retain history/nuisance dependence.
- KL is nats per position (controls average equally over current actions); every fit uses complete held-out episodes.
- R2 and commutation are descriptive held-out probes, not evidence of causal use. No uncertainty is invented from aggregate metrics.
- The descriptive minimum test KL across classes is not used in any selection.

## Condition verdicts

- **Trivial Quotient: partially recovered** (3/6 rows recovered; 6/6 have fit evidence).
- **Reward-State Quotient: partially recovered** (3/6 rows recovered; 6/6 have fit evidence).
- **Full-State Quotient: partially recovered** (0/6 rows recovered; 6/6 have fit evidence).
- **Marginal Reward: partially recovered** (0/6 rows recovered; 6/6 have fit evidence).
- **Joint Reward: partially recovered** (1/6 rows recovered; 6/6 have fit evidence).

## Seed-level recovery

Provisional/not-recovered pending rows are not negative scientific conclusions.

| Condition | Mode | Seed | Fits | k / true | Class | Val KL | Test KL | Verdict / assessment |
|---|---|---:|---:|---:|---|---:|---:|---|
| Trivial Quotient | policy | 42 | 28/28 | 0 / 0 | affine | -3.782e-08 | -3.253e-08 | recovered; complete-grid assessment |
| Trivial Quotient | policy | 43 | 28/28 | 0 / 0 | affine | -2.214e-08 | -2.336e-08 | recovered; complete-grid assessment |
| Trivial Quotient | policy | 44 | 28/28 | 0 / 0 | affine | 3.774e-08 | 4.372e-08 | recovered; complete-grid assessment |
| Trivial Quotient | control | 42 | 28/28 | 1 / 2 | gru | 0.0005803 | 0.0004656 | partially recovered; complete-grid assessment |
| Trivial Quotient | control | 43 | 28/28 | 1 / 2 | gru | 0.0005629 | 0.0004644 | partially recovered; complete-grid assessment |
| Trivial Quotient | control | 44 | 28/28 | 1 / 2 | gru | 0.0004651 | 0.0004709 | partially recovered; complete-grid assessment |
| Reward-State Quotient | policy | 42 | 28/28 | 4 / 1 | affine | 0.001983 | 0.001911 | partially recovered; complete-grid assessment |
| Reward-State Quotient | policy | 43 | 28/28 | 3 / 1 | affine | 0.0009159 | 0.0009183 | partially recovered; complete-grid assessment |
| Reward-State Quotient | policy | 44 | 28/28 | 4 / 1 | affine | 0.001107 | 0.001225 | partially recovered; complete-grid assessment |
| Reward-State Quotient | control | 42 | 28/28 | 2 / 2 | hmm | 6.972e-06 | 7.108e-06 | recovered; complete-grid assessment |
| Reward-State Quotient | control | 43 | 28/28 | 2 / 2 | gru | 1.002e-05 | 9.671e-06 | recovered; complete-grid assessment |
| Reward-State Quotient | control | 44 | 28/28 | 2 / 2 | hmm | 6.106e-06 | 5.81e-06 | recovered; complete-grid assessment |
| Full-State Quotient | policy | 42 | 28/28 | 5 / 2 | affine | 0.005244 | 0.005335 | partially recovered; complete-grid assessment |
| Full-State Quotient | policy | 43 | 28/28 | 6 / 2 | affine | 0.00356 | 0.003487 | partially recovered; complete-grid assessment |
| Full-State Quotient | policy | 44 | 28/28 | 6 / 2 | affine | 0.003479 | 0.003391 | partially recovered; complete-grid assessment |
| Full-State Quotient | control | 42 | 28/28 | 2 / 2 | hmm | 4.087e-06 | 4.4e-06 | partially recovered; complete-grid assessment |
| Full-State Quotient | control | 43 | 28/28 | 2 / 2 | hmm | 4.213e-06 | 4.333e-06 | partially recovered; complete-grid assessment |
| Full-State Quotient | control | 44 | 28/28 | 2 / 2 | hmm | 3.934e-06 | 3.888e-06 | partially recovered; complete-grid assessment |
| Marginal Reward | policy | 42 | 44/44 | 8 / 2 | affine | 0.00483 | 0.004748 | partially recovered; complete-grid assessment |
| Marginal Reward | policy | 43 | 44/44 | 8 / 2 | affine | 0.004695 | 0.004612 | partially recovered; complete-grid assessment |
| Marginal Reward | policy | 44 | 44/44 | 9 / 2 | affine | 0.005134 | 0.004949 | partially recovered; complete-grid assessment |
| Marginal Reward | control | 42 | 44/44 | 3 / 4 | gru | 0.000545 | 0.0005681 | partially recovered; complete-grid assessment |
| Marginal Reward | control | 43 | 44/44 | 3 / 4 | gru | 0.000282 | 0.000286 | partially recovered; complete-grid assessment |
| Marginal Reward | control | 44 | 44/44 | 3 / 4 | gru | 0.0005801 | 0.0005914 | partially recovered; complete-grid assessment |
| Joint Reward | policy | 42 | 44/44 | 9 / 4 | affine | 0.02088 | 0.02026 | partially recovered; complete-grid assessment |
| Joint Reward | policy | 43 | 44/44 | 9 / 4 | affine | 0.02067 | 0.02097 | partially recovered; complete-grid assessment |
| Joint Reward | policy | 44 | 44/44 | 9 / 4 | affine | 0.02052 | 0.02066 | partially recovered; complete-grid assessment |
| Joint Reward | control | 42 | 44/44 | 4 / 4 | gru | 8.423e-05 | 8.493e-05 | recovered; complete-grid assessment |
| Joint Reward | control | 43 | 44/44 | 4 / 4 | gru | 7.132e-05 | 7.133e-05 | partially recovered; complete-grid assessment |
| Joint Reward | control | 44 | 44/44 | 4 / 4 | gru | 6.356e-05 | 6.104e-05 | partially recovered; complete-grid assessment |

## Validation envelopes

![KL versus k](kl_vs_k.pdf)

One curve per seed and mode; at each k the class is selected by validation KL. Negative roundoff KL is clipped to zero for plotting only. Missing k are gaps. Vertical lines mark policy/control task dimensions.

## Curves and selected-state diagnostics

Each curve cell is validation / paired test KL. Best is the validation-selected class, not a test-selected model. Full per-class selected geometry and aligned matrix entries are in recovery.json.

### Trivial Quotient — policy, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | -3.782e-08 / -3.253e-08 | -3.782e-08 / -3.253e-08 | -3.782e-08 / -3.253e-08 | -3.782e-08 / -3.253e-08 | affine: -3.782e-08 / -3.253e-08 |
| 1 | -4.429e-08 / -4.337e-08 | -5.297e-08 / -5.246e-08 | -5.74e-08 / -5.608e-08 | -4.835e-08 / -4.559e-08 | tanh: -5.74e-08 / -5.608e-08 |
| 2 | -4.724e-08 / -4.646e-08 | -5.764e-08 / -5.718e-08 | -5.529e-08 / -5.456e-08 | -5.274e-08 / -4.995e-08 | gru: -5.764e-08 / -5.718e-08 |
| 3 | -4.824e-08 / -4.764e-08 | -4.965e-08 / -4.915e-08 | -5.569e-08 / -5.508e-08 | -5.764e-08 / -5.701e-08 | hmm: -5.764e-08 / -5.701e-08 |
| 4 | -3.391e-08 / -3.263e-08 | -5.204e-08 / -5.169e-08 | -5.787e-08 / -5.733e-08 | -5.854e-08 / -5.789e-08 | hmm: -5.854e-08 / -5.789e-08 |
| 5 | -3.779e-08 / -3.716e-08 | -5.15e-08 / -5.117e-08 | -5.857e-08 / -5.807e-08 | -5.507e-08 / -5.443e-08 | tanh: -5.857e-08 / -5.807e-08 |
| 6 | -2.776e-08 / -2.719e-08 | -5.154e-08 / -5.071e-08 | -5.462e-08 / -5.424e-08 | -5.512e-08 / -5.344e-08 | hmm: -5.512e-08 / -5.344e-08 |

Class-specific elbows: affine k=0, gru k=0, tanh k=0, hmm k=0.

Selected commutation: `{"mse": 0.003409926204351681, "constant_decoder_mse": 0.00340992604105832, "invalid_simplex_fraction": 0.0, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | -6.014e-05 | -6.014e-05 | — | — |
| quotient | -2.333e-05 | -2.334e-05 | — | — |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.1932 | 0.362 | in JSON |
| 2 | full | 0.2192 | 0.1523 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Trivial Quotient — policy, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | -2.214e-08 / -2.336e-08 | -2.214e-08 / -2.336e-08 | -2.214e-08 / -2.336e-08 | -2.214e-08 / -2.336e-08 | affine: -2.214e-08 / -2.336e-08 |
| 1 | -3.044e-08 / -3.171e-08 | -3.972e-08 / -3.96e-08 | -4.791e-08 / -4.671e-08 | -3.952e-08 / -3.961e-08 | tanh: -4.791e-08 / -4.671e-08 |
| 2 | -3.071e-08 / -3.151e-08 | -4.016e-08 / -4.026e-08 | -4.597e-08 / -4.531e-08 | -3.942e-08 / -3.939e-08 | tanh: -4.597e-08 / -4.531e-08 |
| 3 | -3.754e-08 / -3.829e-08 | -3.861e-08 / -3.871e-08 | -4.483e-08 / -4.449e-08 | -4.156e-08 / -4.131e-08 | tanh: -4.483e-08 / -4.449e-08 |
| 4 | -3.445e-08 / -3.498e-08 | -3.764e-08 / -3.785e-08 | -4.44e-08 / -4.397e-08 | -4.085e-08 / -4.044e-08 | tanh: -4.44e-08 / -4.397e-08 |
| 5 | -2.29e-08 / -2.374e-08 | -3.781e-08 / -3.759e-08 | -4.29e-08 / -4.284e-08 | -4.112e-08 / -4.073e-08 | tanh: -4.29e-08 / -4.284e-08 |
| 6 | -2.111e-08 / -2.188e-08 | -3.823e-08 / -3.817e-08 | -4.214e-08 / -4.172e-08 | -4.162e-08 / -4.114e-08 | tanh: -4.214e-08 / -4.172e-08 |

Class-specific elbows: affine k=0, gru k=0, tanh k=0, hmm k=0.

Selected commutation: `{"mse": 0.003133556714539704, "constant_decoder_mse": 0.003133556517636983, "invalid_simplex_fraction": 0.0, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | -0.0002629 | -0.0002629 | — | — |
| quotient | -0.0001864 | -0.0001864 | — | — |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.2187 | 0.3398 | in JSON |
| 2 | full | 0.2192 | 0.152 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Trivial Quotient — policy, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 3.774e-08 / 4.372e-08 | 3.774e-08 / 4.372e-08 | 3.774e-08 / 4.372e-08 | 3.774e-08 / 4.372e-08 | affine: 3.774e-08 / 4.372e-08 |
| 1 | -3.473e-08 / -3.469e-08 | -4.69e-08 / -4.659e-08 | -5.002e-08 / -4.665e-08 | -7.724e-09 / -2.348e-09 | tanh: -5.002e-08 / -4.665e-08 |
| 2 | -3.692e-08 / -3.628e-08 | -5.66e-08 / -5.514e-08 | -5.751e-08 / -5.514e-08 | -3.82e-08 / -3.329e-08 | tanh: -5.751e-08 / -5.514e-08 |
| 3 | -3.992e-08 / -3.979e-08 | -5.403e-08 / -5.439e-08 | -5.671e-08 / -5.378e-08 | -4.507e-08 / -4.088e-08 | tanh: -5.671e-08 / -5.378e-08 |
| 4 | -8.026e-09 / -7.795e-09 | -5.708e-08 / -5.698e-08 | -5.481e-08 / -5.263e-08 | -4.829e-08 / -4.465e-08 | gru: -5.708e-08 / -5.698e-08 |
| 5 | -6.979e-09 / -6.812e-09 | -5.009e-08 / -4.945e-08 | -5.705e-08 / -5.493e-08 | -4.629e-08 / -4.257e-08 | tanh: -5.705e-08 / -5.493e-08 |
| 6 | -2.205e-08 / -2.14e-08 | -5.355e-08 / -5.286e-08 | -5.65e-08 / -5.457e-08 | -4.293e-08 / -3.837e-08 | tanh: -5.65e-08 / -5.457e-08 |

Class-specific elbows: affine k=0, gru k=0, tanh k=0, hmm k=0.

Selected commutation: `{"mse": 0.003243778640284201, "constant_decoder_mse": 0.003243778601888826, "invalid_simplex_fraction": 0.0, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | -0.0001142 | -0.0001142 | — | — |
| quotient | -8.004e-05 | -8.004e-05 | — | — |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.2077 | 0.3544 | in JSON |
| 2 | full | 0.2192 | 0.1497 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Trivial Quotient — control, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.005942 / 0.005033 | 0.005942 / 0.005033 | 0.005942 / 0.005033 | 0.005942 / 0.005033 | affine: 0.005942 / 0.005033 |
| 1 | 0.0006311 / 0.0005562 | 0.0005803 / 0.0004656 | 0.0005845 / 0.0004687 | 0.001092 / 0.0009348 | gru: 0.0005803 / 0.0004656 |
| 2 | 0.0001047 / 0.0001104 | 4.659e-06 / 4.665e-06 | 5.799e-05 / 5.596e-05 | 0.00015 / 0.0001262 | gru: 4.659e-06 / 4.665e-06 |
| 3 | 2.125e-05 / 2.148e-05 | 4.134e-06 / 4.09e-06 | 2.76e-05 / 2.742e-05 | 0.0003576 / 0.0002898 | gru: 4.134e-06 / 4.09e-06 |
| 4 | 1.393e-05 / 1.603e-05 | 2.643e-06 / 2.755e-06 | 1.852e-05 / 1.727e-05 | 9.467e-06 / 1.027e-05 | gru: 2.643e-06 / 2.755e-06 |
| 5 | 1.295e-05 / 1.545e-05 | 2.425e-06 / 2.548e-06 | 1.303e-05 / 1.18e-05 | 7.307e-06 / 8.018e-06 | gru: 2.425e-06 / 2.548e-06 |
| 6 | 6.974e-06 / 7.396e-06 | 2.073e-06 / 2.011e-06 | 1.37e-05 / 1.125e-05 | 6.463e-06 / 6.914e-06 | gru: 2.073e-06 / 2.011e-06 |

Class-specific elbows: affine k=1, gru k=1, tanh k=1, hmm k=2.

Selected commutation: `{"mse": 0.004646722071310845, "constant_decoder_mse": 0.00340992604105832, "invalid_simplex_fraction": 0.5453247189638318, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6852 | 0.8104 | 0.8295 | 0.9204 |
| quotient | 0.5612 | 0.6995 | 0.5611 | 0.5649 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.02345 | 0.1982 | in JSON |
| 2 | full | 0.06983 | 0.05558 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Trivial Quotient — control, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.005753 / 0.004777 | 0.005753 / 0.004777 | 0.005753 / 0.004777 | 0.005753 / 0.004777 | affine: 0.005753 / 0.004777 |
| 1 | 0.0006313 / 0.0005487 | 0.0005629 / 0.0004644 | 0.0005689 / 0.0004607 | 0.001102 / 0.0009096 | gru: 0.0005629 / 0.0004644 |
| 2 | 0.0001059 / 8.869e-05 | 4.6e-06 / 4.164e-06 | 5.671e-05 / 5.078e-05 | 0.0005716 / 0.0004564 | gru: 4.6e-06 / 4.164e-06 |
| 3 | 2.181e-05 / 1.908e-05 | 3.556e-06 / 3.379e-06 | 2.599e-05 / 2.511e-05 | 2.818e-05 / 2.266e-05 | gru: 3.556e-06 / 3.379e-06 |
| 4 | 1.582e-05 / 1.503e-05 | 2.716e-06 / 2.598e-06 | 1.625e-05 / 1.452e-05 | 1.288e-05 / 1.201e-05 | gru: 2.716e-06 / 2.598e-06 |
| 5 | 1.094e-05 / 8.218e-06 | 2.481e-06 / 2.205e-06 | 1.463e-05 / 1.379e-05 | 7.068e-06 / 6.417e-06 | gru: 2.481e-06 / 2.205e-06 |
| 6 | 6.172e-06 / 5.612e-06 | 2.295e-06 / 2.09e-06 | 1.543e-05 / 1.437e-05 | 4.955e-06 / 4.539e-06 | gru: 2.295e-06 / 2.09e-06 |

Class-specific elbows: affine k=1, gru k=1, tanh k=1, hmm k=2.

Selected commutation: `{"mse": 0.0034555324515027175, "constant_decoder_mse": 0.003133556517636983, "invalid_simplex_fraction": 0.5776018756109482, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6958 | 0.783 | 0.8451 | 0.9489 |
| quotient | 0.5039 | 0.6723 | 0.5038 | 0.5084 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.01342 | 0.205 | in JSON |
| 2 | full | 0.09645 | 0.1504 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Trivial Quotient — control, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.00516 / 0.004972 | 0.00516 / 0.004972 | 0.00516 / 0.004972 | 0.00516 / 0.004972 | affine: 0.00516 / 0.004972 |
| 1 | 0.0005455 / 0.0005443 | 0.0004651 / 0.0004709 | 0.0004675 / 0.0004745 | 0.0009258 / 0.0009145 | gru: 0.0004651 / 0.0004709 |
| 2 | 0.0001064 / 9.64e-05 | 5.094e-06 / 4.256e-06 | 6.991e-05 / 6.168e-05 | 0.0004687 / 0.0004739 | gru: 5.094e-06 / 4.256e-06 |
| 3 | 2.118e-05 / 2.016e-05 | 4.25e-06 / 4.037e-06 | 2.288e-05 / 2.172e-05 | 0.0004454 / 0.0004482 | gru: 4.25e-06 / 4.037e-06 |
| 4 | 1.329e-05 / 1.239e-05 | 2.587e-06 / 2.385e-06 | 1.429e-05 / 1.295e-05 | 1.153e-05 / 1.11e-05 | gru: 2.587e-06 / 2.385e-06 |
| 5 | 1.311e-05 / 1.153e-05 | 2.036e-06 / 1.77e-06 | 1.56e-05 / 1.458e-05 | 9.12e-06 / 7.947e-06 | gru: 2.036e-06 / 1.77e-06 |
| 6 | 3.965e-06 / 3.701e-06 | 2.274e-06 / 2.09e-06 | 1.176e-05 / 1.122e-05 | 5.807e-06 / 4.724e-06 | gru: 2.274e-06 / 2.09e-06 |

Class-specific elbows: affine k=1, gru k=1, tanh k=1, hmm k=1.

Selected commutation: `{"mse": 0.004002698599028013, "constant_decoder_mse": 0.003243778601888826, "invalid_simplex_fraction": 0.572794477028348, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6806 | 0.7848 | 0.83 | 0.9407 |
| quotient | 0.5299 | 0.69 | 0.5299 | 0.5321 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.01831 | 0.201 | in JSON |
| 2 | full | 0.08631 | 0.15 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — policy, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.5893 / 0.5974 | 0.5893 / 0.5974 | 0.5893 / 0.5974 | 0.5893 / 0.5974 | affine: 0.5893 / 0.5974 |
| 1 | 0.04343 / 0.0457 | 0.0413 / 0.04332 | 0.04805 / 0.0506 | 0.04717 / 0.04974 | gru: 0.0413 / 0.04332 |
| 2 | 0.004458 / 0.004427 | 0.004531 / 0.004502 | 0.006952 / 0.007072 | 0.01785 / 0.01855 | affine: 0.004458 / 0.004427 |
| 3 | 0.003042 / 0.00297 | 0.002889 / 0.002866 | 0.005953 / 0.006012 | 0.008327 / 0.008687 | gru: 0.002889 / 0.002866 |
| 4 | 0.001983 / 0.001911 | 0.001992 / 0.001967 | 0.00413 / 0.004138 | 0.006492 / 0.006758 | affine: 0.001983 / 0.001911 |
| 5 | 0.001315 / 0.001271 | 0.00164 / 0.001627 | 0.002948 / 0.002914 | 0.005845 / 0.006111 | affine: 0.001315 / 0.001271 |
| 6 | 0.008729 / 0.008644 | 0.001406 / 0.001389 | 0.002756 / 0.00278 | 0.004379 / 0.00446 | gru: 0.001406 / 0.001389 |

Class-specific elbows: affine k=4, gru k=4, tanh k=5, hmm k=6.

Selected commutation: `{"mse": 0.01597918722852919, "constant_decoder_mse": 0.02857271370123685, "invalid_simplex_fraction": 0.1381048387096774, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.5388 | 0.587 | 0.5139 | 0.2197 |
| quotient | 0.8353 | 0.9389 | 0.5128 | 0.2537 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.004302 | 0.002112 | in JSON |
| 2 | full | 0.01238 | 0.00135 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — policy, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.6041 / 0.6033 | 0.6041 / 0.6033 | 0.6041 / 0.6033 | 0.6041 / 0.6033 | affine: 0.6041 / 0.6033 |
| 1 | 0.007233 / 0.007221 | 0.00686 / 0.006897 | 0.008941 / 0.008876 | 0.0132 / 0.01328 | gru: 0.00686 / 0.006897 |
| 2 | 0.002067 / 0.002132 | 0.002551 / 0.002613 | 0.004643 / 0.004645 | 0.005214 / 0.005307 | affine: 0.002067 / 0.002132 |
| 3 | 0.0009159 / 0.0009183 | 0.001288 / 0.001303 | 0.003316 / 0.003382 | 0.00336 / 0.003473 | affine: 0.0009159 / 0.0009183 |
| 4 | 0.0006144 / 0.0006155 | 0.0008026 / 0.0008116 | 0.001807 / 0.001817 | 0.002399 / 0.002417 | affine: 0.0006144 / 0.0006155 |
| 5 | 0.0004501 / 0.0004528 | 0.0007516 / 0.0007695 | 0.001536 / 0.001533 | 0.002134 / 0.00216 | affine: 0.0004501 / 0.0004528 |
| 6 | 0.0003958 / 0.00039 | 0.0005926 / 0.0005913 | 0.001223 / 0.001221 | 0.002161 / 0.002184 | affine: 0.0003958 / 0.00039 |

Class-specific elbows: affine k=3, gru k=3, tanh k=4, hmm k=4.

Selected commutation: `{"mse": 0.012012583945739197, "constant_decoder_mse": 0.026194157055792804, "invalid_simplex_fraction": 0.07143893572825025, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.5525 | 0.554 | 0.7564 | 0.7722 |
| quotient | 0.9484 | 0.9593 | 0.7559 | 0.7657 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.009099 | 0.06986 | in JSON |
| 2 | full | 0.1035 | 0.03615 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — policy, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.6134 / 0.6153 | 0.6134 / 0.6153 | 0.6134 / 0.6153 | 0.6134 / 0.6153 | affine: 0.6134 / 0.6153 |
| 1 | 0.03275 / 0.0327 | 0.02949 / 0.03035 | 0.04803 / 0.04801 | 0.04905 / 0.05018 | gru: 0.02949 / 0.03035 |
| 2 | 0.003127 / 0.003323 | 0.003766 / 0.003977 | 0.02025 / 0.02025 | 0.02568 / 0.02639 | affine: 0.003127 / 0.003323 |
| 3 | 0.0018 / 0.001885 | 0.001974 / 0.002038 | 0.003814 / 0.003967 | 0.005117 / 0.005369 | affine: 0.0018 / 0.001885 |
| 4 | 0.001107 / 0.001225 | 0.001186 / 0.001218 | 0.002472 / 0.002518 | 0.003614 / 0.003755 | affine: 0.001107 / 0.001225 |
| 5 | 0.0009527 / 0.000982 | 0.000991 / 0.001037 | 0.001973 / 0.002015 | 0.003297 / 0.003394 | affine: 0.0009527 / 0.000982 |
| 6 | 0.01015 / 0.009957 | 0.0007256 / 0.0007491 | 0.001702 / 0.001753 | 0.002298 / 0.002371 | gru: 0.0007256 / 0.0007491 |

Class-specific elbows: affine k=3, gru k=4, tanh k=4, hmm k=5.

Selected commutation: `{"mse": 0.013259018424223426, "constant_decoder_mse": 0.027581217194957743, "invalid_simplex_fraction": 0.08316914100684263, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.58 | 0.5957 | 0.6416 | 0.5306 |
| quotient | 0.9157 | 0.9524 | 0.6424 | 0.5324 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.004812 | 0.0002443 | in JSON |
| 2 | full | 0.08556 | 0.1103 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — control, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01462 / 0.01449 | 0.01462 / 0.01449 | 0.01462 / 0.01449 | 0.01462 / 0.01449 | affine: 0.01462 / 0.01449 |
| 1 | 0.00225 / 0.002327 | 0.004433 / 0.004674 | 0.00343 / 0.003613 | 0.004506 / 0.004724 | affine: 0.00225 / 0.002327 |
| 2 | 8.907e-05 / 9.156e-05 | 1.278e-05 / 1.308e-05 | 5.807e-05 / 5.844e-05 | 6.972e-06 / 7.108e-06 | hmm: 6.972e-06 / 7.108e-06 |
| 3 | 3.055e-05 / 3.115e-05 | 4.431e-06 / 4.624e-06 | 2.29e-05 / 2.453e-05 | 4.947e-06 / 4.962e-06 | gru: 4.431e-06 / 4.624e-06 |
| 4 | 1.456e-05 / 1.565e-05 | 5.245e-06 / 5.47e-06 | 1.726e-05 / 1.708e-05 | 2.108e-06 / 2.175e-06 | hmm: 2.108e-06 / 2.175e-06 |
| 5 | 9.943e-06 / 9.769e-06 | 3.351e-06 / 3.368e-06 | 1.741e-05 / 1.75e-05 | 1.342e-06 / 1.548e-06 | hmm: 1.342e-06 / 1.548e-06 |
| 6 | 6.343e-06 / 6.138e-06 | 2.362e-06 / 2.511e-06 | 1.321e-05 / 1.291e-05 | 1.041e-06 / 1.134e-06 | hmm: 1.041e-06 / 1.134e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.00951953085315574, "constant_decoder_mse": 0.02857271370123685, "invalid_simplex_fraction": 0.1545049792277615, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9162 | 0.9537 | 0.9403 | 0.9558 |
| quotient | 0.8853 | 0.9269 | 0.2722 | 0.2734 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.03718 | 0.1031 | in JSON |
| 2 | full | 0.004717 | 0.007596 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — control, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01373 / 0.01362 | 0.01373 / 0.01362 | 0.01373 / 0.01362 | 0.01373 / 0.01362 | affine: 0.01373 / 0.01362 |
| 1 | 0.002418 / 0.002478 | 0.002384 / 0.002417 | 0.003455 / 0.003455 | 0.004757 / 0.004805 | gru: 0.002384 / 0.002417 |
| 2 | 0.0001039 / 0.0001077 | 1.002e-05 / 9.671e-06 | 7.369e-05 / 7.372e-05 | 1.556e-05 / 1.573e-05 | gru: 1.002e-05 / 9.671e-06 |
| 3 | 3.496e-05 / 3.507e-05 | 5.398e-06 / 5.628e-06 | 2.201e-05 / 2.202e-05 | 3.495e-06 / 3.592e-06 | hmm: 3.495e-06 / 3.592e-06 |
| 4 | 1.646e-05 / 1.648e-05 | 4.428e-06 / 4.391e-06 | 1.825e-05 / 1.832e-05 | 1.231e-06 / 1.343e-06 | hmm: 1.231e-06 / 1.343e-06 |
| 5 | 6.658e-06 / 6.845e-06 | 3.568e-06 / 3.573e-06 | 2.825e-05 / 2.833e-05 | 1.052e-06 / 1.095e-06 | hmm: 1.052e-06 / 1.095e-06 |
| 6 | 5.474e-06 / 5.507e-06 | 2.317e-06 / 2.402e-06 | 2.025e-05 / 2.085e-05 | 1.151e-06 / 1.201e-06 | hmm: 1.151e-06 / 1.201e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.0034683065257982027, "constant_decoder_mse": 0.026194157055792804, "invalid_simplex_fraction": 0.2103494623655914, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9707 | 0.9785 | 0.9708 | 0.9733 |
| quotient | 0.9716 | 0.9793 | 0.4269 | 0.4255 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.04418 | 0.117 | in JSON |
| 2 | full | 0.004433 | 0.007774 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Reward-State Quotient — control, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01443 / 0.01446 | 0.01443 / 0.01446 | 0.01443 / 0.01446 | 0.01443 / 0.01446 | affine: 0.01443 / 0.01446 |
| 1 | 0.002268 / 0.002263 | 0.004338 / 0.004346 | 0.003393 / 0.003428 | 0.004404 / 0.00441 | affine: 0.002268 / 0.002263 |
| 2 | 8.422e-05 / 8.257e-05 | 1.544e-05 / 1.586e-05 | 5.692e-05 / 5.655e-05 | 6.106e-06 / 5.81e-06 | hmm: 6.106e-06 / 5.81e-06 |
| 3 | 3.267e-05 / 3.184e-05 | 5.395e-06 / 5.795e-06 | 2.686e-05 / 2.703e-05 | 4.709e-06 / 4.506e-06 | hmm: 4.709e-06 / 4.506e-06 |
| 4 | 1.447e-05 / 1.41e-05 | 4.759e-06 / 5.053e-06 | 1.688e-05 / 1.802e-05 | 2.366e-06 / 2.275e-06 | hmm: 2.366e-06 / 2.275e-06 |
| 5 | 9.495e-06 / 9.655e-06 | 3.558e-06 / 3.976e-06 | 1.448e-05 / 1.535e-05 | 1.568e-06 / 1.65e-06 | hmm: 1.568e-06 / 1.65e-06 |
| 6 | 5.546e-06 / 5.849e-06 | 2.454e-06 / 2.85e-06 | 1.292e-05 / 1.399e-05 | 1.481e-06 / 1.51e-06 | hmm: 1.481e-06 / 1.51e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.009308220531715995, "constant_decoder_mse": 0.027581217194957743, "invalid_simplex_fraction": 0.1568838587487781, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9173 | 0.9554 | 0.9359 | 0.9535 |
| quotient | 0.8897 | 0.9305 | 0.3325 | 0.3355 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient | 0.03928 | 0.1058 | in JSON |
| 2 | full | 0.004704 | 0.005284 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — policy, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.791 / 0.8095 | 0.791 / 0.8095 | 0.791 / 0.8095 | 0.791 / 0.8095 | affine: 0.791 / 0.8095 |
| 1 | 0.09559 / 0.1 | 0.09765 / 0.1009 | 0.1292 / 0.133 | 0.152 / 0.156 | affine: 0.09559 / 0.1 |
| 2 | 0.02673 / 0.02799 | 0.02532 / 0.02615 | 0.0361 / 0.0367 | 0.02987 / 0.03092 | gru: 0.02532 / 0.02615 |
| 3 | 0.009512 / 0.009689 | 0.01038 / 0.01068 | 0.02541 / 0.026 | 0.02143 / 0.02214 | affine: 0.009512 / 0.009689 |
| 4 | 0.006942 / 0.007037 | 0.007258 / 0.007387 | 0.0109 / 0.0111 | 0.01624 / 0.01687 | affine: 0.006942 / 0.007037 |
| 5 | 0.005244 / 0.005335 | 0.005884 / 0.006055 | 0.01023 / 0.01065 | 0.01241 / 0.01273 | affine: 0.005244 / 0.005335 |
| 6 | 0.004396 / 0.004471 | 0.004887 / 0.004907 | 0.008267 / 0.008443 | 0.01207 / 0.01232 | affine: 0.004396 / 0.004471 |

Class-specific elbows: affine k=5, gru k=5, tanh k=6, hmm k=5.

Selected commutation: `{"mse": 0.0041875939717906496, "constant_decoder_mse": 0.027879855566216926, "invalid_simplex_fraction": 0.13052526270772238, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9318 | 0.9506 | 0.6523 | 0.7148 |
| quotient | 0.8536 | 0.904 | 0.3274 | 0.3756 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.1626 | 0.006954 | in JSON |
| 2 | full | 0.002388 | 0.004052 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — policy, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.7391 / 0.719 | 0.7391 / 0.719 | 0.7391 / 0.719 | 0.7391 / 0.719 | affine: 0.7391 / 0.719 |
| 1 | 0.09694 / 0.09473 | 0.08815 / 0.08615 | 0.1242 / 0.1209 | 0.1497 / 0.1458 | gru: 0.08815 / 0.08615 |
| 2 | 0.02643 / 0.02494 | 0.02214 / 0.02117 | 0.03101 / 0.03012 | 0.02923 / 0.02818 | gru: 0.02214 / 0.02117 |
| 3 | 0.009938 / 0.009436 | 0.009396 / 0.009129 | 0.02213 / 0.02128 | 0.01916 / 0.01852 | gru: 0.009396 / 0.009129 |
| 4 | 0.006288 / 0.006026 | 0.006928 / 0.006746 | 0.01177 / 0.01139 | 0.01544 / 0.01479 | affine: 0.006288 / 0.006026 |
| 5 | 0.004696 / 0.004559 | 0.005148 / 0.005018 | 0.009734 / 0.00937 | 0.01242 / 0.01187 | affine: 0.004696 / 0.004559 |
| 6 | 0.00356 / 0.003487 | 0.004285 / 0.004145 | 0.008618 / 0.008216 | 0.01063 / 0.01002 | affine: 0.00356 / 0.003487 |

Class-specific elbows: affine k=6, gru k=5, tanh k=6, hmm k=6.

Selected commutation: `{"mse": 0.0031895275405935653, "constant_decoder_mse": 0.026609729000151287, "invalid_simplex_fraction": 0.10821038000977518, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9392 | 0.9467 | 0.773 | 0.8006 |
| quotient | 0.9015 | 0.9206 | 0.5936 | 0.5895 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.1545 | 0.004186 | in JSON |
| 2 | full | 0.002078 | 0.004534 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — policy, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.7706 / 0.7697 | 0.7706 / 0.7697 | 0.7706 / 0.7697 | 0.7706 / 0.7697 | affine: 0.7706 / 0.7697 |
| 1 | 0.09877 / 0.09735 | 0.09527 / 0.09448 | 0.1301 / 0.1282 | 0.1513 / 0.1491 | gru: 0.09527 / 0.09448 |
| 2 | 0.03335 / 0.03252 | 0.03007 / 0.02958 | 0.04139 / 0.04109 | 0.03706 / 0.03677 | gru: 0.03007 / 0.02958 |
| 3 | 0.01089 / 0.01068 | 0.01067 / 0.01058 | 0.01984 / 0.01989 | 0.02072 / 0.02037 | gru: 0.01067 / 0.01058 |
| 4 | 0.006547 / 0.006424 | 0.006975 / 0.00688 | 0.01068 / 0.01043 | 0.0135 / 0.01333 | affine: 0.006547 / 0.006424 |
| 5 | 0.004582 / 0.004456 | 0.005268 / 0.005247 | 0.009542 / 0.009342 | 0.0132 / 0.013 | affine: 0.004582 / 0.004456 |
| 6 | 0.003479 / 0.003391 | 0.004167 / 0.004121 | 0.008548 / 0.008284 | 0.01117 / 0.01115 | affine: 0.003479 / 0.003391 |

Class-specific elbows: affine k=6, gru k=6, tanh k=5, hmm k=6.

Selected commutation: `{"mse": 0.003464754304999159, "constant_decoder_mse": 0.027351036395852957, "invalid_simplex_fraction": 0.09142839687194526, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9375 | 0.9482 | 0.6378 | 0.7187 |
| quotient | 0.8933 | 0.9191 | 0.4226 | 0.4643 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.1749 | 0.005439 | in JSON |
| 2 | full | 0.002595 | 0.004576 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — control, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01473 / 0.01463 | 0.01473 / 0.01463 | 0.01473 / 0.01463 | 0.01473 / 0.01463 | affine: 0.01473 / 0.01463 |
| 1 | 0.001694 / 0.001745 | 0.001968 / 0.002009 | 0.002522 / 0.002566 | 0.004523 / 0.0047 | affine: 0.001694 / 0.001745 |
| 2 | 7.33e-05 / 7.486e-05 | 1.554e-05 / 1.664e-05 | 8.04e-05 / 8.25e-05 | 4.087e-06 / 4.4e-06 | hmm: 4.087e-06 / 4.4e-06 |
| 3 | 3.573e-05 / 3.854e-05 | 5.906e-06 / 6.359e-06 | 3.097e-05 / 3.186e-05 | 2.52e-06 / 2.741e-06 | hmm: 2.52e-06 / 2.741e-06 |
| 4 | 1.046e-05 / 1.072e-05 | 5.658e-06 / 5.967e-06 | 1.182e-05 / 1.25e-05 | 1.428e-06 / 1.62e-06 | hmm: 1.428e-06 / 1.62e-06 |
| 5 | 6.566e-06 / 7.027e-06 | 3.886e-06 / 3.898e-06 | 2.557e-05 / 2.624e-05 | 1.4e-06 / 1.479e-06 | hmm: 1.4e-06 / 1.479e-06 |
| 6 | 4.699e-06 / 4.625e-06 | 2.942e-06 / 3.036e-06 | 1.951e-05 / 2.06e-05 | 1.397e-06 / 1.499e-06 | hmm: 1.397e-06 / 1.499e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.01192810400600444, "constant_decoder_mse": 0.027879855566216926, "invalid_simplex_fraction": 0.15825849217986315, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.8978 | 0.9428 | 0.936 | 0.9516 |
| quotient | 0.8669 | 0.8893 | 0.1716 | 0.2843 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.05242 | 0.1142 | in JSON |
| 2 | full | 0.008257 | 0.01428 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — control, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01489 / 0.01508 | 0.01489 / 0.01508 | 0.01489 / 0.01508 | 0.01489 / 0.01508 | affine: 0.01489 / 0.01508 |
| 1 | 0.002072 / 0.002037 | 0.002219 / 0.002186 | 0.00296 / 0.002904 | 0.004963 / 0.004717 | affine: 0.002072 / 0.002037 |
| 2 | 8.416e-05 / 8.132e-05 | 1.553e-05 / 1.515e-05 | 9.856e-05 / 9.805e-05 | 4.213e-06 / 4.333e-06 | hmm: 4.213e-06 / 4.333e-06 |
| 3 | 4.357e-05 / 4.555e-05 | 7.9e-06 / 7.727e-06 | 4.205e-05 / 4.147e-05 | 2.843e-06 / 2.955e-06 | hmm: 2.843e-06 / 2.955e-06 |
| 4 | 1.83e-05 / 1.718e-05 | 5.157e-06 / 4.988e-06 | 2.329e-05 / 2.335e-05 | 7.317e-07 / 8.007e-07 | hmm: 7.317e-07 / 8.007e-07 |
| 5 | 1.489e-05 / 1.592e-05 | 4.677e-06 / 4.714e-06 | 2.487e-05 / 2.435e-05 | 1.471e-06 / 1.567e-06 | hmm: 1.471e-06 / 1.567e-06 |
| 6 | 6.777e-06 / 7.416e-06 | 3.326e-06 / 3.339e-06 | 1.599e-05 / 1.58e-05 | 1.455e-06 / 1.487e-06 | hmm: 1.455e-06 / 1.487e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.013100246361501333, "constant_decoder_mse": 0.026609729000151287, "invalid_simplex_fraction": 0.16523857526881722, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.901 | 0.9451 | 0.9308 | 0.9435 |
| quotient | 0.876 | 0.8999 | 0.3124 | 0.3093 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.04975 | 0.09999 | in JSON |
| 2 | full | 0.006242 | 0.01018 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Full-State Quotient — control, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.01462 / 0.01474 | 0.01462 / 0.01474 | 0.01462 / 0.01474 | 0.01462 / 0.01474 | affine: 0.01462 / 0.01474 |
| 1 | 0.001861 / 0.001854 | 0.002163 / 0.002175 | 0.002756 / 0.002703 | 0.004651 / 0.004644 | affine: 0.001861 / 0.001854 |
| 2 | 7.729e-05 / 7.793e-05 | 1.869e-05 / 1.881e-05 | 8.793e-05 / 9.234e-05 | 3.934e-06 / 3.888e-06 | hmm: 3.934e-06 / 3.888e-06 |
| 3 | 3.736e-05 / 3.7e-05 | 7.864e-06 / 8.165e-06 | 3.677e-05 / 3.77e-05 | 2.736e-06 / 2.727e-06 | hmm: 2.736e-06 / 2.727e-06 |
| 4 | 1.683e-05 / 1.798e-05 | 5.612e-06 / 5.733e-06 | 1.31e-05 / 1.32e-05 | 8.753e-07 / 7.58e-07 | hmm: 8.753e-07 / 7.58e-07 |
| 5 | 1.48e-05 / 1.453e-05 | 4.656e-06 / 4.936e-06 | 2.183e-05 / 2.229e-05 | 1.676e-06 / 1.513e-06 | hmm: 1.676e-06 / 1.513e-06 |
| 6 | 6.13e-06 / 6.036e-06 | 3.361e-06 / 3.486e-06 | 1.984e-05 / 2.048e-05 | 1.348e-06 / 1.198e-06 | hmm: 1.348e-06 / 1.198e-06 |

Class-specific elbows: affine k=2, gru k=2, tanh k=2, hmm k=2.

Selected commutation: `{"mse": 0.011271847127083214, "constant_decoder_mse": 0.027351036395852957, "invalid_simplex_fraction": 0.15387493890518084, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.9041 | 0.9465 | 0.933 | 0.9434 |
| quotient | 0.8709 | 0.8908 | 0.3471 | 0.3452 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | exact_reward_state_quotient (counterfactual) | 0.04891 | 0.09368 | in JSON |
| 2 | full | 0.005433 | 0.008354 | in JSON |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |

### Marginal Reward — policy, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.8742 / 0.8673 | 0.8742 / 0.8673 | 0.8742 / 0.8673 | 0.8742 / 0.8673 | affine: 0.8742 / 0.8673 |
| 1 | 0.135 / 0.1351 | 0.1427 / 0.1408 | 0.1773 / 0.1757 | 0.1997 / 0.1972 | affine: 0.135 / 0.1351 |
| 2 | 0.05226 / 0.05271 | 0.05038 / 0.05072 | 0.06605 / 0.06631 | 0.06507 / 0.06516 | gru: 0.05038 / 0.05072 |
| 3 | 0.01951 / 0.01932 | 0.02307 / 0.02297 | 0.03112 / 0.03088 | 0.03439 / 0.03417 | affine: 0.01951 / 0.01932 |
| 4 | 0.01252 / 0.01244 | 0.01222 / 0.01221 | 0.01941 / 0.01944 | 0.02486 / 0.02445 | gru: 0.01222 / 0.01221 |
| 5 | 0.00776 / 0.007699 | 0.00816 / 0.00805 | 0.01256 / 0.01247 | 0.01842 / 0.01792 | affine: 0.00776 / 0.007699 |
| 6 | 0.006444 / 0.006509 | 0.006829 / 0.006791 | 0.0109 / 0.0107 | 0.01697 / 0.01682 | affine: 0.006444 / 0.006509 |
| 7 | 0.005541 / 0.005499 | 0.006294 / 0.006287 | 0.009262 / 0.0091 | 0.01502 / 0.015 | affine: 0.005541 / 0.005499 |
| 8 | 0.00483 / 0.004748 | 0.005477 / 0.005406 | 0.008611 / 0.008563 | 0.01269 / 0.01256 | affine: 0.00483 / 0.004748 |
| 9 | 0.004409 / 0.004402 | 0.005085 / 0.005023 | 0.007994 / 0.008047 | 0.01151 / 0.01145 | affine: 0.004409 / 0.004402 |
| 10 | 0.003902 / 0.003893 | 0.004633 / 0.004615 | 0.007704 / 0.007694 | 0.01081 / 0.01064 | affine: 0.003902 / 0.003893 |

Class-specific elbows: affine k=8, gru k=8, tanh k=8, hmm k=9.

Selected commutation: `{"mse": 0.011912889820796099, "constant_decoder_mse": 0.018160386235515146, "invalid_simplex_fraction": 0.3994913856304985, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.4715 | 0.4789 | 0.7138 | 0.76 |
| factor_1 | 0.9227 | 0.9444 | 0.7113 | 0.7375 |
| factor_2 | 0.05828 | 0.05706 | 0.02585 | 0.04143 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.04169 | 0.05824 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Marginal Reward — policy, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.8386 / 0.8297 | 0.8386 / 0.8297 | 0.8386 / 0.8297 | 0.8386 / 0.8297 | affine: 0.8386 / 0.8297 |
| 1 | 0.1192 / 0.1159 | 0.1131 / 0.1104 | 0.1588 / 0.1549 | 0.1815 / 0.1778 | gru: 0.1131 / 0.1104 |
| 2 | 0.0349 / 0.03391 | 0.03102 / 0.03063 | 0.04091 / 0.03998 | 0.03853 / 0.03738 | gru: 0.03102 / 0.03063 |
| 3 | 0.01513 / 0.01475 | 0.01445 / 0.01422 | 0.02149 / 0.0214 | 0.02302 / 0.02303 | gru: 0.01445 / 0.01422 |
| 4 | 0.01062 / 0.01052 | 0.01131 / 0.01112 | 0.01584 / 0.0154 | 0.01979 / 0.01973 | affine: 0.01062 / 0.01052 |
| 5 | 0.007393 / 0.007329 | 0.008337 / 0.008166 | 0.01377 / 0.01355 | 0.01668 / 0.01675 | affine: 0.007393 / 0.007329 |
| 6 | 0.005862 / 0.005795 | 0.007309 / 0.007184 | 0.01137 / 0.01123 | 0.01469 / 0.01463 | affine: 0.005862 / 0.005795 |
| 7 | 0.005174 / 0.00507 | 0.00629 / 0.006335 | 0.01077 / 0.01061 | 0.01335 / 0.0133 | affine: 0.005174 / 0.00507 |
| 8 | 0.004695 / 0.004612 | 0.005675 / 0.005709 | 0.0094 / 0.009238 | 0.01293 / 0.01283 | affine: 0.004695 / 0.004612 |
| 9 | 0.004104 / 0.004092 | 0.00527 / 0.005244 | 0.008311 / 0.008176 | 0.01182 / 0.01172 | affine: 0.004104 / 0.004092 |
| 10 | 0.003774 / 0.003796 | 0.004805 / 0.004851 | 0.008912 / 0.008846 | 0.01189 / 0.01175 | affine: 0.003774 / 0.003796 |

Class-specific elbows: affine k=8, gru k=8, tanh k=9, hmm k=8.

Selected commutation: `{"mse": 0.008262277662715445, "constant_decoder_mse": 0.01947623747572097, "invalid_simplex_fraction": 0.27295637829912023, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.5996 | 0.6071 | 0.683 | 0.7501 |
| factor_1 | 0.9335 | 0.9474 | 0.6819 | 0.7258 |
| factor_2 | 0.01243 | 0.01391 | 0.003322 | 0.005702 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.03314 | 0.05904 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Marginal Reward — policy, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.8706 / 0.8618 | 0.8706 / 0.8618 | 0.8706 / 0.8618 | 0.8706 / 0.8618 | affine: 0.8706 / 0.8618 |
| 1 | 0.1334 / 0.1324 | 0.1247 / 0.1216 | 0.1767 / 0.1725 | 0.1959 / 0.1913 | gru: 0.1247 / 0.1216 |
| 2 | 0.04104 / 0.03988 | 0.04037 / 0.03942 | 0.04935 / 0.04812 | 0.04738 / 0.04576 | gru: 0.04037 / 0.03942 |
| 3 | 0.02154 / 0.02086 | 0.02305 / 0.02236 | 0.03608 / 0.03511 | 0.03511 / 0.03426 | affine: 0.02154 / 0.02086 |
| 4 | 0.01416 / 0.01376 | 0.016 / 0.01564 | 0.021 / 0.02011 | 0.02889 / 0.02805 | affine: 0.01416 / 0.01376 |
| 5 | 0.00931 / 0.009007 | 0.01117 / 0.01077 | 0.01884 / 0.01816 | 0.02446 / 0.02364 | affine: 0.00931 / 0.009007 |
| 6 | 0.007376 / 0.007106 | 0.009257 / 0.008913 | 0.01321 / 0.01285 | 0.01858 / 0.01797 | affine: 0.007376 / 0.007106 |
| 7 | 0.006502 / 0.006189 | 0.007926 / 0.007606 | 0.01121 / 0.01075 | 0.01628 / 0.01567 | affine: 0.006502 / 0.006189 |
| 8 | 0.00573 / 0.005482 | 0.007484 / 0.007196 | 0.01051 / 0.01014 | 0.01545 / 0.01497 | affine: 0.00573 / 0.005482 |
| 9 | 0.005134 / 0.004949 | 0.006985 / 0.00661 | 0.01047 / 0.01015 | 0.01401 / 0.01351 | affine: 0.005134 / 0.004949 |
| 10 | 0.004617 / 0.004441 | 0.00614 / 0.005839 | 0.009516 / 0.009163 | 0.01353 / 0.01317 | affine: 0.004617 / 0.004441 |

Class-specific elbows: affine k=9, gru k=9, tanh k=8, hmm k=9.

Selected commutation: `{"mse": 0.012030680720084421, "constant_decoder_mse": 0.018528903249328803, "invalid_simplex_fraction": 0.35063080400782015, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.4584 | 0.4629 | 0.6924 | 0.7311 |
| factor_1 | 0.9269 | 0.9449 | 0.6904 | 0.7257 |
| factor_2 | 0.03038 | 0.02817 | 0.01188 | 0.0132 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.04715 | 0.06357 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Marginal Reward — control, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.02445 / 0.02418 | 0.02445 / 0.02418 | 0.02445 / 0.02418 | 0.02445 / 0.02418 | affine: 0.02445 / 0.02418 |
| 1 | 0.01013 / 0.009919 | 0.00995 / 0.009733 | 0.01147 / 0.01112 | 0.01369 / 0.01323 | gru: 0.00995 / 0.009733 |
| 2 | 0.002604 / 0.002655 | 0.002429 / 0.002472 | 0.003384 / 0.003423 | 0.005148 / 0.005062 | gru: 0.002429 / 0.002472 |
| 3 | 0.0008182 / 0.0008292 | 0.000545 / 0.0005681 | 0.0007623 / 0.000798 | 0.001127 / 0.001121 | gru: 0.000545 / 0.0005681 |
| 4 | 0.0002579 / 0.0002524 | 5.12e-05 / 5.243e-05 | 0.0003494 / 0.0003564 | 0.0008664 / 0.0008571 | gru: 5.12e-05 / 5.243e-05 |
| 5 | 0.0002054 / 0.0002002 | 4.756e-05 / 4.803e-05 | 0.0002875 / 0.0002922 | 0.0007695 / 0.0007423 | gru: 4.756e-05 / 4.803e-05 |
| 6 | 9.353e-05 / 9.912e-05 | 2.967e-05 / 3.046e-05 | 0.0002066 / 0.0002 | 0.0007083 / 0.0006977 | gru: 2.967e-05 / 3.046e-05 |
| 7 | 7.05e-05 / 7.243e-05 | 2.642e-05 / 2.75e-05 | 0.0001507 / 0.0001487 | 0.0004015 / 0.0003701 | gru: 2.642e-05 / 2.75e-05 |
| 8 | 5.77e-05 / 6.007e-05 | 2.813e-05 / 2.861e-05 | 0.0001176 / 0.0001176 | 0.0003707 / 0.0003428 | gru: 2.813e-05 / 2.861e-05 |
| 9 | 3.687e-05 / 3.91e-05 | 1.688e-05 / 1.708e-05 | 0.0001248 / 0.0001234 | 0.0003661 / 0.0003442 | gru: 1.688e-05 / 1.708e-05 |
| 10 | 3.122e-05 / 3.096e-05 | 1.813e-05 / 1.833e-05 | 0.0001101 / 0.0001073 | 0.000458 / 0.0004567 | gru: 1.813e-05 / 1.833e-05 |

Class-specific elbows: affine k=3, gru k=3, tanh k=3, hmm k=3.

Selected commutation: `{"mse": 0.08697936404199795, "constant_decoder_mse": 0.018160381174211263, "invalid_simplex_fraction": 0.8267160007331378, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.7381 | 0.9261 | 0.9726 | 0.9746 |
| factor_1 | 0.9648 | 0.9728 | 0.5304 | 0.5277 |
| factor_2 | 0.9103 | 0.9155 | 0.4821 | 0.4888 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.03272 | 0.05711 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Marginal Reward — control, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.02061 / 0.02094 | 0.02061 / 0.02094 | 0.02061 / 0.02094 | 0.02061 / 0.02094 | affine: 0.02061 / 0.02094 |
| 1 | 0.00723 / 0.007536 | 0.007773 / 0.008073 | 0.008353 / 0.008628 | 0.01049 / 0.01068 | affine: 0.00723 / 0.007536 |
| 2 | 0.002134 / 0.002134 | 0.002199 / 0.002218 | 0.002963 / 0.002952 | 0.004494 / 0.004496 | affine: 0.002134 / 0.002134 |
| 3 | 0.0004532 / 0.0004651 | 0.000282 / 0.000286 | 0.0003864 / 0.0003903 | 0.0006424 / 0.0006696 | gru: 0.000282 / 0.000286 |
| 4 | 0.0001949 / 0.0001925 | 3.625e-05 / 3.516e-05 | 0.0002161 / 0.0002152 | 0.0004624 / 0.0004857 | gru: 3.625e-05 / 3.516e-05 |
| 5 | 0.0001373 / 0.0001374 | 2.574e-05 / 2.474e-05 | 0.0002209 / 0.0002196 | 0.0004206 / 0.0004385 | gru: 2.574e-05 / 2.474e-05 |
| 6 | 7.063e-05 / 7.585e-05 | 2.567e-05 / 2.526e-05 | 0.0001029 / 0.0001007 | 0.0003976 / 0.0004126 | gru: 2.567e-05 / 2.526e-05 |
| 7 | 5.547e-05 / 5.784e-05 | 1.813e-05 / 1.797e-05 | 9.729e-05 / 9.615e-05 | 0.000372 / 0.0003856 | gru: 1.813e-05 / 1.797e-05 |
| 8 | 4.326e-05 / 4.601e-05 | 1.63e-05 / 1.661e-05 | 8.653e-05 / 8.661e-05 | 0.0003248 / 0.0003357 | gru: 1.63e-05 / 1.661e-05 |
| 9 | 3.545e-05 / 3.572e-05 | 1.575e-05 / 1.542e-05 | 9.16e-05 / 8.958e-05 | 0.0001771 / 0.000187 | gru: 1.575e-05 / 1.542e-05 |
| 10 | 2.32e-05 / 2.424e-05 | 1.595e-05 / 1.642e-05 | 7.547e-05 / 7.456e-05 | 0.0002119 / 0.000225 | gru: 1.595e-05 / 1.642e-05 |

Class-specific elbows: affine k=3, gru k=3, tanh k=3, hmm k=3.

Selected commutation: `{"mse": 1.4482352338722728, "constant_decoder_mse": 0.019476237350486304, "invalid_simplex_fraction": 0.8274758675464321, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.7984 | 0.939 | 0.9594 | 0.9634 |
| factor_1 | 0.9529 | 0.9561 | 0.718 | 0.7192 |
| factor_2 | 0.9341 | 0.9457 | 0.2465 | 0.2531 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.02502 | 0.06383 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Marginal Reward — control, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.02386 / 0.02413 | 0.02386 / 0.02413 | 0.02386 / 0.02413 | 0.02386 / 0.02413 | affine: 0.02386 / 0.02413 |
| 1 | 0.009787 / 0.009872 | 0.009924 / 0.01002 | 0.01118 / 0.01131 | 0.01348 / 0.01355 | affine: 0.009787 / 0.009872 |
| 2 | 0.002514 / 0.002499 | 0.002448 / 0.002422 | 0.003231 / 0.003108 | 0.005614 / 0.005559 | gru: 0.002448 / 0.002422 |
| 3 | 0.0008205 / 0.0008313 | 0.0005801 / 0.0005914 | 0.0007888 / 0.0007923 | 0.001132 / 0.001138 | gru: 0.0005801 / 0.0005914 |
| 4 | 0.0002558 / 0.0002528 | 5.171e-05 / 5.2e-05 | 0.0004548 / 0.0004532 | 0.0008388 / 0.0008433 | gru: 5.171e-05 / 5.2e-05 |
| 5 | 0.0002022 / 0.0001985 | 4.507e-05 / 4.491e-05 | 0.0002531 / 0.000258 | 0.000726 / 0.0007246 | gru: 4.507e-05 / 4.491e-05 |
| 6 | 0.0001123 / 0.0001115 | 3.41e-05 / 3.43e-05 | 0.000132 / 0.0001308 | 0.0006178 / 0.0006217 | gru: 3.41e-05 / 3.43e-05 |
| 7 | 6.747e-05 / 6.627e-05 | 2.809e-05 / 2.749e-05 | 0.0001571 / 0.0001565 | 0.0004079 / 0.0004049 | gru: 2.809e-05 / 2.749e-05 |
| 8 | 5.971e-05 / 5.926e-05 | 2.331e-05 / 2.339e-05 | 0.0001164 / 0.0001156 | 0.0005083 / 0.0005002 | gru: 2.331e-05 / 2.339e-05 |
| 9 | 4.337e-05 / 4.16e-05 | 1.98e-05 / 1.972e-05 | 8.804e-05 / 8.754e-05 | 0.0003335 / 0.0003315 | gru: 1.98e-05 / 1.972e-05 |
| 10 | 3.355e-05 / 3.165e-05 | 1.842e-05 / 1.861e-05 | 0.0001021 / 0.0001012 | 0.0001947 / 0.0001911 | gru: 1.842e-05 / 1.861e-05 |

Class-specific elbows: affine k=3, gru k=3, tanh k=3, hmm k=3.

Selected commutation: `{"mse": 0.07550717845511058, "constant_decoder_mse": 0.01852895900339112, "invalid_simplex_fraction": 0.8230961326979472, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.7278 | 0.9362 | 0.9709 | 0.9722 |
| factor_1 | 0.9665 | 0.9717 | 0.6391 | 0.6388 |
| factor_2 | 0.909 | 0.913 | 0.354 | 0.3572 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.03932 | 0.06209 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — policy, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 1.201 / 1.188 | 1.201 / 1.188 | 1.201 / 1.188 | 1.201 / 1.188 | affine: 1.201 / 1.188 |
| 1 | 0.3571 / 0.3488 | 0.3459 / 0.3399 | 0.3865 / 0.3781 | 0.4899 / 0.4791 | gru: 0.3459 / 0.3399 |
| 2 | 0.2159 / 0.2084 | 0.2293 / 0.2212 | 0.2637 / 0.2574 | 0.2949 / 0.2873 | affine: 0.2159 / 0.2084 |
| 3 | 0.1283 / 0.1238 | 0.1331 / 0.1286 | 0.1591 / 0.1529 | 0.181 / 0.1749 | affine: 0.1283 / 0.1238 |
| 4 | 0.07578 / 0.07368 | 0.07428 / 0.07221 | 0.09117 / 0.09007 | 0.1043 / 0.1034 | gru: 0.07428 / 0.07221 |
| 5 | 0.04502 / 0.0443 | 0.04813 / 0.04744 | 0.06117 / 0.05986 | 0.07234 / 0.071 | affine: 0.04502 / 0.0443 |
| 6 | 0.03358 / 0.03308 | 0.03773 / 0.03712 | 0.05201 / 0.05128 | 0.05907 / 0.05787 | affine: 0.03358 / 0.03308 |
| 7 | 0.02774 / 0.02697 | 0.03235 / 0.0317 | 0.04527 / 0.04445 | 0.05458 / 0.05366 | affine: 0.02774 / 0.02697 |
| 8 | 0.0238 / 0.02321 | 0.02834 / 0.02764 | 0.04216 / 0.04124 | 0.05253 / 0.05126 | affine: 0.0238 / 0.02321 |
| 9 | 0.02088 / 0.02026 | 0.02595 / 0.02527 | 0.04005 / 0.03917 | 0.05022 / 0.04886 | affine: 0.02088 / 0.02026 |
| 10 | 0.01945 / 0.01891 | 0.02396 / 0.02344 | 0.03562 / 0.03491 | 0.0481 / 0.04683 | affine: 0.01945 / 0.01891 |

Class-specific elbows: affine k=9, gru k=9, tanh k=10, hmm k=8.

Selected commutation: `{"mse": 3.5069137439781066, "constant_decoder_mse": 0.01823535091444276, "invalid_simplex_fraction": 0.9254834127565983, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6141 | 0.8567 | 0.7135 | 0.7292 |
| factor_1 | 0.8787 | 0.9284 | 0.3776 | 0.3816 |
| factor_2 | 0.8501 | 0.92 | 0.3257 | 0.3292 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.03289 | 0.04662 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — policy, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 1.281 / 1.276 | 1.281 / 1.276 | 1.281 / 1.276 | 1.281 / 1.276 | affine: 1.281 / 1.276 |
| 1 | 0.3268 / 0.329 | 0.3243 / 0.3263 | 0.3466 / 0.348 | 0.4488 / 0.4495 | gru: 0.3243 / 0.3263 |
| 2 | 0.2076 / 0.2098 | 0.2114 / 0.2101 | 0.2441 / 0.2457 | 0.2975 / 0.3009 | affine: 0.2076 / 0.2098 |
| 3 | 0.1198 / 0.1199 | 0.1216 / 0.1225 | 0.1517 / 0.1527 | 0.1807 / 0.1814 | affine: 0.1198 / 0.1199 |
| 4 | 0.0635 / 0.06478 | 0.06575 / 0.06616 | 0.08595 / 0.08642 | 0.0985 / 0.09863 | affine: 0.0635 / 0.06478 |
| 5 | 0.0423 / 0.04348 | 0.04508 / 0.04565 | 0.0639 / 0.06455 | 0.07182 / 0.07273 | affine: 0.0423 / 0.04348 |
| 6 | 0.033 / 0.03403 | 0.03692 / 0.03788 | 0.04999 / 0.05058 | 0.06248 / 0.06309 | affine: 0.033 / 0.03403 |
| 7 | 0.026 / 0.02646 | 0.02916 / 0.02977 | 0.04486 / 0.04579 | 0.05373 / 0.05485 | affine: 0.026 / 0.02646 |
| 8 | 0.0231 / 0.02347 | 0.02694 / 0.02733 | 0.03915 / 0.04016 | 0.05228 / 0.05344 | affine: 0.0231 / 0.02347 |
| 9 | 0.02067 / 0.02097 | 0.02511 / 0.02559 | 0.03711 / 0.03799 | 0.04641 / 0.04735 | affine: 0.02067 / 0.02097 |
| 10 | 0.01884 / 0.01904 | 0.02336 / 0.02387 | 0.03379 / 0.03481 | 0.0438 / 0.04485 | affine: 0.01884 / 0.01904 |

Class-specific elbows: affine k=9, gru k=9, tanh k=9, hmm k=9.

Selected commutation: `{"mse": 1.2576919045360202, "constant_decoder_mse": 0.01854856586405921, "invalid_simplex_fraction": 0.8960700757575758, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6339 | 0.8582 | 0.6878 | 0.6843 |
| factor_1 | 0.8792 | 0.9164 | 0.4001 | 0.3981 |
| factor_2 | 0.8644 | 0.9118 | 0.2784 | 0.2764 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.04088 | 0.04587 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — policy, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 1.271 / 1.275 | 1.271 / 1.275 | 1.271 / 1.275 | 1.271 / 1.275 | affine: 1.271 / 1.275 |
| 1 | 0.3423 / 0.3404 | 0.3358 / 0.3338 | 0.3845 / 0.3825 | 0.4699 / 0.4644 | gru: 0.3358 / 0.3338 |
| 2 | 0.2145 / 0.2108 | 0.2141 / 0.2125 | 0.2472 / 0.2458 | 0.2934 / 0.292 | gru: 0.2141 / 0.2125 |
| 3 | 0.1126 / 0.1104 | 0.1162 / 0.1143 | 0.1421 / 0.1389 | 0.1584 / 0.1554 | affine: 0.1126 / 0.1104 |
| 4 | 0.06686 / 0.06748 | 0.06538 / 0.06655 | 0.08175 / 0.08241 | 0.09267 / 0.09351 | gru: 0.06538 / 0.06655 |
| 5 | 0.03832 / 0.03902 | 0.04202 / 0.04268 | 0.05351 / 0.05382 | 0.06617 / 0.06691 | affine: 0.03832 / 0.03902 |
| 6 | 0.03089 / 0.03131 | 0.03536 / 0.03556 | 0.04783 / 0.04854 | 0.05841 / 0.05934 | affine: 0.03089 / 0.03131 |
| 7 | 0.02604 / 0.02607 | 0.03119 / 0.03153 | 0.04518 / 0.04556 | 0.05254 / 0.05329 | affine: 0.02604 / 0.02607 |
| 8 | 0.02333 / 0.02358 | 0.02737 / 0.02747 | 0.0416 / 0.04211 | 0.0502 / 0.05066 | affine: 0.02333 / 0.02358 |
| 9 | 0.02052 / 0.02066 | 0.02462 / 0.02481 | 0.03895 / 0.03895 | 0.04642 / 0.04679 | affine: 0.02052 / 0.02066 |
| 10 | 0.01873 / 0.01898 | 0.02359 / 0.02386 | 0.03453 / 0.03449 | 0.04295 / 0.04306 | affine: 0.01873 / 0.01898 |

Class-specific elbows: affine k=9, gru k=9, tanh k=10, hmm k=9.

Selected commutation: `{"mse": 0.10579395285257988, "constant_decoder_mse": 0.01877148535406763, "invalid_simplex_fraction": 0.8952300219941349, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6226 | 0.8572 | 0.6998 | 0.6939 |
| factor_1 | 0.8446 | 0.9183 | 0.3768 | 0.3775 |
| factor_2 | 0.8787 | 0.9197 | 0.307 | 0.3015 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.03909 | 0.05842 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — control, seed 42

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.03058 / 0.03092 | 0.03058 / 0.03092 | 0.03058 / 0.03092 | 0.03058 / 0.03092 | affine: 0.03058 / 0.03092 |
| 1 | 0.01462 / 0.01461 | 0.01577 / 0.01549 | 0.0175 / 0.01713 | 0.02077 / 0.02061 | affine: 0.01462 / 0.01461 |
| 2 | 0.00547 / 0.005408 | 0.005621 / 0.005529 | 0.007392 / 0.007332 | 0.01124 / 0.01099 | affine: 0.00547 / 0.005408 |
| 3 | 0.002244 / 0.002231 | 0.002355 / 0.002334 | 0.003191 / 0.003125 | 0.004808 / 0.004593 | affine: 0.002244 / 0.002231 |
| 4 | 0.0002985 / 0.0002977 | 8.423e-05 / 8.493e-05 | 0.0003208 / 0.0003153 | 0.001254 / 0.001296 | gru: 8.423e-05 / 8.493e-05 |
| 5 | 0.0002131 / 0.0002099 | 5.746e-05 / 5.825e-05 | 0.0002505 / 0.0002499 | 0.0006819 / 0.0006977 | gru: 5.746e-05 / 5.825e-05 |
| 6 | 0.0001713 / 0.0001693 | 4.684e-05 / 4.58e-05 | 0.0001792 / 0.00018 | 0.000688 / 0.000696 | gru: 4.684e-05 / 4.58e-05 |
| 7 | 0.0001295 / 0.0001263 | 2.958e-05 / 2.996e-05 | 0.0001784 / 0.0001753 | 0.0006291 / 0.0006441 | gru: 2.958e-05 / 2.996e-05 |
| 8 | 9.629e-05 / 9.446e-05 | 2.604e-05 / 2.629e-05 | 0.0001545 / 0.0001531 | 0.0004133 / 0.0004278 | gru: 2.604e-05 / 2.629e-05 |
| 9 | 6.983e-05 / 6.984e-05 | 2.368e-05 / 2.364e-05 | 0.0001188 / 0.0001142 | 0.0003573 / 0.0003869 | gru: 2.368e-05 / 2.364e-05 |
| 10 | 5.398e-05 / 5.546e-05 | 2.136e-05 / 2.186e-05 | 0.0001406 / 0.000138 | 0.0005441 / 0.0005648 | gru: 2.136e-05 / 2.186e-05 |

Class-specific elbows: affine k=4, gru k=4, tanh k=4, hmm k=4.

Selected commutation: `{"mse": 11.42119328177609, "constant_decoder_mse": 0.01823535091444276, "invalid_simplex_fraction": 0.9560308223362659, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6715 | 0.957 | 0.9693 | 0.9689 |
| factor_1 | 0.977 | 0.9786 | 0.4805 | 0.4801 |
| factor_2 | 0.9675 | 0.9731 | 0.4886 | 0.4877 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.04598 | 0.05641 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — control, seed 43

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.03073 / 0.03055 | 0.03073 / 0.03055 | 0.03073 / 0.03055 | 0.03073 / 0.03055 | affine: 0.03073 / 0.03055 |
| 1 | 0.0133 / 0.01347 | 0.01429 / 0.01427 | 0.01639 / 0.01629 | 0.02053 / 0.02037 | affine: 0.0133 / 0.01347 |
| 2 | 0.004862 / 0.00488 | 0.005099 / 0.005104 | 0.007033 / 0.006944 | 0.01078 / 0.01056 | affine: 0.004862 / 0.00488 |
| 3 | 0.002013 / 0.002046 | 0.002065 / 0.002067 | 0.002734 / 0.002686 | 0.00419 / 0.004233 | affine: 0.002013 / 0.002046 |
| 4 | 0.0002767 / 0.0002749 | 7.132e-05 / 7.133e-05 | 0.0003809 / 0.0003858 | 0.001058 / 0.001078 | gru: 7.132e-05 / 7.133e-05 |
| 5 | 0.0001992 / 0.0002021 | 5.067e-05 / 5.02e-05 | 0.0002377 / 0.0002429 | 0.0007559 / 0.0007782 | gru: 5.067e-05 / 5.02e-05 |
| 6 | 0.0001378 / 0.0001366 | 4.382e-05 / 4.332e-05 | 0.0001622 / 0.0001608 | 0.0005899 / 0.0006021 | gru: 4.382e-05 / 4.332e-05 |
| 7 | 0.0001138 / 0.0001113 | 2.916e-05 / 2.834e-05 | 0.0001717 / 0.0001727 | 0.0007301 / 0.0007649 | gru: 2.916e-05 / 2.834e-05 |
| 8 | 8.688e-05 / 8.766e-05 | 2.245e-05 / 2.145e-05 | 0.0001323 / 0.0001331 | 0.000409 / 0.0004004 | gru: 2.245e-05 / 2.145e-05 |
| 9 | 6.155e-05 / 6.299e-05 | 2.429e-05 / 2.363e-05 | 0.0001038 / 0.0001047 | 0.0004368 / 0.0004589 | gru: 2.429e-05 / 2.363e-05 |
| 10 | 4.674e-05 / 4.52e-05 | 2.018e-05 / 1.94e-05 | 0.0001185 / 0.0001189 | 0.0004881 / 0.0005096 | gru: 2.018e-05 / 1.94e-05 |

Class-specific elbows: affine k=4, gru k=4, tanh k=4, hmm k=4.

Selected commutation: `{"mse": 0.22254472233459843, "constant_decoder_mse": 0.01854856586405921, "invalid_simplex_fraction": 0.9532128238025416, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.6771 | 0.931 | 0.9745 | 0.9732 |
| factor_1 | 0.9772 | 0.9792 | 0.6469 | 0.6461 |
| factor_2 | 0.9675 | 0.9737 | 0.3275 | 0.3252 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.05081 | 0.05801 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |

### Joint Reward — control, seed 44

| k | Affine | GRU | Tanh | HMM | Best class: val / test |
|---:|---|---|---|---|---|
| 0 | 0.03028 / 0.03035 | 0.03028 / 0.03035 | 0.03028 / 0.03035 | 0.03028 / 0.03035 | affine: 0.03028 / 0.03035 |
| 1 | 0.01412 / 0.01404 | 0.01437 / 0.01443 | 0.01575 / 0.01577 | 0.02013 / 0.02057 | affine: 0.01412 / 0.01404 |
| 2 | 0.004899 / 0.004876 | 0.005262 / 0.005163 | 0.006962 / 0.006928 | 0.01029 / 0.01023 | affine: 0.004899 / 0.004876 |
| 3 | 0.002062 / 0.002062 | 0.002013 / 0.002066 | 0.002869 / 0.003006 | 0.004337 / 0.0043 | gru: 0.002013 / 0.002066 |
| 4 | 0.0002545 / 0.0002481 | 6.356e-05 / 6.104e-05 | 0.0002986 / 0.0002966 | 0.000867 / 0.0008777 | gru: 6.356e-05 / 6.104e-05 |
| 5 | 0.0002117 / 0.0002089 | 4.975e-05 / 4.943e-05 | 0.0002245 / 0.0002228 | 0.000583 / 0.0005782 | gru: 4.975e-05 / 4.943e-05 |
| 6 | 0.0001573 / 0.000151 | 3.716e-05 / 3.825e-05 | 0.0001614 / 0.0001562 | 0.0005087 / 0.0005165 | gru: 3.716e-05 / 3.825e-05 |
| 7 | 0.0001143 / 0.0001083 | 2.407e-05 / 2.295e-05 | 0.000157 / 0.0001579 | 0.0004488 / 0.0004554 | gru: 2.407e-05 / 2.295e-05 |
| 8 | 9.057e-05 / 8.751e-05 | 1.846e-05 / 1.795e-05 | 0.0001469 / 0.0001419 | 0.0004593 / 0.0004577 | gru: 1.846e-05 / 1.795e-05 |
| 9 | 7.097e-05 / 6.7e-05 | 2.216e-05 / 2.165e-05 | 0.0001471 / 0.0001479 | 0.0004112 / 0.0004023 | gru: 2.216e-05 / 2.165e-05 |
| 10 | 4.862e-05 / 4.687e-05 | 1.929e-05 / 1.886e-05 | 9.82e-05 / 9.608e-05 | 0.0003557 / 0.0003604 | gru: 1.929e-05 / 1.886e-05 |

Class-specific elbows: affine k=4, gru k=4, tanh k=4, hmm k=4.

Selected commutation: `{"mse": 0.13362502892090353, "constant_decoder_mse": 0.01877148535406763, "invalid_simplex_fraction": 0.952949352394917, "nonfinite_fraction": 0.0}`.

| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |
|---|---:|---:|---:|---:|
| belief | 0.673 | 0.937 | 0.9675 | 0.9679 |
| factor_1 | 0.9701 | 0.9722 | 0.6033 | 0.6033 |
| factor_2 | 0.9701 | 0.9748 | 0.3656 | 0.3656 |

HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):

| k | Reference | Transition MSE | Emission MSE | Matrix entries |
|---:|---|---:|---:|---|
| 0 | different state counts | — | — | not applicable |
| 1 | different state counts | — | — | not applicable |
| 2 | different state counts | — | — | not applicable |
| 3 | different state counts | — | — | not applicable |
| 4 | different state counts | — | — | not applicable |
| 5 | different state counts | — | — | not applicable |
| 6 | different state counts | — | — | not applicable |
| 7 | different state counts | — | — | not applicable |
| 8 | full | 0.04376 | 0.06025 | in JSON |
| 9 | different state counts | — | — | not applicable |
| 10 | different state counts | — | — | not applicable |
