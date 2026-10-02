# RockSample d128_kl0.1 belief probes

## Main measurements

All probe points below use identical final-policy histories for initialization and final representations.

| Seed | Fresh return ± SEM | M init 1−R² | M final 1−R² | J init 1−R² | J final 1−R² |
|---|---:|---:|---:|---:|---:|
| 0 | 44.082 ± 0.582 | 0.5138 | 0.0937 | 0.7945 | 0.3732 |
| 1 | 39.297 ± 0.546 | 0.5539 | 0.2195 | 0.8865 | 0.6112 |
| 2 | 44.004 ± 0.574 | 0.4923 | 0.0926 | 0.7712 | 0.3603 |
| 3 | 44.395 ± 0.571 | 0.4967 | 0.0728 | 0.8001 | 0.3370 |

## Per-rock diagnostics

Undefined fits have zero target variance or no eligible rows. Rock indices are zero based.

| Seed | Rock | Check episodes | Good-sample episodes | Prior at end | On-policy R² (all) | On-policy R² (informed/relevant) | Forced-check R² (informed/relevant) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 (1, 0) | 0.996 | 0.500 | 0.004 | 0.9360 | 0.9345 | 0.9745 |
| 0 | 1 (2, 1) | 0.998 | 0.541 | 0.002 | 0.9566 | 0.9509 | 0.9703 |
| 0 | 2 (1, 2) | 0.998 | 0.473 | 0.002 | 0.7401 | 0.6929 | 0.9363 |
| 0 | 3 (2, 2) | 1.000 | 0.473 | 0.000 | 0.9279 | 0.9274 | 0.9700 |
| 0 | 4 (4, 2) | 1.000 | 0.516 | 0.000 | 0.9625 | 0.9759 | 0.9781 |
| 0 | 5 (0, 3) | 1.000 | 0.480 | 0.000 | 0.6045 | 0.7363 | 0.1236 |
| 0 | 6 (3, 4) | 1.000 | 0.504 | 0.000 | 0.9272 | 0.9335 | 0.9692 |
| 1 | 0 (1, 0) | 0.834 | 0.000 | 0.174 | -0.1254 | -0.0732 | 0.5826 |
| 1 | 1 (2, 1) | 0.998 | 0.543 | 0.002 | 0.9547 | 0.9560 | 0.9741 |
| 1 | 2 (1, 2) | 1.000 | 0.477 | 0.000 | 0.8250 | 0.7709 | 0.8468 |
| 1 | 3 (2, 2) | 1.000 | 0.471 | 0.000 | 0.9441 | 0.9406 | 0.9774 |
| 1 | 4 (4, 2) | 0.996 | 0.498 | 0.000 | 0.9588 | 0.9728 | 0.9820 |
| 1 | 5 (0, 3) | 1.000 | 0.484 | 0.000 | 0.8251 | 0.8250 | 0.5352 |
| 1 | 6 (3, 4) | 0.996 | 0.504 | 0.004 | 0.9654 | 0.9663 | 0.9787 |
| 2 | 0 (1, 0) | 0.994 | 0.494 | 0.006 | 0.9128 | 0.9195 | 0.9541 |
| 2 | 1 (2, 1) | 0.998 | 0.539 | 0.002 | 0.9569 | 0.9585 | 0.9617 |
| 2 | 2 (1, 2) | 1.000 | 0.475 | 0.000 | 0.7198 | 0.6866 | 0.7947 |
| 2 | 3 (2, 2) | 1.000 | 0.471 | 0.000 | 0.9502 | 0.9536 | 0.9607 |
| 2 | 4 (4, 2) | 1.000 | 0.516 | 0.000 | 0.9476 | 0.9556 | 0.9806 |
| 2 | 5 (0, 3) | 0.998 | 0.482 | 0.002 | 0.6149 | 0.6328 | 0.0325 |
| 2 | 6 (3, 4) | 0.998 | 0.504 | 0.002 | 0.9543 | 0.9610 | 0.9754 |
| 3 | 0 (1, 0) | 0.996 | 0.500 | 0.004 | 0.9199 | 0.9216 | 0.9425 |
| 3 | 1 (2, 1) | 1.000 | 0.549 | 0.000 | 0.9202 | 0.9097 | 0.9460 |
| 3 | 2 (1, 2) | 0.998 | 0.480 | 0.000 | 0.8981 | 0.8870 | 0.8154 |
| 3 | 3 (2, 2) | 0.996 | 0.465 | 0.004 | 0.9047 | 0.9018 | 0.9540 |
| 3 | 4 (4, 2) | 0.998 | 0.508 | 0.002 | 0.9571 | 0.9669 | 0.9627 |
| 3 | 5 (0, 3) | 1.000 | 0.488 | 0.000 | 0.8592 | 0.8478 | 0.5795 |
| 3 | 6 (3, 4) | 1.000 | 0.504 | 0.000 | 0.9563 | 0.9639 | 0.9680 |

## Interpretation and limits

M is the seven-dimensional vector of current Good probabilities. J is the 128-configuration product posterior. An affine map to M need not linearly represent the nonlinear product J.

R² is 1 − held-out MSE / held-out coordinate-averaged target variance; predictions are unprojected. Raw MSE, target variance, sample coverage, training-only SVD-cutoff selection, and episode-bootstrap comparisons are in the seed JSONs.

Relevant means unsampled and P(Good)>0. Informed/relevant additionally excludes the unchanged 0.5 prior. Certainty at 1 remains relevant. Subset scores evaluate the unrestricted probe without refitting on test rows.

The primary distribution is independently reset episodes under each final stochastic policy. The matched initialization encoder sees the same actions and observations. Initial-policy rollouts provide an additional initialization-distribution baseline, recorded separately; task returns always use each policy's own rollouts.

The common-check control uses identical histories across seeds: check all rocks remotely, visit each rock, check at distance zero, sample, then exit. Probes are refit on disjoint episodes in this off-policy distribution. It tests accessibility when evidence is supplied, not causal use or on-policy transfer.

The initialization parameters are reconstructed from archived runtime seed 42 and both original key splits, with equality tested against ppo.init. There was no saved initialization pickle. Only the step-zero and 30,408,704-step probes exist; connecting lines are visual guides, not intermediate measurements.

Joint-posterior controls include observable/current-input plus sampled-status/time features, next-sensor Good probabilities, training-label permutations, and training-covariance Gaussian feature nulls (three repetitions). Prediction of sensor readings is closely tied to marginals and is not an independent representation claim.

Coordinate traces select the first test episode by the fixed episode split. Check-update metrics measure decoded differences before/after informative checks; no terminal observation is included. Privileged qualities appear only in calibration diagnostics; rewards appear only in behavioral summaries.

Calibration bins and predictive sensor totals are descriptive aggregates of correlated episode rows. Bootstrap intervals resample episodes with fitted probes held fixed; they do not capture training-seed variability or refit uncertainty.

Evaluation uses 512 complete episodes per policy/distribution, split equally for probe fitting and evaluation. No warmup is removed; exact cache replay begins at reset and retains the complete model receptive field.
