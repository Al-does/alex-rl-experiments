# Token-guess cycle 2 hyperparameter sweeps

Each RLlib Tune grid used seed 42 and approximately 2.5 million environment
steps per point. Selection uses the final RLlib episode window's mean token
accuracy.

| sweep | grid | best point | best accuracy |
|---|---|---:|---:|
| PPO learning rate | `3e-5, 1e-4, 3e-4, 1e-3` | `1e-4` | 68.8839% |
| predictive-loss coefficient | `0.01, 0.03, 0.1, 0.3` | `0.3` | 68.6077% |
| Kelly-loss coefficient | `0.1, 0.3, 1.0, 3.0` | `1.0` | 66.9872% |

The learning-rate grid spans 33.3x. Its `1e-4` result is 0.0739 percentage
points below the exact finite-context Bayes ceiling (68.9577%), but `3e-5` is
only 0.0378 points behind `1e-4`. A single seed does not distinguish those two
rates reliably.

The coefficient sweeps are one-factor-at-a-time controls and therefore retain
the original `3e-4` learning rate; they do not use the winning PPO rate. Both
coefficient grids are strongly non-monotonic, with unusually poor `0.03`
predictive and `0.3` Kelly runs. Replicate the selected points and those failure
points across multiple seeds before promoting defaults. A subsequent joint
validation can combine `1e-4` with predictive coefficient `0.3` and Kelly
coefficient `1.0`.

Provenance:

- experiment commit: `516f2b654e54f14617500b26a1a0d8c96a143fd1`
- RL-Harness commit: `27713b1b8cf6dc6a5af45c9018a64abe26a108a2`
- full Tune outputs and compact results: hash-manifested in B2 under each run ID
