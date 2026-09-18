# RRXOR supervised paper replication

- Analyzed checkpoint: update 1,000,000
- Exact length-10-sequence Bayesian floor: 0.586501 nats
- Exact validation loss: 0.586902 nats (gap +0.000400)
- Concatenated-layer affine probe: MSE 0.027689, R² 0.662606
- Pairwise-distance R²: belief 0.7067, next-token 0.3472
- Scientific checks: PASS

Training used only shifted next-token cross-entropy. Exact beliefs were used only for post-training analysis.
