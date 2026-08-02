# MESS3 supervised replication: swish-mlp-decoder-2-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802531 nats (gap +0.000020)
- Final pre-LN affine probe: MSE 0.000204966, R² 0.998156
- Scientific gate: PASS
- Active optimization: 1847.2s at 33.8 updates/s
- End-to-end training wall time: 1849.9s
- Probe/plot wall time: 3.8s
- Total experiment wall time: 1853.9s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
