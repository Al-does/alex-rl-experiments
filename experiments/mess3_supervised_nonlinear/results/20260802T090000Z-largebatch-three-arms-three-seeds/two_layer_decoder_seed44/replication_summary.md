# MESS3 supervised replication: swish-mlp-decoder-2-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802543 nats (gap +0.000031)
- Final pre-LN affine probe: MSE 0.000276914, R² 0.997509
- Scientific gate: PASS
- Active optimization: 1855.4s at 33.7 updates/s
- End-to-end training wall time: 1858.0s
- Probe/plot wall time: 3.9s
- Total experiment wall time: 1862.1s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
