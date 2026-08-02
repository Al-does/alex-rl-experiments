# MESS3 supervised replication: swish-mlp-decoder-4-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802546 nats (gap +0.000034)
- Final pre-LN affine probe: MSE 0.000226662, R² 0.997961
- Scientific gate: PASS
- Active optimization: 1946.7s at 32.1 updates/s
- End-to-end training wall time: 1949.5s
- Probe/plot wall time: 3.8s
- Total experiment wall time: 1953.5s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
