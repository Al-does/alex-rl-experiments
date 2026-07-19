# MESS3 supervised replication: large-batch-sqrt-scaled

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802556 nats (gap +0.000044)
- Final pre-LN affine probe: MSE 0.000241159, R² 0.997830
- Scientific gate: PASS
- Active optimization: 1356.0s at 46.1 updates/s
- End-to-end training wall time: 1358.6s
- Probe/plot wall time: 4.5s
- Total experiment wall time: 1364.4s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
