# MESS3 supervised replication: large-batch-sqrt-scaled

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802556 nats (gap +0.000044)
- Final pre-LN affine probe: MSE 0.000230488, R² 0.997926
- Scientific gate: PASS
- Active optimization: 1367.3s at 45.7 updates/s
- End-to-end training wall time: 1370.6s
- Probe/plot wall time: 4.6s
- Total experiment wall time: 1379.9s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
