# MESS3 supervised replication: linear-decoder-control

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802530 nats (gap +0.000019)
- Final pre-LN affine probe: MSE 0.000315097, R² 0.997165
- Scientific gate: PASS
- Active optimization: 1816.7s at 34.4 updates/s
- End-to-end training wall time: 1819.3s
- Probe/plot wall time: 4.2s
- Total experiment wall time: 1823.7s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
