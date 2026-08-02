# MESS3 supervised replication: swish-mlp-decoder-2-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802533 nats (gap +0.000021)
- Final pre-LN affine probe: MSE 0.000263217, R² 0.997632
- Scientific gate: PASS
- Active optimization: 1874.7s at 33.3 updates/s
- End-to-end training wall time: 1877.2s
- Probe/plot wall time: 3.9s
- Total experiment wall time: 1881.3s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
