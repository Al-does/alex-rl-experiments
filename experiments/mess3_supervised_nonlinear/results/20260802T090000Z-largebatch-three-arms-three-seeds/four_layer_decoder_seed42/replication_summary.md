# MESS3 supervised replication: swish-mlp-decoder-4-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802533 nats (gap +0.000022)
- Final pre-LN affine probe: MSE 0.000502114, R² 0.995482
- Scientific gate: PASS
- Active optimization: 1949.5s at 32.1 updates/s
- End-to-end training wall time: 1952.3s
- Probe/plot wall time: 3.9s
- Total experiment wall time: 1956.3s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
