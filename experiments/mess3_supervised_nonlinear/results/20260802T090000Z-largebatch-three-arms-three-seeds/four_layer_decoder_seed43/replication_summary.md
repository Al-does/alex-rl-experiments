# MESS3 supervised replication: swish-mlp-decoder-4-layer

- Analyzed checkpoint: update 61,446
- Exact Bayesian floor: 0.802512 nats
- Exact validation loss: 0.802558 nats (gap +0.000046)
- Final pre-LN affine probe: MSE 0.000374058, R² 0.996634
- Scientific gate: PASS
- Active optimization: 1935.4s at 32.3 updates/s
- End-to-end training wall time: 1938.1s
- Probe/plot wall time: 3.8s
- Total experiment wall time: 1942.1s

The model was trained only on next-token cross-entropy. Belief targets were used only after training by the affine probe.
