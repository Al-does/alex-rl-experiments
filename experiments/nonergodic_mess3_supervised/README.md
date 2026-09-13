# Nonergodic MESS3 supervised reproduction

This study recreates the supervised next-token experiment in
[The geometry of nonergodic composition](https://simplex.pub/nonergodic-geometry/).
It uses the article's two MESS3 components, 50/50 sequence-level mixture, BOS
plus 127 emissions, four-layer transformer, AdamW recipe, and post-hoc linear
readout of the weighted six-state belief.

The two leaves correspond to the two article checkpoints called out in the
training appendix:

- `loss_convergence`: 10,000 optimizer steps, where the next-token loss is
  reported to be within a few `1e-4` nats/token of Bayes optimal.
- `geometry_checkpoint`: 45,000 optimizer steps, the checkpoint used for the
  article's geometry figures.

Both conditions save log-spaced model-only checkpoints under ignored
`artifacts/`, fit probes on independent held-out sequences, and write compact
metrics and figures under `results/`.
