# MESS3 supervised next-token prediction

These experiments reproduce the supervised MESS3 belief-geometry result and
compare large-batch SGD with Muon.

## Conditions

- `paper_supervised_replication`: paper-scale SGD recipe.
- `large_batch_replication`: batch 16,384, SGD learning rate 0.16, 62,500
  updates.
- `muon_large_batch_replication`: matched large-batch budget, Muon learning
  rate 0.02 for 2D parameters and AdamW learning rate 3e-4 for non-2D
  parameters.

Both completed large-batch runs use seed 0. The retained manifests preserve
their original module paths and remote artifact URIs because those are
historical provenance records.

## Results

- Large-batch SGD: probe R² 0.99783, probe MSE 0.000241, exact weighted
  next-token accuracy 68.59%.
- Large-batch Muon: probe R² 0.98432, probe MSE 0.001742, exact weighted
  next-token accuracy 68.58%.
- Bayes-optimal weighted next-token accuracy: 68.59%.

The checkpoint-wise R² comparison is under
`muon_large_batch_replication/results/20260719-sgd-muon-r2-comparison-v2/`.
