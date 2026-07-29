# Bootstrap assessment

The checkpoint results already contain the bootstrap calculation recommended by
`analysis/probes/README.md`: each MSE uses 1,000 percentile-bootstrap resamples
clustered by complete environment episode. The per-run charts use those 95%
intervals directly. The fitted probe remains fixed, so these intervals estimate
evaluation-rollout sampling uncertainty, not probe-fit uncertainty.

No additional bootstrap should be applied to individual timesteps or training
checkpoints: timesteps are correlated within episodes, and checkpoints are
repeated measurements of one trained model.

The combined condition chart keeps independently trained model-seed variability
separate. It shows all three seed values and mean ± population SD. A bootstrap
over only three model seeds would be coarse and potentially misleading, so it is
not used. If inferential condition comparisons become important, run more model
seeds and then use paired seed differences (the same seed set is shared by every
condition); a hierarchical seed-then-episode bootstrap would be appropriate only
with enough model seeds and retained per-episode probe errors.

The existing held-out permutation nulls answer a different question—whether the
activation/target association generalizes—and should not be used as MSE error
bars.
