# Cassandra belief-factoring study

## Question

Does a transformer trained on Cassandra's canonical global-action maintenance
task represent each named component's posterior, or only a coarser
permutation-invariant summary sufficient for global decisions?

The environment has one machine with four hidden components, not four
independent machines. Each component has four conditions. The exact joint
belief has 255 degrees of freedom; concatenated component marginals have 12.

## Policy and training

- PPO with the reusable causal `TransformerModel`.
- Canonical hidden 16-symbol observations.
- Policy input is the current symbol one-hot plus the preceding action one-hot.
  Action history is necessary because the Bayesian filter is
  action-conditioned.
- Reward is not included in the model input because the environment's public
  belief diagnostic does not condition on reward.
- Canonical global `operate`, `inspect`, `repair`, and `replace` actions.
- Canonical discount `0.999`.

The transformer never receives exact belief, factored belief, or hidden state.

## Probe protocol

The protocol adapts *Transformers learn factored representations*
(<https://arxiv.org/abs/2602.02385>) and the repository's MESS3 affine-probe
practice:

1. Save and probe the random initialization before any optimizer step.
2. Probe log-spaced training checkpoints.
3. Teacher-force checkpoint-independent behavior trajectories, with disjoint
   train/test seed streams. This gives initialization and trained models the
   same history distribution.
4. Extract the decision-token residual before final LayerNorm.
5. Fit held-out affine ridge probes.
6. Compare against a linear probe on the current policy observation itself.

The fixed behavior policy uses probabilities `(0.68, 0.20, 0.08, 0.04)` for
operate, inspect, repair, and replace. It emphasizes natural degradation while
still visiting every intervention.

## Targets

- Full 256-state joint posterior.
- Four labeled component marginals in 12 independent simplex coordinates.
- Identity residual: each component marginal minus the mean component
  marginal. This is the primary "detail beyond aggregate" target.
- Mean component marginal in three independent coordinates.
- Labeled expected component condition.
- Sorted expected component condition, a permutation-invariant control.
- Next-operate pass probability.
- Expected immediate reward for every action.
- Distribution over the number of broken components.
- Total correlation of the joint belief.

## Geometry

For activations and every target, report:

- PCA spectrum;
- cumulative explained variance;
- dimensions at 90%, 95%, and 99% CEV;
- participation ratio.

For each component's whitened marginal target, fit a readout and report the
paper's normalized principal-angle overlap:

`||Q_i.T @ Q_j||_F^2 / min(rank_i, rank_j)`.

High overlap indicates shared readout directions; low overlap indicates
separate component subspaces.

## Interpretation

Evidence for a coarse representation requires a pattern, not one statistic:

1. aggregate belief gains more R² over initialization than the identity
   residual;
2. sorted component health decodes better than labeled component health;
3. component readout subspaces overlap strongly;
4. these differences exceed the current-observation baseline.

PCA rank alone is not semantic evidence. A rich permutation-invariant
multiset representation can require more than three dimensions, and policy,
value, position, or action information can dominate activation variance.

## Causal follow-up

The strongest follow-up is a matched action-scope intervention:

1. canonical global actions;
2. indexed action aliases whose index is ignored;
3. genuinely addressable per-component repair and replacement.

The alias arm controls action-vocabulary size. The causal prediction is that
addressable actions increase identity-residual decodability and reduce
component-subspace overlap relative to the alias arm.

Additional post-smoke interventions:

- component-label permutation equivariance on teacher-forced histories;
- aggregate-matched history pairs with different labeled marginals;
- pairs with equal marginals but different joint correlation;
- activation patching in each inferred component subspace.

These require either an environment action-scope variant or a controlled
counterfactual history generator, so they are intentionally not conflated
with the first canonical training condition.
