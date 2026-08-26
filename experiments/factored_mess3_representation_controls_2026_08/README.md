# Factored MESS3 representation controls

This study matches the completed PPO baseline conditions with transformers
trained only on delayed joint-token next-token cross-entropy:

- two factors, 64 residual dimensions;
- three factors, 64 residual dimensions;
- five factors, 64 residual dimensions;
- five factors, 120 residual dimensions.

Every full recipe consumes exactly 50,000,000 next-token examples. Smoke mode
consumes exactly 4,096. All conditions use seed 42 by default, independent
passive MESS3 factors (`alpha=0.85`), and length-11 context. The model is the
paper residual encoder plus one joint-token classification head; it has no PPO
loss, policy surrogate, value loss, or critic head.

Each trained checkpoint runs held-out reverse activation encoding, reduced-rank
predictive curves, controlled vary-one geometry, and token-embedding additive
decomposition. Checkpoints remain under ignored `artifacts/`; compact metrics
and plots are written under `results/`.

`RunContext.resume_from` on a runnable supervised leaf must point to that
leaf's `final.pt` or containing checkpoint directory. To apply the same suite
to a completed PPO baseline, call the explicit experiment-local
`analysis.analyze_ppo_checkpoint(context, n_factors=..., d_model=...)`; in that
case `context.resume_from` must point to the matching public RLlib Algorithm
checkpoint. Factor count and width are selected in Python rather than added as
scientific CLI flags.
