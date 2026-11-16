# reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11

Requested by the human user: the `reward_factors_persistent_nuisance_v2_jax_temp15_2026_11`
transformer arms with a 32-frame context (the ladder's), not the original factors
study's 10. Two leaves: `reward_both_transformer`, `reward_factor_1_transformer`.

Everything else is inherited: v2 environment (two independent variant-3 factors,
Kronecker joint, uniform 1/9 init), 2B steps, constant entropy 0.0, temperature 1.5 in
rollout and PPO likelihoods, no epsilon, residual joint/marginal probes. The ε=0.05 twin
is `reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11`.

```bash
uv run --group jax rl-harness experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11.reward_both_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
