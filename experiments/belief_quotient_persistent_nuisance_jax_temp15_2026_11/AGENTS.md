# belief_quotient_persistent_nuisance_jax_temp15_2026_11

Transformer leaves (`variant_{1,2,3}_transformer`) of
`belief_quotient_persistent_nuisance_jax_2b_2026_11` rerun from scratch for the same
2B-step budget with no entropy bonus (coefficient 0.0) and `TrainConfig.temperature = 1.5`:
rollout actions are sampled from `softmax(logits / 1.5)` and the PPO ratio uses the same
tempered log-likelihoods. Greedy evaluation and probe rollouts are unaffected. Process,
model, lr, batch, probes and checkpoint cadence are the parent's; recorded in
`resolved_recipe.json` under `training.temperature` / `training.entropy_schedule`.

```bash
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.variant_2_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.variant_2_transformer.experiment --seed 42 --hardware-profile cuda4090
```
