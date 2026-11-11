# belief_quotient_persistent_nuisance_jax_entropy_2026_11

Transformer arms of `belief_quotient_persistent_nuisance_jax_2b_2026_11` rerun from scratch for
the same 2B-step budget with a lower constant entropy coefficient (0.01 or 0.005 from step 0,
no annealing) instead of the baseline 0.015. Sampling temperature is 1.0 and epsilon-greedy is
off (`epsilon=0.0`). Leaves: `variant_{1,2,3}_transformer_{ent0p01,ent0p005}`, recorded in
`resolved_recipe.json` under the training entropy schedule.

```bash
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_entropy_2026_11.variant_2_transformer_ent0p005.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_entropy_2026_11.variant_2_transformer_ent0p01.experiment --seed 42 --hardware-profile cuda4090
```
