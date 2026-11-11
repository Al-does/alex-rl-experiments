# belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11

Transformer arms of `belief_quotient_persistent_nuisance_ppo_jax_2026_11` (standard PPO with a
linear critic and GAE) at 10x the budget (2B env steps, 101 rollouts) with the entropy bonus
lowered from 0.015 to a constant 0.005 or 0.0 (no annealing). Temperature 1.0; everything else
(PPO coefficients, batch geometry, checkpoints, probes) is inherited unchanged. Leaves:
`variant_{1,2,3}_transformer_{ent0p005,ent0}`; the coefficient is recorded in
`resolved_recipe.json` under `training.entropy_coeff`.

```bash
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11.variant_2_transformer_ent0.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run --group jax-cuda rl-harness experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11.variant_2_transformer_ent0p005.experiment --seed 42 --hardware-profile cuda4090
```
