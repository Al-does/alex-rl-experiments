# belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_kl_2026_11

Same six transformer arms as `belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11`
(standard PPO, 2B env steps, constant entropy 0.005 or 0.0, temperature 1.0) with the
RLlib-style adaptive KL penalty from `ppo.py` turned on: `use_kl_loss=True`, initial
`kl_coeff` 0.2, `kl_target` 0.01 (x1.5 above 2x target, x0.5 below half, once per rollout).
Leaves: `variant_{1,2,3}_transformer_{ent0p005,ent0}`.

```bash
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_kl_2026_11.variant_2_transformer_ent0.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run --group jax-cuda rl-harness experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_kl_2026_11.variant_2_transformer_ent0p005.experiment --seed 42 --hardware-profile cuda4090
```
