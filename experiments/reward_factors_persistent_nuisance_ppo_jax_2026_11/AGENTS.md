# Persistent-nuisance reward factors, standard PPO (pure JAX)

Eight leaves: {reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}.
Source: `reward_factors_persistent_nuisance_jax_2026_11` (PR #208). Same
two-factor process (Kronecker product of the variant-3 persistent-nuisance
factor), encoders (transformer context 10), probes on the pre-final-LayerNorm
residual and checkpoints; the critic-free trainer is replaced by the standard
PPO trainer and coefficients in
`belief_quotient_persistent_nuisance_ppo_jax_2026_11` (`ppo.py`, `AGENTS.md`).

```bash
uv run --group jax rl-harness experiments.reward_factors_persistent_nuisance_ppo_jax_2026_11.reward_both_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
