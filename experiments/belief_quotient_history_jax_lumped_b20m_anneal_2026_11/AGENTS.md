# belief_quotient_history_jax_lumped_b20m_anneal_2026_11

Transformer-only belief-quotient ladder (b20m (192 envs)): `TOTAL_ENV_STEPS = 1,000,000,000`, `NUM_ENVS = 192` (64 minibatches of 3 episodes), temperature 1.5 → 1.0 and entropy 0.015 → 0 over the per-leaf post-plateau windows from `…_lumped_anneal4b_2026_11` (`TrainConfig.temperature_schedule` / `entropy_schedule`, piecewise-linear in env steps). Leaves: `variant_{1,2,3}_transformer`.
Smoke: `uv run rl-harness experiments.belief_quotient_history_jax_lumped_b20m_anneal_2026_11.variant_2_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.
