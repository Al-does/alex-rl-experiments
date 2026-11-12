# reward factors history jax — lumped baseline, small batch (b2m), 500M steps, entropy 0, T=1.5

Companion of `reward_factors_history_jax_lumped_temp15_2026_11`: identical environment, models,
critic-free PPO recipe, probes, entropy 0.0 and temperature 1.5, but
`NUM_ENVS = 1,920` (rollout = 1,966,080 steps, ~1/10 of the parent's 19,922,944)
with the same 64 minibatches per rollout (30 episodes / 30,720 steps each, ~1/10
of the parent's 304 episodes) and a 500,000,000-step budget (255 rollouts,
16,320 Adam steps). Checkpoints/probes keep the parent schedule (step 0,
log-spaced iterations, every 2M steps, final).
Smoke: `uv run rl-harness experiments.reward_factors_history_jax_lumped_temp15_b2m_2026_11.<leaf>.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.
