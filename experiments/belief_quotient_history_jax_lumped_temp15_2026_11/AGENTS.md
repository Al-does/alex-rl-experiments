# belief quotient history jax — lumped baseline, 4B steps, entropy 0, temperature 1.5

Companion of `belief_quotient_history_jax_lumped_2026_11` (same environment, models, critic-free PPO
recipe, probes). Differences: fresh runs of 4,000,000,000 env steps (201 rollouts),
no entropy bonus (coefficient 0.0), and `TrainConfig.temperature = 1.5`: rollout
actions are sampled from `softmax(logits / 1.5)` and the PPO ratio uses the same
tempered log-likelihoods (`reinforce.rollout` / `surrogate_loss`). Greedy
evaluation and probe rollouts are unaffected. Recorded in `resolved_recipe.json`
under `training.temperature` / `training.entropy_schedule`.
Smoke: `uv run rl-harness experiments.belief_quotient_history_jax_lumped_temp15_2026_11.<leaf>.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.
