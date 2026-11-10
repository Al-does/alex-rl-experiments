# reward_factors_history_jax_2b_2026_11

Same 20-arm family as `reward_factors_history_jax_2026_11` with ten times the training budget:
2,000,000,000 env steps per arm (101 rollouts of 19,456 synchronized 1,024-step
episodes, 64 minibatches each). Everything else (environment, architectures,
lr, constant entropy 0.015, probes, checkpoint cadence) is the parent recipe;
`shared.py` only forwards `total_env_steps` and the study name to
`reward_factors_history_jax_2026_11.shared.run_condition`. Leaves mirror the parent names. Fresh runs
only, runtime `--seed` (default 42).

```bash
uv run rl-harness experiments.reward_factors_history_jax_2b_2026_11.reward_both_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.reward_factors_history_jax_2b_2026_11.reward_both_gru128.experiment --seed 42 --hardware-profile cuda4090
```
