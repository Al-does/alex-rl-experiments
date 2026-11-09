# belief_quotient_history_jax_1b_2026_11

Same 20-arm family as `belief_quotient_history_jax_2026_11` with five times the training budget:
1,000,000,000 env steps per arm (55 rollouts of 19,456 synchronized 1,024-step
episodes, 64 minibatches each). Everything else (environment, architectures,
lr, constant entropy 0.015, probes, checkpoint cadence) is the parent recipe;
`shared.py` only forwards `total_env_steps` and the study name to
`belief_quotient_history_jax_2026_11.shared.run_condition`. Leaves mirror the parent names. Fresh runs
only, runtime `--seed` (default 42).

```bash
uv run rl-harness experiments.belief_quotient_history_jax_1b_2026_11.variant_1_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_history_jax_1b_2026_11.variant_1_gru128.experiment --seed 42 --hardware-profile cuda4090
```
