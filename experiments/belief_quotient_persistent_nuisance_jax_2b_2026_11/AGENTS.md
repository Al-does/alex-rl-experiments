# belief_quotient_persistent_nuisance_jax_2b_2026_11

Transformer and memoryless leaves of `belief_quotient_persistent_nuisance_jax_2026_11`
with ten times the training budget: 2,000,000,000 env steps per arm (101 rollouts of
19,456 synchronized 1,024-step episodes, 64 minibatches each). Everything else
(persistent_nuisance_v1 process, architectures, lr, constant entropy 0.015, temperature
1.0, probes, checkpoint cadence) is the parent recipe; `shared.py` only forwards
`total_env_steps` and the study name to the parent `run_condition`. Leaves:
`variant_{1,2,3}_{transformer,memoryless}`. Fresh runs only, runtime `--seed` (default 42).

```bash
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_2b_2026_11.variant_2_memoryless.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_2b_2026_11.variant_2_transformer.experiment --seed 42 --hardware-profile cuda4090
```
