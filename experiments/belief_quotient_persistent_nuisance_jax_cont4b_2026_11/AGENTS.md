# belief_quotient_persistent_nuisance_jax_cont4b_2026_11

Continuation companion of `belief_quotient_persistent_nuisance_jax_2b_2026_11`: each of the
6 leaves (`variant_{1,2,3}_{transformer,memoryless}`) restores the parent leaf's completed,
seed-matched final `step_<N>.npz` (2,012,217,344 steps; local `artifacts/` tree or the B2
prefix in the parent's `run_manifest.json`) and trains for another `ADDITIONAL_ENV_STEPS`
(= the parent budget, 2B), so recorded steps run from ~2.01B to ~4.02B absolute.
Process, model, lr, constant entropy 0.015, temperature, batch and probes are the
parent's. Only parameters are checkpointed, so Adam moments restart from zero at the
join. The first checkpoint report (`is_restored: true`) re-probes the restored
parameters. Smoke runs also restore the real parent checkpoint (needs B2 credentials).

```bash
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_cont4b_2026_11.variant_2_memoryless.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_cont4b_2026_11.variant_2_transformer.experiment --seed 42 --hardware-profile cuda4090
```
