# reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_cont4b_2026_11

Continuation companion of `reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11`
(transformer arms only): each `{reward_both,reward_factor_1}_transformer` leaf restores the
parent leaf's completed, seed-matched final `step_<N>.npz` (2,012,217,344 steps; local
`artifacts/` tree or the B2 prefix in the parent's `run_manifest.json`) and trains for
another `ADDITIONAL_ENV_STEPS` (= the parent budget, 2B), so recorded steps run from
~2.01B to ~4.02B absolute. Epsilon-0.02 emissions, model (context 32), lr, entropy 0.0,
temperature 1.5, batch and probes are the parent's. Only parameters are checkpointed, so
Adam moments restart from zero at the join. The first checkpoint report
(`is_restored: true`) re-probes the restored parameters. Smoke runs also restore the real
parent checkpoint (needs B2 credentials).

```bash
uv run --group jax rl-harness experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_cont4b_2026_11.reward_both_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
