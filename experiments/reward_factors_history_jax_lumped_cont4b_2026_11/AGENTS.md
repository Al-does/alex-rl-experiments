# reward_factors_history_jax_lumped_cont4b_2026_11

Continuation companion of `reward_factors_history_jax_lumped_2026_11` (PR #197 recipe): each of the
8 leaves restores the parent leaf's completed, seed-matched final
`step_<N>.npz` (2,012,217,344 steps; local `artifacts/` tree or the B2 prefix in the
parent's `run_manifest.json`) and trains for another `ADDITIONAL_ENV_STEPS`
(= the parent budget, 2B), so recorded steps run from ~2.01B to ~4.02B absolute.
Environment, model, PPO shell, entropy 0.015, lr 4.2e-4, batch and probes are
the parent's. Only parameters are checkpointed, so Adam moments restart from
zero at the join; per-iteration RNG keys continue past the parent's 101
iterations. The first checkpoint report (`is_restored: true`) re-probes the
restored parameters with this run's probe streams.

```bash
uv run rl-harness experiments.reward_factors_history_jax_lumped_cont4b_2026_11.reward_both_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
