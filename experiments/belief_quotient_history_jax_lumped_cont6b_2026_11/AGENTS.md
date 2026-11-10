# belief_quotient_history_jax_lumped_cont6b_2026_11

Second continuation companion: parent is `belief_quotient_history_jax_lumped_cont4b_2026_11` (PR #197 recipe): each of the
12 leaves restores the parent leaf's completed, seed-matched final
`step_<N>.npz` (4,024,434,688 steps; local `artifacts/` tree or the B2 prefix in the
parent's `run_manifest.json`) and trains for another `ADDITIONAL_ENV_STEPS`
(= the parent budget, 2B), so recorded steps run from ~4.02B to ~6.04B absolute.
Environment, model, PPO shell, entropy 0.015, lr 4.2e-4, batch and probes are
the parent's. Only parameters are checkpointed, so Adam moments restart from
zero at the join; per-iteration RNG keys continue past the parent's 202
iterations. The first checkpoint report (`is_restored: true`) re-probes the
restored parameters with this run's probe streams.

```bash
uv run rl-harness experiments.belief_quotient_history_jax_lumped_cont6b_2026_11.variant_1_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
