# reward factors history jax — small-batch temp15 continuation, +1.5B (500M → 2.0B)

Continuation companion of `reward_factors_history_jax_lumped_temp15_b2m_2026_11`: each leaf restores
the parent's final parameter checkpoint `step_0501350400.npz` (local `artifacts/`
or the parent's B2 prefix from `run_manifest.json`) and trains another
1,500,000,000 steps with the identical recipe (1,920 envs × 1,024, 64 minibatches,
entropy 0.0, temperature 1.5) — 763 rollouts, absolute steps 501,350,400 →
2,001,469,440. Only parameters are restored (Adam moments restart at the join; the
per-iteration RNG keys continue past the parent's 255 iterations). The first probe
report has `is_restored: true`; `continued_from` is recorded in
`resolved_recipe.json` and `condition_summary.json`.
Smoke: `uv run rl-harness experiments.reward_factors_history_jax_lumped_temp15_b2m_cont2b_2026_11.<leaf>.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.
