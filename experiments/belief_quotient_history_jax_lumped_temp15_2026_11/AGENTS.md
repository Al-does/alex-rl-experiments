# belief quotient history jax — lumped baseline, 4B steps, entropy 0, temperature 1.5

Companion of `belief_quotient_history_jax_lumped_2026_11` (same environment, models, critic-free PPO
recipe, probes). Differences: fresh runs of 4,000,000,000 env steps (201 rollouts),
no entropy bonus (coefficient 0.0), and `TrainConfig.temperature = 1.5`: rollout
actions are sampled from `softmax(logits / 1.5)` and the PPO ratio uses the same
tempered log-likelihoods (`reinforce.rollout` / `surrogate_loss`). Greedy
evaluation and probe rollouts are unaffected. Recorded in `resolved_recipe.json`
under `training.temperature` / `training.entropy_schedule`.
Smoke: `uv run rl-harness experiments.belief_quotient_history_jax_lumped_temp15_2026_11.<leaf>.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.

## Post-hoc target probes

`uv run python -m experiments.belief_quotient_history_jax_lumped_temp15_2026_11.reprobe --leaf variant_2_gru92`
downloads seed-42 parameter checkpoints from the run manifest's B2 prefix (cached
in ignored artifacts), and writes `reprobe_targets.json` beside the original
summary. Existing outputs are never overwritten. Default: all 202 archived
checkpoints, original greedy embedding probes, 60K/80K independent fit/test rows,
64-step warmup, and the original fixed probe key. `--steps 0 ...` selects an
explicit subset; `--smoke` uses smoke budgets (not an archive reproduction);
`--out PATH` selects a separate output. Timings include fetch, load, and fitting.

Targets are full belief, reward-state probability, fine-state contrast, and the
coarsened C/not-C quotient probability. The quotient is filtered over whole
episodes before dropping warmup, with no transition at reset and the preceding
executed action thereafter. It is asserted lumpable for Trivial Quotient and
Reward-State Quotient; omitted for Full-State Quotient. These descriptive
linear-accessibility measurements do not establish causal use. Check with
`uv run pytest -q tests/test_history_jax_reprobe.py tests/test_history_jax_temperature.py`.

Uniform nonreward destinations and symmetric emissions make A/B tokens carry identical reward-class likelihood, so c[B] equals rho for Trivial Quotient and Reward-State Quotient (up to numerical tolerance).
