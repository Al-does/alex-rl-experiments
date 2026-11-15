# reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11

Requested by the human user: `reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11`
with epsilon-greedy exploration, epsilon 0.05, as in
`belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11`. Two leaves:
`reward_both_transformer`, `reward_factor_1_transformer` (32-frame context).

Rollout actions are sampled from `(1 - 0.05) * softmax(logits / 1.5) + 0.05 / 9` (nine joint
actions), and the PPO likelihoods use the same mixture. Greedy evaluation and probes are
unchanged. Everything else is inherited from the parent study.

```bash
uv run --group jax rl-harness experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11.reward_both_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```

`plot_training_curves.py` reads the committed results of this study and of the no-epsilon
parent study and regenerates `results/figures/training_curves_{no_eps,eps05}.png` (one panel
per condition and seed; linear env steps on x; factor 1, factor 2 and joint 1−R² on a log left
axis shared across all eight panels; reward per step on the right):

```bash
uv run python -m experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11.plot_training_curves
```
