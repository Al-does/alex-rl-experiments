# belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11

Requested by the human user: the `belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11`
transformer arms (v2 environment, 2B steps, constant entropy 0.0, temperature 1.5)
with epsilon-greedy exploration, epsilon 0.05. Three `variant_{1,2,3}_transformer` leaves.

Rollout actions are sampled from `(1 - 0.05) * softmax(logits / 1.5) + 0.05 / 3`, and the
PPO likelihoods (recorded `logp` and the surrogate ratio) use that same mixture, so the
update stays on-policy for the behaviour policy. The entropy metric is that of the tempered
softmax. Greedy evaluation and probes are unchanged (argmax ignores both knobs).
Everything else is inherited from the parent study.

```bash
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11.variant_1_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
