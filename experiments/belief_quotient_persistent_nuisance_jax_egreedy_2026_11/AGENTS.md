# belief_quotient_persistent_nuisance_jax_egreedy_2026_11

Transformer arms of `belief_quotient_persistent_nuisance_jax_2b_2026_11` rerun from scratch for
the same 2B-step budget with no entropy bonus (coefficient 0.0), sampling temperature 1.0 and
epsilon-greedy exploration (`TrainConfig.epsilon`). Rollout actions are sampled from
`(1 - eps) * softmax(logits) + eps / 3`, and the PPO ratio uses the same mixture's
log-likelihoods (`reinforce.policy_log_probs`), so updates stay on-policy for the behaviour
policy. Greedy evaluation and probe rollouts are unaffected. Leaves:
`variant_{1,2,3}_transformer_{eps0p05,eps0p2}` (eps = 0.05 or 0.2), recorded in
`resolved_recipe.json` under `training.epsilon`.

```bash
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_egreedy_2026_11.variant_2_transformer_eps0p2.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_persistent_nuisance_jax_egreedy_2026_11.variant_2_transformer_eps0p05.experiment --seed 42 --hardware-profile cuda4090
```
