# Full-State Quotient seed 42: A/B observation intervention (positive control)

This is the positive control for the Reward-State Quotient result in
`../../variant_2_ctx32_ent/results/ab_token_intervention_seed42.md`. Under
Full-State Quotient, Tilt X is optimal in state 1 and Tilt Y in state 2, so the
policy must distinguish A from B. The same randomization should therefore
change actions and lower reward-state occupancy.

The final module (`checkpoint_000025`, 10,192,528 agent steps) of
`mess3_reward_state_action_symmetry_cycle_6-variant_3_ctx32_ent-seed42-10m` was
restored from B2 and its three files were checked against the run's canonical
durability manifest. The protocol, evaluation seed (20260929), episode seeds
and A/B coin streams are identical to the Reward-State Quotient evaluation:
64 greedy episodes of 1,024 decisions, paired environment copies, a shadow
policy state on the baseline trajectory, and 10,000 paired-episode bootstrap
resamples. Exact hashes are in `ab_token_intervention_seed42.json`.

| Measure | Baseline | Randomized A/B |
| --- | ---: | ---: |
| Reward-state occupancy | 0.314987 | 0.216293 |
| Greedy action fractions (Noop / Tilt X / Tilt Y) | 0.318 / 0.342 / 0.340 | 0.242 / 0.379 / 0.379 |
| Paired occupancy difference (95% episode bootstrap interval) | | -0.098694 [-0.101318, -0.096069] |

The resampling changed 49.89% of A/B tokens (22,281 of 44,664). Every one of
those flips changed the shadow policy's greedy action (22,281 of 22,281), and
no unflipped A/B or C input did. Mean total variation between shadow and
baseline action distributions was 0.975 on flipped inputs and 0.334 over all
decisions. In closed loop, 42.83% of greedy actions differed from baseline and
occupancy fell in all 64 episodes (per-episode change -0.124 to -0.076).

Identity control (`ab_token_identity_control_seed42.json`, `--identity-control`)
passes inputs through unchanged on the same seeds and reproduces the baseline
exactly: zero action differences and a paired difference of exactly 0.

For comparison, the Reward-State Quotient seed-42 policy under the identical
intervention changed 0 of 65,536 greedy actions, with mean total variation
0.0029 on flipped inputs. The intervention is therefore able to produce a
large behavioral effect when the policy uses A/B identity. This is one seed
and greedy actions only.

## Reproduce locally

Restore the module as in the Reward-State Quotient write-up, substituting
`variant_3_ctx32_ent` and variant `3`, then run:

```bash
uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_token_intervention \
  --checkpoint experiments/mess3_reward_state_action_symmetry_cycle_7/variant_3_ctx32_ent/artifacts/ab_intervention/seed42_final \
  --run-manifest experiments/mess3_reward_state_action_symmetry_cycle_7/variant_3_ctx32_ent/results/mess3_reward_state_action_symmetry_cycle_6-variant_3_ctx32_ent-seed42-10m/run_manifest.json \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_3_ctx32_ent/artifacts/ab_intervention/reproduced.json
```

The script reads the condition (variant 2 or 3) from the run manifest.
