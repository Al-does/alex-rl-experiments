# A/B randomization at the checkpoint before the penultimate one

## Result

Yes: checkpoint 23 has higher total variation than both checkpoints 24 and 25
on identical histories and randomization draws.

| Checkpoint | Sampled steps | Mean total variation | 95% episode-bootstrap interval | RMS Noop-minus-Tilt-X logit change |
| --- | ---: | ---: | --- | ---: |
| 23 | 9,406,096 | 0.00204741 | [0.00200580, 0.00209018] | 0.335186 |
| 24 (penultimate) | 9,799,312 | 0.00139071 | [0.00136638, 0.00141631] | 0.279352 |
| 25 (final) | 10,192,528 | 0.00130353 | [0.00128274, 0.00132452] | 0.226139 |

Checkpoint 23's mean TV is 47.22% above checkpoint 24 and 57.07% above
checkpoint 25. Paired differences are:

- 23 minus 24: +0.00065670, interval [0.00063644, 0.00067824].
- 23 minus 25: +0.00074388, interval [0.00071489, 0.00077457].

There are no original-versus-randomized greedy-action disagreements at any
checkpoint on the 57,344 scored decisions. Checkpoint 23 also has none over
all 65,536 decisions including warmup. Its original greedy actions match
the final model's original greedy actions on every scored decision.

Both logit sensitivity and probability sensitivity decrease over these last
three checkpoints. Mean maximum action probability increases from 0.982006
to 0.985070 to 0.986722. The TV decline therefore accompanies both smaller
logit perturbations and increased confidence; the comparison does not
isolate softmax saturation as its cause. It does not test whether the
remaining sensitivity serves reward-state inference.

## Protocol and verification

- Same training seed 42 and exact run as the preceding comparison.
  Checkpoint 23 is iteration 24, immediately before checkpoint 24/iteration 25.
- Downloaded native module files were verified against the canonical B2
  durability manifest. Module-state SHA256:
  `415118945953743695e96ed2c81791175edc9f4a647fb31f1236d742c8c0ac5c`.
- Reused the preceding evaluation's 64 complete 1,024-step final-policy
  trajectories (evaluation seed 20261104, split 2), including exactly the
  same independently randomized A/B symbols. C observations and executed
  final-policy actions are preserved.
- Each model has separate original/randomized caches, initialized at reset.
  Exclude the first 128 decisions after processing full prefixes, leaving
  57,344 scored decisions. Bootstrap 2,000 paired episodes with
  decision-weighted means.
- The lean replay implementation was checked against four complete final-model
  episodes from the original saved replay: maximum absolute logit error
  1.91e-6. Checkpoint 24/25 TV values reproduce the previous report.
- The source, shared histories, original replay, and new raw replay have
  recorded hashes in the adjacent JSON. One new regression test covers
  separate caches, reset handling, and partial batches; all 11 focused tests
  pass.
- This is a requested, descriptive extension after observing 24 > 25, using
  one seed and frozen final-policy histories. It is not an independent
  confirmation of a general or monotonic training-wide trend.

Raw replay arrays and checkpoints stay in ignored `artifacts/`.
The manuscript is unchanged.

## Reproduce

From the experiment repository with the verified downloads already present:

```bash
.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_checkpoint_tv \
  --checkpoint experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_checkpoint23_tv_seed42/module \
  --download-report experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_checkpoint23_tv_seed42/download.json \
  --previous-report experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/results/ab_pattern_checkpoint_seed42_20261104.json \
  --histories experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_pattern_checkpoint_seed42/full_histories.npz \
  --previous-replay experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_pattern_checkpoint_seed42/full_replay.npz \
  --final-checkpoint experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_logit_evidence_seed42/module \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_checkpoint23_tv_seed42/reproduced.json \
  --raw-output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_checkpoint23_tv_seed42/reproduced.npz
```

Outputs refuse overwriting.
