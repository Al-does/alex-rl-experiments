# Does A/B sensitivity track additional reward-state evidence?

## Result

The test did **not** support the predicted positive relationship. For the
original Reward-State Quotient seed-42 final checkpoint, the held-out partial
correlation between original-minus-randomized reward-state evidence and
original-minus-randomized Noop-versus-Tilt-X logit was -0.002386
(95% paired-episode bootstrap interval [-0.011749, 0.007232]).

| Evaluation subset | Test decisions | Partial correlation | 95% interval |
| --- | ---: | ---: | --- |
| All (primary) | 57,344 | -0.002386 | [-0.011749, 0.007232] |
| Same current token | 38,106 | 0.003110 | [-0.009989, 0.016327] |
| Current token C | 18,712 | -0.003449 | [-0.018716, 0.011729] |

The evidence term improved held-out logit-change prediction by only
0.000000407 in R-squared over the control model (interval spans zero).
The controls alone explained 80.4476% of the logit-change variance.
The unadjusted correlation was also small (-0.008715; descriptive diagnostic,
not the primary comparison). The test trajectories had no greedy-action
disagreements and mean action-distribution total variation 0.001409.

## Protocol

- Restore native `checkpoint_000025`, at 10,192,528 sampled training steps.
  Module SHA256:
  `6ff2b60203e298c7967cd16e2a0ad5676fbc22c288e02e2515a38a06a635cc5b`.
  Downloaded module files were checked against the canonical B2 manifest.
- Collect 64 fitting and 64 independent test episodes, 1,024 decisions each,
  evaluation seed 20261103 with disjoint split/episode seed keys.
- The original greedy policy controls the environment. The comparison policy
  receives independently resampled fair A/B inputs, preserving C and original
  executed actions. Both caches start at reset and evolve separately.
- Refilter both histories with the original three-state model; validate the
  baseline filter against public environment belief diagnostics at every step.
  Their coarse observation/action histories are identical. Subtracting the
  full-filter reward-state probabilities therefore also subtracts their
  additional evidence relative to the common coarse belief.
- Randomized-history filtering uses the original emission model. It is an
  "as if these were ordinary symbols" comparison, not the posterior for a
  known randomization channel.
- Exclude the first 128 decisions, covering the full encoder receptive field.
  Fit cubic coarse-belief functions within each original-token,
  randomized-token, previous-action cell to both evidence and logit differences.
  Fit controls and the incremental linear slope on fitting episodes only.
- Score on held-out episodes. Bootstrap 2,000 paired episodes, without
  refitting the training models. Positive association was the predicted sign;
  no model or feature selection used test outcomes.

## Interpretation

A/B identity does improve exact reward-state inference, but the small policy
probability changes did not track that improvement in this test. Residual
token sensitivity is not, by itself, evidence of useful finer inference.
This does not rule out nonlinear relationships or other uses of A/B history,
and it does not identify the network's internal computation. The checkpoint
and altered-history evaluation distribution limit generalization.

## Reproduce

From the experiment repository, with the hash-verified module at the path below:

```bash
.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_logit_evidence \
  --checkpoint experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_logit_evidence_seed42/module \
  --run-manifest experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/results/mess3_reward_state_action_symmetry_cycle_6-variant_2_ctx32_ent-seed42-10m/run_manifest.json \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_logit_evidence_seed42/reproduced.json \
  --raw-output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_logit_evidence_seed42/reproduced.npz
```

Outputs refuse overwriting. Add `--smoke` for four episodes per split and
256 decisions per episode; smoke is an execution check, not the reported test.
The adjacent JSON records training/analysis revisions, source and data hashes,
runtime versions, fitting diagnostics, prediction comparisons, and intervals.
Raw arrays and checkpoints remain under ignored `artifacts/`.
