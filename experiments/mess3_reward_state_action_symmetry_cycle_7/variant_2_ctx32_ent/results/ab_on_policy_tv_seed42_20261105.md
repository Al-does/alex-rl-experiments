# A/B sensitivity on each checkpoint's own trajectories

## Result

The ordering persists: checkpoint 23 > 24 > 25 under both greedy and sampled
action selection, with each checkpoint controlling its own environment.

| Checkpoint | Training steps | Own greedy trajectories: mean TV | Own sampled trajectories: mean TV |
| --- | ---: | ---: | ---: |
| 23 | 9,406,096 | 0.00202775 | 0.00206938 |
| 24, penultimate | 9,799,312 | 0.00139074 | 0.00140306 |
| 25, final | 10,192,528 | 0.00131206 | 0.00130697 |

All six estimates use 64 complete 1,024-decision episodes, processing from
reset and excluding the first 128 decisions. Each estimate scores 57,344
decisions. This is a fresh evaluation cohort, not the preceding saved final-
policy histories.

| Paired mean-TV difference | Estimate | 95% paired-episode bootstrap interval |
| --- | ---: | --- |
| Greedy: 23 minus 24 | 0.00063702 | [0.00062100, 0.00065420] |
| Greedy: 24 minus 25 | 0.00007867 | [0.00007240, 0.00008525] |
| Sampled: 23 minus 24 | 0.00066633 | [0.00064958, 0.00068456] |
| Sampled: 24 minus 25 | 0.00009609 | [0.00008892, 0.00010371] |

## Did the histories actually differ?

Under greedy action selection, all three independently generated the same
selected actions and observation tokens at every decision, including warmup.
Thus, own-policy greedy evaluation does not create different histories for
these checkpoints when environment seeds match.

Under sampled action selection, the trajectories do differ. Comparing
checkpoints 23 and 25:

- Selected action histories differ in 62/64 episodes (424 differing decisions
  including warmup).
- Observation histories differ in 51/64 episodes (156 differing tokens including
  warmup).
- After warmup, actions differ at 0.623% of decisions and observations at 0.228%.
  Common random numbers keep most decisions coupled, but the histories are not
  forced to match.

The corresponding episode counts are 60/64 action histories and 36/64
observation histories for checkpoints 23 versus 24, and 57/64 and 31/64
for 24 versus 25.

## Policy responses

RMS A/B-induced Noop-minus-Tilt-X logit changes also decrease:

| Checkpoint | Greedy trajectories | Sampled trajectories |
| --- | ---: | ---: |
| 23 | 0.332765 | 0.334819 |
| 24 | 0.277146 | 0.279406 |
| 25 | 0.224357 | 0.226732 |

There are zero original-versus-shadow greedy-choice disagreements on scored
decisions in all six evaluations. In the sampled evaluation, the native
policies select a non-greedy action on 1,019, 848, and 749 scored decisions,
respectively. Zero greedy-choice disagreements does not imply that draws
from the original and shadow action distributions would always match.

The declining TV is therefore not specific to the final policy's visitation
distribution: it persists under each checkpoint's sampled on-policy
visitation. This remains a descriptive result for one training seed and three
late checkpoints. It does not identify the role of residual A/B information,
prove a general training-wide trend, or establish sampled-policy invariance.

## Protocol

- Training seed 42, Reward-State Quotient, the same verified native modules
  as the preceding checkpoint comparison. Module-state hashes are checked
  before restoration.
- Evaluation seed 20261105. Each episode's environment, symbol randomization,
  and action uniforms use independent streams
  `SeedSequence([evaluation_seed, episode, stream])`, with stream IDs 0, 1,
  and 2. Streams are common across checkpoints and action modes.
- In each run, the unmodified native policy alone controls the environment.
  A separate cache receives fair, independently resampled A/B inputs;
  C and the preceding executed native action are preserved. The shadow
  never controls an environment. TV compares these distributions at each
  decision on that checkpoint's own trajectory.
- Sampled actions use the original, unmodified softmax and inverse-CDF
  sampling with common uniforms across checkpoints. No temperature adjustment,
  fitting, training, or checkpoint selection is performed.
- Process complete episode prefixes, then apply the 128-decision warmup mask.
  Bootstrap 2,000 paired episode IDs, with decision-weighted means. Different
  checkpoint trajectories remain paired by episode seed, not by identical
  observations. Bootstrap seed is 20261104.

## Verification

- Real-checkpoint smoke covers all three checkpoints under both modes.
- Three new tests cover inverse-CDF boundaries/zero-probability actions,
  native-action execution, and preserving native feedback in the shadow under
  both action modes. Across the relevant test files, 25 tests pass; the known
  duplicate-run campaign-discovery test is excluded.
- Independently replaying two complete checkpoint-23 episodes from each mode
  reproduces original and shadow logits within maximum absolute error 1.91e-6.
- The adjacent JSON stores all estimates and episode-bootstrap intervals,
  trajectory-disagreement counts, source/checkpoint/raw-data hashes, and
  repository/runtime provenance. Raw arrays remain under ignored `artifacts/`.
- The manuscript is unchanged.

## Reproduce

With the three already-verified local checkpoint downloads:

```bash
.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_on_policy_tv \
  --scratch experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_on_policy_tv_reproduction \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_on_policy_tv_reproduction/report.json
```

Outputs refuse overwriting. Add `--smoke` for four 256-decision episodes per
checkpoint and mode; smoke estimates are not the reported results.
