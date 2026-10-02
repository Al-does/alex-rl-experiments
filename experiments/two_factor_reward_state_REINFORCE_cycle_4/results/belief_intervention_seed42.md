# Joint Reward: seed-42 factor-belief intervention

The 30M-step Joint Reward checkpoint is
`two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-30m`.
Its native `default_policy` RLModule is under
`s3://slop-bucket/experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-30m/step_checkpoints/steps_030000000/learner_group/learner/rl_module/default_policy/`.
All three module files were checked against `metadata/durability_manifest.json`;
`module_state.pkl` has SHA-256
`2ce92d4502837894e0024c4aaa000f9bb1b520993d1d6063ef667686a4db8d76`.

On independent greedy on-policy fit and test rollouts (12,000 scored steps
each, excluding the first eight steps of every episode), fit affine maps
from the final-block residual to the two independent factor beliefs, and from
the first two coordinates of each factor belief to the residual. Held-out
decoder \(R^2\) is **0.9823 / 0.9856** (factors 1 / 2); belief-to-residual
\(R^2\) is **0.6407**. Principal cosines between the two factor-specific
belief-to-residual two-dimensional spans are **0.1544 / 0.0279**. This
identifies largely distinct linear directions without asserting that the
rest of the residual is explained by beliefs. The top activation coordinates
by the norm of each factor's two belief-to-residual coefficients are, in
zero-based indexing, **[22, 44, 14, 24, 58, 19, 10, 52]** for factor 1 and
**[1, 55, 34, 42, 62, 37, 40, 9]** for factor 2. These are rankings of
distributed fitted directions, not individually identified belief neurons;
the complete affine weights and biases are in the JSON report.

For each decision, decode the *policy's own* current residual into both
factor beliefs; use the fitted belief-to-residual weights to add
`(target_factor - decoded_factor) @ weights_for_factor` before final layer
normalization. The actor never receives diagnostic beliefs or hidden states.
`opposite_2` targets probability one on whichever **nonreward** factor-2
state has *lower* decoded probability, and zero on the other states. Its
counterfactual target is a valid simplex vertex. Other conditions replace
factor 2 by its fit mean (`erase_2`), exchange its two nonreward probabilities
(`swap_2`), or erase factor 1; the random-direction control applies a
per-step norm-matched shift orthogonal to the decoder. `opposite_2_clamp_1`
is a separate **action-splice diagnostic**: retain the intact policy's
factor-1 action and the steered policy's factor-2 action on the same rollout.

Thirty-two paired environment seeds, one complete 1,024-step episode each,
greedy policies, evaluation seed 1042. Occupancy is the fraction of decisions
in each factor's reward state, i.e. expected per-factor reward per step.
The uncertainty shown is the standard error across episodes.

| Intervention | Factor 1 occupancy | Factor 2 occupancy |
| --- | ---: | ---: |
| Intact | 0.3140 ± 0.0033 | 0.3124 ± 0.0025 |
| Uniform random joint actions | 0.1667 ± 0.0024 | 0.1618 ± 0.0024 |
| Erase factor 2 | 0.3140 ± 0.0033 | 0.3118 ± 0.0026 |
| Exchange factor-2 nonreward states | 0.2962 ± 0.0033 | 0.2049 ± 0.0021 |
| Opposite factor-2 target | 0.2594 ± 0.0038 | **0.0854 ± 0.0018** |
| Opposite factor-2 target, factor-1 action clamped | 0.3140 ± 0.0033 | **0.0772 ± 0.0017** |
| Erase factor 1 | 0.3139 ± 0.0033 | 0.3122 ± 0.0025 |
| Matched-norm random direction | 0.3140 ± 0.0033 | 0.3124 ± 0.0025 |

The unspliced factor-2 intervention pushes factor-2 occupancy below the
uniform-random baseline, but also reduces factor-1 occupancy: decoding-level
factor separation alone does not guarantee behavioral separation in the
joint-action head. The action-splice diagnostic restores factor-1 occupancy
while retaining the factor-2 effect; it should not be mistaken for a pure
activation-only intervention. The opposite target is an extreme
counterfactual belief, so its steered activations may be off the natural
on-policy activation manifold. A below-random occupancy is evidence of
causal task impairment in this seed, not evidence that the agent learned an
explicit objective to avoid reward. The result is one frozen seed, 32
evaluation episodes, and a preselected family of interventions explored
locally; it is not a cross-seed result.

The full compact report is `belief_intervention_seed42_embedding1.json`.
With the B2 credentials configured, reproduce it into a new ignored
`artifacts/` report path:

```bash
uv run python scripts/download_b2_prefix.py \
  --prefix experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-30m/step_checkpoints/steps_030000000/learner_group/learner/rl_module/default_policy/ \
  --dest experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42/module
uv run python -m experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_intervention \
  --checkpoint experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42/module \
  --output experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42/reproduction.json \
  --steps 12000 --episodes 32 --mapping belief_embedding --strength 1 \
  --evaluation-seed 1042
```
