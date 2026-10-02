# Joint Reward seed-42: belief interventions after 300M steps

## Question and protocol

Does the advantage of joint-belief-aligned steering over marginal-belief-aligned
steering persist after the seed-42 Joint Reward agent is continued from 30M to
300M steps? This repeats the 30M causal-localization protocol without changing
its code, target definitions, seeds, strength grid, or evaluation horizon. The
module comes from the continuation run
`two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-300m`, final
`steps_300000000` checkpoint sampled at **300,071,594** agent steps. The native
module-state SHA-256 is
`db4be080db6288e2c743ac408a8edb638bbcc9838c7cf68098b640e00f23dd13`;
all three module files were checked against the run's B2 durability manifest.

As at 30M, independent 12,000-step greedy-rollout groups (seeds 42 and 43,
excluding eight warmup steps per episode) fit and test affine maps for four
marginal-belief coordinates and eight joint-belief coordinates at each site.
Sixteen complete 1,024-step episodes (seed 1042) tune strength at the last
pre-normalization block. A different 32 complete episodes (seed 2042) confirm
the chosen strengths at each layer, including after final LayerNorm. Random
actions and both map-specific, norm-matched decoder-null directions are
controls. Occupancy is the fraction of steps in the rewarded state for each
factor; ± values below are standard errors across episodes. Executed actions
come entirely from one steered forward pass, without action splicing.

The joint target keeps the factor-1 marginal **decoded from the joint map**
and concentrates factor 2 on the less likely nonreward state; it is a
synthetic counterfactual, not a fresh Bayesian update. The marginal target
changes factor-2 marginal coordinates instead. Both maps are fitted against
the environment's exact Bayesian beliefs. See the 30M report for the hook
locations and further methodological details.

## Strength tuning

The tuning-set random-action factor-2 occupancy was **0.1648**. The selection
rule takes the smallest tested strength at or below random, falling back to
the lowest factor-2 occupancy if none qualifies. Values are factor-1 / factor-2
occupancy on the *tuning* episodes:

| Strength | Marginal map | Joint map |
|---:|---:|---:|
| 0 | 0.3080 / 0.3164 | 0.3080 / 0.3164 |
| 0.50 | 0.3080 / 0.3164 | **0.3080 / 0.1298** |
| 0.75 | 0.3080 / 0.3163 | 0.3080 / 0.0667 |
| 0.80 | 0.3079 / 0.3152 | 0.3080 / 0.0667 |
| 0.85 | 0.3079 / 0.3113 | 0.3080 / 0.0667 |
| 0.90 | 0.3080 / 0.3062 | 0.3080 / 0.0667 |
| 0.95 | 0.3080 / 0.3044 | 0.3080 / 0.0667 |
| 1.00 | **0.3075 / 0.3016** | 0.3080 / 0.0667 |

Selected strengths: **1.0 marginal (fallback; none reached random)** and
**0.5 joint (first below random)**. These same strengths are reused at every
site; earlier layers were not individually retuned. A strength of 1.0 in the
two maps does not imply equal activation-shift magnitude.

## Independent confirmation and 30M comparison

At 300M, intact factor-1 / factor-2 occupancy was **0.3130 ± 0.0024 /
0.3135 ± 0.0027**; random actions yielded **0.1633 ± 0.0018 /
0.1617 ± 0.0021**. Values below use the selected strengths above. The 30M
columns show the earlier confirmation with that checkpoint's separately
selected strengths (0.9 marginal, 0.5 joint), on the same evaluation seeds.

| Site | 300M marginal F1 / F2 | 300M joint F1 / F2 | 30M marginal F1 / F2 | 30M joint F1 / F2 |
|---|---:|---:|---:|---:|
| After block 1 | 0.2585 / 0.1632 | **0.3130 / 0.1522** | 0.2748 / 0.0817 | 0.3127 / 0.1255 |
| After block 2 | 0.2629 / 0.1680 | **0.3130 / 0.1485** | 0.2755 / 0.0772 | 0.3127 / 0.1194 |
| After block 3 | 0.2930 / 0.2454 | **0.3130 / 0.1370** | 0.2766 / 0.0874 | 0.3130 / 0.1173 |
| After block 4, before final LayerNorm | 0.3125 / 0.3007 | **0.3130 / 0.1304** | 0.2835 / 0.1357 | 0.3130 / 0.1087 |
| After final LayerNorm | 0.3129 / 0.3044 | **0.3130 / 0.1417** | 0.2910 / 0.1587 | 0.3130 / 0.1252 |

At 300M the joint intervention reduces factor-2 occupancy below the random
baseline at **all five sites** while retaining essentially intact factor-1
occupancy. Factor-1 actions change on 0–0.018% of decisions with the joint
map; factor-2 actions change on 68–75%. At the last pre-normalization block,
factor-2 occupancy falls from 0.3135 to 0.1304, with factor 1 at 0.3130.
The marginal intervention at this site barely changes behavior even at
strength 1 (0.3125 / 0.3007); at the first block it reaches 0.1632 for
factor 2, just above the confirmation random baseline, but reduces factor-1
occupancy to 0.2585 and changes 24% of factor-1 actions. The matched
decoder-null controls at the last block changed neither occupancy nor actions.

**The preference has not flipped:** under this fixed intervention design,
joint-belief-aligned steering is still much more factor-selective than
marginal-belief-aligned steering for this seed. The 300M joint effect is
somewhat weaker than at 30M with strength 0.5 (e.g., factor-2 occupancy
0.1304 versus 0.1087 at the last block), while the 300M marginal map is far
less effective at that site even at strength 1. These are comparisons of
different fitted maps *and* target constructions, rather than an identification
of the policy's exact internal Bayesian update. They do not establish that
separate marginals are absent, and a small factor-1 occupancy difference is
not a certificate of identical behavior on every state.

Held-out marginal / joint probe R² at 300M: **0.991 / 0.983** after block 1,
**0.990 / 0.982** before final LayerNorm, and **0.990 / 0.981** after final
LayerNorm. R² values for four marginal and eight joint coordinates should
not be compared directly across targets; decoding is separate evidence from
the behavioral interventions. This is one training seed with deliberately
off-policy synthetic targets and no natural donor-activation patching.
Per-site uncertainty, action-change rates, fit metrics and tuning results are
recorded in `belief_causal_followup_seed42_300m.json`.

## Reproduction

With `B2_*` credentials and the seed-42 continuation results available, fetch
the native module under ignored `reward_both/artifacts/` (the checkpoint root
is `step_checkpoints/steps_300000000/learner_group/learner/rl_module/default_policy/`):

```bash
uv run python scripts/download_b2_prefix.py \
  --prefix experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-300m/step_checkpoints/steps_300000000/learner_group/learner/rl_module/default_policy/ \
  --dest experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42_300m/module
uv run python -m experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_causal_followup \
  --checkpoint experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42_300m/module \
  --output experiments/two_factor_reward_state_REINFORCE_cycle_4/results/belief_causal_followup_seed42_300m_REPRO.json
```

The analysis refuses to overwrite an existing result; use a fresh output
filename when reproducing it.
