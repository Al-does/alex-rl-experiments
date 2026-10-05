# Joint Reward seed-42: factor-specific causal-localization follow-up

## Protocol

The frozen seed-42 Joint Reward RLModule is identified by SHA-256
`2ce92d4502837894e0024c4aaa000f9bb1b520993d1d6063ef667686a4db8d76`.
Greedy rollouts provide separate 12,000-step fitting and 12,000-step probe-test
groups (seeds 42 and 43; the first eight steps of each episode are excluded).
Independent 16-episode tuning (seed 1042) selects intervention strength; 32
complete 1,024-step episodes (seed 2042) confirm it. Each comparison uses the
same evaluation environment seeds. Occupancy is the fraction of steps in the
factor's rewarded state; standard errors are across complete episodes.

At each representation site, separate affine regressions fit activations to
either four marginal-belief coordinates or eight of nine joint-belief
coordinates, and fit the corresponding belief-to-activation map. The joint
counterfactual is the decoded factor-1 marginal outer-producted with a
factor-2 point mass on the less likely **nonreward** state; the ninth joint
coordinate is inferred from normalization. The marginal counterfactual uses
the corresponding two factor-2 coordinates. In both cases the fitted
belief-to-activation map converts the target difference to a residual shift.
These targets are extrapolations from ordinary on-policy beliefs.

All executed actions come from the *single steered policy forward pass* at
each step. A separate unmodified pass from the same current state only
measures action changes; its actions and cache are never executed. Earlier
block shifts are applied to the output of the block's MLP, equivalently to
its post-block residual, before subsequent blocks. Thus subsequent blocks'
KV caches incorporate the shifted residual; the intervened block's own KV
cache is computed before that shift. The final-block and post-normalization
shifts do not rewrite earlier recurrent memories.

## Strength tuning

The smallest tested strength at or below the *tuning-set* random-action
factor-2 occupancy (0.165) was **0.9 for the marginal map** and **0.5 for
the joint map**. Intermediate tuning occupancies (factor 1, factor 2):

| Strength | Marginal map | Joint map |
|---:|---:|---:|
| 0 | 0.308, 0.316 | 0.308, 0.316 |
| 0.50 | 0.308, 0.316 | 0.308, 0.106 |
| 0.75 | 0.307, 0.283 | 0.308, 0.066 |
| 0.80 | 0.302, 0.238 | 0.308, 0.066 |
| 0.85 | 0.293, 0.180 | 0.308, 0.066 |
| 0.90 | 0.281, 0.137 | 0.308, 0.066 |
| 0.95 | 0.264, 0.103 | 0.308, 0.066 |
| 1.00 | 0.251, 0.083 | 0.308, 0.066 |

## Held-out confirmation

Intact occupancy was **0.3130 ± 0.0024** for factor 1 and **0.3134 ±
0.0027** for factor 2. Uniform random joint actions yielded **0.1633 ±
0.0018** and **0.1617 ± 0.0021**, respectively. The table uses each map's
tuning-selected strength; uncertainties are standard errors of episode
occupancy, not training-seed uncertainty.

| Intervention site | Marginal map F1 / F2 | Joint map F1 / F2 |
|---|---:|---:|
| After block 1 | 0.2748 / 0.0817 | **0.3127 / 0.1255** |
| After block 2 | 0.2755 / 0.0772 | **0.3127 / 0.1194** |
| After block 3 | 0.2766 / 0.0874 | **0.3130 / 0.1173** |
| After block 4, before final LayerNorm | 0.2835 / 0.1357 | **0.3130 / 0.1087** |
| After final LayerNorm | 0.2910 / 0.1587 | **0.3130 / 0.1252** |

The joint map has negligible factor-1 action changes at all sites (0 to
0.14% of decisions) and changes factor-2 actions on 76–82% of decisions.
The marginal map changes factor-1 actions on 10–17% of decisions. At the
final pre-normalization site, a stepwise norm-matched random direction
orthogonal to each fitted belief decoder changed neither actions nor
occupancy for either map.

Held-out probe R² for marginal/joint targets was 0.989/0.981 after block 1,
0.984/0.973 before final LayerNorm, and 0.983/0.971 after final LayerNorm.
These scores are within-target measures: a four-coordinate marginal R²
and an eight-coordinate joint R² are not directly interchangeable.

The confirmation supports a **factor-selective causal effect of steering
through the fitted joint-belief map**, including at earlier blocks. It does
not establish that the network internally computes that exact joint
posterior or that these off-manifold counterfactuals operate exclusively by
changing beliefs rather than action-relevant features correlated with them.
Testing natural on-manifold donor-activation patches, more checkpoints, and
multiple training seeds would narrow those alternatives. The shared fixed
strengths were selected at the final pre-normalization site and were *not*
retuned for each earlier layer. The null direction tests a decoder-orthogonal
direction of matched norm, not every possible unrelated perturbation.

Full per-site probe MSE/target variance, per-condition occupancy standard
errors, action-change rates, seeds, and strength-grid values are in
`belief_causal_followup_seed42.json`.

## Reproduction

Restore the seed-42 native module under the ignored
`reward_both/artifacts/intervention_seed42/module/` directory as in the
original `belief_intervention_seed42.md`, then run from the experiment repo:

```bash
uv run python -m experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_causal_followup \
  --checkpoint experiments/two_factor_reward_state_REINFORCE_cycle_4/reward_both/artifacts/intervention_seed42/module \
  --output experiments/two_factor_reward_state_REINFORCE_cycle_4/results/belief_causal_followup_seed42.json
```

The command refuses to overwrite an existing report; use a new output path
to reproduce independently.
