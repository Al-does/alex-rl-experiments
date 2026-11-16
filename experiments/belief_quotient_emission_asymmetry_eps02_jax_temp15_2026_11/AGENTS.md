# belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11

Approximately lumpable rungs between the Reward-State and Full-State conditions
of the PR #234 ladder (`belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11`).
Written for the deep-beliefs reviewer request to "perturb the symmetry (e.g.,
slightly different emission probabilities for states 1 and 2) and report which
target the representation tracks: the approximate quotient, the full belief, or
neither". No full training has been run; smoke outputs are not findings.

## Process

The source `variant_2` process with one change. States 1 and 2 emit C with
probability `0.85 - kappa` and `0.85 + kappa`; each keeps its 3:1 own-token to
other-token ratio. Transitions, tilts (effect size 2.5), uniform reset, reward
state 3 and the state-3 emission row `[.01, .01, .98]` are the source's.

| leaf | kappa | not-C rate, states 1 / 2 | state-1 row | state-2 row |
| --- | --- | --- | --- | --- |
| source `variant_2_transformer` | 0 | .15 / .15 | .1125 .0375 .85 | .0375 .1125 .85 |
| `asym_025_transformer` | .025 | .175 / .125 | .13125 .04375 .825 | .03125 .09375 .875 |
| `asym_050_transformer` | .05 | .20 / .10 | .15 .05 .80 | .025 .075 .90 |
| `asym_075_transformer` | .075 | .225 / .075 | .16875 .05625 .775 | .01875 .05625 .925 |
| `asym_100_transformer` | .10 | .25 / .05 | .1875 .0625 .75 | .0125 .0375 .95 |

At `kappa = 0` the classes `{1,2}` / `{3}` with tokens merged to not-C / C are an
exact two-state POMDP (the manuscript's quotient filter). For `kappa > 0` the two
non-reward states still share every reward-class transition probability but
emit the merged token at different rates (emission lumpability defect
`2 * kappa`), so no two-state filter is exact: the reward-state posterior depends
on which non-reward state the agent is in. The equal-weight two-class model is
the `kappa = 0` quotient at every level, so the approximate-quotient target is
the manuscript's closed-form filter applied unchanged.

Token B is equally likely from states 1 and 2 at `kappa = .075` and likelier
from state 1 above it; identity is then carried by token A and by C-run lengths.

## Recipe

Each leaf is the PR #234 `variant_2_transformer` recipe: 2B-step target
(2,012,217,344 actual), entropy 0, temperature 1.5, no epsilon exploration,
transformer 64x4 with banded RoPE window 32, same checkpoint and probe cadence. A test
asserts the config, model spec, initial distribution, transitions and reward
equal the source leaf's. Training-time probes add `reward_class` and `identity`
to the full `belief` target. Seeds are runtime `--seed` values; PR #234 used
42-56. Source runs took about 20 minutes each on an RTX 4090.

```bash
uv run --group jax pytest -q tests/test_emission_asymmetry_eps02.py
uv run --group jax rl-harness experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.asym_050_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run --group jax rl-harness experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.asym_050_transformer.experiment --seed 42 --hardware-profile cuda4090
```

## Design diagnostics (no training)

`design.py` simulates two controllers per level with exact float64 filters:
Tilt X while a statistic is below a threshold, Noop otherwise, the statistic
being the quotient filter or the full belief's reward marginal. Thresholds are
selected on one seed and confirmed on another. `results/design_summary.json`
(8,192 complete 1,024-step episodes per evaluation, seed 0):

| kappa | quotient controller | reward-marginal controller | quotient regret | reward-marginal variance not explained by the quotient |
| --- | --- | --- | --- | --- |
| 0 | .4541 | .4554 | .0014 (0.3%) | .003 |
| .025 | .4483 | .4507 | .0024 (0.5%) | .013 |
| .05 | .4291 | .4384 | .0093 (2.1%) | .056 |
| .075 | .3949 | .4162 | .0213 (5.1%) | .156 |
| .10 | .3487 | .3837 | .0350 (9.1%) | .349 |

Episode Monte Carlo standard errors are about .001 per controller and .0005
for the paired regret. These are simulated feasible controllers, not
Bayes-optimality bounds, and threshold rules are not shown to be optimal within
either information class.

```bash
uv run --group jax python -m experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.design --output NEW.json
```

## Offline analysis

`analysis.py` takes one completed run of this study, or a PR #234
`variant_2_transformer` run (the `kappa = 0` anchor) or `variant_3_transformer`
run (Full-State control), and never trains. It reads the process from
`resolved_recipe.json`, verifies the step-0 and final checkpoints against the
run's durability manifest (downloading from B2 into `--checkpoint-dir` when
absent), and writes a JSON report that is never overwritten.

```bash
uv run --group jax python -m experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.analysis \
  --run-dir experiments/<study>/<leaf>/results/<run-id> \
  --checkpoint-dir experiments/belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11/artifacts/analysis_inputs/<run-id> \
  --output experiments/belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11/results/analysis/<label>_seed<seed>.json
uv run --group jax python -m experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.report \
  experiments/belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11/results/analysis/*.json
```

`--smoke` shrinks every budget; local-only runs without a durability manifest
need `--allow-unverified`. Protocol: final policy, greedy, 64 fit / 128 test
complete 1,024-step episodes from independent seeds, warmup 128 (the encoder's
receptive field), post-final-LayerNorm site, grouped SVD-cutoff affine probes.
The run's own step-0 encoder is replayed on the same observations. Targets are
refiltered in float64 over complete episode prefixes and checked against the
rollout filter. Intervals are 95% episode bootstraps with fixed probes: they
describe evaluation trajectories for one training seed, not seed variation.

Readouts, each within a run so that visitation differences across levels do not
enter:

1. **Identity.** Change from initialization in held-out R² of `b1 - b2`.
2. **Which reward statistic.** Held-out R² of the quotient filter minus that of
   the exact reward marginal on identical rows, and its change from
   initialization (`which_target.*.training_change`). The occupancy-weighted
   two-class filter is scored the same way as a stronger quotient candidate.
3. **Quotient gap.** Change from initialization in R² of
   `reward marginal - quotient`, the part of the reward belief that needs identity.
4. **Action rule.** Held-out agreement of majority-action lookups on the
   quotient, the reward marginal, and each joined with the state-1/state-2
   share; the gain from adding identity to the quotient, and which rule the
   agent follows on rows where the two one-dimensional rules disagree.
5. **Behavior.** Greedy expected reward (2,048 episodes) against the two
   simulated controllers for the run's own process.

Reading the outcome:

- Approximate quotient: identity change stays at or below the `kappa = 0`
  anchor, readout 2 does not move toward the reward marginal, identity adds
  nothing to the action rule, reward does not exceed the quotient controller.
- Full belief: identity change rises toward the Full-State control, readout 2
  turns negative, identity improves the action rule, reward exceeds the
  quotient controller.
- Neither: quotient and reward-marginal R² both fall toward the
  token/previous-action baseline or below initialization.

Mixed patterns (identity accessible but unused by the action rule) are a result
and should be reported as such. Power differs by level: the two reward targets
share 98.7% of their variance at `kappa = .025`, so only readouts 1, 3 and 4 are
informative there; readout 2 becomes informative from `kappa = .05`. Probe R²
is affine accessibility, not causal use.

## Pipeline check on existing PR #234 seed-42 checkpoints

Run from an uncommitted tree, outputs in ignored `artifacts/validation/`; one
training seed, so these verify the pipeline and are not findings.

| run | identity R² change | readout 2 change from init | action agreement gained from identity | agent reward |
| --- | --- | --- | --- | --- |
| `variant_2` (kappa 0) | -.179 [-.187, -.171] | -.0003 [-.0007, +.0001] | +.003 [+.003, +.004] | .439 |
| `variant_3` (Full-State) | +.051 [+.048, +.054] | -.0101 [-.0111, -.0091] | +.166 [+.151, +.179] | .415 |

The seed-42 anchor trails the quotient controller (.455) by .015, more than the
quotient regret at `kappa <= .05`. Behavior can therefore separate the outcomes
only at `kappa >= .075`; at lower levels it is a consistency check.

## Scope

One perturbation family on one rung. It does not perturb action effects, vary
the architecture, or bound Bayes-optimal return. Suggested minimum: seeds
42-46 on all four leaves (20 runs) plus the analysis of the matching PR #234
`variant_2` and `variant_3` seeds; 42-56 matches the source campaign.
