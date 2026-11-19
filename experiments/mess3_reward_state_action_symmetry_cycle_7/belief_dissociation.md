# Rho/delta dissociation and single-step pulses

```sh
uv run pytest -q tests/test_fsq_belief_steering.py tests/test_belief_dissociation.py
uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.belief_dissociation --smoke --seeds 42
uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.belief_dissociation
```

This additive follow-up never overwrites the original Full-State Quotient
reports. It restores final Reward-State Quotient and Full-State Quotient
modules for training seeds 42–46 through the existing manifest-verified
downloader. All three native files must match the canonical B2 SHA-256 records;
the Reward-State Quotient seed-42 module also has a pinned manuscript hash.
Artifacts, checkpoints, smoke outputs, and execution logs are ignored.
Every output refuses replacement. Completed outputs are not silently reused.

## Sustained steering

The fit/test, map coordinates, hooks, eight-step probe warmup, 12,000 retained
steps per split, seeds 42/43, tuning seed 1042 (16 complete episodes), and
confirmation seed 2042 (32 complete episodes) match the preceding FSQ driver.
Episodes are 1,024 decisions; occupancy includes reset decisions. The entire
prespecified lambda grid is run at block 4 on confirmation episodes. Each
target's smallest tuning lambda at/below random occupancy is frozen across
sites; if none qualify the lowest-occupancy lambda wins (smaller lambda breaks
ties). A genuine null can therefore select lambda zero. Inspect lambda-one
curves and their nonzero shift norms rather than a zero-strength control alone.

Both agents receive point/mirror delta targets and rho_flip/rho_point targets.
For the rho targets, nonreward mass is divided in the raw decoded b1:b2 ratio.
If their sum is nonpositive, division falls back to equal parts. No clipping or
simplex projection is applied. rho_point chooses reward mass one exactly when
decoded rho is below 0.5, otherwise zero. Thus raw decoded targets can be
off-manifold, just as for the original affine intervention.

At every grid point the reports include mean actual installed residual-shift
norm, mean pre-shift residual norm, and their ratio of means. These are measured
on each condition's own closed-loop histories, not a shared intact activation
distribution. The shift includes lambda; lambda zero has zero norm by design.
The held-out coordinate R² values separately score b1, b2, derived rho=1-b1-b2,
and delta=b1-b2. A weak delta probe in the Reward-State Quotient agent remains
a qualification on interpreting its delta intervention; a large shift alone
does not prove a semantically calibrated write.

The existing prediction module now supports both agents' controlled matrices
and all four targets. Full-State Quotient uses the original directions;
Reward-State Quotient uses Tilt X (+1,+1,-1), Tilt Y (-1,-1,-1).
Each reference uses the same N=60, gamma=0.97, 800-iteration nearest-grid value
iteration and exact closed-loop filter, with target applied to the true belief.
It is a feasible near-optimal reference, not certified Bayes optimality.
Aggregate uncertainty is SE across five training-seed means; each individual
report retains episode SE. Random/identical greedy behaviors on matched seeds
can have zero training-seed SE without zero Monte Carlo uncertainty.

## Pulses

Intact greedy rollouts use 32 complete episodes, fresh seed 3042. Cache
snapshots are taken **before** processing decisions 256, 512, and 768. Each
branch receives a single lambda-one mirror or rho_flip pulse at block 1, 2,
or 3, then no more shifts. It receives exactly the intact observation vectors,
including intact executed preceding-action features, for lags 0–48. Branch
actions never enter the environment, filter, or subsequent observation vector.
Every branch begins from its own cloned intact cache. Comparisons use the
intact decoder readout and logits from these same inputs and episode batch.

The readout is the affine block-4 decoder fitted on intact data. Signed
differences and mean absolute differences in decoded rho and delta are saved
at all lags, alongside greedy-action disagreement and action-distribution TV.
Repeated pulses are averaged within an episode before computing episode SE.
The five-seed aggregate retains all per-seed fitted rates and across-seed SEs
for signed and absolute curves.

Full-filter predictions start at the exact posterior for decision t0, replace
it by target(b_t0), and then consume **future** tokens and intact preceding
actions. They do not reprocess the token already observed at t0. Reward-State
Quotient predictions additionally reuse `coarse_filter_spec()` to replay the
exact lumped filter over the entire coarse-token prefix. That separate
posterior, not full-filter rho, initializes its own perturbed/unperturbed pair.
The quotient only predicts rho; it has no delta coordinate. Its delta pulse
leaves rho unchanged. Predictions are posterior differences, not learned-map
reconstructions, and do not assert that a residual shift installs exactly that
posterior in the network.

Reported decay multipliers are exp(slope) from ordinary least squares on
log(mean absolute difference) over lags 1–48. Values below
max(1e-12, peak amplitude × 1e-8) are excluded. Fewer than three valid lags
produces no fitted rate. A multiplier below one indicates decay, above one
growth; fit R² and fitted lag endpoints are retained. These descriptive fits
can hide nonexponential transients or the transformer's per-layer band cutoff;
the complete lag curves are the primary evidence. No causal Bayesian-state
claim follows merely from persistence or accessibility.
