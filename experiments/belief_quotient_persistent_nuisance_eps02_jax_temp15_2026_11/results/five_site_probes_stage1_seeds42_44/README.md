# Five-site probes: Stage 1

Offline CPU-only analysis of training seeds 42–44 in all three transformer
conditions, at actual step-0 initialization and final step 2,012,217,344.
No training or paid compute. No historical results are overwritten.

Run from the experiment repository:

```sh
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.five_site_probes
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.summarize_five_site_probes
```

Both commands refuse existing outputs. The first requires the frozen source
revisions below in the local Git object database. The second creates `tables.md`,
`per_seed.csv`, `summary.json`, and `hashes.json` from the 18 per-checkpoint reports.

## Provenance and protocol

Run identities and final checkpoint hashes are selected from the exact
`belief_steering_v2_seed{42,43,44}_stable.json` reports at
`45517b13a67d746048a002f9db52ea55a654b219`. Both actual checkpoints are downloaded
from B2 and verified for byte count and SHA-256 against each run's live canonical
durability manifest before loading. The manifest hash, object key, run ID, source
revisions, source/adapter hashes and runtime versions are recorded per report.
Checkpoints remain in ignored `artifacts/arch_coarsening/<run-id>/`.

The adapter executes the unchanged architecture-coarsening `flatten`, `score`,
`merged_filter`, and `verified_checkpoint` functions from
`30608e20d125402d55e78b6451a2c04a0d36a2ee`. Greedy own-checkpoint rollouts use
60,000 fit and 80,000 test rows from independent 1,024-step episodes, discard
64 decisions per episode, and retain the original episode-major row ordering.
Keys are `split(fold_in(PRNGKey(42), 2))` for every checkpoint and training seed.
Ridge is 1e-6 with an unpenalized intercept; JAX matmul precision is highest.
The protocol therefore compares each initialized/trained agent on its own
greedy histories, not initialized activations on trained-agent histories.

The observed histories are replayed with the steering analysis's unchanged
`instrumented_step` at intervention site -1 (no intervention), collecting the
latest-position residual after blocks 1–4 and after final normalization.
Final-norm activations are checked against the original rollout's `embedding`
on every episode and timestep, including warmup. Maximum discrepancy is zero.

`b` is the complete three-coordinate decision-time posterior given tokens and
executed actions. `c` is the reward-state coordinate of the filter with A/B
merged, using complete histories. It is an exact two-state quotient target for
Trivial Quotient and Reward-State Quotient. For Full-State Quotient it is the
existing three-state merged-token filter control, **not an exact quotient**.
Using only its reward-state coordinate reproduces the two-coordinate normalized
error where the exact quotient exists. `rho` is b's reward-state probability;
`delta` is the difference of its two nonreward-state probabilities. Each target
is directly fitted by the same affine ridge routine. All four targets have
`error` (1−R²), `r_squared`, MSE and target variance. Multivariate b error pools
the squared errors and centered target sums of squares over all coordinates.

## Validation and differences from steering

Final-norm b/c scores from replayed sites are compared with separate calls to
the original analysis's score function on its original rollout `embedding`
(not replayed features), using the original full-belief and merged-token targets.
For training seed 42, initialization/final scores are additionally checked
against the archived architecture-coarsening reports at `30608e20`.
Seeds 43/44 are **new final-norm-only reproductions**, not archived results.

Steering probes instead use 12,000 fit/test retained rows, eight-step warmup,
separate rollout seeds 42/43 and an augmented affine least-squares fit with
relative SVD cutoff 1e-6, without ridge. They decode the first two belief
coordinates and derive rho/delta from those predictions; this analysis fits
rho/delta directly with ridge. Therefore their coordinate R² need not reproduce
numerically under this protocol. The original five-site steering scores are
retained in each final-checkpoint report for comparison; they are not
initialization controls or protocol-matched reference values.

No additional seeds beyond Stage 1 have been evaluated. Affine probe results
describe accessibility, not information absence or causal use.
