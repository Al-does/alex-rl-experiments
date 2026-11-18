# Fifteen-seed five-site probes

Stage 2 extends Stage 1 with training seeds 45–56, all three conditions,
at initialization (step 0) and final step 2,012,217,344. The 18 Stage 1
reports for seeds 42–44 are copied byte-for-byte; Stage 1 is not modified.
`stage1_snapshot.json` records hashes of every original Stage 1 file.

Run from the repository root:

```sh
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.five_site_probes_stage2
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.five_site_probes_stage2 --summarize
```

The runner refuses an existing result directory; the summary refuses existing
summary outputs. The analysis reuses `five_site_probes.run_one` unchanged,
including CPU-only execution, frozen source revisions, checkpoint selection,
manifest byte-count/SHA-256 verification, independent fit/test rollouts,
warmup, targets, affine ridge probes, and final-norm validation. See the
Stage 1 README for the complete protocol and differences from steering.
New checkpoints remain under ignored `artifacts/arch_coarsening/`.

The frozen architecture source is `30608e20d125402d55e78b6451a2c04a0d36a2ee`;
steering source/run identities are from `45517b13a67d746048a002f9db52ea55a654b219`.
Seed 42 agrees with archived final-norm scores at both checkpoints; all other
seeds have new original-method final-norm-only reproductions on the original
rollout embeddings. These are separate scoring calls, not separate rollout runs.
Each new report records the unchanged Stage 1 adapter's hash; `hashes.json`
also records the Stage 2 runner/summary hash.

## Tables and uncertainty

`tables.md` gives arithmetic means of normalized errors (1−R²), each with
a two-sided 95% percentile seed-bootstrap interval. `summary.json` has
full-precision aggregates and bootstrap metadata. `per_seed.csv` retains all
1,800 site/target measurements, including R², MSE and target variance.

For each interval, resample 15 training seeds with replacement 100,000 times
using NumPy `default_rng(42)`, calculate the seed mean each time, then take
the 2.5th and 97.5th percentiles with linear quantile interpolation. The same
sampled seed indices are used across sites, targets, conditions and checkpoints.
These are pointwise intervals, not simultaneous confidence bands. They reflect
training-seed variation conditional on the fixed evaluation protocol, not
additional rollout or probe-fitting uncertainty. No clipping or error-floor
adjustment is applied.

`reward_state_paired.csv` gives, for every Reward-State Quotient seed and site,
initialization and final full-belief errors, their difference (final minus init),
and the strict comparison `final_b_below_init`. These are within-seed comparisons
on each checkpoint's own greedy histories, not a common-history intervention.

For Trivial Quotient and Reward-State Quotient, c is the exact merged-token
quotient target. For Full-State Quotient, c is the existing merged-token
filter control, **not an exact quotient**. Probe accessibility is not task
performance, information absence, or evidence of causal use.

No training, paid compute, or commits are part of this analysis.

## Large initialization outlier

Under the unchanged fitter, initialization seed 51 has very large normalized
errors in Trivial Quotient and Reward-State Quotient. Block-1 full-belief
errors are 686.277551 and 686.278807, respectively (MSE about 77.4117 and
target variance about 0.112799). Block-2 full-belief errors are about 12.5351;
block-3 errors are about 1.766. The same seed also has large c/rho errors,
including c error about 2.717 at final normalization. These unclipped values
are retained in all means and intervals and dominate the early-block means.
The outlier's cause has not been diagnosed; no regularizer change, refit,
seed exclusion, winsorization, or alternative numerical protocol was applied.
Do not interpret these extreme means as an established information deficit.
