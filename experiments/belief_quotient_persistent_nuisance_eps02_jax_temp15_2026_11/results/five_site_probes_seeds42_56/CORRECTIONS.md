# Selective simplex correction

Run `uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.constrain_five_site_probes`
from the experiment repository. Existing correction outputs are refused.

All 90 original reports and previous Stage 1/2 summaries are retained unchanged.
Use `corrected_tables.md`, `corrected_summary.json`, `corrected_per_seed.csv`
and `corrected_paired.csv` for the corrected view. The auditable overlay is
`degenerate_fit_replacements.json`, including every original/reproduced/new
score, method, checkpoint hash, raw-input hashes, and diagnostics.
`corrected_hashes.json` hashes reports, correction outputs and the adapter.

Exactly 32 of 1,800 scores meet the original error > 1 selection rule, all at
initialization seed 51 in Trivial Quotient or Reward-State Quotient. The other
1,768 scores are unchanged. Final checkpoints and Full-State Quotient need no
replacements. Every original selected score is reproduced on exactly the
same deterministic fitting/testing rows, with no regularizer or solver change.

The Euclidean simplex projection function is loaded directly from
`deep-beliefs/experiments/persistent_nuisance_v2/constrain_degenerate.py` and its
source hash is recorded. Selected b scores use projected three-coordinate
predictions. Selected c scores project their own two-state (1-c,c) predictions
and score the reward coordinate with the original metric. Selected rho/delta
scores derive from projected b predictions. Only scores originally above 1
are replaced, even when the auxiliary b projection is computed for a selected
coordinate at a site where the original b score itself is below 1.

Means and pointwise percentile seed-bootstrap intervals use the same 15 seeds,
100,000 resamples, default_rng(42), shared seed indices and linear quantile
interpolation as Stage 2. `corrected_paired.csv` includes all 375 within-seed
comparisons for Reward-State Quotient b/c/delta and Full-State Quotient b/delta;
counts use strict final error < initialization error. Histories remain each
checkpoint's own greedy histories, not a common-history intervention.

Run `uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.summarize_five_site_robust`
after correction to append median tables to `corrected_tables.md` and add
`median` to each of the 120 aggregates in `corrected_summary.json`. It also
adds `sensitivity_excluding_seed51`: the 14-seed arithmetic means for the two
initialization block-1 c cells, excluding only seed 51. The primary 15-seed
means/CIs are not replaced. All statistics are computed from the corrected
per-seed CSV; the correction hash manifest is updated. Repeated application
is refused. The sensitivity means are 0.249601 (Trivial Quotient) and 0.370117
(Reward-State Quotient).

## Diagnosis and remaining caveat

The executed fitter uses float64 SVD-backed ridge least squares, not normal
equations. Replayed final-norm activations exactly match original rollout
embeddings; original scores reproduce. For seed 51, block-1 centered fitting
features have condition number 1.385e7. A weakly supported singular direction
has singular value 0.001499 and coefficient norm about 7,441; its maximum
held-out coordinate (0.2651) is 1,679 times its fitting maximum (0.0001579).
Unconstrained predictions reach about 3,040; ten held-out rows account for
99.72% of squared error. This is observed held-out extrapolation in an
ill-conditioned probe. The diagnostic seed-50 control is similarly
ill-conditioned (1.384e7) but predictions reach only 1.199 and the ten worst
rows account for 0.22% of squared error. Thus condition number alone is not
sufficient to explain the extreme error. Why seed 51's histories generate
this feature-support mismatch has not been further diagnosed.

Projection does not guarantee positive R². At block 1, projected c error for
seed 51 remains 39.6266 / 39.6184 in Trivial / Reward-State Quotient: projected
MSE is about 1.306e-4 against target variance 3.296e-6. These bounded-prediction
scores are retained, not clipped or replaced by the mean predictor; they still
dominate the initialization c mean/CI at that site.

Selection is retrospective on held-out error. Selected predictions are
constrained while unselected ones remain unconstrained, so corrected results
are not a uniformly constrained-probe benchmark. No training, paid compute,
commits or pushes were performed.
