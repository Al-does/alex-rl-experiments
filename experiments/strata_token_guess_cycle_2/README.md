# Strata token-guess cycle 2

This cycle repeats the single-HMM Strata token-guess study with:

- `alpha = 0.98`
- `t0 = 0.30`
- `t1 = 0.80`

The recipe pins these values in `process.py`, passes them to
`envs.strata.model.strata_model`, and reuses them when constructing the
analysis emission geometry.
