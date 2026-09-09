# Strata token-guess cycle 1

This study is **deprecated**. Existing runs in this folder used the Strata HMM parameters:

- `alpha = 0.97`
- `t0 = 0.38`
- `t1 = 0.54`

These values remain in `experiments.strata_token_guess_cycle_1.process` and are passed to `envs.strata.model.strata_model`.

For new work, use the updated Strata defaults in `rl-harness` (`alpha = 0.98`, `t0 = 0.30`, `t1 = 0.80`).
