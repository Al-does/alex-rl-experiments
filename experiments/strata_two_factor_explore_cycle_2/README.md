# Strata two-factor explore cycle 2

This study is **deprecated**. Existing runs in this folder used the Strata HMM parameters:

- `alpha = 0.97`
- `t0 = 0.38`
- `t1 = 0.54`

These values remain in `experiments.strata_two_factor_explore_cycle_2.process` and `experiments.strata_two_factor_explore_cycle_2.shared` and are passed to both the factor model factory (`envs.strata.model.strata_model`) and the reward task (`envs.strata.tasks.reward_state.StrataRewardTask`).

For new work, use the updated Strata defaults in `rl-harness` (`alpha = 0.98`, `t0 = 0.30`, `t1 = 0.80`).
