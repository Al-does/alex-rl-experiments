# Strata two-factor explore cycle 2

The Strata HMM parameters used for this study have been updated.

- **Deprecated parameters** used for existing runs in this folder:  
  `alpha = 0.97`, `t0 = 0.38`, `t1 = 0.54`
- **Current parameters**:  
  `alpha = 0.98`, `t0 = 0.30`, `t1 = 0.80`

These values are passed through `experiments.strata_two_factor_explore_cycle_2.process.environment_config` and `experiments.strata_two_factor_explore_cycle_2.shared.build_config` to both the factor model factory (`envs.strata.model.strata_model`) and the reward task (`envs.strata.tasks.reward_state.StrataRewardTask`).
