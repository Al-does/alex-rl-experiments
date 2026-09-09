# Strata token-guess cycle 1

The Strata HMM parameters used for this study have been updated.

- **Deprecated parameters** used for existing runs in this folder:  
  `alpha = 0.97`, `t0 = 0.38`, `t1 = 0.54`
- **Current parameters**:  
  `alpha = 0.98`, `t0 = 0.30`, `t1 = 0.80`

These values are passed through `experiments.strata_token_guess_cycle_1.process.environment_config` to `envs.strata.model.strata_model`.
