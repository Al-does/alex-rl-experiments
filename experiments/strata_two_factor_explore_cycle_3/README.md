# Strata two-factor explore cycle 3

This cycle repeats the two-factor Strata exploration study with:

- `alpha = 0.98`
- `t0 = 0.30`
- `t1 = 0.80`

The recipe pins these values in `process.py` and passes them to both the
factor-model factory (`envs.strata.model.strata_model`) and the reward task
(`envs.strata.tasks.reward_state.StrataRewardTask`). The same values feed the
analytic design and probe emission geometry.
