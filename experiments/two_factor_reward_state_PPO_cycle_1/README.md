# Two-factor reward-state PPO, cycle 1

This is the PPO comparison for PR 65's two-factor reward-state SAC study. It
asks whether a recurrent transformer trained for control linearly represents
both independent MESS3 beliefs when reward depends on both factors, and whether
it selectively represents the rewarded factor when only one factor contributes
reward.

## Preregistered design

- The process, flat nine-action product, reward arms, 20,000,000-step budget,
  seed policy, demand audit, checkpoint schedule, and probe targets are the
  same as `two_factor_reward_state_SAC_cycle_1`.
- Two independent three-state MESS3 factors use emission accuracy `alpha=0.55`
  and baseline transition rows
  `[[.75,.15,.10],[.15,.75,.10],[.30,.30,.40]]`.
- The policy sees only one of nine joint symbols plus the preceding flat action.
  It never sees either subtoken, latent state, belief, or product structure.
- PPO uses a stateful, pre-LayerNorm transformer with a 64-dimensional residual
  stream, four layers, four heads, a 256-dimensional ReLU MLP, learned absolute
  positions, and a 64-frame context.
- Unlike stateless SAC, PPO carries history in recurrent module state. Each
  observation therefore contains one aligned joint-token/action frame rather
  than a flattened copy of all 64 frames.
- PPO uses `gamma=0.99`, GAE `lambda=0.95`, learning rate `3e-4`, six epochs,
  a 32,768 train batch, and 4,096-sample minibatches.

## Required pretraining audit

The exact PR 65 audit compares an exact-filter QMDP policy to reactive and
constant policies using at least 4,096 chains, 6,000 steps, and 500 burn-in
steps. Training aborts unless the demand gap is at least `0.015` and its
chain-level standard error is at most `5e-4`.

## Analysis

Initial, power-of-two-iteration, and final checkpoints are probed at the shared
actor-critic transformer's last block residual before final LayerNorm.
Independent held-out rollouts fit linear readouts to the exact joint predictive
belief and both exact factor marginals. Reports include held-out MSE, RMSE, R²,
and principal-component counts at 95% cumulative explained variance.
