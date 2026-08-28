# Two-factor reward-state SAC, cycle 1

This study asks whether a recurrent-history transformer trained for control
linearly represents both independent MESS3 beliefs when reward depends on both
factors, and whether it selectively represents the rewarded factor when only
one factor contributes reward.

## Preregistered design

- Two independent three-state MESS3 factors use emission accuracy `alpha=0.55`.
- Each factor has baseline transition rows
  `[[.75,.15,.10],[.15,.75,.10],[.30,.30,.40]]`.
- The policy sees only one of nine joint emission symbols. It never sees either
  subtoken, latent state, belief, or the pair structure of the action space.
- A factor shift `k` moves every old destination `j` to `(j+k) mod 3`.
- One flat `Discrete(9)` action indexes the exact pair order in `task.py`.
- The policy observation contains a 64-step history of joint-token one-hots and
  preceding flat-action one-hots. This is required because RLlib's SAC module
  is stateless.
- Rewards are current-state occupancy indicators: both factors additively,
  factor 1 only, or factor 2 only.
- Every arm trains discrete SAC for 20,000,000 environment steps at seed 42
  unless the runtime supplies another seed.

## Required pretraining audit

`design.demand_audit()` compares an exact-filter QMDP policy to reactive and
constant policies using at least 4,096 chains, 6,000 steps, and 500 burn-in
steps. Training aborts unless the demand gap is at least `0.015` and its
chain-level standard error is at most `5e-4`.

## Analysis

Initial, power-of-two-iteration, and final checkpoints are probed at the actor's
last transformer-block residual before final LayerNorm. Independent held-out
rollouts fit linear readouts to:

1. the exact nine-state joint predictive belief;
2. the exact three-state factor-1 marginal;
3. the exact three-state factor-2 marginal.

These are transducer targets: the exact filter applies the previously executed
action pair's controlled transition before conditioning on the new joint token.
Reports include held-out MSE, RMSE, R², and principal-component counts at 95%
cumulative explained variance (CEV) for actor activations and all targets.
