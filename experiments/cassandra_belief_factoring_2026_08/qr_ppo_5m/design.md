# Cassandra fixed-quantile PPO comparison

## Comparison

Train two otherwise matched seed-42 policies for 5,000,000 environment steps:

1. cardinality-matched global maintenance aliases;
2. component-targeted maintenance.

Both use the successful narrow transformer (`d_model=64`, four layers, one
head), the improved 64-step transformer context/BPTT window, all-good initial
states, discount `0.990`, entropy coefficient `0.03`, and no KL penalty.

## Distributional critic

The critic predicts 64 fixed return quantiles. Their mean remains PPO's scalar
baseline for GAE and bootstrap values. PPO's scalar value MSE is excluded from
the objective (`vf_loss_coeff=0.0`); quantile Huber regression, weighted by
`0.01`, is the sole critic loss.

QR learns quantile locations, unlike C51, so it does not require fixed minimum
or maximum support values. The environment's immediate reward ranges are:

- global aliases: `[-15.0, 0.9985**4]`;
- targeted: `[-3.75, 0.9985**4]`.

At discount `0.990` and the 1,000-step horizon, conservative discounted-return
bounds are approximately:

- global aliases: `[-1499.94, 99.40]`;
- targeted: `[-374.98, 99.40]`.

The quantile Huber threshold is `10.0` return units for both conditions. This
keeps a quadratic region spanning the order of the maintenance costs while
making larger early-training return errors linear and robust. Unlike RLlib's
scalar squared-error clamp, Huber errors above this threshold retain a nonzero
linear gradient; the QR objective is not hard-clipped.

The original QR recipe used a loss coefficient of `0.5`. Observed raw QR loss
was about `30` early and `6.6` after 2.8M steps, so its weighted contribution
was roughly `15` to `3.3`—orders of magnitude larger than the policy and
entropy terms. Coefficient `0.01` instead contributes roughly `0.30` to
`0.066`, comparable to the successful scalar baseline's weighted critic
update. This directly adopts the value-update scale lesson without importing
scalar PPO's clipping mechanism into quantile regression.

`vf_clip_param=100` is retained only for RLlib's scalar value-MSE diagnostic.
Because `vf_loss_coeff=0.0`, it has no effect on gradients or the QR objective.
Keeping the same threshold and quantile count preserves the matched
environment comparison.

## Commands

```bash
uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.global_alias_qr.experiment \
  --seed 42

uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.targeted_qr.experiment \
  --seed 42
```
