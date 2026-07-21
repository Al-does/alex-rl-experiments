# MESS3 Kelly cycle 3

## Question

At the conventional PPO horizon (`gamma=0.99`, `lambda=0.95`), how do plain
PPO, IQN PPO, and action-conditional decoupled Kelly shaping compare?

## Controlled design

All arms use passive MESS3 (`alpha=0.85`), token-only observations, `delay=1`,
the same 3-layer transformer, 2.5 million sampled steps, seeds 42–44, no warm
start, and no predictive auxiliary loss.

1. `ppo`: binary token-correctness reward and a scalar mean-value critic.
2. `iqn`: the same actor reward with an IQN critic.
3. `conditional_decoupled_kelly_mean`: correctness PPO plus three
   action-conditional wager outputs trained by selected-action direct Kelly
   utility; scalar critic.
4. `conditional_decoupled_kelly_iqn`: the same conditional Kelly setup with an
   IQN critic.

The probe streams are identical to cycle 2. Per-iteration training curves and
Tune summaries are intentionally omitted from tracked results; compact recipes,
final probes, manifests, findings, and comparison figures remain.
