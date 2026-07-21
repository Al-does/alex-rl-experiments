# MESS3 Kelly cycle 2

## Question

Can direct Kelly wager gradients preserve the representation shaping seen in
cycle 1 while ordinary correctness reward restores token-policy accuracy? Does
an IQN critic improve either outcome relative to a scalar mean-value critic?

## Controlled design

All arms use passive MESS3 (`alpha=0.85`), token-only observations, `delay=1`,
the same 3-layer transformer, PPO, 2.5 million sampled steps, no warm start, and
no predictive auxiliary loss. `gamma=0` and `lambda=0` intentionally make the
task a contextual bandit: current actions do not alter future MESS3 states.

Each actor condition has a scalar-critic and IQN-critic variant:

1. `correctness`: PPO receives binary token correctness; no wager head.
2. `coupled_kelly`: PPO receives Kelly log growth and a scalar wager head gets
   direct realized-Kelly loss.
3. `decoupled_kelly`: PPO receives binary correctness while the scalar wager
   head independently gets direct realized-Kelly loss.
4. `conditional_decoupled_kelly`: as above, but the transformer emits one wager
   per token and only the selected token's wager is trained each step.

The direct Kelly loss remains connected to the shared transformer, preserving
the candidate representation-shaping pressure. The decoupled arms remove wager
size from categorical-policy credit, while action-conditional wagers remove the
cycle-1 scalar head's policy-average confidence ambiguity.

IQN arms replace the scalar critic loss with 32 sampled quantiles and use the
mean of 64 fixed quantiles as PPO's baseline. With `gamma=0`, IQN models the
distribution of immediate correctness or Kelly growth rather than a
long-horizon return.
