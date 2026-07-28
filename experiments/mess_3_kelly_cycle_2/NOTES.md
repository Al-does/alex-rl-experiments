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

## Three-seed results

| condition | belief R² | token accuracy | expected log growth |
|---|---:|---:|---:|
| `correctness_mean` | 0.9834 ± 0.0015 | 0.6830 ± 0.0027 | — |
| `correctness_iqn` | **0.9857 ± 0.0005** | 0.6836 ± 0.0033 | — |
| `coupled_kelly_mean` | 0.9375 ± 0.0047 | 0.6760 ± 0.0039 | **0.1254 ± 0.0061** |
| `coupled_kelly_iqn` | 0.9467 ± 0.0060 | 0.6779 ± 0.0040 | 0.1148 ± 0.0043 |
| `decoupled_kelly_mean` | 0.9529 ± 0.0038 | **0.6860 ± 0.0018** | 0.0333 ± 0.0186 |
| `decoupled_kelly_iqn` | 0.9572 ± 0.0035 | 0.6856 ± 0.0025 | 0.0243 ± 0.0054 |
| `conditional_decoupled_kelly_mean` | 0.9559 ± 0.0075 | 0.6858 ± 0.0022 | -0.0277 ± 0.0101 |
| `conditional_decoupled_kelly_iqn` | 0.9491 ± 0.0066 | 0.6858 ± 0.0028 | -0.0443 ± 0.0079 |

The main finding is that myopic correctness PPO itself produces a nearly linear
belief representation. Moving from the prior `gamma=0.99` reward-only result
(`R²=0.8552`) to `gamma=0` gives `R²=0.9834` without an auxiliary objective.
IQN adds only `0.0022` mean R² on this immediate-reward task, much less than its
gain with longer-horizon returns.

Decoupling token reward from wager size restores accuracy to roughly 68.6%, so
cycle 1's low accuracy was a credit-assignment problem rather than a necessary
Kelly tradeoff. However, direct wager losses do not improve on the gamma-zero
correctness representation. They also calibrate poorly: scalar decoupled wagers
have roughly `0.38` RMSE against Bayes Kelly, while conditional wagers exceed
`0.40` and lose expected wealth.

A likely mechanism is per-sample overfitting. PPO reuses each realized binary
outcome for six epochs, and direct Kelly loss can push the wager toward an
extreme based on that one stochastic outcome. Action-conditional heads receive
even sparser feedback. The scalar coupled arm remains the most profitable
learned bettor here, but the cleanest belief representation comes from ordinary
gamma-zero correctness PPO.
