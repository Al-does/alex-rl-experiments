# MESS3 Kelly cycle 1

## Question

Can Kelly-style reward variants make a transformer encode the exact Bayesian
belief state under reinforcement learning, without next-token supervision or
warm-starting?

## Controlled recipe

All conditions use the passive three-state MESS3 process (`alpha=0.85`),
token-only observations, `delay=1`, the same 3-layer causal transformer, PPO,
seed 42 by default, and 2.5 million sampled agent steps. The categorical token
action remains the only environment action.

For a selected token with wager fraction `f`, fair three-way odds give:

- correct: `log(1 + 2f)`
- incorrect: `log(1 - f)`
- analytical Kelly fraction: `clip((3p - 1) / 2, 0, 0.9999)`

`gamma=1` preserves the additive log-wealth interpretation. No condition uses a
predictive auxiliary loss or a pretrained checkpoint.

## Conditions

1. `fixed_full`: always wager `0.9999`.
2. `policy_implied_kelly`: derive `f` from the selected categorical policy
   probability.
3. `learned_kelly`: an independent sigmoid head emits `f`; realized Kelly
   growth directly trains this head while PPO trains the token action.
4. `bayes_oracle`: the environment's exact Bayesian filter sizes the wager for
   the policy's selected token, but the belief is never included in the model
   observation.

The three policy-sized conditions form log-growth rewards on the learner device
before GAE. This retains the standard `Discrete(3)` rollout interface and avoids
CPU transfers in the training hot path.

## Primary measurements

- Held-out rank-2 affine belief-probe R².
- Greedy token accuracy.
- Mean expected and realized log growth.
- Mean wager and fraction of wagers below `0.01`.
- Wager RMSE against the exact Bayesian Kelly fraction.

Operational wager collapse means final mean wager below `0.01` and more than
95% of evaluated wagers below `0.01`. Learning curves are retained so temporary
early abstention can be distinguished from terminal collapse.

## Seed-42 results

| condition | belief R² | token accuracy | mean wager | expected log growth |
|---|---:|---:|---:|---:|
| `fixed_full` | 0.9281 | 0.6754 | 0.9999 | -2.240269 |
| `policy_implied_kelly` | 0.8829 | 0.6489 | 0.6761 | 0.042845 |
| `learned_kelly` | 0.9490 | 0.5210 | 0.3628 | 0.184927 |
| `bayes_oracle` | 0.8770 | 0.6729 | 0.5108 | 0.287087 |

No learned condition ended in wager collapse, so no warm start, wager floor, or
other anti-collapse intervention was added. The learned head temporarily put
roughly 60–67% of behavior wagers below `0.01` around 0.13–0.20M steps, then
recovered; only 2.58% of held-out greedy wagers were below `0.01` at the end.

The learned wager head produced the strongest belief representation in this
single seed (`R²=0.9490`) and captured 63.0% of the Bayes-oracle policy ceiling
in expected log growth. Its lower unweighted token accuracy is not inconsistent
with positive growth: correct high-confidence wagers and abstention on weak
states matter more than accuracy under the Kelly objective.
