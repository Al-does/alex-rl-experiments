# Targeted 50M policy action-selection diagnostic

## Question

Does the targeted policy select components using its history, or does it
shotgun repair/replacement across all four components?

## Protocol

- Final targeted checkpoint at 50,069,504 training steps.
- 64 seeded stochastic-policy evaluation episodes (64,000 environment steps).
- All-good starts, 1,000-step episodes.
- Diagnostics compare chosen targets with the exact hidden component state and
  the policy-history Bayesian belief. Hidden state and belief are not policy
  inputs.

## Policy behavior

| Action group | Fraction |
|---|---:|
| Operate | 0.855 |
| Inspect | 0.000 |
| Repairs | 0.117 |
| Replacements | 0.029 |

Mean stochastic-policy return was **507.36**.

## Targeting quality

| Diagnostic | Chosen target | Random component baseline |
|---|---:|---:|
| Repair target truly bad/fair | **0.577** | 0.520 |
| Belief probability target is bad/fair | **0.585** | 0.523 |
| Replacement target truly broken | **0.274** | 0.146 |
| Replacement target truly non-good | **0.738** | 0.456 |
| Belief probability target is broken | **0.279** | 0.145 |
| Belief expected condition gain | **1.519** | 0.870 |

The chosen replacement component maximized posterior broken probability in
67.9% of replacement actions and maximized expected condition gain in 74.9%.
Repair targeting was weaker: the chosen component maximized posterior
repairability in 34.3% of repair actions.

## Shotgun test

Across 9,247 contiguous maintenance streaks:

- mean streak length: 1.006;
- maximum streak length: 2;
- fraction of streaks length at least four: **0**;
- fraction touching all four components: **0**;
- fraction containing a four-action component sweep: **0**.

The policy is not cycling through all four actions to guarantee a hit. It
usually performs one maintenance action and returns to production. Replacement
actions are strongly targeted; repair actions are only modestly better than a
random target.

The policy never uses `inspect`. Its identity information therefore comes from
its own component-specific intervention history and elapsed/degradation
history, not explicit noisy component inspection.

Full metrics:
`targeted_50m_policy_action_diagnostic/results/20260823T230352Z-f260906b/`.
