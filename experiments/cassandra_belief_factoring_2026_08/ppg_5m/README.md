# Cassandra PPG: matched 5M action-scope comparison

This campaign contains exactly two seed-42 training conditions:

| Leaf | Action scope |
|---|---|
| `global_alias` | Ten actions: indexed aliases of global repair/replacement |
| `targeted` | Ten actions: component-addressable repair/replacement |

Both recipes train for 5,000,000 environment steps from the all-good initial
state with the same width-64, four-layer, one-head transformer. Every PPG and
PPO setting is matched: `N_pi=32`, six auxiliary epochs, clone coefficient
`1.0`, raw half-MSE value coefficients `0.003`, policy/auxiliary learning rate
`3e-4`, and entropy annealed from `0.03` to `0.008` over 2.5M steps.

Run the conditions separately with:

```bash
uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.ppg_5m.global_alias.experiment \
  --seed 42

uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.ppg_5m.targeted.experiment \
  --seed 42
```
