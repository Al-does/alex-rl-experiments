# Targeted Cassandra PPG 5M campaign

This isolated campaign compares a 2×2 Phasic Policy Gradient grid at seed 42:

| Leaf | `N_pi` | Auxiliary value coefficients |
|---|---:|---:|
| `npi16_value_0p003` | 16 | 0.003 |
| `npi16_value_0p01` | 16 | 0.01 |
| `npi32_value_0p003` | 32 | 0.003 |
| `npi32_value_0p01` | 32 | 0.01 |

Every full recipe trains for 5,000,000 targeted-environment steps from the
all-good initial state with the stable width-64, four-layer, one-head
transformer. Entropy anneals from `0.03` to `0.008` over 2.5M steps. The PPG
smoke configuration uses two policy batches and one auxiliary epoch so a smoke
run exercises both phases rather than only PPO's policy phase.

Targeted immediate rewards lie in `[-3.75, 0.9985**4]`. With `gamma=0.99`,
discounted values are bounded by roughly `[-375, 99]`; because the reusable
PPG implementation uses raw half-MSE, `0.003` and `0.01` are deliberately
smaller than the paper's reward-normalized coefficient.

Run any leaf with:

```bash
uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.targeted_ppg_5m.<leaf>.experiment \
  --seed 42
```
