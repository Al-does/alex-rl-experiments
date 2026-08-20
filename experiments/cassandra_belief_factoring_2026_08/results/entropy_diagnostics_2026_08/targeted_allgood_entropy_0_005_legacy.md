# Targeted PPO: all-good starts, entropy 0.005, legacy sequence length

## Purpose

Original seed-42 targeted-action PPO reference before randomized starts and
BPTT-256. This is the strongest pre-entropy-tuning targeted result.

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo/results/20260818T193443Z-e37dbc13` |
| Seed | 42 |
| Environment steps | 5,015,952 |
| Episode length | 1,000 |
| Initial state | all components good |
| Action scope | 10 component-targeted actions |
| Observation | 16-way symbol + previous 10-way action; no reward or belief |
| Algorithm | PPO |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | `0.005` constant |
| KL loss | enabled (RLlib adaptive default) |
| Train batch / minibatch | 32,768 / 2,048 |
| Epochs | 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 64 / 32 |
| Environments per runner | 24 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Final sampled mean return | **109.37** |
| Final sampled range | 0.48 to 212.28 |
| Final policy entropy | 0.653 |
| Final value explained variance | 0.620 |
| Training wall time | 823 s |

The policy learned beyond the always-operate reference (39.06) but remained
below the exact-belief QMDP heuristic (180.04). No independent frozen-policy
evaluation was retained for this run.
