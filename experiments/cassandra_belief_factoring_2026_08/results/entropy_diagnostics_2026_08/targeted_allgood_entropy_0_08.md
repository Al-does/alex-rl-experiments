# Targeted PPO: all-good starts, entropy 0.08

## Purpose

Test a moderate constant entropy coefficient after coefficient 0.8 proved too
large and coefficient 0.005 collapsed.

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_0_08_all_good/results/20260819T235236Z-b2d130a3` |
| Seed | 42 |
| Environment steps | 5,013,504 |
| Episode length | 1,000 |
| Initial state | all components good |
| Action scope | targeted |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | **`0.08` constant** |
| KL loss | enabled; target 0.01 |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| Effective raw lookback | 768 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Final mean return | **160.52** |
| Final sampled range | -38.48 to 294.19 |
| Final policy entropy | 1.351 |
| Final value explained variance | -1.0 |
| Mean KL | 0.00194 |
| Training wall time | 2,477 s |

Reward was strongest around 2.3–2.7M steps (roughly 185–241) and generally
declined into the 135–205 range by 5M. Constant entropy 0.08 prevented collapse
but continued to force exploration after the best-performing region.
