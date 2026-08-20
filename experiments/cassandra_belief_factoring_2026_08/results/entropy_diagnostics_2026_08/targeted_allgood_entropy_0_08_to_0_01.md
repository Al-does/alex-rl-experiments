# Targeted PPO: all-good starts, entropy 0.08 annealed to 0.01

## Purpose

Preserve exploration through 2.5M steps, then reduce exploration pressure so
the policy can exploit what it learned.

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_anneal_all_good/results/20260820T012606Z-a674abc3` |
| Seed | 42 |
| Environment steps | 5,013,504 |
| Episode length | 1,000 |
| Initial state | all components good |
| Action scope | targeted |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy schedule | 0.08 through 2.5M; linear to 0.01 at 5M |
| KL loss | enabled; initial coefficient 0.2, target 0.01 |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| Effective raw lookback | 768 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Final mean return | **234.65** |
| Final sampled range | 108.29 to 346.58 |
| Final entropy coefficient | 0.01 |
| Final policy entropy | 1.218 |
| Final value explained variance | -1.0 |
| Mean KL | 0.00326 |
| Training wall time | 3,052 s |

This was the strongest targeted result. Reward climbed sharply after roughly
4M steps and exceeded the 180.04 QMDP reference in the final reporting window,
although the critic remained poorly fit.
