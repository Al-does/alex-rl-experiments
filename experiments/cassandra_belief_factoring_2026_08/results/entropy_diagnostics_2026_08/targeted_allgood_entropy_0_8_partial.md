# Targeted PPO: all-good starts, entropy 0.8 (terminated early)

## Purpose

Test whether very strong entropy regularization prevents the targeted policy
from collapsing to operate-only behavior.

## Recipe

| Setting | Value |
|---|---|
| Experiment leaf | `targeted_ppo_entropy_0_8_all_good` |
| Run ID | `20260819T220826Z-eef12233` |
| Vast instance | 48153876 |
| Seed | 42 |
| Requested steps | 5,000,000 |
| Episode length | 1,000 |
| Initial state | all components good |
| Action scope | targeted |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | **`0.8` constant** |
| KL loss | enabled (RLlib adaptive default) |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

The run was manually terminated because it was far worse than the preceding
targeted run at the same point.

| Metric | Value |
|---|---:|
| Last sampled steps | 1,212,416 |
| Last reporting window | 1,179,648 steps |
| Mean return | **-1,264.92** |
| Policy entropy | 2.253 |
| Value explained variance | approximately 0 |

The maximum possible entropy for ten actions is `ln(10) = 2.303`; coefficient
0.8 kept the policy nearly uniform and prevented useful specialization.
Partial compact results were not retained when the box was terminated.
