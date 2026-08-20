# Targeted PPO: standard transformer, entropy 0.03, gamma 0.990, 10M

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_0_03_gamma_0_990_all_good_10m/results/20260820T060058Z-f3c36627` |
| Seed / steps | 42 / 10,027,008 |
| Initial state / episode length | all-good / 1,000 |
| Entropy coefficient | `0.03` constant |
| Gamma / lambda | `0.990` / `0.95` |
| KL loss | disabled; coefficient 0 |
| Learning rate / PPO clip | `3e-4` / `0.2` |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Peak mean return | **48.26 at 2,457,600 steps** |
| Last reporting mean return | 44.70 at 9,994,240 steps |
| Final policy entropy | 0.000020 |
| Final value explained variance | -1.0 |
| Training wall time | 4,974 s |

The standard transformer collapsed to a nearly deterministic low-return policy
despite constant entropy coefficient 0.03.
