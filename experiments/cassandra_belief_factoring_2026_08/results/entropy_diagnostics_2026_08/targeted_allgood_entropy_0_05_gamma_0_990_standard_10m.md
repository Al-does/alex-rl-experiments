# Targeted PPO: standard transformer, entropy 0.05, gamma 0.990, 10M

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_0_05_gamma_0_990_all_good_10m/results/20260820T073122Z-e586a193` |
| Seed / steps | 42 / 10,027,008 |
| Initial state / episode length | all-good / 1,000 |
| Entropy coefficient | `0.05` constant |
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
| Peak mean return | **167.09 at 7,274,496 steps** |
| Last/final mean return | 30.25 at 10,027,008 steps |
| Entropy at peak | 0.495 |
| Final policy entropy | 0.000571 |
| Final value explained variance | approximately 0 |
| Training wall time | 4,382 s |

Entropy coefficient 0.05 delayed collapse and enabled a much higher peak than
0.03, but it did not prevent late operate-only convergence in the standard
transformer.
