# Targeted PPO: width-64 four-layer transformer, entropy 0.03, 10M

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m/results/20260820T060017Z-3d46dd5a` |
| Seed / steps | 42 / 10,027,008 |
| Initial state / episode length | all-good / 1,000 |
| Entropy coefficient | `0.03` constant |
| Gamma / lambda | `0.990` / `0.95` |
| KL loss | disabled; coefficient 0 |
| Learning rate / PPO clip | `3e-4` / `0.2` |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | **width 64, 4 layers, 1 head** |
| Per-layer context / BPTT | 256 / 256 |
| Effective raw lookback | 1,024 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Peak mean return | **392.52 at 9,863,168 steps** |
| Last reporting mean return | 368.08 at 9,994,240 steps |
| Final policy entropy | 0.916 |
| Final value explained variance | 0.067 |
| Training wall time | 10,928 s |

The narrower, deeper transformer avoided policy collapse and continued
improving through the end of the 10M budget. Host connector performance made
this run unusually slow.
