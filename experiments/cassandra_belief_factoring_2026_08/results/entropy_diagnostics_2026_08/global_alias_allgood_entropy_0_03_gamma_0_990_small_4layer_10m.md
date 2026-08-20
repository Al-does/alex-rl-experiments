# Global-alias PPO: width-64 four-layer transformer, entropy 0.03, 10M

## Recipe

| Setting | Value |
|---|---|
| Source run | `global_alias_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m/results/20260820T081146Z-1d660cb4` |
| Seed / steps | 42 / 10,027,008 |
| Initial state / episode length | all-good / 1,000 |
| Action scope | 10 cardinality-matched global aliases |
| Entropy coefficient | `0.03` constant |
| Gamma / lambda | `0.990` / `0.95` |
| KL loss | disabled; coefficient 0 |
| Learning rate / PPO clip | `3e-4` / `0.2` |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 64, 4 layers, 1 head |
| Per-layer context / BPTT | 256 / 256 |
| Effective raw lookback | 1,024 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Peak mean return | **465.47 at 9,863,168 steps** |
| Last reporting mean return | 452.65 at 9,994,240 steps |
| Final policy entropy | 0.0689 |
| Final value explained variance | 0.124 |
| Training wall time | 4,421 s |

The small global-alias model remained stable and continued improving through
the end of training. Its lower final entropy than the small targeted model is
consistent with the easier identity-invariant control problem.
