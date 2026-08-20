# Global-alias small model: continuation to 20M, entropy 0.03 → 0.008

## Recipe

| Setting | Value |
|---|---|
| Source run | `global_alias_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m`, 10,027,008 steps |
| Continuation run | `global_alias_ppo_small_entropy_anneal_continue_10m/results/20260820T182145Z-b4be6471` |
| Seed / final steps | 42 / 20,054,016 |
| Entropy schedule | 0.03 at restore; linear to 0.008 over 2.5M steps; then constant |
| Gamma / lambda | 0.990 / 0.95 |
| KL loss | disabled |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 64, 4 layers, 1 head |
| Per-layer context / BPTT | 256 / 256 |
| Initial state | all-good |
| Hardware | RTX 4090; 80GB disk |

## Outcome

| Metric | Value |
|---|---:|
| Peak continuation mean return | **478.97 at 10,813,440 steps** |
| Last reporting mean return | 449.25 at 20,021,248 steps |
| Final policy entropy | 0.0623 |
| Final entropy coefficient | 0.008 |
| Final value explained variance | 0.365 |
| Continuation wall time | 3,537 s |

The global-alias policy remained stable throughout the continuation. Its peak
occurred early, and final performance remained close to the original 10M
policy's 452.65 last reporting return.
