# Global-alias small model: continuation to 50M, entropy 0.008

## Recipe

| Setting | Value |
|---|---|
| Source run | `global_alias_ppo_small_entropy_anneal_continue_10m`, 20,054,016 steps |
| Continuation run | `global_alias_ppo_small_continue_30m/results/20260820T204934Z-79b17e50` |
| Seed / final steps | 42 / 50,069,504 |
| Entropy coefficient | `0.008` constant |
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
| Peak continuation mean return | **478.41** |
| Final mean return | **446.23** |
| Final policy entropy | 0.0657 |
| Final value explained variance | 0.517 |
| Continuation wall time | 2:48:56 |

The global-alias policy remained stable but did not improve beyond its earlier
peak near 479. Its critic remained substantially better fit than the targeted
critic.
