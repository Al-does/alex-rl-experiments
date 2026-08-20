# Targeted small model: continuation to 20M, entropy 0.03 → 0.008

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m`, 10,027,008 steps |
| Continuation run | `targeted_ppo_small_entropy_anneal_continue_10m/results/20260820T193259Z-b0a45f73` |
| Seed / final steps | 42 / 20,054,016 |
| Entropy schedule | 0.03 at restore; linear to 0.008 over 2.5M steps; then constant |
| Gamma / lambda | 0.990 / 0.95 |
| KL loss | disabled |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 64, 4 layers, 1 head |
| Per-layer context / BPTT | 256 / 256 |
| Initial state | all-good |
| Hardware | RTX 4090; 503GiB host RAM; 80GB disk |

## Outcome

| Metric | Value |
|---|---:|
| Peak continuation mean return | **413.78 at 19,890,176 steps** |
| Last reporting mean return | 371.43 at 20,021,248 steps |
| Final policy entropy | 0.568 |
| Final entropy coefficient | 0.008 |
| Final value explained variance | -0.0067 |
| Continuation wall time | 3,749 s |

The policy remained stable and improved beyond its 10M peak of 392.52. The
critic's final explained variance remained near zero despite strong policy
reward. An initial retry on a 64GB host failed during restore; the high-memory
replacement completed without worker restarts.
