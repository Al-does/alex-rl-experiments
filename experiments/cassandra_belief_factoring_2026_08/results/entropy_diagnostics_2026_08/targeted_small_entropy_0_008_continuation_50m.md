# Targeted small model: continuation to 50M, entropy 0.008

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo_small_entropy_anneal_continue_10m`, 20,054,016 steps |
| Continuation run | `targeted_ppo_small_continue_30m/results/20260820T205945Z-e2f72153` |
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
| Peak continuation mean return | **518.63** |
| Final mean return | **501.29** |
| Final policy entropy | 0.156 |
| Final value explained variance | 0.276 |
| Continuation wall time | 2:56:29 |

The targeted policy remained stable and improved substantially from its 20M
peak of 413.78. Its final return exceeded the corresponding global-alias policy
for the first time in this campaign.
