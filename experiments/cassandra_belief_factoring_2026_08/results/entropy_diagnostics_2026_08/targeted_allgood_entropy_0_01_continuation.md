# Targeted PPO: continuation after annealing, entropy fixed at 0.01

## Purpose

Restore the 5M-step annealed-policy checkpoint and test whether reward would
continue climbing with another five million requested steps.

## Recipe

| Setting | Value |
|---|---|
| Source checkpoint run | `20260820T012606Z-a674abc3` |
| Continuation result | `targeted_ppo_entropy_anneal_continue_5m/results/20260820T045117Z-26211dc7` |
| Seed | 42 |
| Source steps | 5,013,504 |
| Requested total target | 10,013,504 |
| Manual stop | 9,830,400 |
| Episode length | 1,000 |
| Initial state | all components good |
| Action scope | targeted |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | **`0.01` constant after restore** |
| KL loss | enabled; target 0.01 |
| Train batch / minibatch / epochs | 32,768 / 8,192 / 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Window | Mean return | Policy entropy | Value explained variance | Mean KL |
|---|---:|---:|---:|---:|
| 4.5–5.0M | 196.22 | 1.270 | -0.912 | 0.00553 |
| 5.0–6.0M | 78.34 | 0.636 | -0.234 | 0.01534 |
| 6.0–7.0M | 36.41 | 0.0077 | 0.054 | ~0.000002 |
| 7.0–9.0M | 37–39 | 0.0026–0.0053 | 0.07–0.17 | ~0.000001 |
| 9.0–9.83M | 35.80 | 0.0018 | 0.020 | approximately 0 |

The policy rapidly collapsed toward operate-only behavior after restoration.
KL briefly spiked while entropy and reward fell, then approached zero once the
policy became nearly deterministic. The apparent critic improvement after
collapse reflects a simpler, low-return policy rather than recovery. The run
was manually interrupted and its metrics were durably published.
