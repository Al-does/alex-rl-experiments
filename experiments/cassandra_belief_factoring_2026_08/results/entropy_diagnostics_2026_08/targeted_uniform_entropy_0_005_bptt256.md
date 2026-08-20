# Targeted PPO: uniformly randomized starts, entropy 0.005, BPTT-256

## Purpose

Seed-42 cardinality-controlled targeted run after switching to uniformly
randomized component states and BPTT-256.

## Recipe

| Setting | Value |
|---|---|
| Source run | `targeted_ppo/results/20260819T140744Z-5d2a0a96` |
| Seed | 42 |
| Environment steps | 5,013,504 |
| Episode length | 1,000 |
| Initial state | each component uniform over broken/bad/fair/good |
| Action scope | 10 component-targeted actions |
| Observation | 16-way symbol + previous 10-way action; no reward or belief |
| Algorithm | PPO |
| Learning rate | `3e-4` |
| Gamma / lambda | `0.999` / `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | `0.005` constant |
| KL loss | enabled; initial coefficient 0.2, target 0.01 |
| Train batch / minibatch | 32,768 / 8,192 |
| Epochs | 4 |
| Model | width 96, 3 layers, 4 heads |
| Per-layer context / BPTT | 256 / 256 |
| Effective raw lookback | 768 |
| EnvRunners × envs/runner | 16 × 4 |
| Hardware | RTX 4090 |

## Outcome

| Metric | Value |
|---|---:|
| Final sampled mean return | **3.04** |
| Final sampled range | -3.75 to 26.71 |
| Deterministic frozen-policy return | 4.00 |
| Deterministic actions | 100% operate |
| Final policy entropy | 0.000748 |
| Final value explained variance | -0.001 |
| Training wall time | 2,503 s |

The policy smoothly converged to operate-only behavior. Environment, belief,
targeted-action, and cached-versus-training-forward invariants passed; no
environment bug was found.
