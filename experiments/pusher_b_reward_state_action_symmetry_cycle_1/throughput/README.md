# Pusher-B throughput study

This folder preserves the science of
`pusher_b_reward_state_action_symmetry_cycle_1` while isolating operational
sampling experiments from its historical code and results.

## Hypothesis

The RTX 4090 run is bottlenecked by CPU EnvRunner sampling rather than learner
updates. Increasing independent CPU EnvRunners while holding the complete
episode batch, model, PPO settings, seed, task, and reward fixed should reduce
sampling wall time without materially changing early learning.

## Matched benchmark

- Condition: `b10`, reward state `B`, variant `2`
- Seed: `42`
- Budget: 524,288 requested environment steps (two complete PPO iterations)
- Baseline: at most 16 CPU EnvRunners
- Candidates: at most 32 or 52 CPU EnvRunners
- Environments per runner: recomputed to preserve at least one 262,144-step
  complete-episode sampling batch. The 16, 32, and 52 runner layouts all
  collect exactly 2,080 complete episodes per sampling round on the target host.
- Learner: one CUDA GPU under the `cuda4090` hardware profile
- Comparison: sampling and learner timers, sampled steps per second, CPU/GPU
  utilization, wall time, and the two initial training-curve points

Each runnable leaf writes to its own result and artifact paths. Nothing in the
historical experiment or its result directories is modified.

## RTX 5090 benchmark

The three layouts were run sequentially at seed 42 on one Vast.ai RTX 5090
host with 272 visible CPUs, a 261.12-CPU cgroup quota, and 62 GB RAM. Absolute
times are host-specific; the matched comparison isolates the runner layout.

| Layout | Sampling time | Iteration time | Sample throughput | Shell wall time | Mean GPU util. | RAM util. |
|---|---:|---:|---:|---:|---:|---:|
| 16 × 130 | 569 s | 656 s | 387 steps/s | 1,478 s | 0.16% | 51.5% |
| 32 × 65 | 355 s | 448 s | 531 steps/s | 1,090 s | 0.20% | 62.2% |
| 52 × 40 | 223 s | 312 s | 858 steps/s | 775 s | 0.30% | 72.9% |

The 52-runner layout reduced sampling time by 61% and total iteration time by
52% relative to the 16-runner baseline. Learner updates remained approximately
flat at 87–93 seconds, confirming that CPU sampling dominated the baseline.
GPU utilization remained below 0.3% on average because the learner was active
only after each long sampling phase.

Mean returns across the first two iterations were 32.7→34.5, 33.1→35.2, and
33.7→35.5 for 16, 32, and 52 runners respectively. This single-seed,
two-iteration check found no material early-learning divergence, but it is not
a statistical equivalence test. The 52-runner layout is the fastest measured
option on a host with sufficient RAM; 32 runners is the lower-memory fallback.
