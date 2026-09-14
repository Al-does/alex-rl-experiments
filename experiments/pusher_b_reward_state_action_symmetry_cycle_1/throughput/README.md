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
