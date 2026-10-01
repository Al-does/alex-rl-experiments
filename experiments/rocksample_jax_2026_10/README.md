# RockSample in JAX (2026-10)

Pure-JAX port of RockSample[5,7] plus an end-to-end jitted PPO trainer that
does not use RLlib or Ray. The goal is throughput: many 3-5M-step PPO runs
(e.g. the inner loop of `rocksample_meta_rl_2026_09`) on one GPU.

- `env.py` — fixed-layout RockSample (`envs.rocksample` semantics: moves,
  east exit, sample, noisy checks with efficiency `2^{-d/h}`, 100-step
  truncation, same observation vector) as pure functions; `step_autoreset`
  returns `final_obs` for truncation bootstrapping. Randomised training
  layouts are not ported.
- `model.py` — the RL-Harness transformer (pre-LN, RoPE, hard causal band of
  `context_len=32`, 3 layers, 4 heads) with a windowed training path and a
  KV-cached rollout path; tests check the two agree.
- `ppo.py` — PPO matching `rocksample_ppo_2026_09` (clip 0.2, vf 0.5,
  entropy 0.05, adaptive KL, GAE 0.99/0.95, global-norm clip 0.5, Adam 3e-4,
  4 epochs). Training sequences carry `n_layers * context_len` steps of
  history so the training forward sees the same context as the rollout.
  `init`/`train_chunk` are pure, so a population of runs is a `jax.vmap`.
- `benchmark.py` — throughput benchmark, old vs new.

## Benchmark (Vast 53737726, 1x RTX 4090, 32 effective CPUs, 2026-10-01)

`results/benchmark_2026-10-01_rtx4090.jsonl` has the raw lines. JAX numbers
exclude compilation (`compile_s`); RLlib excludes its first two iterations.

| path | setup | env steps/s |
|---|---|---|
| old env (numpy) | 1 env, random actions | 102k |
| JAX env | 1,024 / 16,384 / 131,072 envs, random actions | 62M / 747M / 3.5B (env-only upper bound; XLA fuses the reduction) |
| old PPO (RLlib + Torch) | 16 runners x 24 envs, GPU learner, batch 8,192 | 4.8k |
| JAX PPO | 64 envs x 128 steps (batch 8,192, minibatch 1,024) | 240k |
| JAX PPO | same, 16 seeds vmapped | 490k total (31k per seed) |
| JAX PPO | 512 envs (batch 65,536), 1 seed | 581k |

Sanity check on learning: 16 vmapped seeds to 3.5M steps took 98 s (+51 s
compile); final batch return per seed ranged 10.0-24.5 (median ~19.5), the
same range as the RLlib PPO runs, which plateau near 25.

Run: `uv sync --group jax-cuda` on a GPU box, then
`python -m experiments.rocksample_jax_2026_10.benchmark <mode>` (use
`.venv/bin/python` for `train-rllib`; Ray workers break under `uv run`).
