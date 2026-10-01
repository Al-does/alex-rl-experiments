# RockSample in JAX (2026-10)

Pure-JAX port of RockSample[5,7] plus an end-to-end jitted PPO trainer that
does not use RLlib or Ray. The goal is throughput: many 3-5M-step PPO runs
on one GPU, and the meta-RL study built on them.

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
  The policy objective is pluggable (`Objective`), with its parameters in
  `RunnerState.meta` so runs with different objectives vmap together.
- `lpo.py` — LPO learned drift `F(r, A)` (port of
  `rocksample_meta_rl_2026_09/learned_objective.py`); initial parameters give
  exactly PPO's clipped surrogate and entropy 0.05.
- `es.py` — antithetic OpenAI-ES with centred ranks and Adam (copied from
  `rocksample_meta_rl_2026_09`).
- `meta.py` + `lpo_es_drift/experiment.py` — the meta-RL study with no
  RLlib: each generation trains the whole ES population (8 antithetic pairs
  + the centre, 3M steps each, from scratch) as one jitted `vmap`; pair
  members share inner seeds. Fitness is mean return over the final 20% of
  updates. Writes `meta_progress.jsonl` and `meta_state.json` each
  generation; `--resume-from meta_state.json` continues.
- `ppo_n5_k7_d64/experiment.py` — plain PPO baseline with the RLlib recipe's
  hyperparameters: 16 seeds vmapped, 612 updates x 8,192 = 5.0M steps each.
  Writes `training_curves.jsonl` (per-seed lists per update), `summary.json`
  (final-20% return per seed) and `artifacts/final_params.npz`.
- `benchmark.py` — throughput benchmark, old vs new.

## PPO baseline run

    uv sync --group jax-cuda && python -m harness.cli \
        experiments.rocksample_jax_2026_10.ppo_n5_k7_d64.experiment --seed 42

## Meta-RL run

    uv sync --group jax-cuda && python -m harness.cli \
        experiments.rocksample_jax_2026_10.lpo_es_drift.experiment --seed 42

Vast's bootstrap only runs plain `uv sync`, so the launch command must add
the `jax-cuda` group itself. Smoke (`--smoke`): 2 pairs + centre, 2,048
steps per run, 2 generations; about 15 s on CPU with `--group jax`.

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

Only the 4.8k vs 240k rows are like-for-like (same batch 8,192). The 581k
row is a single run with an 8x larger batch, not comparable to RLlib. The
490k row is aggregate over 16 independent runs, the relevant regime for ES.

Sanity check on learning: 16 vmapped seeds to 3.5M steps took 98 s (+51 s
compile); final batch return per seed ranged 10.0-24.5 (median ~19.5), the
same range as the RLlib PPO runs, which plateau near 25.

Run: `uv sync --group jax-cuda` on a GPU box, then
`python -m experiments.rocksample_jax_2026_10.benchmark <mode>` (use
`.venv/bin/python` for `train-rllib`; Ray workers break under `uv run`).
