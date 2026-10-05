# Pusher-B in JAX (2026-10)

Pure-JAX port of the Pusher-B next-token CE ("supervised") and token-guess
PPO experiments in `experiments/pusher_b`. RLlib, Ray and PyTorch are not
used. The legacy path is left unchanged as the fallback. Its short
counterparts for comparisons are in `legacy_short/`.

- `process.py`: the b10/b90 HMMs, same edge matrices as
  `pusher_b/process.py`. The env is RL-Harness `envs.hmm.jax_env.JaxHMMEnv`
  (delay 1, 127 steps, truncation), with reward
  `action == raw_token_before`.
- `model.py`: the legacy transformer (4 layers, d128, 4 heads, d_mlp 512,
  pre-RMSNorm, RoPE, gated GELU, bias-free heads). It has a full causal
  forward and a KV-cached rollout path. The tests check the two agree.
- `supervised.py`: the recipe of `pusher_b/supervised.py`. BOS + 127
  tokens, AdamW 1e-3, batch 512, 10k updates, validation against the exact
  Bayesian predictive. Sequences are sampled on device inside the jitted
  update.
- `ppo.py`: the recipe of `pusher_b/rl.py`. gamma = lambda = 0,
  standardised advantages, clip 0.2, value loss 0.25·min(mse, 10), 6
  epochs, 64-episode minibatches, and lr/entropy schedules in lifetime env
  steps. A batch is 2,048 synchronised complete episodes (260,096 steps,
  vs RLlib's ~262k). `next_token_aux=True` adds the auxiliary CE head from
  `pusher_b/learning.py`.
- Leaves: `supervised_b10`, `supervised_b90`, `ppo_b10`, `ppo_b90`,
  `ppo_b10_aux_ce`.
- `benchmark.py`: short JAX runs that write JSON lines.
- `results/batch_sweep/`: JAX PPO batch/minibatch/lr sweep (b10, b90, 2
  seeds, 10M steps). The default (2,048 episodes, minibatch 64, base lr)
  was best; larger batches were no faster and learned worse. Smaller
  batches were not tested.
- `ModelSpec(compute_dtype="bfloat16")` exists but was ~16x slower for CE
  on an RTX 4090; keep float32.

Run (GPU):

    uv sync --group jax-cuda && python -m harness.cli \
        experiments.pusher_b_jax_2026_10.ppo_b10.experiment --seed 42

`--smoke` runs on CPU with `uv sync --group jax`.
