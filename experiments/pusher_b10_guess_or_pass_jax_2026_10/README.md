# Pusher-B b10 guess-or-pass in JAX

Pure-JAX PPO transfer campaign port of
`experiments/pusher_b10_guess_or_pass_2026_10`; the RLlib study remains the
fallback. Actions 0/1 guess the pending token and action 2 passes. Rewards are
`+1`, `-cost`, and `0`.

The converter maps the RLlib encoder's token and BOS embeddings, RMSNorm
scales, QKV/output projections, and gated-GELU MLP weights into the matching
JAX trunk. Every encoder key and tensor shape is checked; non-encoder
parameters are listed as ignored. Each transfer also compares encoder
activations on 16 fixed-seed sequences and requires maximum absolute error
below `1e-4`.

List and launch the campaign:

```bash
uv run python -m experiments.pusher_b10_guess_or_pass_jax_2026_10.campaign --list
uv run python -m experiments.pusher_b10_guess_or_pass_jax_2026_10.campaign \
  --run-index 0 21 --frozen-cost 1.5 --smoke --hardware cpu --no-upload-artifacts
```

GPU setup and a scratch run:

```bash
uv sync --group jax-cuda
uv run rl-harness experiments.pusher_b10_guess_or_pass_jax_2026_10.ppo_c2.experiment
```

Four CPU smoke runs (2,032 steps each) measured 2,663–3,919 active env
steps/s. The full-trunk PPO transfer parity error was `3.10e-6`; the frozen
PPO+NTP transfer error was `6.68e-6`, and both frozen-trunk/head-update checks
passed. No GPU throughput was measured.
