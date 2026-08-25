# Cassandra PPG: matched 5M action-scope comparison

This campaign contains exactly two seed-42 training conditions:

| Leaf | Action scope |
|---|---|
| `global_alias_ppg` | Ten actions: indexed aliases of global repair/replacement |
| `targeted_ppg` | Ten actions: component-addressable repair/replacement |

Both recipes train for 5,000,000 environment steps from the all-good initial
state with the same width-64, four-layer, one-head transformer and **BPTT 64**
(`context_len=64`, `max_seq_len=64`). PPG-specific settings are held fixed:
`N_pi=32`, six auxiliary epochs, clone coefficient `1.0`, raw half-MSE auxiliary
value coefficients `0.003`, policy/auxiliary learning rate `3e-4`.

Critic and entropy settings follow the current Cassandra best-run baseline
(`best_run_parameters.json`):

- `vf_clip_param=100`, `vf_loss_coeff=0.01`
- constant entropy coefficient `0.03`

Run the conditions separately with:

```bash
uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.ppg_5m.global_alias_ppg.experiment \
  --seed 42 --hardware cuda4090

uv run rl-harness \
  experiments.cassandra_belief_factoring_2026_08.ppg_5m.targeted_ppg.experiment \
  --seed 42 --hardware cuda4090
```

On Vast, `--run` enables durable teardown by default (compact `results/` to
GitHub, `artifacts/` to B2, then destroy after verified transfer).
