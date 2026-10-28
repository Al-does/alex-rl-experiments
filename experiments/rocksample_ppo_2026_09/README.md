# RockSample PPO size comparison

Six independent experiments cross the fixed RockSample[5,7] default with
RockSample[4,4] (`n=4, k=4`) and transformer widths 64, 128, and 192. Both
instances use the environment's default fixed layouts, 100-step episode cap,
and randomized rock qualities and sensor readings on each episode. The policy
observes normalized positions, the latest Good/Bad/None symbol, and the
previous action; hidden rock qualities are not policy inputs.

Each arm uses PPO, a three-layer, four-head transformer with `context_len=32`,
seed 42 by default, `gamma=0.99`, constant entropy coefficient 0.05, and a
1,000,000-environment-step budget. PPO's dimensionless policy ratio clipping
is 0.2. RLlib's `vf_clip_param` caps the **squared value error**, so its
default of 10 would suppress learning from a single ±10 reward. The chosen
value of 1,000,000 is above the squared error between the worst discounted
100-step trajectory of repeated -10 samples (approximately -634 at 0.99) and
the optimistic +80 upper bound from seven good rocks and exit. This keeps
value clipping out of the expected return range. [4,4] uses the same setting
for a matched comparison. Truncated episodes retain value bootstrapping.

Run one arm with `uv run rl-harness
experiments.rocksample_ppo_2026_09.n5_k7_d64.experiment --smoke --hardware
cpu --no-upload-artifacts` (substitute any leaf below). A smoke run trains for
2,048 environment steps in two 1,024-step batches and writes disposable
`.smoke/` results; omit `--smoke` for its full budget. Smoke return is only a
pipeline check. Published reference returns use `gamma=0.95`, so they cannot
be compared directly with these `gamma=0.99` training results.

| Instance | Width 64 | Width 128 | Width 192 |
|---|---|---|---|
| Default [5,7] | `n5_k7_d64` | `n5_k7_d128` | `n5_k7_d192` |
| [4,4] | `n4_k4_d64` | `n4_k4_d128` | `n4_k4_d192` |
