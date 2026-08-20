# Targeted PPO entropy diagnostics

Dedicated run records:

1. [`targeted_allgood_entropy_0_005_legacy.md`](targeted_allgood_entropy_0_005_legacy.md)
2. [`targeted_uniform_entropy_0_005_bptt256.md`](targeted_uniform_entropy_0_005_bptt256.md)
3. [`targeted_allgood_entropy_0_8_partial.md`](targeted_allgood_entropy_0_8_partial.md)
4. [`targeted_allgood_entropy_0_08.md`](targeted_allgood_entropy_0_08.md)
5. [`targeted_allgood_entropy_0_08_to_0_01.md`](targeted_allgood_entropy_0_08_to_0_01.md)
6. [`targeted_allgood_entropy_0_01_continuation.md`](targeted_allgood_entropy_0_01_continuation.md)
7. [`targeted_allgood_entropy_0_03_gamma_0_990_standard_10m.md`](targeted_allgood_entropy_0_03_gamma_0_990_standard_10m.md)
8. [`targeted_allgood_entropy_0_03_gamma_0_990_small_4layer_10m.md`](targeted_allgood_entropy_0_03_gamma_0_990_small_4layer_10m.md)
9. [`targeted_allgood_entropy_0_05_gamma_0_990_standard_10m.md`](targeted_allgood_entropy_0_05_gamma_0_990_standard_10m.md)
10. [`global_alias_allgood_entropy_0_03_gamma_0_990_small_4layer_10m.md`](global_alias_allgood_entropy_0_03_gamma_0_990_small_4layer_10m.md)
11. [`small_final_checkpoint_probe_comparison.md`](small_final_checkpoint_probe_comparison.md)
12. [`targeted_small_entropy_anneal_0_03_to_0_008_continuation_20m.md`](targeted_small_entropy_anneal_0_03_to_0_008_continuation_20m.md)
13. [`global_alias_small_entropy_anneal_0_03_to_0_008_continuation_20m.md`](global_alias_small_entropy_anneal_0_03_to_0_008_continuation_20m.md)

These are single-seed diagnostics. Raw returns are not directly comparable
across different initial-state distributions.

The records are intentionally self-contained: each lists its scientific and
optimization settings, source run ID, and outcome. One-off campaign recipes,
raw `progress.jsonl`, checkpoints, duplicate durability manifests, and
per-checkpoint plots remain on the research branch/B2 rather than this PR.

[`compact_training_curves.json`](compact_training_curves.json) retains the
episode-reporting rows needed to reproduce campaign reward and optimization
plots without committing the raw RLlib histories. Each row contains sampled
steps, training iteration, return mean/min/max, entropy coefficient, policy
entropy, value explained variance, mean KL, and policy loss. Non-reporting
windows and repeated flattened configuration/connector payloads are omitted.
