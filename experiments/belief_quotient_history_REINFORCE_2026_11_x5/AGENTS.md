# 5x continuation of the history-dependent Belief-Quotient Ladder

Twelve leaves mirror `belief_quotient_history_REINFORCE_2026_11` (`variant_{1,2,3}_{transformer,memoryless,gru128,gru92}`).
Each leaf locates the parent leaf's completed run for the runtime `--seed` in
the parent's checked-in `results/`, downloads that run's final Tune checkpoint
from the B2 prefix in its `run_manifest.json`, restores it with
`Algorithm.restore_from_path` inside `on_algorithm_init`, and trains to
50M environment steps (5 x 10M). Weights, optimizer state, lifetime step
counters, and the training iteration continue from the parent, so Tune's
stopping metric and the checkpoint cadence (powers-of-two iterations, 2M-step
thresholds beyond the parent's last one, final) pick up where the parent left
off. The parent's entropy schedule reached zero at 9M steps; the continuation
therefore trains with entropy 0. Environment RNG streams restart from the seed.

The first probe report is the restored checkpoint (`restored_checkpoint`,
labelled with the parent's actual final step), which doubles as a restore
fidelity check against the parent's final report. `condition_summary.json` and
`resolved_recipe.json` record the parent under `continued_from`; plot a full
0-50M trajectory by concatenating parent and continuation reports.

Smoke (needs B2 credentials to fetch the parent checkpoint):

```bash
uv run rl-harness experiments.belief_quotient_history_REINFORCE_2026_11_x5.variant_2_gru128.experiment --smoke --hardware cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_history_REINFORCE_2026_11_x5.variant_2_gru128.experiment --seed 42 --hardware cuda4090
```

`--resume-from` is still rejected; continuation is defined by the recipe.
Context-32 transformer arms need a box with >=100 GB disk (Ray object-store spill).
