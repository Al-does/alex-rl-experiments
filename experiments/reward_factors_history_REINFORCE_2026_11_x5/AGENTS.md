# 5x continuation of the history-dependent Reward Relevance study

Eight leaves mirror `reward_factors_history_REINFORCE_2026_11` (`{reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}`).
Each restores the parent leaf's completed seed-matched final checkpoint from B2
(located via the parent's checked-in `run_manifest.json`) and trains to 150M
environment steps (5 x 30M) with the parent's configuration (no entropy
bonus). Mechanics are shared with `belief_quotient_history_REINFORCE_2026_11_x5`; see that study's `AGENTS.md`
and `belief_quotient_history_REINFORCE_2026_11/continuation.py`.

```bash
uv run rl-harness experiments.reward_factors_history_REINFORCE_2026_11_x5.reward_both_gru92.experiment --smoke --hardware cpu --no-upload-artifacts
uv run rl-harness experiments.reward_factors_history_REINFORCE_2026_11_x5.reward_both_gru92.experiment --seed 42 --hardware cuda4090
```
