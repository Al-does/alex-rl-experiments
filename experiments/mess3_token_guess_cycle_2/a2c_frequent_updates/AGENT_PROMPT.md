# Agent prompt: run corrected A2C for three seeds

Copy everything below into a new agent after this PR is merged.

---

Run the corrected frequent-update A2C pilot in
`experiments/mess3_token_guess_cycle_2/a2c_frequent_updates` for seeds 42, 43,
and 44.

Scientific recipe:

- exactly 1,000,000 sampled environment steps per seed;
- `train_batch_size_per_learner=128`;
- one epoch over each fresh batch, no minibatching or replay;
- LR `1e-4`, gamma/lambda `0`, and the existing A2C objective/model;
- do not run the legacy `a2c` leaf—it is update-starved and retained only for
  reproduction.

Procedure:

1. Pull the latest `main` in `/workspace` and `/rl-harness`.
2. Read the repository `AGENTS.md` files plus the `runpod-flash` and
   `vast-provisioning` skills.
3. Run:

   ```bash
   cd /workspace
   uv run pytest -q tests/test_mess3_token_guess_cycle_2.py
   uv run rl-harness \
     experiments.mess3_token_guess_cycle_2.a2c_frequent_updates.experiment \
     --seed 42 --smoke
   ```

4. Use the existing launcher, dry-run first:

   ```bash
   cd /workspace
   uv run python experiments/mess3_token_guess_cycle_2/server_jobs.py \
     --conditions a2c_frequent_updates \
     --seeds 42 43 44 \
     --mode parallel-seeds \
     --backend flash-then-vast \
     --dry-run
   ```

5. If the dry run is correct, repeat it with `--yes`. Use the current pushed
   experiment and harness SHAs. Ensure B2/Git result durability is enabled by
   the launcher. Monitor all three jobs to terminal success; destroy any
   infrastructure rented directly through Vast when finished.
6. Verify every `resolved_recipe.json` records:
   `total_env_steps=1000000`, `train_batch_size_per_learner=128`,
   `num_epochs=1`, and `minibatch_size=null`.
7. Verify all three runs have successful Tune trials and final checkpoints.
   Collect compact `condition_summary.json` and `checkpoint_probe_curve.json`
   outputs. Report per-seed optimizer-update count, final belief MSE/R², token
   accuracy, and the mean ± SD across seeds. Compare the ~1M checkpoint against
   the legacy A2C and PPO runs at the same approximate environment-step budget.

Do not alter hyperparameters during this run. If execution fails, diagnose and
retry infrastructure failures without silently changing the scientific recipe.

---
