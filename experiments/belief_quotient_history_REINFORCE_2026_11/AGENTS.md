# History-dependent Belief-Quotient Ladder

This is a new study, not a continuation or replacement of historical results.
The baseline is `0.75 * I + 0.25 * cycle_5_baseline`, applied before the
unchanged reward-state odds tilts of strength 1.5. Emission accuracy is 0.75.
The resulting baseline rows are `(0.9375, 0.0375, 0.025)`,
`(0.0375, 0.9375, 0.025)`, and `(0.075, 0.075, 0.85)`.

The 12 leaves are `variant_{1,2,3}_{transformer,memoryless,gru128,gru92}`.
Use Trivial Quotient, Reward-State Quotient, and Full-State Quotient in prose.
The memoryless MLP reads the current token only. Other architectures read the
current token and previous action, with recurrent/context state carried as in
the source recipes. The transformer uses four width-64 layers, one head, and
per-layer context 32; the MLP is width 64 with six residual blocks; GRUs are
128×2 and 92×4 with 32-step truncated BPTT.

Full runs target 10M environment steps. Baseline-free REINFORCE retains learning
rate 4.2e-4, gamma 0.99, and entropy 0.03 annealed to zero at 9M steps.
Seeds are runtime `--seed` values; no seed count is hardcoded.
All recipes start fresh and reject `--resume-from` rather than accidentally
restoring a historical environment.

Requires the shared harness's GRU/ResidualMLP support, merged in RL-Harness
PR 87 (`9e4fc16f3015013f1b94a7919e9da4d725922ff5`) or a later descendant.
An older local editable harness may not contain these models.

From the experiment repository:

```bash
uv run rl-harness experiments.belief_quotient_history_REINFORCE_2026_11.variant_2_transformer.experiment --smoke --hardware cpu --no-upload-artifacts
uv run rl-harness experiments.belief_quotient_history_REINFORCE_2026_11.variant_2_transformer.experiment --seed 42 --hardware cuda4090
```

Replace the leaf for other conditions/architectures. Smoke targets 4,096 steps,
with real initialization and trained checkpoint probes. Smoke output is ignored
under the leaf's `.smoke/`; it is only a wiring check, not scientific evidence.

The shared runner saves the actual training initialization, powers-of-two
iterations, 2M-step checkpoints, and the final checkpoint. Checkpoints/raw logs
stay under ignored `artifacts/`; compact manifests, resolved recipes, learning
curves, and initialization-to-final probe reports go under `results/`.
The training-time checkpoint storage is log-spaced rather than the historical
ladder's every-iteration retention. Initialization, powers-of-two iterations,
and final probing are preserved, with additional probes at the 2M-step
thresholds. Reports label the actual sampled step, not the nominal threshold.

Primary probes retain the historical post-final-LayerNorm site, 60K independent
fit rows, 80K held-out test rows, and warmup 64. Inherited smoke probes use
4,096 rows per split and warmup 4. Targets use the restored checkpoint's actual
environment and token/action history, never rewards or hidden truth as actor
inputs. Compare initialization-to-final changes within architecture; own-policy
visitation also changes and is not a fixed-history encoder control.

`design.analytic_design_summary()` recomputes the 27 deterministic current-token
policies, privileged-state references, and frozen feasible three-token lookups.
At these parameters the latter attain 0.3202297379 and 0.3023403483 in the two
nontrivial conditions, versus 0.2950888739 and 0.2668669030 for the best
deterministic current-token policies. These are stationary expected rewards,
not Bayes optima or empirical caps.

`models.py` and `training.py` are also used by the companion
`reward_factors_history_REINFORCE_2026_11` study. Do not change historical source
recipes or shared harness code to specialize either new study.

Local CPU verification passed all 12 leaves at seed 42 using the merged harness
revision above. Each smoke produced reports at actual steps 0, 2,970, and 5,018;
complete-episode sampling overshoots the 4,096-step target. The ignored
`.smoke/verification.json` indexes the individual smoke summaries. No full
training runs or broader test suite were executed.
