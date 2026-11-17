# Seed-42 architecture coarsening comparison

Offline reviewer-response analysis only. No training or paid compute. Source
recipes and completed transformer results are from `b3db18a1` (PR #234);
non-transformer results are imported by an **uncommitted** merge of PR #238's
`d20d13dc172b4959477c4ea12f919520209144f2` into a new local branch off main.
No commit/push was requested or performed.

```sh
base=experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_coarsening
uv run python -m "$base.analyze" --architecture transformer --curves
uv run python -m "$base.analyze" --architecture gru128 --curves
uv run python -m "$base.analyze" --architecture gru92 --curves
uv run python -m "$base.analyze" --architecture memoryless --curves
uv run python -m "$base.summarize" --paper ../deep-beliefs
uv run pytest -q tests/test_arch_coarsening_eps02.py tests/test_history_jax_persistent_nuisance_eps02.py
```

Omit `--curves` for finals only. Runs resume existing per-checkpoint reports;
use a fresh output directory for any scientific protocol change. Checkpoints
are cached under ignored study `artifacts/arch_coarsening/`; each checkpoint
is byte-count and SHA-256 checked against its live canonical B2 durability
manifest before restoring. `results/` contains only compact metrics and
provenance, never activations or checkpoints. All 26 indices (0–4, every
fifth, and final 101) are evaluated for every arm.

`summarize.py` validates the grid, reproduces transformer b and available c
scores against PR #23's `ladder_reprobes.json`, and checks final Reward-State
and Full-State b/c against its `coarse_comparison.json`. It records the exact
paper branch revision and reference-file hashes. `results/summary.json`
includes finals, curves, probe settings, and the reward gate. Per-checkpoint
files include every B2 checkpoint URI, SHA-256, byte count, durability-manifest
key/hash, run ID and training revisions. All final checkpoints are at
2,012,217,344 environment steps, not a separately selected best checkpoint.

## Targets and representation sites

- b is the exact full three-state belief from all A/B/C observations and
  previous executed actions.
- c is the state-3 posterior from complete observation histories in which
  A/B are merged. It is an exact two-state quotient filter for Trivial and
  Reward-State Quotient; Full-State uses PR #23's three-state merged-token
  filter control (not an exact quotient). Using its state-3 coordinate rather
  than the complementary two-vector leaves normalized error unchanged.
- Transformer: post-final-LayerNorm embedding, exactly as PR #23.
- GRUs: last layer recurrent hidden state (the model's `residual`, before
  final LayerNorm), not the normalized head input or concatenated layers.
- Memoryless: post-final-LayerNorm 64-wide residual MLP embedding, which
  depends on the current token only.

The probe streams are greedy 60k/80k independent fit/test rows from 1,024-step
episodes; exclude 64 warmup decisions per episode. Key:
`split(fold_in(PRNGKey(42), 2))`. Affine ridge 1e-6, highest matmul precision,
same harness fit and R² routines as PR #23. c is filtered in NumPy float64
using the saved recipe's process arrays. b comes from the original JAX
rollout filter. No collapsed full-belief comparator or difference target is
computed. Rewards are pre-transition state-3 occupancy, measured here on the
held-out post-warmup rows, not on full episodes. The descriptive low-reward
flag is reward below 90% of the same-variant transformer; it is not a test
of statistical significance. Always inspect the continuous reward gap too.

The saved training recipes use critic-free Monte Carlo policy gradients with
clipped PPO surrogate over 64 whole-episode minibatches; calling them pure
single-batch REINFORCE would be inaccurate. They share Adam lr 4.2e-4,
gamma .99, one epoch, temperature 1.5, zero entropy bonus and exploration,
uniform prior and the same epsilon-0.02 process. GRUs carry hidden state
through full episodes at evaluation, despite 32-step truncated training BPTT.

## Evidence boundary

Single training seed, no cross-seed uncertainty. Both GRUs show much lower c
than b error in Reward-State Quotient, but neither reproduces the transformer's
Full-State reversal (c remains slightly better). The memoryless nontrivial
conditions have much lower reward and near-no-information probe errors.
Architecture-specific greedy histories and target variances differ; these
are not probes on matched histories. Errors are target-normalized, so b/c
comparisons also change the target variance and dimension. Affine accessibility
is not information absence or causal use. A paper-side draft figure and
method record live under `deep-beliefs/figures/` without manuscript edits.

All final CPU rewards equal the archived GPU probe rewards. At intermediate
checkpoints the maximum absolute reward difference is 0.00287500024; both
measurements are retained. Its cause was not investigated. The validation
initially required exact archived reward agreement at every checkpoint;
that inappropriate CPU/GPU equivalence requirement was replaced with a
reported discrepancy, not a change to measurements or probe settings.
