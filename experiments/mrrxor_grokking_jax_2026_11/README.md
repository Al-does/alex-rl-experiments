# Finite-data MRRXOR PPO grokking

Pure JAX/Optax PPO, reusing the Pusher-B transformer, actor/value heads,
PPO loss, affine ridge probe and asynchronous NPZ/B2 checkpoint writer.
MRRXOR follows [the reference implementation](https://github.com/artemyk/belief-state/blob/main/mrrxor.py)
and [the belief-state post](https://www.lesswrong.com/posts/4f9p7brLdnTMFuwfB).

The fixed seed draws 1,024 training and 4,096 held-out length-32 sequences;
held-out rows are rejected if they match any training row. Each episode samples
a training row uniformly with replacement. Input at decision n is BOS followed
by tokens strictly before n; actions never enter the input or change the process.
Exact beliefs start at stationarity. No reward or latent state reaches the model.

Defaults: four layers, width 128, MLP 512; AdamW constant learning rate 3e-4,
weight decay 1.0, four PPO epochs, clip 0.2, gamma=lambda=0,
standardized advantages, value coefficient .25 and entropy coefficient .01.
Budget is 100M steps rounded up to a complete update. Configure scientific
hyperparameters through `PPOConfig` in the leaf, not runtime CLI overrides.
Smoke uses 2,048 steps and smaller pools, but the same default transformer.
Checkpoints include actual initialization, updates 1/2/5 per decade, every
1,000 updates, and final. Snapshots contain parameters, not optimizer state;
resume is explicitly unsupported. Pools and recipe are saved beside the index.
Harness manifests record repository provenance; per-checkpoint SHA256 is in
the offline analysis. B2 follows the template's optional upload policy.

```bash
uv run pytest -q tests/test_mrrxor_grokking_jax_2026_11.py
uv run rl-harness experiments.mrrxor_grokking_jax_2026_11.ppo_grokking.experiment --smoke --hardware cpu --no-upload-artifacts
uv run python -m experiments.mrrxor_grokking_jax_2026_11.analysis --run-dir ARTIFACT_RUN_DIRECTORY --output NEW_REPORT.json
```

Use `artifact_run_dir` printed in the training summary (not the results directory).
Omit `--smoke` only for an intentional full training run. No full run has been
established by this recipe's smoke validation.

## Reading diagnostics

Sampled-policy reward is its exact conditional expectation on the fixed binary
labels (no extra action-sampling noise). Greedy accuracy on positions with
Bayes max probability below .6 is the primary memorization signal: a large
train−held-out gap suggests learning fixed labels on intrinsically uncertain
prefixes. The held-out reference is approximately, not exactly, .5.

Excess CE is realized-label cross entropy minus realized-label Bayes NLL;
it may be negative on memorized training data. The separate Bayes-to-policy KL
is nonnegative and compares distributions. An RL policy is not generally a
calibrated predictor: maximizing expected binary reward favors deterministic
Bayes guesses, not Bayes probabilities, so high CE need not mean poor RL reward.

Offline analysis fits process-weighted PCA once on all 65,536 stationary
length-16 histories with their exact probabilities. It selects the PC most
affinely predictive of next-token probability, rather than assuming PC3.
Final-normalized residual activations are probed to the full belief and each
PC separately, fitting only training-pool rows and scoring held-out sequences.
Fixed ridge 1e-6 is not tuned on held-out data. Every prefix including BOS is
included; these short episodes have complete causal context, no warmup removal.
Policy log-odds regressions fit on training histories and report train and
held-out R² (and their gap) using NTP PC alone versus full belief.

The PNG plots held-out belief/NTP-PC probe error and train/held-out excess CE,
plus random-bit accuracy. Initialization appears at x=1 on the log axis.
These are descriptive accessibility diagnostics, not a causal or exclusive
identification of the model's representation, and a smoke run cannot establish
grokking. Prefixes within a sequence are correlated; no confidence intervals
or checkpoint-selection claims are made. Per-layer probing and specificity
controls beyond the requested NTP/full-belief comparison are not included.

## Single-GPU sequential sweep

```bash
uv run python -m experiments.mrrxor_grokking_jax_2026_11.sweep --smoke --hardware cpu --no-upload-artifacts
# On an already-provisioned GPU, omit --smoke and choose its hardware profile:
uv run python -m experiments.mrrxor_grokking_jax_2026_11.sweep --hardware cuda4090
```

The `ppo_sweep` leaf follows the existing RockSample `Arm` / `SweepRecipe`
convention; unlike the Pusher heldout ladder's separate leaves, it runs all arms
sequentially inside one process, with one common seed from `context.seed`.
The four arms vary only train sequences {256, 1024, 4096, 16384}; all use
weight decay 1.0 and constant learning rate 3e-4, also the `PPOConfig` defaults.
Edit `RECIPE` or construct a
`SweepRecipe` with replaced `PPOConfig`s to narrow the grid or reduce budgets.
The single-run `ppo_grokking` leaf remains unchanged.

Each arm costs **100,003,840 environment steps** (24,415 updates × 4,096
steps, rounded up from 100M), plus checkpoint evaluation and offline analysis.
All four arms cost about **400 million agent steps**; this is not a wall-time guarantee
and has not been GPU-benchmarked. Reduce per-arm `total_env_steps` for a quick
pilot. Smoke runs two width-16, one-layer arms (`n16`, `n32`) with the same
weight decay and learning rate, at 2,048 steps each, and analyzes
all three checkpoints of each arm on CPU.

Arms use labels `n256`, `n1024`, `n4096`, `n16384`. Each gets checkpoints,
recipe, pools, curve, analysis JSON and PNG. Root `summary.json` and `summary.md`
rank arms by descending final held-out
random-bit accuracy, breaking ties by held-out expected reward, and include
train/held-out random-bit accuracy, excess CE and held-out NTP-PC probe error.
The summary is refreshed after each completed arm; failures stop the sweep.
Do not interpret this ranking on one seed as a confirmed hyperparameter result.

Training pools are runtime arguments rather than compiled constants; learning
rates and weight decay are dynamic scalars/arrays. Jitted PPO updates are reused
across arms sharing pool and model shapes, and evaluation/PCA are cached too.
Arms are not vmapped, keeping peak training memory at one arm's state. AdamW
is composed from Optax Adam moments plus decoupled weight decay; its learning
rate is constant within each rollout/update, scheduled in environment steps.

### One folder per sweep run

The study launcher resolves paths **before** calling the shared harness CLI,
so even its initial/final provenance manifests live inside the run folder.
Non-smoke runs use `experiments/mrrxor_grokking_jax_2026_11/runs/<run-id>/`;
unpublished smoke runs use `experiments/mrrxor_grokking_jax_2026_11/.smoke/<run-id>/`.
Use `--run-id NAME` to choose the final folder name; directory overrides are
rejected by this launcher. The ordinary `rl-harness ...ppo_sweep.experiment`
command without explicit matching paths is not supported: its default context
would split metadata into leaf-local `results/<id>` and `artifacts/<id>` trees.
The single-run `ppo_grokking` command still uses the usual harness layout.

```text
runs/<run-id>/
  run_manifest.json       # harness provenance, completion/upload status
  resolved_recipe.json
  summary.json
  summary.md
  artifacts/              # harness-level heavy-output destination
  n256/
    resolved_recipe.json
    curve.json
    summary.json
    analysis.json
    analysis.png
    remote_artifacts.json # optional full B2 index, ignored
    artifacts/
      metrics.jsonl
      pools.npz
      recipe.json
      checkpoint_index.json
      param_checkpoints/update_*/params.npz
  n1024/ ...
  n4096/ ...
  n16384/ ...
```

Each arm's complete heavy subtree is uploaded using the existing harness
upload helper when enabled, in addition to asynchronous checkpoint uploads.
The harness retains its own top-level metadata/upload lifecycle. No shared
harness or storage code is modified. Repository `.gitignore` already excludes
`experiments/**/artifacts/` at any depth and all `.smoke/` trees; the study's
`.gitignore` also excludes full B2 indexes in `runs/`. Thus checkpoints, pools,
and verbose logs stay ignored, while compact curves, analysis, summary tables,
and provenance under this single run folder remain committable.
