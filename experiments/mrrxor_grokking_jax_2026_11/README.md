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

Defaults: four layers, width 128, MLP 512; AdamW learning rate 1e-4,
weight decay 0.1, four PPO epochs, clip 0.2, gamma=lambda=0,
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
