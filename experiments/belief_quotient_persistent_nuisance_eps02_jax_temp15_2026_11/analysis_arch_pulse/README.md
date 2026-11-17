# Seed-42 recurrent single-write pulse comparison

Frozen-policy reviewer follow-up to `analysis_arch_coarsening/`. No training,
paid compute, uploads, manuscript edits, commits, or pushes. Only Reward-State
Quotient (`variant_2`) and Full-State Quotient (`variant_3`), seed 42, exact
final checkpoint `checkpoints/step_2012217344.npz`.

## Sources and reproducibility

- Analysis branch starts at fetched `origin/main`
  `4b575146ec27c69d12b09c6fcea9eb00ede5c116`. That tree has **no** eps02
  steering adapter or v2 `steering_core.py`/`steering_pulse.py`. Consequently
  the reference remains PR #239 revision
  `45517b13a67d746048a002f9db52ea55a654b219`, not a newer main driver.
- Recipes and completed run manifests are read from frozen result revision
  `d20d13dc172b4959477c4ea12f919520209144f2`; no result branch merge is needed.
  The six policies are the same seed-42 checkpoints as the earlier coarsening
  analysis. Each is reverified against the canonical B2 manifest's byte count
  and SHA-256 **before** loading. Every report records exact run ID, checkpoint
  URI/hash/size, manifest key/hash, and training revisions.
- `reference.py` imports the pinned PR #239 core/pulse modules from an ignored,
  detached worktree. They are unmodified. Their SHA-256s also match the source
  hashes in the original Transformer stable reports. Trainer `models.py` and
  `env.py` were compared between the pinned source and the analysis base and
  have no differences.
- Runtime: Python 3.13.18, JAX 0.11.1, NumPy 2.5.1, `cpu:0`;
  shared harness revision `a267508fdeb523e95b3ef3fc42119102540bbe38`.
- `recurrent.py` replaces the selected layer output after its GRU cell update.
  That replacement feeds the next layer immediately **and** becomes that
  layer's next-step recurrent carry. The top layer is included. Readout is
  the top hidden state **before** final LayerNorm, not the post-norm embedding.
  `analyze.py` uses the original Transformer path for the two reference runs.

From the repository root, with the result and PR #239 revisions available:

```bash
git fetch origin
git worktree add --detach \
  experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/artifacts/arch_pulse_reference \
  45517b13a67d746048a002f9db52ea55a654b219
```

Set `PULSE_REFERENCE` to an existing detached checkout instead if desired.
The loader requires its HEAD to equal the pinned revision. B2 credentials
are needed to read manifests/checkpoints and original full reports, not for
the focused tests. With the repository environment already installed:

```bash
for architecture in transformer gru128 gru92; do
  for variant in 2 3; do
    uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.analyze \
      --architecture "$architecture" --variant "$variant"
  done
done
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.summarize
uv run pytest -q \
  experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/analysis_arch_pulse/tests/test_pulse.py \
  tests/test_history_jax_persistent_nuisance_eps02.py
```

The analyzer refuses to overwrite reports. Preserve historical outputs before
re-running into these paths. The summary generator intentionally rewrites only
its derived `summary.json`/`summary.md`. Tests explicitly skip if the required
reference checkout has not been prepared; the recorded run below did not skip.
Checkpoints and downloaded full reference reports stay under ignored
`artifacts/`, while all six compact reports and summaries are under `results/`.

## Protocol

- Greedy probe histories: fit seed 42, test seed 43; 14 independent 1,024-step
  episodes each, drop eight steps per episode, flatten episode-major, retain
  first 12,000 rows independently per split. These are the pulse driver's
  settings, **not** the earlier coarsening probe's 60k/80k rows or ridge fit.
- At every site fit affine hidden→`b[:2]` and affine `b[:2]`→hidden maps with
  NumPy least squares, `rcond=1e-6`. Store float32 maps; highest-precision
  matmul for decoding/write displacement. Report held-out coordinate R² for
  ρ=`b3` and δ=`b1-b2`. No clipping/projection of decoded beliefs.
- Pulse seed 3042, 32 episodes, `t0={256,512,768}`, lags 0–128 inclusive.
  Full-strength single write:
  `h + (target(decoded_b)[:2] - decoded_b[:2]) @ encoder`.
  The encoder is a reverse affine regression, **not** the decoder's inverse;
  thus full strength need not realize the target exactly at readout.
- δ mirror swaps latent states 1/2. ρ flip maps ρ to `1-ρ`, preserving the
  conditional ratio between states 1/2 (equal split fallback if their decoded
  mass is nonpositive), exactly as the source driver does.
- Both copies replay the **same saved unsteered observations**, including
  previous executed actions. Branch outputs never evolve the environment.
  Both start from the intact pre-decision carry/cache at `t0`; only lag zero
  writes. Measure mean absolute coordinate difference across 32 episodes ×
  three pulse times at the final readout. Retention is the ratio of these two
  aggregate means, `mean|change(+1)| / mean|change(0)|`, not the mean of ratios.
  Ratios above one are allowed: downstream/recurrent propagation can amplify
  the final-site difference. This is not a retained probability mass.
- Per-agent exact full-filter traces use that agent's own saved histories and
  the eps02 transition/emission model. The four **paper** exact-filter rows
  are separately labeled references, not identical-history GRU estimates.

## Results and limits

`results/summary.md` has the paper-format all-site table, retained fractions,
lags 2/8/32/128, per-site R², CPU-versus-archived Transformer comparisons,
and the exact-filter/paper 15-seed reference values. `summary.json` also
records all six checkpoint provenances and report hashes. Each individual
report retains full signed/absolute 129-lag curves in coordinate order
`[rho, delta]`, including exact-filter curves.

GRU top-layer writes persist to +1 much more than Transformer block-3 writes.
Full-State GRU128 retains 98.4% of the δ change and 81.7% of the ρ change;
GRU92 retains 24.8% and 88.2%. Reward-State GRU128 retains 58.7%/27.7%, and
GRU92 34.1%/90.6%. This differs from the Transformer's rapid next-step
readout suppression, but does **not** establish exact Bayesian recursion,
task use of a coordinate, or persistence indefinitely. All requested GRU top-site
changes are below 0.0035 by lag 128. Reward-State GRU δ R² is 0.600–0.862,
versus Full-State 0.941–0.971; interpret δ pulses in light of that difference.
These remain descriptive, one-training-seed comparisons, with different
on-policy histories/probe sites. Reward-State GRU92's earlier held-out reward
was 8.9% below the Transformer's; this task does not reevaluate success.

### Transformer sanity limitation — not an exact numerical pass

Both seed-42 Transformer variants ran through the original pinned pulse
driver. Archived GPU stable-report payloads were downloaded and verified by
their compact reports' SHA-256 and byte count. CPU re-fits give close, but
**not identical**, pulse values. The largest difference in a requested
coordinate is 0.00791520 (Reward-State block-1 ρ at pulse:
0.335897535 CPU versus 0.327982336 archived). Across both coordinates the
maximum is 0.02694732; the full comparison is retained, not filtered away.
At block 3 the requested-coordinate pulse/+1 values are:

| Condition | Pulse | Original GPU at/+1 | CPU refit at/+1 |
|---|---|---:|---:|
| Full-State Quotient | δ mirror | 0.476026982/0.007073729 | 0.476078987/0.007036861 |
| Full-State Quotient | ρ flip | 0.546114385/0.001839551 | 0.546339691/0.001837723 |
| Reward-State Quotient | δ mirror | 0.300086766/0.004069077 | 0.299877018/0.004019434 |
| Reward-State Quotient | ρ flip | 0.636387527/0.001617417 | 0.636461735/0.001607740 |

The core/pulse source hashes match the archived run, and full-filter curves
agree within 1.8e-6 across all lags, but the fitted probes differ slightly.
Archived reports identify `cuda:0`; these runs identify `cpu:0`. This is
consistent with numerical sensitivity, but we did **not** isolate activation
roundoff versus probe fitting; its exact cause remains unverified. No
archived probe maps were available in the reports to hold that link fixed.
Do not claim exact reproduction or silently substitute archived numbers for
the fresh Transformer runs. A strict exact-number sanity requirement remains
unmet; the discrepancy is small compared with the observed GRU retention
contrast and does not change its direction.

## Checks

```text
35 passed in 15.24s
Validated 6 reports, 36 pulse/site curves, 129 lags each; checkpoint and analysis-source hashes match.
Transformer requested-coordinate maximum absolute curve difference: 0.007915198802947998
```

Tests cover actual GRU128/GRU92 site shapes and parity with the trainer,
same-step downstream and next-step carry effects at every test layer,
single-write fixed-input replay, pre-decision snapshot timing, rollout parity
with PR #239's recurrent instrumentation, affine fit/R², constant targets,
empty/invalid retained samples and pulse windows, all-site short pulse output,
and exact-filter token/preceding-action timing. No visual deliverable was
requested or generated for this pulse analysis.
