# First-, top-, and all-layer GRU pulse writes

New driver: `write_modes.py`; new output directory:
`results/write_site_comparison_seed42/`. No existing results, summary or figure
are overwritten. The driver refuses a nonempty output directory.

```bash
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.write_modes \
  --output-dir experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/analysis_arch_pulse/results/NEW_WRITE_MODE_DIRECTORY
```

## Held fixed

Seed-42 GRU128 and GRU92, Full-State Quotient (`variant_3`) and Reward-State
Quotient (`variant_2`), exact same final `step_2012217344.npz` checkpoints.
Each checkpoint is rechecked against canonical B2 manifest bytes/SHA-256
before loading. Recipes/manifests are still read from frozen revision
`d20d13dc172b4959477c4ea12f919520209144f2`; pinned paper core/pulse driver
revision is `45517b13a67d746048a002f9db52ea55a654b219`.

Probes: refit per-layer affine hidden→belief and reverse belief→hidden maps,
least-squares `rcond=1e-6`, seeds 42/43, 12,000 retained greedy rows per split,
eight-step exclusion, episode-major selection from 14 × 1024-step episodes.
Pulse: seed 3042, 32 episodes, times 256/512/768, lag zero plus 128 future
steps, full strength once. Both branches replay intact-run tokens and previous
executed actions. Readout: top recurrent hidden state before final LayerNorm.
Saved curve coordinate order is `[rho, delta]`.

## Write conventions

**First layer only:** exactly the existing `recurrent.step(..., site=0)`
protocol. Only the first cell output is shifted; upper cells immediately
process its modified output, and all resulting hidden states become the next
carry. Only layer 1 is explicitly written. Existing `*_layer1` pulse reports
are therefore reusable. The driver freshly reruns the entire original pulse
path and requires zero difference from all saved signed/absolute measured and
filter curves before reuse. The same check covers top-only reuse.

**All layers:** run an intact shadow branch on the same inputs. At pulse time,
for each layer in bottom-to-top order:

1. Compute its cell from the already steered lower output and its own carry.
2. Decode its intact shadow output with that layer's forward affine map;
   construct the mirror or ρ-flip target from that belief.
3. Shift the current steered cell output by
   `(target[:2] - decode(current_output)[:2]) @ layer_encoder`.
4. Pass the modified output upward and store it as that layer's next carry.

The target's source is the intact shadow, **not** an upper layer's already
steered belief estimate. This is the coordinated-writes convention tested in
the rebound audit, generalized from two to four layers. At later lags the
write strength is zero, while every altered recurrent state is carried.
No clipping or inverse-decoder substitution is introduced.

Each report includes first/top/all signed and absolute full 129-lag curves,
the policy's own full-filter reference, exact checkpoint provenance, source
hashes, per-layer probe scores, and hashes of the refitted map arrays. The
new `write_site_comparison.{json,md}` tabulates lags 0/1/2/4/8/16/32/64 and
records the hashes of protected existing outputs. Filter rows are not shared
between architectures: intact histories differ.

## Tests and figure

```bash
uv run pytest -q \
  experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/analysis_arch_pulse/tests/test_write_modes.py \
  experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/analysis_arch_pulse/tests/test_pulse.py \
  tests/test_history_jax_persistent_nuisance_eps02.py
```

Recorded output: `54 passed in 35.73s`.

Focused tests cover two- and four-layer mirror/ρ writes, each layer's modified
carry, steered-input ordering rather than independent parallel writes,
intact-shadow versus already-steered targets, zero-strength parity with the
trainer, single-write timing/fixed inputs, and invalid/empty pulse windows.

New paper figure/script:
`deep-beliefs/figures/fig_gru_pulse_write_site.pdf` and
`plot_gru_pulse_write_site.py`. The existing architecture-decay figure is
preserved. No manuscript edits, training, paid compute, uploads, commits or
pushes are part of this analysis. All comparisons remain descriptive and
single-seed; an affine target write need not be on-manifold or implement a
Bayesian update.
