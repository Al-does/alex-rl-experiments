# Overleaf draft

Upload `draft.tex` and the `figures/` directory to Overleaf. The document is
standalone and uses packages available in the standard TeX Live image.

The committed `data/` directory is the compact manuscript data snapshot:

- `token_guess_cycle2_mse_curves.json` copies the definitive 15-seed
  token-guess trajectories.
- `action_symmetry_cycle4_mse_curves.json` contains the seven retained
  checkpoints for variants 1–3 and seeds 42–46.
- `action_symmetry_cycle4_summary.json` preserves final MSE, action-mix, and
  occupancy aggregates.

Regenerate both figures from the repository root with:

```bash
uv run --frozen python \
  writeups/token_guess_and_action_symmetry/plot_mse.py
```

`figures/figure_data_summary.json` records the exact plotted means and
population standard deviations.

## Data provenance

Token-guess source:

```text
experiments/mess3_token_guess_cycle_2/results/
  20260729T165100Z-fifteen-seeds-0p66m/
  mse_over_training/mse_over_training_summary.json
```

Cycle-4 source:

```text
results commit: 3d0b709
experiment ref: 8cd58c4f313c812d318863144e26cdad93d80fe2
harness ref:    38f5136f3085743e5b3ba2e89d7e7fc247ba5b5e
```

`snapshot_cycle4_data.py` reduces the 15 source
`checkpoint_probe_curve.json` files to only the fields used by this write-up.
It validates that all three variants and all five seeds are present.

## Architecture note

The experiments do not use one numerically identical trunk:

- token guess: 64-wide, 4 layers, 1 attention head, 10-token context;
- action symmetry: 96-wide, 3 layers, 4 attention heads, 64-step attention
  band per layer.

The draft therefore describes each architecture within its own experiment
section rather than presenting a shared architecture.
