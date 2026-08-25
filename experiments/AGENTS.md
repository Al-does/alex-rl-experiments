# `experiments/` — complete scientific recipes

This repository is the composition root for Alex's studies. It depends on the
shared `rl-harness` library (editable sibling checkout). Generic packages must
never import experiments.

## Layout

```text
experiments/study_name_2026_07/
  shared.py
  condition_name/
    experiment.py
    analyze.py
    findings.md
    results/
    artifacts/
```

Each leaf has exactly one `experiment.py` exporting `run(context)`.

## Custom code and promotion

First configure existing library components. If a missing abstraction is a
reusable RL concept, implement it in the `rl-harness` checkout and open a PR
there; keep only a small adapter here. Idiosyncratic code stays beside the
experiment until reuse proves an abstraction.

## Results and artifacts

Track compact findings under `results/`. Ignore large/raw data under
`artifacts/` locally; optional Backblaze B2 upload records durable URIs in
`results/` when configured (see the library's `docs/artifact_storage.md`).

### What belongs in git

| Path | Commit from remote boxes? | Notes |
|------|---------------------------|-------|
| `experiments/**/results/**` | Yes | JSON summaries, manifests, small plots |
| `experiments/**/artifacts/**` | **Never** | Checkpoints (`.pt`, `.pkl`), tune trees, raw logs |
| `*.pth`, `*.pt`, `checkpoint_*` | **Never** | Always under `artifacts/` or B2 |
| Source code (`analysis.py`, etc.) | No | Land on your feature branch via normal PR, not from Vast |

Remote GPU boxes publish **only** new files under `experiments/**/results/**`.
They push to the **launch branch** (the `--branch` you passed to `provision
up`), not to `main`. Merge to `main` manually in a PR when the campaign is
ready. If two agents push to the same branch concurrently, git merge (not
rebase) is used; resolve any content conflicts on a workstation — do not rely
on silent rebases on the box.

Do not use a shared orphan `results` branch unless you have a deliberate reason.
Prefer one feature branch per campaign attempt so parallel agents stay isolated.

## Storage conventions (this repo owns naming)

The harness writes verbose per-iteration metrics to ignored
`artifacts/<run-id>/metrics.jsonl`. **This repo** decides what compact projections
land in Git-tracked `results/`. See `experiments/storage/training_curves.py`.

| File | Location | Git? | Description |
|------|----------|------|-------------|
| `metrics.jsonl` | `artifacts/` | no | harness verbose flattened Ray metrics (B2) |
| `training_curves.jsonl` | `results/` | yes | compact curves (`iteration`, `return_mean`, …) |
| `progress.jsonl` | `results/` | **legacy — never commit** | pre-cleanup verbose dump; read from B2 instead |
| `run_manifest.json`, `tune_summary.json` | `results/` | yes | provenance and final trial metrics |

After `run_tune()`, call `write_training_curves(context)` from
`experiments.storage.training_curves` to materialize compact curves.

### Agent guide: what to read

| Goal | Read | Avoid |
|------|------|-------|
| Final return | `tune_summary.json`, `*_summary.json` | bulk-reading all of `results/` |
| Learning curve | `training_curves.jsonl` | `progress.jsonl`, `remote_artifacts.json` |
| Provenance | `run_manifest.json` | — |
| Perf debugging | B2 `compact-results/progress.jsonl` or `artifacts/metrics.jsonl` | legacy Git `progress.jsonl` |
