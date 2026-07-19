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
