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
