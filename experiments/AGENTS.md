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
    results/                 # study-level digests (optional)
    artifacts/               # ignored locally
  condition_name/
    results/<run-id>/        # compact per-run outputs (tracked)
    artifacts/<run-id>/      # heavy per-run outputs (ignored → B2)
```

Each leaf has exactly one `experiment.py` exporting `run(context)`.

## Custom code and promotion

First configure existing library components. If a missing abstraction is a
reusable RL concept, implement it in the `rl-harness` checkout and open a PR
there; keep only a small adapter here. Idiosyncratic code stays beside the
experiment until reuse proves an abstraction.

## Results and artifacts

Every run writes two sibling trees under the experiment leaf:

| Tree | Git | Durability | Purpose |
|------|-----|------------|---------|
| `results/<run-id>/` | **Yes** | pushed from remote boxes | compact science + provenance |
| `artifacts/<run-id>/` | **No** | Backblaze B2 when configured | checkpoints, Tune trees, verbose logs |

Remote GPU boxes publish **only** new files under `experiments/**/results/**`.
They push to the **launch branch** (the `--branch` you passed to `provision
up`), not to `main`. Merge to `main` manually in a PR when the campaign is
ready.

### Standard per-run files (`results/<run-id>/`)

These names are owned by the harness unless noted. Prefer them over ad-hoc
filenames so agents and analysis tools can find outcomes without scanning the
tree.

| File | Required | Description |
|------|----------|-------------|
| `run_manifest.json` | yes | provenance: commits, command, seed, hardware, status, B2 summary |
| `training_curves.jsonl` | yes (Tune/direct RLlib) | **compact** per-iteration learning curves (~1 KB/iter). Short keys: `iteration`, `steps`, `return_mean`, `entropy`, losses, timing |
| `tune_summary.json` | Tune runs | final trial metrics + checkpoint path |
| `remote_artifacts.json` | when B2 configured | index of uploaded `artifacts/` files with URIs and hashes |
| `durability_manifest.json` | when B2 configured | canonical copy of the B2 index (same content as `remote_artifacts.json`) |
| `<condition>_summary.json` | optional | experiment-specific final snapshot (e.g. `grid_summary.json`, `intervention_summary.json`) |
| `resolved_recipe.json` | optional | resolved hyperparameters / model config for the run |
| `progress.jsonl` | **legacy** | pre-split verbose dump; **do not read** — use `training_curves.jsonl` instead |

**Do not** write checkpoints, Tune trial directories, or multi-megabyte JSON
into `results/`. Those belong under `artifacts/` (B2).

### Standard per-run files (`artifacts/<run-id>/`)

| Path | Description |
|------|-------------|
| `progress.jsonl` | **verbose** per-iteration Ray metrics (full flattened dump). On B2 only — never in Git |
| `tune/` | Tune trial tree (`progress.csv`, checkpoints, events) |
| `checkpoints/` | harness-managed checkpoint saves |

### Study- or campaign-level files (`results/` above the run id)

Use these for human-facing synthesis across many runs. They are intentionally
small and agent-friendly.

| File | Description |
|------|-------------|
| `<topic>_findings.md` | narrative write-up (tables, recommendations) |
| `<topic>_summary.json` | machine-readable final numbers for a campaign |
| `<topic>_training_curves.json` | optional multi-run curve bundle (see `entropy_diagnostics_2026_08/compact_training_curves.json`) |
| plots (`.png`) | only when deliberately part of the published finding |

Keep study digests **useful but not cramped**: include final returns, deltas
vs control, key hyperparameters, and run-id pointers — not every intermediate
metric.

### What belongs in git

| Path | Commit from remote boxes? |
|------|---------------------------|
| `experiments/**/results/**` | Yes — compact JSON/JSONL, manifests, small plots |
| `experiments/**/artifacts/**` | **Never** |
| Source (`experiment.py`, `shared.py`, …) | via normal PR, not from Vast |

Do not use a shared orphan `results` branch unless you have a deliberate reason.
Prefer one feature branch per campaign.

---

## Agent guide: what to read (avoid context blow-ups)

Agents should **never** bulk-read `progress.jsonl` or `remote_artifacts.json`.
Those files are large (long JSON lines or hundreds of B2 entries) and will
dominate context.

Use this lookup table instead:

| Goal | Read this | Avoid |
|------|-----------|-------|
| Final return / did the run finish? | `tune_summary.json`, `<condition>_summary.json`, or study `*_summary.json` | scanning all of `results/` |
| Learning curve / return vs steps | `training_curves.jsonl` | `progress.jsonl`, Tune `progress.csv` |
| Compare many runs in one study | study `*_summary.json`, `compact_training_curves.json`, findings `.md` | every per-run `training_curves.jsonl` unless plotting |
| Provenance / seed / hardware | `run_manifest.json` | — |
| Checkpoint location | `run_manifest.json` → `remote_artifacts.json` → filter `files` for `checkpoint` | reading entire manifest into context; use `jq`/grep for one path |
| Live run health on Vast | Vast logs / SSH tail | downloading artifacts mid-run |
| Perf debugging (connector timers) | `artifacts/<run-id>/progress.jsonl` on B2 or the box | `results/` tree in Git |

When plotting, read **`training_curves.jsonl` only** (typically ~150 short lines,
~150 KB). If a run predates the split and only has legacy `results/.../progress.jsonl`,
use `tune_summary.json` for the final point or fetch verbose progress from B2.

When searching, prefer **globbing filenames** (`training_curves.jsonl`,
`tune_summary.json`, `*_summary.json`) over `grep` across all of `results/`.
