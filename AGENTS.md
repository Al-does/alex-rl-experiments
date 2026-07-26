# Cloud agents

This repo is the personal experiment composition root. Cloud agents need the
shared [`rl-harness`](https://github.com/Al-does/RL-Harness) library checked out
as an editable sibling.

## Cursor Cloud specific instructions

Environment config lives in `.cursor/environment.json`. On startup, the `install`
script runs `./scripts/bootstrap_local.sh`, which:

1. Clones or links `rl-harness` at `/rl-harness` (sibling of `/workspace`)
2. Runs `uv sync --group dev` in this repo

Expected layout after install:

```text
/
  workspace/     # this repo (alex-rl-experiments)
  rl-harness/    # shared library (editable dependency)
```

### Where to make changes

- **Experiment recipes** (`experiments/…`): commit and open PRs in this repo.
- **Reusable library code**: edit `/rl-harness`, branch there, and open a PR in
  `Al-does/RL-Harness`.

### Run and test

```bash
uv run pytest -q -m "not slow"
uv run rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --smoke
```

### Vast.ai from Cloud Agents

The Cloud image installs `openssh-client`, and `bootstrap_local.sh` ensures a
local `~/.ssh/id_rsa` keypair exists. Both are required by
`devops.vast.provision` (readiness probes SSH into each box). If `ssh` is
missing on an old snapshot, re-run `./scripts/bootstrap_local.sh` or
`apt-get install -y openssh-client`.

### RunPod Pods from Cloud Agents

The sibling harness also provides `devops/runpod/pods/` for on-demand,
non-interruptible Community Cloud Pods. It uses RunPod APIs and does not need
SSH. Run a dry run first, then launch from this experiment checkout:

```bash
uv run python -m devops.runpod.pods.provision up \
  --run "rl-harness experiments.study.condition.experiment --upload-artifacts" \
  --forward-b2 --self-destruct --dry-run
```

RunPod jobs clone this repository and the harness at explicitly recorded refs.
Configure `RUNPOD_API_KEY` and `GH_TOKEN` as Cursor Runtime Secrets. Configure
the existing `B2_*` settings too when checkpoints must survive Pod teardown;
RunPod network volumes are unavailable on Community Cloud.

### Optional secrets

For Backblaze B2 artifact upload, add the same `B2_*` variables you use locally
as Cursor dashboard secrets (see `README.md` and `rl-harness/docs/artifact_storage.md`).

### More context

See `experiments/AGENTS.md` for experiment layout and promotion rules.
