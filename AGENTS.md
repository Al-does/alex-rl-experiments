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

Remote boxes publish **compact results only** (`experiments/**/results/**`) back
to the **launch branch** (`--branch cursor/...`). They do not rebase onto
`main` or a shared `results` branch. Merge findings to `main` manually in a
PR. Never commit checkpoints or `artifacts/` trees from a box — see
`experiments/AGENTS.md`.

### RunPod Pods from Cloud Agents

The sibling harness also provides `devops/runpod/pods/` for on-demand,
non-interruptible Community Cloud Pods. Batch runs use RunPod APIs and do not
need SSH. Run a dry run first, then launch from this experiment checkout:

```bash
uv run python -m devops.runpod.pods.provision up \
  --run "rl-harness experiments.study.condition.experiment --upload-artifacts" \
  --forward-b2 --self-destruct --dry-run
```

RunPod jobs clone this repository and the harness at explicitly recorded refs.
Configure `RUNPOD_API_KEY` and `GH_TOKEN` as Cursor Runtime Secrets. Configure
the existing `B2_*` settings too when checkpoints must survive Pod teardown;
RunPod network volumes are unavailable on Community Cloud.

For opt-in terminal profiling, `--interactive` injects only this Cloud Agent's
generated SSH public key; the private key remains on the VM. Use the harness
`logs POD_ID --follow` and `ssh POD_ID` subcommands, then explicitly destroy the
Pod. The provider hard ceiling remains active.

### Optional secrets

For Backblaze B2 artifact upload, add the same `B2_*` variables you use locally
as Cursor dashboard secrets (see `README.md` and `rl-harness/docs/artifact_storage.md`).

### More context

See `experiments/AGENTS.md` for experiment layout and promotion rules.

### Wing study verification

`wing_two_factor_explore_cycle_1` requires the sibling harness's edge-emitting
`envs.wing` support; a state-emitting approximation is not equivalent to Wing.
Its six leaves are `{reward_both,reward_factor_1}_state_{0,1,2}`. From this repo:

```bash
uv run pytest -q tests/test_wing_two_factor_explore_cycle_1.py tests/test_wing_probe_analysis.py tests/test_wing_design.py
uv run rl-harness experiments.wing_two_factor_explore_cycle_1.reward_both_state_0.experiment --smoke --hardware cpu --no-upload-artifacts
uv run python -m experiments.wing_two_factor_explore_cycle_1.design --analytic-only
```

Omit `--analytic-only` for the deterministic vectorized reference-policy audit
(no neural-network training). Smoke runs execute checkpoint probing as well as
PPO updates and write ignored `.smoke/<run-id>/` outputs. The study uses a strict
32-frame RoPE encoder (base 10,000, applied to queries and keys); the newer banded
cached transformer's per-layer window is not an equivalent 32-frame receptive
field. Original reproduction recipes retain learned absolute positions by
default. Full Wing budgets are 30 million steps for each single-rewarded-factor
arm and 50 million for each both-rewarded-factor arm; smoke remains 2,048 steps.
Temperature scaling must be identical in rollout and PPO likelihoods. Belief
targets condition on tokens and executed actions, never rewards.
