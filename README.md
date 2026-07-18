# alex-rl-experiments

Alex's personal experiment recipes for the shared
[`rl-harness`](https://github.com/Al-does/RL-Harness) library.

**Colleagues:** do not fork this repo for new work. Start from the template
[`rl-experiments-template`](https://github.com/Al-does/rl-experiments-template)
instead (Use this template → `./scripts/bootstrap_local.sh`).

## Layout

```text
XOR/
  rl-harness/              # shared library (or symlink to "RL Harness")
  alex-rl-experiments/     # this repo
```

Science lives here. Reusable code lives in the library. Library changes are
PRs to `rl-harness`; experiment-only work stays here.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
./scripts/bootstrap_local.sh
# or, if the library sibling is already present:
uv sync --group dev
```

## Run

```bash
uv run rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke
```

## Stay on library `main`

```bash
git -C ../rl-harness pull
```

Run manifests record both this repo's commit and the library commit.

## Contribute a library change

```bash
cd ../rl-harness
git checkout -b alex/my-change
# edit losses/, learners/, etc.
git push -u origin HEAD
gh pr create
```

## Tests

```bash
uv run pytest -q -m "not slow"
```

## vast.ai

```bash
uv run --group devops python -m devops.vast.provision up -n 1 --dry-run
```
