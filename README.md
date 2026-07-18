# alex-rl-experiments

Personal experiment recipes for Alex, built on the shared
[`rl-harness`](https://github.com/Al-does/RL-Harness) library.

## Layout

```text
XOR/
  rl-harness/              # shared library (or symlink to "RL Harness")
  alex-rl-experiments/     # this repo
```

Science lives here. Reusable `harness/`, `learners/`, `losses/`, `envs/`,
`analysis/`, and `devops/` live in the library. Library changes are PRs to
`rl-harness`; experiment-only work stays in this repo.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
./scripts/bootstrap_local.sh
# or, if the library sibling is already present:
uv sync --group dev
```

This editable-installs `../rl-harness` so library edits are live.

## Run

```bash
uv run rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke
```

## Stay on library `main`

```bash
git -C ../rl-harness pull
# re-run experiments as needed; no version pin yet
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

Keep the matching experiment changes in this repo (second commit/push).

## Tests

```bash
uv run pytest -q -m "not slow"
```

## vast.ai

Provisioning still uses the library's `devops.vast` package. From this repo:

```bash
uv run --group devops python -m devops.vast.provision up -n 1 --dry-run
```

Boxes clone **this** repo (results push target) and the library as a sibling
editable install. See `rl-harness` `devops/vast/README.md`.
