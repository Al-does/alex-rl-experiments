# alex-rl-experiments

Alex's personal experiment recipes for the shared
[`rl-harness`](https://github.com/Al-does/RL-Harness) library.

**Colleagues:** start at [`rl-experiments`](https://github.com/Al-does/rl-experiments)
— **fork** it (no rename), clone your fork, run `./scripts/bootstrap_local.sh`.
Do not fork this science repo for new work.

## Layout

```text
XOR/
  rl-harness/              # shared library (or symlink to "RL Harness")
  alex-rl-experiments/     # this repo (Alex's ongoing studies)
```

## Setup

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

## Contribute a library change

```bash
cd ../rl-harness
git checkout -b alex/my-change
git push -u origin HEAD
gh pr create
```

## Tests

```bash
uv run pytest -q -m "not slow"
```
