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

### Optional: Backblaze B2 artifact backup

Large run outputs (checkpoints, `.pt` weights, Tune trees) live under ignored
`artifacts/<run-id>/`. When B2 credentials are configured, the harness uploads
that tree at the end of each run and records `s3://…` URIs in tracked
`results/<run-id>/`.

One-time setup on this machine:

```bash
./scripts/setup_b2_env.sh
```

The script writes `~/.rl_harness_b2_env`, offers to add auto-load to
`~/.zshrc` (default: yes — press `n` to opt out), and is the easiest way to
get local runs and vast provisioning wired up.

Useful helpers after setup:

```bash
# Run an experiment with secrets loaded (no manual export)
./scripts/run_harness.sh \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke

# Rent a vast box and forward the same B2 vars to the remote container
./scripts/run_vast.sh up -n 1 \
  --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0 --smoke" \
  --self-destruct --yes

# Install or remove ~/.zshrc auto-load later
./scripts/b2_shell_autoload.sh install
./scripts/b2_shell_autoload.sh remove
```

Skip zshrc auto-load during setup with
`./scripts/setup_b2_env.sh --no-shell-autoload`. Re-run
`./scripts/setup_b2_env.sh --shell-autoload-only` if you skipped it the first time.

For Cursor Cloud Agents, add the same `B2_*` variables as dashboard secrets.
Full reference (bucket setup, object layout, download): `docs/artifact_storage.md`
in your sibling `rl-harness` checkout.

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
