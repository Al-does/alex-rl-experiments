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

`wing_two_factor_explore_cycle_2` has only `reward_both_state_0` and
`reward_factor_1_state_0`. It preserves the cycle-1 architecture and budgets,
sets rotation strength to 1.0, and uses `vf_clip_param=1e9` because RLlib caps
squared value error rather than the size of a value update. Its probes explicitly
use the cycle-2 environment. Verify with
`uv run pytest -q tests/test_wing_two_factor_explore_cycle_2.py`, then smoke either
leaf using the command above with `cycle_2` substituted for `cycle_1`.

### Offline Wing specificity controls

The analysis modules also expose a supplementary, non-interventional CLI for
existing checkpoints; this does not change the historical training-time probe
battery or retrain agents:

```bash
uv run python -m experiments.wing_token_guess_cycle_1.analysis --checkpoint CHECKPOINT --output NEW_RESULT.json --smoke
uv run python -m experiments.wing_two_factor_explore_cycle_2.analysis --condition reward_both --checkpoint CHECKPOINT --output NEW_RESULT.json --smoke
uv run pytest -q tests/test_wing_control_data.py tests/test_wing_control_analysis.py
```

Use `reward_factor_1` for the other cycle-2 arm. Omit `--smoke` for the default
20,000 independently collected fit and test samples, five null repetitions,
and 200 episode-bootstrap resamples. `--steps` changes the per-split analysis
sample budget. Each invocation writes the full report plus a small
`<output-stem>_summary.json` with a fixed last-layer view. Outputs refuse
overwriting either existing file. Reports include source/checkpoint hashes,
repository revisions, and runtime versions. Use new `results/` paths for compact
reports; checkpoints and scratch outputs stay in `artifacts/`.

`CHECKPOINT` accepts either an Algorithm root containing
`learner_group/learner/rl_module/default_policy` or that complete RLModule
subdirectory itself. `RLModule.from_checkpoint` restores these Wing modules on
CPU without starting Ray; no portable export or full Algorithm restore is
needed. B2 final-module downloads need only three files and can be verified
against the run's canonical durability manifest.

The delayed token-guess NTP control predicts the pending hidden token from the
filtered source belief, not from `info.belief_current @ emission_matrix` (which
is one transition too far). Controlled Wing uses preceding executed actions;
its rotations occur after emission, so next-token probabilities coincide across
current actions. The battery includes both per-factor marginal and full joint
NTP/log-NTP baselines; policy probabilities are separate RL nuisance features.
Alternative models are selected using training histories only,
with a next-observation KL constraint and affine/permutation-equivalence checks.
History shuffling recomputes both activations and targets and is a distribution
shift test, not a null required to have zero R². All comparisons are descriptive
linear accessibility tests, not evidence of causal use.

### Wing task-success tables and figures

Generate fresh complete-episode success estimates and Bayes references, then
render the checked-in probe results without loading model checkpoints:

```bash
uv run python -m experiments.wing_two_factor_explore_cycle_1.evaluate_success --output NEW_SUCCESS.json
uv run python -m experiments.wing_two_factor_explore_cycle_1.report_controls --success NEW_SUCCESS.json --output-dir NEW_REPORT_DIRECTORY
uv run pytest -q tests/test_wing_success_evaluation.py tests/test_wing_success_benchmarks.py tests/test_wing_control_report.py
```

Evaluation restores the final module subtrees recorded by the three completed
runs, uses 128 independent 1,024-step episodes per model, includes reset steps,
and reports conditional expected token accuracy or mean rewarded-factor arrival
occupancy under the learned stochastic policy. It does not divide variable-length
training returns by a fixed horizon. Result percentages and probe R² are distinct.

At alpha=0.94, x=0.4, token 1 dominates in every state. Always guessing joint
(1,1) therefore attains the exact passive Bayes maximum, 72.19334444%; the
Monte Carlo exact-filter calculation is only a cross-check. Cycle-2 control
strength is 1.0: use its freshly computed finite-horizon belief-grid upper bound
(about 68.3441%), not the old cycle-1 strength-0.15 benchmark. Grid refinement is
not an exact-optimality certificate. The benchmark policy uses full history,
not a promise of optimality within the transformer's 32-frame context.

The report generator writes scientific Markdown/CSV tables, PNG/SVG charts,
and a hash manifest. Specificity inputs are schema-2 `validated.json` reports;
superseded `final.json` drafts and all checkpoints stay out of commits.
`--overwrite` is restricted to intact generator-owned outputs.

Task-success reporting must include the saved initialization-to-final checkpoint
trajectory, not only final-checkpoint bars. Curves use the held-out `policy`
metrics in the original `condition_summary.json` checkpoint reports, starting
from the actual zero-step initialization. Do not use `train_policy`, invent
intermediate measurements, or infer confidence intervals from aggregate means.
These archived evaluations exclude 32 warmup steps per episode; keep them
separate from the newer complete-episode expected-success bars. The controlled
Bayes line is a full-episode reference, not an exactly matched post-warmup
ceiling.
