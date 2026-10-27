# Pusher-B b10 guess-or-pass transfer

This study compares scratch agents and encoder warm starts from token-guess
checkpoints with similar greedy next-token accuracy (about 0.6375) and
different probe `1-R²`. Actions 0/1 guess the pending token; action 2 passes.
Correct, wrong, and pass rewards are `+1`, `-cost`, and `0`.

The cost ladder is `c=1` (negative control), `c=1.5`, and `c=2` (primary).
Scratch expected per-step rewards:

| Cost | Bayes | Last-1 | Last-4 | Last-6 |
| --- | ---: | ---: | ---: | ---: |
| 1.0 | 0.276 | 0.275 | 0.275 | 0.275 |
| 1.5 | 0.149 | 0.096 | 0.137 | 0.146 |
| 2.0 | 0.082 | 0.000 | 0.062 | 0.075 |

## Source catalog

| Label | Algorithm / architecture | Env steps | Probe `1-R²` |
| --- | --- | ---: | ---: |
| `ppo_iter4` | PPO | 1,081,024 | 0.555 |
| `ppo_iter8` | PPO | 2,162,048 | 0.444 |
| `ppo_iter16` | PPO | 4,324,096 | 0.188 |
| `ppo_iter32` | PPO | 8,648,192 | 0.084 |
| `ppo_iter56` | PPO | 15,134,336 | 0.059 |
| `ppo_iter143` | PPO continuation | 115,901,216 | 0.038 |
| `ppo_ce_iter4` | PPO + next-token CE | 1,137,920 | 0.509 |
| `ppo_ce_iter8` | PPO + next-token CE | 2,275,840 | 0.1098 |
| `ppo_ce_iter211` | PPO + next-token CE | 60,025,280 | 0.0396 |
| `sac_split_iter32/64/128/512` | SAC, split transformers | 455,168 / 910,336 / 1,820,672 / 7,282,688 | 0.396 / 0.124 / 0.062 / 0.058 |
| `sac_shared_iter32/64/512` | SAC, shared trunk | 455,168 / 910,336 / 7,282,688 | 0.522 / 0.167 / 0.055 |

## PPO vs PPO+CE campaign

Six seed-42 source checkpoints have nearly matched greedy accuracy but span
belief-probe qualities:

| Source | Pretraining objective | Env steps | Greedy accuracy | Probe `1-R²` | Checkpoint |
| --- | --- | ---: | ---: | ---: | --- |
| `ppo_iter4` | PPO | 1,081,024 | 0.6375 | 0.555 | iter 4 |
| `ppo_iter8` | PPO | 2,162,048 | 0.6375 | 0.444 | iter 8 |
| `ppo_iter143` | PPO | 115,901,216 | 0.6375 | 0.038 | final continuation |
| `ppo_ce_iter4` | PPO + next-token CE | 1,137,920 | 0.6375 | 0.509 | `log_spaced_checkpoints/iteration_000004_steps_001137920` |
| `ppo_ce_iter8` | PPO + next-token CE | 2,275,840 | 0.6375 | 0.1098 | `log_spaced_checkpoints/iteration_000008_steps_002275840` |
| `ppo_ce_iter211` | PPO + next-token CE | 60,025,280 | 0.6373 | 0.0396 | `interval_checkpoints/iteration_000211_steps_060025280` |

The campaign has three scratch controls (`ppo_c2`, `ppo_c1p5`, `ppo_c1`),
18 full-trunk transfers (all six sources at all three costs), and six frozen
transfers (all sources at `c=2` only), for 27 runs total. Frozen transfer is
limited to `c=2` because that cost has the largest Bayes-versus-last-token
gap: `0.082` versus `0.000`. Runs are ordered as scratch, then all full-trunk
transfers, then frozen transfers. Target-task fine-tuning is plain PPO with no
CE loss or auxiliary head, with a 10M-env-step target (smoke length unchanged).

At iteration 8, PPO+CE is already past its probe-score drop (`0.509` at iter 4
to `0.1098` at iter 8), while PPO iter 8 is at `0.444`, before PPO's drop to
`0.188` at iter 16. The comparisons include early/intermediate/late checkpoints
within each objective at `c=2`, PPO versus PPO+CE at approximately matched
probe quality, the `c=1` negative control, scratch versus transfer, and frozen
versus full trunk.

Print all 27 launch commands:

```bash
uv run python -m experiments.pusher_b10_guess_or_pass_2026_10.campaign --list
```

Run one zero-based entry, for example (index 26 is `ppo_ce_iter8`, frozen,
`ppo_c2`):

```bash
uv run python -m experiments.pusher_b10_guess_or_pass_2026_10.campaign --run-index 26 --smoke --hardware cpu --no-upload-artifacts
```

This is a single-seed campaign. The late PPO+CE source has about 60M
pretraining steps versus 116M for late PPO, so its comparison also has a
pretraining-duration/plasticity confound.

The leaves are `ppo_c1`, `ppo_c1p5`, `ppo_c2`, `sac_c1`, `sac_c1p5`, and
`sac_c2`. PPO copies the shared encoder; SAC copies the actor and critic
encoders and initializes target copies from the fresh critics. Heads are
always initialized for the three-action task. `--trunk frozen` excludes
encoder parameters from optimization and checks encoder/head changes after the
first update.

Scratch CPU smoke:

```bash
uv run rl-harness experiments.pusher_b10_guess_or_pass_2026_10.ppo_c2.experiment --smoke --hardware cpu --no-upload-artifacts
```

Warm-start CPU smoke:

```bash
uv run python -m experiments.pusher_b10_guess_or_pass_2026_10.transfer experiments.pusher_b10_guess_or_pass_2026_10.ppo_c2.experiment --source ppo_iter4 --trunk full --smoke --hardware cpu --no-upload-artifacts
uv run python -m experiments.pusher_b10_guess_or_pass_2026_10.transfer experiments.pusher_b10_guess_or_pass_2026_10.sac_c2.experiment --source sac_split_iter64 --trunk frozen --smoke --hardware cpu --no-upload-artifacts
```

Use `--source-dir LOCAL` to load a local RLModule checkpoint. SAC transfer
defaults to the source architecture; pass `--architecture shared_trunk` for a
shared-trunk source. B2 downloads are restricted to the RLModule subtree and
verified against the source durability manifest. Full training runs have not
been performed.
