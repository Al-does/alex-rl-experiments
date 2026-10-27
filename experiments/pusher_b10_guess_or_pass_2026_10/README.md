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
| `sac_split_iter32/64/128/512` | SAC, split transformers | 455,168 / 910,336 / 1,820,672 / 7,282,688 | 0.396 / 0.124 / 0.062 / 0.058 |
| `sac_shared_iter32/64/512` | SAC, shared trunk | 455,168 / 910,336 / 7,282,688 | 0.522 / 0.167 / 0.055 |

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
