# Stage 3: action-sensitive writes

## Methods

`action` stacks logit Jacobians at every 16th post-warmup fit activation and
uses the leading right singular vectors. Ladder uses raw action logits (ranks
1–3). Factor tasks use marginal logits formed with logsumexp over joint actions:
moving uses factor 2's Jacobians projected off the leading rank-3 factor-1
subspace; XOR stacks both factors' marginal-logit Jacobians (ranks 1–6).
Numerically unsupported directions are dropped at relative tolerance 1e-6.
Rank selection maximizes pooled belief R² on independent seed-43 trajectories;
seed-44 diagnostic data are not used for selection. Both affine decoder and
belief-to-activation regression write are restricted to the chosen subspace.
The projection is an approximate global selectivity constraint, not an exact
per-activation non-target null space.

`behavioural` optimizes KL(map(target belief) || agent policy after the write),
plus beta times non-target KL from intact and .01 times mean squared activation
shift. Moving uses target factor 2; XOR and ladder use the complete policy.
`behavioural_action` restricts the shift to the selected action basis. Both use
60 Adam steps, learning rate .03, beta=1 on moving and zero when no factor is
preserved. The current belief comes from the exact filter and the target policy
from the calibrated stage-2 affine belief-to-logit map. These are **filter-assisted,
upper-bound-style behavioural references**, not demonstrations of decoding
belief from activations and not mathematically certified upper bounds.

## Seed-42 local pilot

Same stage-2 budgets: eight episodes per split for ladder/moving, two for the
others; 1,024 decisions. Parentheses are wall seconds, including fits and tuning.
Ladder/Full-State cells show rho-flip occupancy / mirror actions changed; other
cells show reward per step. These are descriptive one-agent pilot results.

| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |
|---|---:|---:|---:|---:|---:|
| action | .1434 / .0033 (33) | .9774 (25) | .5327 (8) | .6064 (16) | .1255 / .4521 (17) |
| behavioural reference | .1982 / .0538 (38) | .9659 (27) | .6011 (16) | .7251 (20) | .2695 / .3691 (20) |
| behavioural action reference | .4010 / .0179 (40) | .8284 (32) | .4883 (17) | .5171 (22) | .2285 / .0400 (19) |

| Test | Full affine R² | Action-subspace R² | Selected rank |
|---|---:|---:|---:|
| Ladder | .8771 | .5616 | 3 |
| Moving | .9022 | .4374 | 3 |
| XOR eps02 | .8335 | .7795 | 6 |
| XOR v2 | .9092 | .8081 | 6 |
| Full-State | .9746 | .8889 | 3 |

The action restriction removes a large amount of decodable belief information.
It nevertheless produces behavioural changes, unlike the passive-probe gradient
failure. This does not imply that all excluded directions are globally ignored:
the Jacobian subspace is bank-averaged and rank-truncated, and layer normalization
makes policy sensitivity activation-dependent. The v2 behavioural-reference gain
is promising only as a reference, with the same calibration caveats as stage 2.

## Campaign launch contract

Three on-demand CPU-rich RTX-4090 boxes, one each for agent seeds 42, 43, 44.
Each executes all 17 methods and all five tests. `--campaign` enforces 16
selection episodes and 32 confirmation episodes, all of length 1,024; fit and
diagnostic budgets are 24 and 12 episodes. Historical Eq. (4) strengths remain
archived. Causal ranks and new action ranks are selected on separate replay or
probe-selection trajectories. The seed runner first smokes the same command
with one episode per split and the two-point lambda grid, then runs full budgets.
Smoke files have a distinct prefix and are excluded from scientific aggregation.

```sh
python -m experiments.reward_factors_causal_decoding_2026_11.seed_campaign \
  --seed SEED --workers 12 --max-cpus 32
```

The box runner explicitly uses JAX CPU: these are many small batched Jacobian/
optimization scans, and the CPU pilot is the measured timing basis. GPU is not
required for this evaluation. Independent processes share up to 32 CPU cores.
Results are resumable per method and atomically saved under the two studies'
`results/` folders. The provisioner publishes only compact results to
`devin/1795210916-causal-decoding` before self-destruction.

Offer discovery uses `VastClient().search_offers(query, limit=500)`, excludes
geolocations ending CN, and requires at least 24 effective CPU cores and 64 GB
CPU RAM. This harness revision does not expose `--min-cpu-ram`; the equivalent
minimum is enforced in SDK search (`cpu_ram>=64`, search uses GB) and checked
against returned MB values before pinning machine IDs. Max age is disabled
(`--max-age 0`) as explicitly requested; only owned instance IDs may be destroyed.
