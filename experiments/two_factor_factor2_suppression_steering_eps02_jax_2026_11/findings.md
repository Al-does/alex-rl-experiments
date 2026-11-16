# Factor-2 suppression: emission ε=0.02 Joint Reward agents

## Result

Across 15 independently trained agents, final-block factored and joint steering
reduce factor-2 reward from 0.40731 to 0.09402 and 0.08960, respectively, near
or below the uniform-random reference of 0.10582. Factor-1 reward changes from
0.40269 to 0.38235 and 0.39560. The matched-norm decoder-orthogonal controls
retain factor-2 rewards of 0.39233 and 0.39115. This is selective behavioral
suppression, not proof that the agents implement a particular Bayesian algorithm.

All 15 agents reach the random baseline during tuning for both targets. On
independent confirmation, 14/15 factored and 13/15 joint interventions are at or
below the random baseline. Selection success is a tuning criterion, not a
guarantee on held-out episodes. Earlier sites are less effective, especially
after block 1.

## Protocol and provenance

- Source: PR #234, Joint Reward (`reward_both_transformer`) in
  `reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11`, seeds
  42–56, final checkpoint step **2,012,217,344**. Each report records its source
  run, checkpoint SHA-256 verified against the B2 durability manifest, full
  environment metadata, model, and experiment/harness revisions.
- The agents' own ε=0.02 process is used and checked against the resolved recipe.
  Both factors reward occupancy of manuscript state 3 (code index 2). Per-step
  factor rewards sum to total reward; total SE uses episode totals and is not
  the sum of component SEs. Complete 1,024-step episodes count reset and warmup
  steps. Greedy actions are taken from the steered forward pass.
- Architecture: four-block, width-64, one-head transformer, context/chunk length
  32. Sites are after blocks 1–4 and after final normalization. The shared v2
  forward, checkpoint loader, row collector, affine solver, target maps, R²,
  null-direction construction, factor-belief helpers, and revision helper are
  imported rather than copied. The eps02 source-run/checkpoint fetcher is reused.
- Separate on-policy greedy fit/test streams use seeds 42/43, 60,000/80,000
  rows, excluding 64 warmup steps per collection episode. Affine decoders and
  embeddings use four factored coordinates or the first eight joint
  probabilities. Final-block mean probe R² is **0.91177 factored / 0.86942
  joint**. These are descriptive accessibility metrics, not reward measures.
- Keep decoded factor 1 fixed. Replace factor 2 with a point mass on its less
  likely nonreward state, using the argmin over code states 0 and 1. The joint
  target is the decoded factor-1 marginal outer-product with this point mass.
  No clipping or simplex projection of decoded beliefs is added. Update
  `h' = h + λ (y* − D(h)) E`, one site at a time.
- Tune at block 4, seed 1042, 64 episodes, grid
  `{0, .25, .5, .75, .8, .85, .9, .95, 1, 1.25, 1.5, 2, 3}`. Select the smallest
  strength whose factor-2 reward is at or below tuning random; otherwise select
  the minimum factor-2 reward and flag failure. Freeze strengths for all sites.
- Confirm with seed 2042, 256 episodes, matched environment random streams for
  all conditions. Each target has one matched-norm decoder-orthogonal random
  direction at block 4; its instantaneous shift norm matches the corresponding
  target shift on the control's own history. Action-change fractions compare
  with an intact forward on the same current history and carry, not a separate
  intact trajectory. Mean final-block F1/F2 action-change fractions are
  0.12518/0.61434 factored and 0.06675/0.66603 joint.
- Driver enforces `jax.default_matmul_precision("highest")`. Maps are fit with
  the reused CPU NumPy float64 least-squares solver, `rcond=1e-10`, even on GPU
  boxes. Thus GPU low-precision least-squares is not used.
- Scientific code revision: `7fef68a9d3339a92e0a00ede173595bc6edb5f49`.
  Later experiment revisions below contain results only, not changes to the
  driver/summary. Harness: `a267508fdeb523e95b3ef3fc42119102540bbe38` for all
  agents. All reports record clean experiment checkouts at analysis start.

## Final-block confirmation

All uncertainties in these aggregate tables are **standard errors across the
15 training seeds**, not episode SEs or confidence intervals. Per-agent JSONs
retain episode SEs. Values are occupancies on a 0–1 scale, not percentages;
total is on a 0–2 scale.

| Condition | F1 | F2 | Total |
| --- | ---: | ---: | ---: |
| Intact | 0.40269 ± 0.00220 | 0.40731 ± 0.00386 | 0.81000 ± 0.00540 |
| Uniform random | 0.09836 ± 0.00000 | 0.10582 ± 0.00000 | 0.20417 ± 0.00000 |
| Factored, block 4 | 0.38235 ± 0.00555 | 0.09402 ± 0.00216 | 0.47637 ± 0.00697 |
| Joint, block 4 | 0.39560 ± 0.00165 | 0.08960 ± 0.00371 | 0.48521 ± 0.00439 |
| Factored matched-null, block 4 | 0.40127 ± 0.00256 | 0.39233 ± 0.00506 | 0.79361 ± 0.00690 |
| Joint matched-null, block 4 | 0.39922 ± 0.00271 | 0.39115 ± 0.00457 | 0.79037 ± 0.00653 |

The random policy does not depend on the trained agent and uses exactly the
same environment/action streams in every report. Its across-agent SE is
therefore zero **by design**, not evidence of zero Monte Carlo uncertainty.

## Per-site confirmation

| Target/site | F1 | F2 | Total |
| --- | ---: | ---: | ---: |
| Factored, block 1 | 0.39720 ± 0.00164 | 0.14948 ± 0.00690 | 0.54669 ± 0.00688 |
| Factored, block 2 | 0.39582 ± 0.00143 | 0.09510 ± 0.00572 | 0.49092 ± 0.00624 |
| Factored, block 3 | 0.39740 ± 0.00274 | 0.09077 ± 0.00338 | 0.48817 ± 0.00384 |
| Factored, block 4 | 0.38235 ± 0.00555 | 0.09402 ± 0.00216 | 0.47637 ± 0.00697 |
| Factored, post-final-norm | 0.38613 ± 0.00415 | 0.09838 ± 0.00532 | 0.48452 ± 0.00832 |
| Joint, block 1 | 0.40018 ± 0.00216 | 0.17420 ± 0.00973 | 0.57438 ± 0.00980 |
| Joint, block 2 | 0.39632 ± 0.00145 | 0.09773 ± 0.00626 | 0.49404 ± 0.00669 |
| Joint, block 3 | 0.39634 ± 0.00200 | 0.08516 ± 0.00343 | 0.48149 ± 0.00390 |
| Joint, block 4 | 0.39560 ± 0.00165 | 0.08960 ± 0.00371 | 0.48521 ± 0.00439 |
| Joint, post-final-norm | 0.39307 ± 0.00169 | 0.09431 ± 0.00438 | 0.48738 ± 0.00383 |

## Per-seed strengths and final-block F2

Both targets reached the tuning random baseline for every row (15/15 each).

| Seed | Factored λ | Joint λ | Intact F2 | Random F2 | Factored F2 | Joint F2 | Factored null F2 | Joint null F2 | Backend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.95 | 0.90 | 0.41049 | 0.10582 | 0.09700 | 0.08522 | 0.38503 | 0.39131 | cpu |
| 43 | 0.90 | 0.80 | 0.41893 | 0.10582 | 0.09899 | 0.11054 | 0.40580 | 0.40226 | gpu |
| 44 | 0.95 | 0.85 | 0.40398 | 0.10582 | 0.08713 | 0.08111 | 0.40277 | 0.40166 | gpu |
| 45 | 0.90 | 0.75 | 0.40448 | 0.10582 | 0.09974 | 0.08825 | 0.40152 | 0.40644 | gpu |
| 46 | 0.95 | 0.75 | 0.41577 | 0.10582 | 0.08156 | 0.09232 | 0.41268 | 0.40846 | gpu |
| 47 | 1.00 | 0.80 | 0.41335 | 0.10582 | 0.08827 | 0.10131 | 0.41890 | 0.41247 | gpu |
| 48 | 0.75 | 0.75 | 0.40390 | 0.10582 | 0.10204 | 0.07194 | 0.38688 | 0.37848 | gpu |
| 49 | 0.95 | 0.90 | 0.40082 | 0.10582 | 0.09649 | 0.08818 | 0.39555 | 0.38228 | gpu |
| 50 | 0.95 | 0.85 | 0.41798 | 0.10582 | 0.08435 | 0.07872 | 0.39609 | 0.39537 | cpu |
| 51 | 0.85 | 0.75 | 0.41322 | 0.10582 | 0.09650 | 0.09452 | 0.38947 | 0.38887 | gpu |
| 52 | 0.75 | 0.75 | 0.35712 | 0.10582 | 0.07965 | 0.06081 | 0.35676 | 0.35446 | gpu |
| 53 | 0.95 | 0.80 | 0.41105 | 0.10582 | 0.10800 | 0.11168 | 0.34576 | 0.35388 | gpu |
| 54 | 0.80 | 0.75 | 0.41699 | 0.10582 | 0.09498 | 0.07695 | 0.40831 | 0.40364 | gpu |
| 55 | 0.90 | 0.85 | 0.40974 | 0.10582 | 0.10382 | 0.09936 | 0.39519 | 0.39452 | cpu |
| 56 | 0.95 | 0.85 | 0.41182 | 0.10582 | 0.09180 | 0.10308 | 0.38434 | 0.39319 | gpu |

## Comparison with the superseded cycle-4 study

Reference: `deep-beliefs/experiments/causal_interventions/README.md`, section
**"Superseded: factor-2 suppression study"**, source revision
`9ecb94542a2d0977e1e369040c956ca8d6353cbe`. That study used a single 30M-step
Joint Reward seed-42 checkpoint, context 10, 12,000 fit/test rows, eight dropped
warmup steps, 16 tuning and 32 confirmation episodes. The same less-likely
nonreward-state target and frozen-across-sites strength rule were used.

| Old cycle-4 final block | F1 (mean ± episode SE) | F2 (mean ± episode SE) |
| --- | ---: | ---: |
| Intact | 0.3130 ± 0.0024 | 0.3134 ± 0.0027 |
| Uniform random | 0.1633 ± 0.0018 | 0.1617 ± 0.0021 |
| Factored, λ=0.9 | 0.2835 ± 0.0022 | 0.1357 ± 0.0024 |
| Joint, λ=0.5 | 0.3130 ± 0.0024 | 0.1087 ± 0.0019 |

The qualitative pattern reproduces: strong factor-2 suppression with much less
factor-1 disruption, especially for joint steering. The present intact and
random levels differ substantially from cycle 4; these are **not controlled
head-to-head effect-size estimates**. Environment, trained agents, training
budget, context, sample budgets, tuning grid, and uncertainty unit differ.
Do not compare old episode SEs directly to new training-seed SEs, substitute
the separate old 300M continuation, or treat the old single-seed table as a
multi-seed result.

## Execution and infrastructure

The complete local seed-42 JSON was committed and pushed before renting
(`a233109f`). Launches used on-demand RTX 4090/5090 offers, pinned machine IDs,
`--max-age 2 --forward-b2 --self-destruct`, the requested launch branch, and
one seed per box. An offer tally used `limit=500`; workers used disjoint spare
machine lists, launches spaced by roughly 40 seconds, at most eight concurrent
workers. The first box passed bootstrap and explicitly reported
`[CudaDevice(id=0)]` before the remaining workers launched.

Twelve seeds completed on GPUs. Seed 50's disjoint candidates became unavailable;
seed 55's first host went offline and its spares were unavailable. Both were
completed locally on CPU, as authorized, rather than renting indefinitely.
Thus **13 total instances were rented, not 14 successful GPU boxes**: twelve
successful analyses and one failed host. No foreign instance was operated on.

| Seed | Instance | GPU | Analysis seconds | Experiment revision |
| --- | --- | --- | ---: | --- |
| 42 | local CPU | — | 263.7 | `7fef68a9` |
| 43 | 60985052 | RTX 4090 | 96.6 | `a233109f` |
| 44 | 60985955 | RTX 4090 | 90.8 | `a233109f` |
| 45 | 60986019 | RTX 4090 | 81.3 | `a233109f` |
| 46 | 60986085 | RTX 4090 | 41.0 | `a233109f` |
| 47 | 60986141 | RTX 5090 | 141.3 | `b4a50a95` |
| 48 | 60985648 | RTX 4090 | 109.4 | `a233109f` |
| 49 | 60985712 | RTX 4090 | 82.8 | `a233109f` |
| 50 | local CPU fallback | — | 254.8 | `9170830a` |
| 51 | 60985872 | RTX 4090 | 40.6 | `a233109f` |
| 52 | 60986532 | RTX 4090 | 105.5 | `9170830a` |
| 53 | 60986341 | RTX 4090 | 67.3 | `bdf05118` |
| 54 | 60986734 | RTX 5090 | 46.5 | `7e3b1248` |
| 55 | local CPU fallback; failed instance 60986481 | RTX 5090 failed | 262.4 CPU | `28d18398` |
| 56 | 60986573 | RTX 5090 | 166.8 | `28d18398` |

The current required-durability teardown hook publishes analysis JSONs, but
prints `required durability found no pending run manifest` and retains the box
because this analysis produces no training-run manifest or new model artifacts.
After verifying publication on the launch branch, these owned instances were
explicitly destroyed. Seed 47's complete JSON was copied back and published
locally to avoid further publisher delay, then its box was destroyed. Its only
runtime warning concerned CUDA kernel-timing accuracy; the evaluation exited
successfully. Source checkpoints remain in their original SHA-verified B2
locations; scratch downloads were not committed or separately uploaded.

Final harness `provision status`: **No tracked instances (state.json is empty).**
All thirteen owned IDs were additionally queried directly and confirmed absent.
Rough GPU rental estimate: **$2–3**, based on quoted rates approximately
$0.365–$0.835/hour and observed bootstrap/evaluation/publication lifetimes;
this is not an invoice and excludes any separate storage/network charges.

## Caveats and reproduction

- Aggregate SEs quantify variation among these 15 trained seeds conditional on
  the fixed fit, tuning, and confirmation streams. They do not integrate over
  new probe-fit seeds or new environment-evaluation seed sets. Shared random
  numbers induce pairing; unpaired SEs are not SEs of differences.
- Maps are fitted on each intact agent's on-policy activation distribution.
  Steering changes the trajectory and may leave that distribution. Decoders
  need not remain calibrated under intervention, and decoded beliefs need not
  lie on the simplex.
- Each null uses a single deterministic decoder-orthogonal random direction;
  these are not an ensemble of all possible perturbation controls. Null reward
  effects are small on average but not identically zero.
- CPU/GPU floating-point execution can differ near greedy-action ties. The
  recorded backend is `cpu` for seeds 42, 50, 55 and `gpu` for all other seeds;
  no cross-backend numerical-equivalence claim is made.
- Results are final-checkpoint intervention measurements, not learning curves,
  optimality certificates, or evidence that a fully factored generative model
  is uniquely used internally.

Generate the machine-readable aggregate (choose a fresh output filename):

```bash
uv run --group jax python -m experiments.two_factor_factor2_suppression_steering_eps02_jax_2026_11.summarize --output NEW_SUMMARY.json
```

`results/summary.json` includes all 14 conditions, F1/F2/total across-agent
statistics, per-agent values, and the counts reaching the tuning baseline.
The fifteen source reports retain all strength sweeps, probes, episode SEs,
action-change fractions, and checkpoint provenance. No plots were generated.
