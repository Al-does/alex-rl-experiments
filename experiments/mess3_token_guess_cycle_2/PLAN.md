# Proposed design for token-guess cycle 2

Draft for discussion. `REVIEW.md` establishes the problems; this is one way to
address them. Everything here is negotiable, and the open questions at the end
are the parts I would not decide alone.

## Principle

Cycle 1 ran a large number of arms, measured each once, and read a ranking off
the resulting table. Cycle 2 should run fewer arms, measure each well enough to
put an interval on it, and pre-commit to which comparisons it is trying to
resolve. The compute cost of doing this is trivial — the numbers are at the end —
so the binding constraint is design discipline, not GPU budget.

## 1. Consolidate into one study

One directory, one `shared.py`, one task class, one probe. Today the same
experiment is spread over four directories with two copies of the task, three
copies of the probe, and three IQN implementations (`mess3_token_guess_cycle_1.
iqn_value.iqn`, `mess3_reward_state_cycle_1.iqn`, and the library's
`IQNValueMixin`).

The library's `IQNValueMixin` should be the only one. Anything else that is a
reusable RL concept goes to `rl-harness`; Kelly wagering stays here.

## 2. Report metrics against their range, always

Never present a bare R². Every table gets the four reference rows from
`references.experiment` alongside the trained arms:

| row | belief R² | source |
|---|---:|---|
| affine probe on raw observations | 0.9668 | analytic, no network |
| randomly initialised transformer | 0.8733 | same probe, same architecture |
| supervised next-token model | 0.9989 | `mess3_supervised` |
| exact Bayesian filter (accuracy) | 0.6883 | analytic |

R² is a bad scale here because the interesting region is 0.967–0.999. I would
report three things instead, and let the audience pick:

- **Probe MSE**, which has real dynamic range (0.0012 at R² = 0.99 against 0.03
  at R² = 0.85) and reads naturally on a log axis.
- **Incremental R²**: fit the probe on the raw-observation window alone, then on
  the window plus the network's activations, and report the increment. This
  answers "what does the network represent that its own inputs did not already
  linearly provide", which is the question the study is actually asking.
- **Position in the usable range**, as `references.experiment` already computes,
  for the one summary slide.

## 3. Fix the checkpoint protocol

Log-spaced checkpoints for every arm, following `rl-harness/docs/
checkpoint_strategy.md`, with the probe run at each. Then either:

- pre-register "the final checkpoint" and additionally report the curve, so the
  reader can see the metric is decaying; or
- pre-register "the mean over the last k checkpoints", which removes the
  ±0.0035 within-run fluctuation from the comparison for free.

I prefer the second for the primary metric and the first in an appendix, but
this is a real choice and it must be made before the runs, not after.

The step budget must be identical across every arm. 2.5M is defensible; so is
5M. What is not defensible is the current mixture of 828K, 2.5M, 3M and 20M.

## 4. Seeds: ten, from a documented spawn

Ten seeds per condition, paired across arms, drawn by spawning from one master
`SeedSequence` and recorded in the manifest.

The planning table below is from `statistics.seeds_for_power`, paired, 80% power,
95% confidence:

| belief-R² gap | sd = 0.002 | sd = 0.005 | sd = 0.010 | sd = 0.020 |
|---:|---:|---:|---:|---:|
| 0.002 | 10 | 52 | 199 | 787 |
| 0.005 | 4 | 10 | 34 | 128 |
| 0.010 | 3 | 5 | 10 | 34 |
| 0.020 | 3 | 3 | 5 | 10 |
| 0.100 | 3 | 3 | 3 | 3 |

Ten seeds resolves any gap of 0.01 or more at the spreads actually observed
(0.001–0.011). It will not resolve the 0.002–0.005 gaps between the middle arms,
and that is the point: those should be reported as ties rather than chased. Going
to the ~150 seeds that would settle them is possible on this compute budget but I
do not think it buys a better talk.

Note the planning numbers are computed from effects observed at n=3 and are
therefore optimistic — the observed gap in a small sample is biased upward. Ten
is already padded for that; I would not go below eight.

## 5. Sweep the coefficients, in two stages

Every intervention currently has exactly one tested strength, so no "X does not
help" claim is available. Proposed grid:

| arm | swept coefficient | values |
|---|---|---|
| all | learning rate | 1e-4, 3e-4, 1e-3 |
| predictive aux | loss weight λ | 0.03, 0.1, 0.3, 1.0 |
| max entropy | reward coefficient α | 0.01, 0.05, 0.2, 0.5 |
| IQN | loss coefficient | 0.125, 0.5, 2.0 |
| Kelly | direct-loss weight | 0.25, 1.0, 4.0 |

Sweeping the learning rate per arm is not optional. Arms differ in parameter
count and in how much auxiliary gradient enters the shared trunk while the
learning rate is pinned at 3e-4, so an arm can currently lose by being
mis-tuned rather than by being a worse idea.

**Two stages, with disjoint seeds.** Stage 1 sweeps on seeds 0–2 and selects one
configuration per arm. Stage 2 re-runs the selected configurations on ten
*fresh* seeds, and only stage 2 enters the results table. Selecting and reporting
on the same seeds is what makes a swept comparison optimistic, and it is easy to
avoid here.

Selection criterion for stage 1 must be pre-registered. I suggest belief-probe
MSE, since accuracy is saturated.

## 6. Pre-register the comparison family

Ranking eight arms is 28 tests. Declare the primary comparisons in advance and
correct within that family; report everything else as exploratory.

Proposed primaries, all on belief-probe MSE at γ = 0.99:

1. reward-only PPO against the raw-observation floor — does task reward degrade
   the representation?
2. reward-only PPO against an untrained network — is the degradation relative to
   initialisation?
3. predictive aux against reward-only — does the auxiliary objective recover it?
4. IQN against reward-only — does distributional value do the same?
5. Kelly against IQN — are they doing the same thing or different things?
6. best combined arm against the best single arm — do they compose?

Six tests, Holm-corrected. Everything else is exploratory and labelled as such.

## 7. Arms

Two axes crossed, with max-entropy as an orthogonal third:

| axis | levels |
|---|---|
| critic | scalar mean, IQN |
| representation pressure | none, predictive aux, Kelly wager |
| entropy in reward | off, on |
| γ | 0, 0.99 |

The full cross is 24. I would run the 2 × 3 × 2 = 12 critic × pressure × γ cells
as the main grid, and treat entropy as a separate 2 × 2 (critic × γ) add-on at
its swept coefficient, giving 16 conditions. The γ = 1.0 and differential
average-reward objectives from cycle 1 are worth keeping as a third γ level if
the budget is there, since the average-reward arm was the one that restored sane
critic diagnostics.

## 8. Compute

Measured from the committed manifests: a 2.5M-step arm costs 6–14 minutes on an
RTX 4090, median about 6.5. Adding eight checkpoint probes puts it near 12.

| stage | runs | GPU-hours | cost at $0.35/hr |
|---|---:|---:|---:|
| stage 1 sweep (7 arms × 12 configs × 3 seeds) | 252 | 50 | ~$18 |
| stage 2 confirmation (16 conditions × 10 seeds) | 160 | 32 | ~$11 |
| references and audit | — | <1 | — |
| **total** | **412** | **82** | **~$30** |

Roughly ten wall-clock hours across eight Vast boxes. The entire committed
history of these four studies is 19.4 GPU-hours, so this is about four times all
prior work on the question, for the price of lunch.

`devops.vast.provision up -n N --run "..."` gives one command per box, so the
sweep matrix needs a small dispatcher that maps (arm, coefficient, seed) to
`--run` strings. That is the only new infrastructure required; there is no
multi-seed or sweep support in the harness today.

## Open questions

1. **Is the degradation result the headline?** The strongest thing in the data is
   that reward-only PPO ends up below both the raw observations and its own
   initialisation. That reframes the study from "which objective induces belief
   geometry" to "task reward destroys it and these objectives protect it". It is
   a better talk, but it is a different talk. Which one do you want to give?

2. **Is belief-probe R² still the right dependent variable?** It was chosen to
   match the supervised replication, but on this task it is compressed into the
   top 3% of its range and it measures linear decodability rather than use. An
   intervention experiment — ablate the probe-identified directions and see
   whether actions change — would answer the question colleagues will actually
   ask. That is more work than a re-run. Worth it before the presentation, or
   after?

3. **How much of the Kelly programme survives?** Cycles 1–3 ran fourteen Kelly
   variants and none of the orderings among them is supported. Cycle 2 could
   carry one Kelly arm (conditional decoupled, the best performer) rather than
   the family. Is the wager-calibration result something you want to present in
   its own right, or is Kelly here purely as a representation-shaping device?

4. **Do you want γ as a swept axis or a fixed choice?** γ is currently confounded
   with the study directory, and the γ = 0 versus γ = 0.99 contrast is one of the
   larger apparent effects. Making it a proper factor doubles the grid. My
   inclination is yes, because a clean γ effect is presentable and the current
   evidence for it is cross-study.

5. **Do we keep the 20M-step observation?** The R² decay from 0.990 to 0.968 over
   20M steps is arguably the most interesting single curve in the repository, and
   it exists for one arm at one seed. Three seeds of a long run for two arms
   would cost about six GPU-hours and would turn an anecdote into a result.

6. **What is the presentation format?** If there is a slide budget, that should
   drive how many arms survive. Sixteen conditions with intervals is a dense
   table; six conditions with intervals and a reference band is a slide.
