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

## 1b. Move the operating point, and pay for it in probe size

Cycle 1 ran at `alpha=0.85` with a self-transition of 0.9, which is close to the
worst choice in this family for metric sensitivity: a fast-mixing chain lets the
belief be approximated by an exponentially weighted average of recent one-hot
observations, which is exactly what an affine probe computes. Slowing the chain
breaks that approximation.

| point | α | p | R² floor | R² range | accuracy range | probe ±95% at 30k steps |
|---|---:|---:|---:|---:|---:|---:|
| cycle 1 | 0.85 | 0.900 | 0.9671 | 0.033 | 0.015 | 0.0006 |
| candidate A | 0.70 | 0.990 | 0.8967 | 0.103 | 0.134 | 0.0037 |
| candidate B | 0.60 | 0.995 | 0.8782 | 0.122 | 0.136 | 0.0061 |
| candidate C | 0.70 | 0.995 | 0.9061 | 0.094 | 0.145 | 0.0046 |

Candidate C is my recommendation: about three times the belief-probe range and
ten times the accuracy range, while keeping the task clearly learnable at a
Bayes-optimal accuracy of 0.68.

The trade is precision, and it is worth being concrete about why.

The probe number is an estimate, computed from a finite rollout. Two rollouts of
the same checkpoint give two slightly different answers, and the spread between
them is the metric's measurement error. That error depends on how many
*independent* samples the rollout holds, which is not the same as how many steps
it has. On a slow chain the hidden state sits still for hundreds of steps, so
consecutive samples are near-duplicates and a long rollout can carry very little
new information.

Measured by direct replication — running the identical probe 60 times on fresh
rollouts and taking the standard deviation of the answers:

| operating point | 30k steps | 120k steps | 500k steps |
|---|---:|---:|---:|
| cycle 1 | 0.0004 | 0.0002 | 0.0001 |
| candidate C | 0.0025 | 0.0012 | 0.0005 |

At 30,000 steps the slow chain's measurement error is 0.0025, which is the same
size as the smallest seed-to-seed spreads observed in the Kelly cycles. Probe
noise would then be indistinguishable from seed variance, and adding seeds would
not fix it.

**Raise the probe to around 500,000 test steps**, where the measurement error
falls to 0.0005 and is safely below anything the design needs to resolve. Probe
rollouts are inference only and cost seconds, so this is the cheapest fix in the
plan and the easiest one to omit by accident.

Two notes on method. The i.i.d. bootstrap used in `REVIEW.md` understates this
error, because resampling individual steps pretends they are independent; a
moving-block bootstrap predicted ±0.0046 at 30k on candidate C and replication
measured ±0.0047. And the simulation behind these numbers does not include the
512-step episode resets the real environment applies, which partially
decorrelate the rollout, so treat 0.0025 as an upper bound and confirm it once
against the real probe by running it repeatedly on a single fixed checkpoint
with different probe seeds.

Two consequences follow. Episode length should rise from 512 so that the reset
back to a uniform belief is a smaller fraction of each trajectory. And at
γ = 0.99 the effective horizon of about 100 steps is now shorter than the state's
persistence, which is worth stating when γ is discussed.

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

## 3. Treat training duration as an axis, not a hyperparameter

Log-spaced checkpoints for every arm, following `rl-harness/docs/
checkpoint_strategy.md`, with the probe run at each.

Duration should not be tuned, because belief-probe R² is not monotonic in it and
the decline is now known to be optimiser drift rather than anything about the
objective. Picking the step budget after seeing the results would be choosing the
ranking. So: pre-register the budget, pre-register whether the primary statistic
is the final checkpoint or the mean over the last k, and report the whole curve
either way. I prefer the mean over the last k, which removes the ±0.0035
within-run fluctuation for free.

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

## 5. Sweep the optimiser and the coefficients, in two stages

Only four things are worth sweeping, and the first one matters more than the
arms do. See `task_parameters/` for the evidence.

| priority | swept | values |
|---|---|---|
| 1 | optimiser | AdamW, Muon |
| 1 | learning rate | 1e-4, 3e-4, 1e-3 |
| 2 | predictive aux weight λ | 0.03, 0.1, 0.3, 1.0 |
| 2 | max-entropy reward coefficient α | 0.01, 0.05, 0.2, 0.5 |
| 2 | IQN loss coefficient | 0.125, 0.5, 2.0 |
| 2 | Kelly direct-loss weight | 0.25, 1.0, 4.0 |

**Optimiser and learning rate move the headline metric more than the arms do.**
Training the study architecture on next-token prediction and probing it as it
goes, belief-probe R² *falls* with continued training while cross-entropy sits at
the Bayes floor — 0.9567 to 0.9318 over 6,000 AdamW steps at the cycle-1
parameters, and 0.9487 to 0.9367 at the slower candidate. That is supervised
training, so the decline over 20M PPO steps in cycle 1 is not an artefact of
reinforcement learning. `mess3_supervised` independently found SGD at 0.9979
against Muon at 0.9843 with identical next-token accuracy, a 0.014 swing from the
optimiser alone that is larger than most of the between-arm gaps cycle 2 is
trying to resolve. An arm comparison run at one fixed optimiser and learning rate
is partly measuring optimiser drift.

Sweeping the learning rate per arm is separately necessary because arms differ in
parameter count and in how much auxiliary gradient enters the shared trunk, so an
arm can lose by being mis-tuned rather than by being a worse idea.

**Do not sweep:** context length (see below — shrink it once and fix it), GAE λ
(irrelevant at γ = 0, low leverage at γ = 0.99), batch size, minibatch size, epoch
count, clip parameter, `d_model`, or `n_layers`. Fix these and record them. γ is a
scientific factor, not a nuisance hyperparameter; keep it in the design rather
than tuning it away.

## 5b. Shrink the context length once, then leave it alone

`CausalTransformerEncoder` applies its causal band at *every layer*, so the
receptive field is `n_layers × context_len`. Confirmed by perturbing inputs at
increasing distance and watching when the output stops changing: three layers at
`context_len=64` reach exactly 192 observations. The belief needs about 32.

The cost is paid twice, because the learner recomputes over
`lookback + max_seq_len = 192 + 32 = 224` positions to produce 32 useful
embeddings, and the cached rollout path attends over the same band.

| context_len | receptive field | learner step | rollout step | belief R² available |
|---:|---:|---:|---:|---:|
| 64 | 192 | 305 ms | 15.9 ms | 1.00000 |
| 32 | 96 | 146 ms (2.1x) | 6.9 ms (2.3x) | 1.00000 |
| 16 | 48 | 82 ms (3.7x) | 4.7 ms (3.4x) | 1.00000 |
| 12 | 36 | 70 ms (4.3x) | 4.6 ms (3.5x) | 0.99997 |
| 8 | 24 | 57 ms (5.4x) | 3.9 ms (4.0x) | 0.99966 |

Belief R² is at candidate C; candidate B needs more, reaching 0.99983 only at a
receptive field of 48.

**Use `context_len=16`.** It costs nothing measurable in available belief R² at
any candidate operating point, leaves a 50% margin over what the belief needs,
and keeps the choice valid if α or p move later. `context_len=12` is defensible
if candidate C is locked in, but the extra saving is small.

`context_len` affects compute only, not capacity — positions are encoded with
RoPE, so the parameter count does not change.

End-to-end this is worth **1.6x**, measured with a real PPO loop at matched
environment steps. It is less than the 3.4–3.7x component figures because
environment stepping is pure NumPy and unaffected. The split differs on a 4090
with 16 env runners, so measure it once there before re-budgeting.

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

At `context_len=16` a run costs about 1.6x less than the measured 6–14 minutes,
so budget roughly 7 minutes per run including checkpoint probes.

| stage | runs | GPU-hours | cost at $0.35/hr |
|---|---:|---:|---:|
| stage 1 sweep (7 arms × 6 optimiser/lr × 3 coefficient × 3 seeds) | 378 | 47 | ~$17 |
| stage 2 confirmation (16 conditions × 10 seeds) | 160 | 20 | ~$7 |
| references, operating point, audit | — | <1 | — |
| **total** | **538** | **67** | **~$24** |

Roughly nine wall-clock hours across eight Vast boxes. The entire committed
history of these four studies is 19.4 GPU-hours, so this is about three times all
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
