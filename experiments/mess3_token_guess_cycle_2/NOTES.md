# MESS3 token-guess cycle 2

Cycle 2 is a planned re-run of the token-guess comparison ahead of presenting it.
It does not yet contain any training conditions.

What is here now is the measurement work that has to precede the re-run:

- `REVIEW.md` — an audit of `mess3_token_guess_cycle_1` and
  `mess_3_kelly_cycle_1/2/3`, which are one experiment split over four
  directories.
- `PLAN.md` — a proposed design for the re-run, with the open questions that
  still need deciding.
- `metric_references.py` and `references/` — the floors and ceilings both
  headline metrics are measured against. Cycle 1 reported bare numbers, and both
  metrics turn out to have task-imposed ranges narrow enough that the bare
  numbers mislead.
- `operating_point.py` and `task_parameters/` — how the metric's range and
  precision vary with the MESS3 transition and emission parameters, so the
  process can be chosen to make the metric sensitive rather than degenerate.
- `statistics.py` and `audit/` — seed-level aggregation with intervals, paired
  comparison, multiplicity correction, and power planning, plus a re-analysis of
  the results already committed.

## What the references say

| reference | belief-probe R² | greedy accuracy |
|---|---:|---:|
| randomly initialised transformer | 0.8733 | 0.3412 |
| affine probe on the raw observations | 0.9668 | — |
| exact Bayesian filter, one observation | — | 0.6732 |
| exact Bayesian filter, saturated | — | 0.6883 |
| supervised next-token replication | 0.9989 | 0.6859 |

Belief-probe R² therefore moves through 0.032 and greedy accuracy through 0.015.
Cycle 1's reward-only and max-entropy arms land below the untrained network on
the first metric, and at the repeat-the-previous-token rule on the second.

## What the audit says

Re-reading the three-seed Kelly cycles with a t-based interval on the mean and
pairing across shared seeds, none of the 34 published pairwise orderings survives
a Holm correction. The reported `±` values were population standard deviations of
three samples, roughly a third of the width of a confidence interval.

## What the operating-point analysis says

Cycle 1's parameters are close to the worst choice in the symmetric MESS3 family
for metric sensitivity. Slowing the chain to a self-transition of 0.995 at
`alpha=0.70` roughly triples the belief-probe range and multiplies the accuracy
range by ten, at the cost of a probe interval that widens from ±0.0006 to
±0.0046 because the rollout decorrelates far more slowly.

It also shows that belief-probe R² declines with continued *supervised* training
while cross-entropy stays at the Bayes floor. The decline cycle 1 saw over 20M
PPO steps is therefore not a property of reinforcement learning, and optimiser
and learning rate move the headline metric more than most of the arms do.

Regenerate all three with:

```bash
uv run rl-harness experiments.mess3_token_guess_cycle_2.references.experiment
uv run rl-harness experiments.mess3_token_guess_cycle_2.task_parameters.experiment
uv run rl-harness experiments.mess3_token_guess_cycle_2.audit.experiment
```

Pass `--no-upload-artifacts` when `B2_*` variables are configured as secrets, or
the bucket name is written into tracked results.
