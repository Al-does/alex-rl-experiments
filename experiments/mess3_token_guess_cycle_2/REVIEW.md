# Review of the MESS3 token-guess studies

Scope: `mess3_token_guess_cycle_1` and `mess_3_kelly_cycle_1/2/3`. These are four
directories but one experiment. All of them train a 3-layer, 96-wide transformer
with PPO on passive MESS3 (`alpha=0.85`, `delay=1`, episode length 512), reward
it 1 for guessing the next emitted token, and then fit a held-out rank-2 affine
probe from its residual stream to the exact Bayesian belief. `NextTokenGuessTask`
and `RawNextTokenTask` are the same task written twice.

The findings below are ordered by how much they would change what you tell your
colleagues. Every number is reproducible from the repository:

```bash
uv run rl-harness experiments.mess3_token_guess_cycle_2.references.experiment
uv run rl-harness experiments.mess3_token_guess_cycle_2.audit.experiment
```

---

## 1. The headline metric is mostly measuring something trivial

Belief-probe R² was reported as a bare number from 0.85 to 0.99, and read as
"how much of the Bayesian belief did this objective induce". It cannot be read
that way, because the metric has a floor that nobody measured.

An affine probe fitted directly to the **one-hot encoded raw observations**, with
no network and no training anywhere in the pipeline, scores **R² = 0.9668**. It
gets there on four observations and is saturated by six.

| probe input | R² |
|---|---:|
| last 1 observation | 0.8043 |
| last 2 observations | 0.9302 |
| last 4 observations | 0.9648 |
| last 8 observations | 0.9669 |
| last 64 observations | 0.9668 |

A **randomly initialised** copy of the study's own transformer, probed by exactly
the same code at exactly the same site, scores **R² = 0.8733**.

The supervised next-token replication in `mess3_supervised` reaches **0.99888**.

So the range this metric can move through is 0.9668 to 0.9989 — a span of
**0.032** — and the published scores land like this:

| condition | reported R² | position in the usable range |
|---|---:|---:|
| `comparison/reward_only` | 0.8552 | −348% |
| `comparison/max_entropy` | 0.8558 | −346% |
| *randomly initialised transformer* | *0.8733* | *−293%* |
| `comparison/predictive_loss` | 0.9319 | −109% |
| *affine probe on raw observations* | *0.9668* | *0%* |
| `iqn_value` | 0.9760 | +29% |
| `kelly_cycle_3/conditional_decoupled_kelly_iqn` | 0.9824 | +49% |
| `kelly_cycle_2/correctness_iqn` | 0.9857 | +59% |
| *supervised next-token replication* | *0.9989* | *100%* |

Two consequences.

**Reward-only PPO and max-entropy PPO are below the untrained network.** After
2.5M steps their residual streams are *less* linearly informative about the
belief than they were at initialisation. That is a real and interesting result —
task-reward training is destroying linearly-available structure — but it is the
opposite of the claim "reward-only PPO reaches R² = 0.855 of the belief".

**All eight Kelly cycle-2 wager arms sit below the no-network floor.** The
cycle-2 note reads their 0.94–0.96 as respectable. Against the floor they are
negative.

This is safe because the task is passive: `NextTokenGuessTask.resolve_action`
returns the same transition matrix for every action, so the distribution of
beliefs the probe sees is identical for every arm, for the untrained network, and
for the raw-observation baseline. The comparison is exact, not approximate.

## 2. Greedy token accuracy is saturated and cannot discriminate

An exact Bayesian filter allowed only the previous observation — the trivial
"repeat the last token" rule — scores **0.6732**. Given eight or more
observations it saturates at **0.6883**. Total usable range: **0.0151**.

Reported accuracies span 0.6733 to 0.6860. So the best arms are already within
0.002 of Bayes-optimal, `reward_only` at 0.6733 is at the repeat-last-token
rule, and the whole comparison lives inside 1.5 percentage points. Meanwhile the
probe's own sampling error at 30,000 test steps is ±0.005 at 95%, and that
assumes independent samples, which consecutive steps of a slowly-mixing chain
are not.

Reporting this metric to four decimal places invites exactly the question you do
not want from the audience.

## 3. None of the published orderings is statistically supported

Cycles 2 and 3 ran three seeds and reported `mean ± std`. That `±` is a
population standard deviation (`ddof=0`), not an interval; for three samples it
sits 18% below the sample standard deviation and about 3x below a 95% confidence
interval on the mean.

Recomputed properly, and paired across the seeds each pair of arms shares:

- **Kelly cycle 3**: 0 of 6 pairwise orderings survive a Holm correction.
- **Kelly cycle 2**: 0 of 28 pairwise orderings survive a Holm correction.

The headline gap is the clearest case. Cycle 3 reported PPO `0.8742 ± 0.0432`
against IQN `0.9688 ± 0.0090`, which reads as decisive. Paired on the three
shared seeds, the difference is +0.095 with a 95% interval of **[−0.017,
+0.206]** and p = 0.067. PPO's own interval, [0.743, 1.006], extends past the
maximum value R² can take.

Some claims in the notes are far past what three seeds can carry. Cycle 3 states
that "conditional Kelly with a scalar critic narrowly exceeds plain IQN in belief
R²". That gap is 0.0029 against a paired spread of 0.013; resolving it needs
**165 seeds**.

## 4. Final-checkpoint scores are drawn from a moving target

Every arm reports the checkpoint at its step budget, with `num_to_keep=1`. The
one arm that kept a checkpoint curve, `iqn_value_20m`, shows why that is unsafe:

| steps | R² | greedy accuracy |
|---:|---:|---:|
| 0.83M | 0.9902 | 68.12% |
| 2.48M | 0.9772 | 68.08% |
| 3.31M | 0.9673 | 68.34% |
| 10.76M | 0.9718 | 68.77% |
| 20.03M | 0.9686 | 68.47% |

Belief R² **peaks at the first retained checkpoint and declines**, while accuracy
slowly rises. They are not merely uncorrelated, they move in opposite directions.
After 3M steps R² fluctuates with a standard deviation of 0.0035 and a range of
0.013 — comparable to most of the between-arm gaps being reported.

So "IQN scores 0.976 at 2.5M" is one draw from a decaying, noisy trajectory. A
different step budget reorders the table. And the budget is not even constant
across the family: 828K, 2.5M, 3M and 20M all appear.

## 5. Every intervention was tested at exactly one strength

| intervention | values tried |
|---|---|
| predictive auxiliary loss λ | 0.1 |
| max-entropy reward coefficient α | 0.05 |
| IQN loss coefficient | 0.5 |
| Kelly direct-loss weight | 1.0 |
| learning rate | 3e-4 everywhere (4.2e-4 in the reward-state battery) |

No conclusion of the form "X does not help" is available from one point. The
max-entropy arm is the clearest example: at α = 0.05 the entropy bonus is at most
`0.05 · ln 3 ≈ 0.055` against a reward of 0 or 1, so the arm was never given
enough signal to do anything, and "max-entropy training produced essentially the
same belief representation as reward-only" is a statement about that one
coefficient.

The arms are also not matched on anything except environment steps. IQN arms set
`vf_loss_coeff=0` and add a quantile loss; Kelly and predictive arms add heads
and extra gradient into the shared trunk. Parameter count and total gradient
magnitude differ between arms while the learning rate is held fixed, so an arm
can win by being better tuned rather than by being a better idea.

## 6. γ is confounded with the study

Discount factor and condition set were varied together across directories.

| γ, λ | where |
|---|---|
| 0.99, 0.95 | token-guess `comparison`, `iqn_value`, Kelly cycle 3 |
| 1.0, 0.95 | Kelly cycle 1, `iqn_gamma_1_3m` |
| 0.0, 0.0 | Kelly cycle 2 |
| 0.0, 0.95 | `iqn_return_objectives/gamma_zero` |

Cycle 2 concludes that "moving from the prior γ = 0.99 reward-only result
(R² = 0.8552) to γ = 0 gives R² = 0.9834 without an auxiliary objective". That
compares n=1 to n=3, across two studies, two probe seed streams, and two
code paths. It is a plausible hypothesis, not a measured effect.

The λ change is harmless in itself — GAE weights terms by `(γλ)^k`, so at γ = 0
the value of λ cannot matter — but it should be made explicit rather than left as
an apparent second difference.

## 7. Smaller items worth fixing

**Probe protocol is not shared.** The RL probe uses a rank-2 constrained affine
map on 30,000 test steps at the post-final-LayerNorm site, from a single
train/test split. The supervised reference uses a full-rank probe on 472,390
positions, sweeps five layer sites and reports the best. Those two numbers should
not be placed in the same table until they are measured the same way. Kelly cycle
2 also uses different probe seed streams (200/201/202) from the token-guess cycle
(100/101/102), so cross-study numbers carry an extra source of variation.

**Probe R² and MSE are reported, but no uncertainty.** Bootstrapping the probe's
test set puts its own sampling noise at ±0.0005, which is small — but it needed
to be measured rather than assumed.

**Pairing does not currently buy anything.** All arms already run seeds 42/43/44,
so paired analysis is free. On the existing data the paired spread is frequently
*larger* than the unpaired spread, meaning seed effects are arm-specific
optimisation noise rather than shared data noise. Three seeds cannot settle this;
keep the paired design because it costs nothing, but do not budget for the
variance reduction.

**Actual step counts differ between arms.** 2,515,821 against 2,512,179 within
Kelly cycle 1, because the stop condition triggers on batch boundaries. Small,
but it is free to record and equalise.

**Run-to-run non-determinism is negligible.** The two reproduction arms re-ran
the same seed and code and differed by 0.00016 in R². Seed variance is roughly
200x larger. Determinism is not the problem here; sample size is.

**Naming and layout.** `mess3_token_guess_cycle_1` and `mess_3_kelly_cycle_*`
differ in spelling, and one experiment is spread over four directories with three
copies of the probe and two copies of the task. `mess_3_kelly_cycle_3/shared.py`
imports IQN from `mess3_token_guess_cycle_1.iqn_value.iqn` while the reward-state
battery imports it from `mess3_reward_state_cycle_1.iqn`, and the library now has
its own `IQNValueMixin`. Three IQN implementations are in play.

**Committed manifests leak the B2 bucket name.** Any run made with `B2_*`
configured writes the bucket and endpoint into tracked `results/`. Pass
`--no-upload-artifacts`, or scrub those fields, when running from an environment
where those are secrets.

---

## What this means for the presentation

The existing work supports a small number of claims very well:

- Training reward-only PPO on this task **degrades** the linear decodability of
  the Bayesian belief relative to both the raw observations and to the network's
  own initialisation. That is a genuinely surprising, presentable result, and it
  is large enough to survive the sample size.
- Adding a predictive or distributional objective **recovers and then exceeds**
  the raw-observation floor. `correctness_iqn` at γ = 0 and
  `conditional_decoupled_kelly_iqn` at γ = 0.99 are the only arms whose intervals
  clear the floor.
- Greedy token accuracy is saturated near Bayes-optimal for every arm, so
  representation quality and task performance are dissociable on this task. This
  is the same dissociation the reward-state battery found, arrived at
  independently.

It does not support any fine-grained ranking among the middle arms, and the
current notes lead with several of those rankings.

`PLAN.md` proposes what cycle 2 should do about it.
