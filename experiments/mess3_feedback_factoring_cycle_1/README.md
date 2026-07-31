# `mess3_feedback_factoring_cycle_1` — when the guess changes the process

## Premise

`mess3_token_guess_cycle_2` asks a gamma-zero agent to guess the token MESS3 is
about to reveal. The guess is scored and then thrown away: the hidden process
never notices it. This cycle keeps that task and adds one thing. **The guess now
changes the transition probabilities, and each of the three guesses changes them
in a different way.**

The scientific question is not whether the agent can steer — under `gamma = 0`
it has no reason to. It is whether the agent *models* its own influence. Good
myopic prediction at step `t + 1` requires knowing what the guess at step `t`
did to the hidden state, so the pressure to represent the feedback is real even
though the pressure to exploit it is zero.

> **Sibling study.** `mess3_feedback_cycle_1` asks the same question with a
> different intervention: the guess pulls the next state toward itself,
> `U_a = (1 - eta) T + eta R_a`. That rule is deliberately minimal and keeps the
> three token labels permutation symmetric. This study instead chooses the one
> intervention that makes the process an exact composition of two factors, so
> that the factored-representation predictions of Shai et al. become testable.
> The two are complementary, not competing.

## The composed generator

The hidden state is a pair `(m, phi)`, nine states indexed `m * 3 + phi`:

- `m` is an untouched passive MESS3 chain under the circulant kernel `T`;
- `phi` is a `Z_3` register that only the agent's own guesses move.

Guess `a` executes `kron(T, R(a))` with `R(a) = (1 - kappa) I + kappa C^a`, where
`C` is the forward cyclic shift. Guess `0` leaves the register alone, guess `1`
rotates it one step and guess `2` two steps, each firing with probability
`kappa`. The joint kernel is therefore always an exact tensor product.

Each factor emits its own sub-token and the observed token is the pair
`(x, rho)` from a nine-symbol alphabet indexed `x * 3 + rho`:

- `x = u + phi (mod 3)` is the composite token the agent is scored on, with `u`
  the passive MESS3 sub-token;
- `rho` reports the register, equal to `phi` with probability `1 - epsilon` and
  uniform noise otherwise.

So

```text
P((x, rho) | m, phi) = E[m, (x - phi) % 3] · [ (1 - eps)·1{rho = phi} + eps/3 ]
```

The reward is unchanged from cycle 2 — one point for naming the current
composite token — so the myopic optimum is still "guess the most likely token"
and only the executed kernel depends on the guess.

### The two axes are not the same knob

`epsilon` is the knob of [*Transformers learn factored
representations*](https://arxiv.org/pdf/2602.02385) (arXiv:2602.02385). At
`eps = 0` the token-labelled operator splits as `A(m) (x) B(phi)`, which is
conditional independence in exactly their Definition 2.1, so a factored
representation is lossless. At `eps = 1` the register sub-token is pure noise,
only the sum `m + phi` is ever observable, and both factor marginals collapse
onto their uniform priors: a factored representation then carries no predictive
information at all. Intermediate `epsilon` mixes a factoring and a non-factoring
operator, exactly as their `eps·T_int + (1 - eps)·(x) T_n`.

`kappa` is a separate axis with no counterpart in that paper. It sets how hard a
guess shoves the process, not how far the belief sits off the product manifold.
It is deliberately *not* called epsilon, because it cannot play that role: the
cost of factoring along `kappa` is a cliff rather than a ramp. With the register
unreported, any nondegenerate mixture of `Z_3` shifts drives both factor
marginals to exactly `log 3` nats by symmetry, so `kappa` offers only "no second
factor" (`kappa = 0`), "lossless" (`kappa = 1`) and "totally lossy" (everything
in between), with nothing in the middle.

## Measured properties of the generator

From `composition_theory` (seed 42, 384 chains of 3072 steps). The cost of
factoring is the extra nats per token paid for predicting from `b_m (x) b_phi`
instead of the exact joint belief.

Register-noise axis at `kappa = 0.7`:

| `epsilon` | 0.0 | 0.15 | 0.3 | 0.5 | 0.7 | 0.85 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cost of factoring (nats) | 0.000 | 0.003 | 0.011 | 0.030 | 0.074 | 0.143 | 0.214 |
| product-state MSE | 0.000 | 0.0002 | 0.0007 | 0.0018 | 0.0040 | 0.0069 | 0.0092 |
| myopic ceiling | 0.622 | 0.619 | 0.614 | 0.616 | 0.614 | 0.614 | 0.613 |

The cost ramps over two orders of magnitude while the ceiling barely moves, so
`epsilon` isolates factorability from raw task difficulty. At `eps = 0` the cost
is exactly zero while the register belief is still a genuine distribution rather
than a delta — product states, not trivial states, which is the case the paper's
geometric prediction is actually about.

Feedback-strength axis at `eps = 1`:

| `kappa` | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| myopic ceiling | 0.690 | 0.618 | 0.569 | 0.590 | 0.635 | 0.690 |

Difficulty is an inverted U. Both endpoints reproduce the exact passive optimum
of `0.68958` from
`mess3_token_guess_cycle_2.analysis.bayesian_optimal_accuracy`, because a
deterministic rotation is the passive process viewed in a rotating frame. That
is the main correctness check on the whole construction. A ten-step context is
already effectively infinite memory here.

`kappa = 0.5` is excluded from the grid. With exact argmax tie-breaking the loop
drives itself into a predictive tie between guesses `1` and `2`, never plays the
register-inert guess `0`, and accuracy collapses to `0.43`; a few hundredths
either side it snaps back. It is a knife-edge artifact of exact ties, not a
feature of the dynamics.

## Does the closed loop collapse to a single HMM?

The motivating conjecture: once the policy is fixed, maybe "an underlying HMM
plus guesses that change it" can be rewritten as one autonomous HMM whose
transition matrix stacks the guess-conditioned kernels and renormalizes them,

```text
Ubar[s, .] = sum_y P(guess = y | state = s) U(y)[s, .].
```

`composition.single_hmm_report` builds `Ubar` from realized `(state, guess)`
statistics and compares the exact distribution over four-token blocks under
`(Ubar, E)` against the closed loop's empirical distribution, with a
sampling-noise floor from splitting the chains in half.

**The answer is no for every `kappa > 0`, but the conjecture's own reasoning
points the right way.** At `kappa = 0` the residual sits at or below the
sampling floor, as it must. Everywhere else it is five to thirty times the
floor. `Ubar` is exact only when the guess is conditionally independent of the
hidden state given the state; a Bayes agent chooses from its *belief*, which is
correlated with the state through shared history, so marginalizing the guess at
the level of the state throws that correlation away. Emitting the guess from the
belief's own distribution rather than greedily does shrink the residual three to
four times, because randomizing weakens the state-guess coupling. It does not
remove it.

## Measurement

Every checkpoint gets one affine read-out per target, all fit on the same
held-out activations (post-final-LayerNorm, greedy rollouts, disjoint train and
test seed streams). The targets form an information ladder over how much of the
agent's own influence they account for:

| target | what the observer knows |
| --- | --- |
| `joint` | the realized guess; the exact nine-state predictive belief, and the minimal sufficient statistic of the process |
| `marginal` | the policy's state-conditioned guess statistics, but not the realized guess |
| `blind` | only that some guess happened, marginalized under a uniform guess law |
| `composite`, `composite_blind` | the same two, aggregated onto `s = m + phi`, the three-state belief that fixes the reward |
| `factor_m`, `factor_phi` | the two factor predictive vectors of the composition hypothesis |

`action_awareness_ratio` is `joint` MSE over `blind` MSE. Below one means the
residual stream is organized around the belief of an agent that knows what its
guesses did.

The blind baseline marginalizes over guesses rather than assuming the
register-inert kernel. Assuming inertness would be a strawman: an accurate
register report contradicts it outright, and the filter degenerates. The
marginalized baseline is a real competitor, which is why untrained models start
slightly *above* one rather than below it — the earlier, weaker baseline made
the contrast look easier than it is.

Degenerate targets are reported, not silently fitted. At `kappa = 0` the
register never moves, so `_fit_target` marks `factor_phi` `"degenerate"` rather
than reporting a meaningless ratio, and `action_awareness_ratio` is exactly
`1.0` because the two targets coincide by construction.

## Arms

All arms share the gamma-zero clipped-PPO recipe of `mess3_token_guess_cycle_2`
(`lr = 1e-4`, `gamma = 0`, `lambda = 0`, paper-scale residual transformer with
`d_model = 64`, four layers, context ten), with `delay = 1` and 2.5M env steps.
Only the generator changes.

| arm | `kappa` | `epsilon` | previous guess visible | role |
| --- | --- | --- | --- | --- |
| `factoring_free` | 0.7 | 0.0 | yes | factoring is lossless; the orthogonal-subspace test |
| `factoring_cheap` | 0.7 | 0.3 | yes | factoring costs 5% of maximum |
| `factoring_costly` | 0.7 | 0.85 | yes | factoring costs two thirds of maximum |
| `factoring_impossible` | 0.7 | 1.0 | yes | factoring is vacuous; only the composite is predictive |
| `no_feedback` | 0.0 | 1.0 | yes | control; reproduces the passive study, both belief targets coincide |
| `deterministic_feedback` | 1.0 | 1.0 | yes | past guesses pin the register without any report |
| `factoring_free_blind` | 0.7 | 0.0 | **no** | counterfactual; hiding the guess should cost little when the register reports itself |
| `factoring_impossible_blind` | 0.7 | 1.0 | **no** | counterfactual; with no report and no visible guess the register is unrecoverable |
| `composition_theory` | sweep | sweep | n/a | network-free ceilings, factoring cost, single-HMM test |
| `battery` | all | all | mixed | one pass over every condition, for smoke validation |

The two blind arms are the decisive controls. They share dynamics with their
sighted partners, so any difference in action awareness comes from the guess
being readable rather than from the process being different.

> **Correction from the campaign.** The original rationale claimed hiding the
> guess should barely matter at `eps = 0` "where the register announces itself."
> That is wrong under `delay = 1`: the observed report describes `phi` at `t - 1`,
> while the guess must anticipate `phi_t = phi_{t-1} + a_{t-1}`. The report is
> always one step stale, so it never substitutes for knowing the guess — it only
> bounds how far back the agent must remember. The measured accuracy cost of
> blinding is the same at both endpoints (0.107 at `eps = 0`, 0.099 at
> `eps = 1`); only the representational cost differs, by 4.7 times.

## Running

```bash
uv run rl-harness experiments.mess3_feedback_factoring_cycle_1.battery.experiment --smoke
uv run rl-harness experiments.mess3_feedback_factoring_cycle_1.composition_theory.experiment
uv run rl-harness experiments.mess3_feedback_factoring_cycle_1.factoring_costly.experiment --seed 42
python experiments/mess3_feedback_factoring_cycle_1/seed_queue.py \
  --condition factoring_costly --seeds 42 43 44 45 46
```

`seed_queue.py` runs one arm across several seeds sequentially on one GPU box,
pushing compact results after each seed so a mid-queue failure still lands the
completed runs. `MESS3_FEEDBACK_FACTORING_C1_MAX_ENV_STEPS` truncates the budget for
cheaper campaigns while keeping the every-ten-iteration checkpoint cadence.

## Predictions

1. Sighted arms drive `action_awareness_ratio` below one over training; the
   blind arms cannot and stay well above it.
2. Hiding the guess costs little in `factoring_free_blind` and a lot in
   `factoring_impossible_blind`, because the register report substitutes for
   knowing the guess only when it is accurate.
3. Task success tracks each arm's own myopic ceiling, not the passive `0.69`,
   except at `kappa` in `{0, 1}` where the two coincide.
4. In `factoring_free`, `factor_subspace_overlap` falls over training as the
   passive chain and the guess-driven register separate into orthogonal
   subspaces — the Factored World Hypothesis holding for a factor the policy
   itself drives.
5. As `epsilon` rises the factored representation should be abandoned: the
   `joint` target stays decodable while `factor_m` and `factor_phi` degrade
   toward their priors. Whether models pass through a factored stage first,
   as the paper's inductive-bias claim predicts, is visible in the
   checkpoint curves.
6. `marginal_belief_mse` stays well above zero wherever `kappa > 0`, matching
   the network-free finding that the loop is not one stacked HMM.

## Outcomes

The eight-arm, five-seed campaign is written up in
[`results/20260731T190000Z-eight-arms-five-seeds/findings.md`](results/20260731T190000Z-eight-arms-five-seeds/findings.md).
Headline scoreboard:

| prediction | verdict |
| --- | --- |
| 1 — sighted arms drive `action_awareness_ratio` below one | **untestable**; a perfectly factored representation scores 1.58, so predictions 1 and 4 are mutually incompatible at `eps = 0` |
| 2 — hiding the guess costs little at `eps = 0` | **half confirmed**; representational cost differs 4.7x, accuracy cost does not differ at all |
| 3 — success tracks each arm's own ceiling | **confirmed**; 98–99% of ceiling sighted, 81–83% blind |
| 4 — `factor_subspace_overlap` falls in `factoring_free` | **falsified as stated**, but the metric sits at ~0.5 everywhere and needs a null baseline |
| 5 — factor targets degrade as `epsilon` rises | **confirmed**; a clean monotone ladder to full undecodability |
| 6 — `marginal_belief_mse` above zero wherever `kappa > 0` | **confirmed**; exactly zero at `kappa = 0`, 31–90% of target variance elsewhere |

The finding that survives is not on the list. Every arm learns the
reward-sufficient composite belief to R² = 0.93–0.98 regardless of `epsilon`,
and in the arms where the composite is *not* sufficient for the joint it is the
only target that improves — the register factor decodes *better* at
initialization (R² = 0.983) than after 2.5M steps (R² = 0.738). Read alongside
the sibling study, the sign of the change in belief decodability is predicted in
all seven cases by whether the world belief exceeds the reward-sufficient
statistic.

Two reported numbers were artifacts: raw cross-arm MSE is invalid because target
variance differs threefold (the ranking inverts under normalization), and
`action_awareness_ratio` penalizes exactly the structure the study set out to
detect. Use `campaign_analysis.py` for both corrections:

```bash
uv run python -m experiments.mess3_feedback_factoring_cycle_1.campaign_analysis
```
