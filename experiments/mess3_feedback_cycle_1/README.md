# `mess3_feedback_cycle_1` — when the guess changes the process

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

## The feedback rule

Write `C` for the forward cyclic shift on `Z_3` and `T` for the passive MESS3
transition matrix. Guess `a` executes

```text
U(a) = T @ R(a),    R(a) = (1 - kappa) I + kappa C^a
```

so guess `0` leaves the process alone, guess `1` rotates the hidden state one
step, and guess `2` rotates it two steps, each firing with probability `kappa`.
`kappa` is the only manipulated variable in the study.

The reward is unchanged from cycle 2 (one point for naming the current token),
so the myopic optimum is still "guess the most likely token". Only the executed
kernel depends on the guess.

### Why a cyclic shift

`T` and the MESS3 emission matrix `E` are both circulant, and so is every
`R(a)`. Everything therefore commutes, and the executed state splits exactly
into two parallel parts:

```text
s_t = m_t + Phi_t (mod 3),   x_t = u_t + Phi_t (mod 3)
```

where `(m, u)` is an untouched passive MESS3 chain with its own token, and
`Phi` is a `Z_3` register whose increments are driven only by the agent's own
guesses. This is precisely the composition of generators studied in
[*Transformers learn factored representations*](https://arxiv.org/pdf/2602.02385)
(arXiv:2602.02385), with one difference that makes it an RL question: **the
policy, not the generator, drives the second factor.**

`kappa` turns out to be an exact analogue of that paper's `epsilon` knob
between decomposable and indecomposable generators, verified numerically in
`composition_theory`:

| `kappa` | Register `Phi` | Joint belief | Factored `(b_m, b_Phi)` |
| --- | --- | --- | --- |
| `0` | frozen at zero | product state | lossless, and trivial |
| `0 < kappa < 1` | hidden walk | off the product manifold | **maximally lossy** |
| `1` | a known function of past guesses | product state | lossless and non-trivial |

The interior case is unusually sharp. Only the sum `m + Phi` is ever observed,
so each factor marginal sits at exactly its uniform prior (register entropy
`log 3` to machine precision) while the joint belief stays fully informative.
A transformer that follows the Factored World Hypothesis into a product
representation here would learn something with *zero* predictive content; the
only useful representation is the composed state.

At `kappa = 1` the composition is lossless and non-trivial, so this is the arm
where the paper's geometric prediction can actually be tested: the passive
MESS3 factor and the agent-driven `Z_3` register should occupy orthogonal
subspaces of the residual stream. `analysis.subspace_overlap` measures this.

## Difficulty is an inverted U in `kappa`

Because a deterministic rotation is just the passive process viewed in a
rotating frame, `kappa = 0` and `kappa = 1` have *identical* Bayes ceilings.
The difficulty peaks in between, where the agent cannot tell whether its own
shift fired. Measured by `composition.myopic_ceiling` (seed 42, 384 chains of
3072 steps):

| `kappa` | 0.0 | 0.1 | 0.2 | 0.3 | 0.35 | 0.4 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ceiling | .6897 | .6494 | .6183 | .5920 | .5773 | .5690 | .5903 | .6128 | .6352 | .6568 | .6898 |

Both endpoints reproduce the exact passive optimum of `0.68958` computed by
`mess3_token_guess_cycle_2.analysis.bayesian_optimal_accuracy`, which is the
main correctness check on the whole construction. A ten-step context is
already effectively infinite memory here: the 10-window and exact-filter
ceilings agree to four decimals at every `kappa`.

`kappa = 0.5` is excluded from the grid. With exact argmax tie-breaking it is a
measure-zero degeneracy in which the loop drives itself into a near-uniform
belief and accuracy collapses to `0.43`.

## Does the closed loop collapse to a single HMM?

The motivating conjecture: once the policy is fixed, maybe the "underlying HMM
plus guesses that change it" can be rewritten as one autonomous HMM whose
transition matrix stacks the guess-conditioned kernels and renormalizes them,

```text
Ubar[s, .] = sum_y P(guess = y | state = s) U(y)[s, .].
```

`composition.single_hmm_report` builds `Ubar` from the realized `(state, guess)`
statistics and compares the exact distribution over four-token blocks under
`(Ubar, E)` against the closed loop's empirical distribution, with a
sampling-noise floor from splitting the chains in half.

**The answer is no for every `kappa > 0`, but the conjecture's own reasoning
points the right way.** Excess total variation above the sampling floor
(seed 42, 384 chains of 3072 steps):

| `kappa` | 0.0 | 0.2 | 0.35 | 0.6 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| myopic argmax | -0.002 | 0.058 | 0.109 | 0.069 | 0.110 | 0.184 |
| probability matching | -0.006 | 0.026 | 0.039 | 0.039 | 0.037 | 0.060 |
| sampling floor | 0.005 | 0.005 | 0.006 | 0.006 | 0.005 | 0.007 |

At `kappa = 0` the residual is negative, meaning the stacked HMM is exact to
within sampling noise, as it must be. Everywhere else it is five to thirty
times the floor.

The reason is structural. `Ubar` is exact only when the guess is conditionally
independent of the hidden state given the state itself. A Bayes agent chooses
from its *belief*, which is correlated with the state through the shared
history, so marginalizing the guess at the level of the state throws away that
correlation. Emitting the token as per the belief vector's distribution — the
probability-matching policy in the conjecture — does cut the residual by
roughly three to four times relative to greedy argmax, because probability
matching randomizes the guess and weakens the state–guess coupling. It does
not remove it.

The trained arms carry the same test at the level of beliefs:
`marginal_belief_mse` in every `probe_metrics.json` is the gap between the
stacked-HMM filter and the exact guess-conditioned filter along the learned
policy's own trajectories.

## Measurement

Every checkpoint is probed with one affine read-out per target, all fit on the
same held-out activations (post-final-LayerNorm, greedy rollouts, disjoint
train and test seed streams):

| target | meaning |
| --- | --- |
| `executed` | exact guess-conditioned Bayesian belief; the transducer target of `rl-harness`, `K(x \| a) = diag(P(x \| s)) U(a)` at delay one |
| `blind` | the same filter run as if every guess executed the passive kernel |
| `marginal` | the stacked single-HMM filter under `Ubar` |
| `joint` | the nine-state `(m, Phi)` filter |
| `factor_m`, `factor_phi` | its two factor marginals |

`action_awareness_ratio` is `executed` MSE over `blind` MSE. Below one means
the residual stream is organized around the belief of an agent that knows what
its guesses did.

Read the absolute ratio with care: an untrained transformer already carries the
previous guess in its residual stream simply because the guess is part of the
observation, so the ratio starts below one in the sighted arms. The
training-driven quantities are the change from the initial checkpoint
(`training_change.action_awareness_ratio_delta`) and `fine_mse_ratio`, which
conditions on the two most recent `(token, guess)` pairs and therefore measures
integration beyond what the newest inputs hand over for free.

Degenerate targets are reported, not silently fitted: for `0 < kappa < 1` the
factor marginals are constant, so `_fit_target` marks them `"degenerate"`
rather than reporting a meaningless ratio.

## Arms

All arms share the gamma-zero clipped-PPO recipe of `mess3_token_guess_cycle_2`
(`lr = 1e-4`, `gamma = 0`, `lambda = 0`, paper-scale residual transformer with
`d_model = 64`, four layers, context ten), with `delay = 1` and the previous
guess observable. Only the environment changes.

| arm | `kappa` | previous guess visible | role |
| --- | --- | --- | --- |
| `no_feedback` | 0.0 | yes | control; reproduces cycle 2, and both belief targets coincide |
| `weak_feedback` | 0.35 | yes | near the difficulty peak; maximally indecomposable |
| `strong_feedback` | 0.70 | yes | ignoring the guess costs nearly the whole target variance |
| `full_feedback` | 1.0 | yes | lossless composition; the arm for the orthogonal-subspace test |
| `strong_feedback_blind` | 0.70 | **no** | ablation: identical dynamics, guess unreadable |
| `awareness_contrast` | 0.70 | both | paired truncated run of the two arms above |
| `composition_theory` | sweep | n/a | network-free ceilings, single-HMM test, factorization loss |
| `battery` | all | mixed | one pass over every condition, for smoke validation |

`strong_feedback` against `strong_feedback_blind` is the decisive control. The
dynamics are identical, so any difference in action awareness comes from the
guess being readable rather than from the process being different.

## Running

```bash
uv run rl-harness experiments.mess3_feedback_cycle_1.battery.experiment --smoke
uv run rl-harness experiments.mess3_feedback_cycle_1.composition_theory.experiment
uv run rl-harness experiments.mess3_feedback_cycle_1.strong_feedback.experiment --seed 42
uv run rl-harness experiments.mess3_feedback_cycle_1.awareness_contrast.experiment --seed 42
```

`MESS3_FEEDBACK_C1_MAX_ENV_STEPS` truncates the 2.5M-step budget for remote
campaigns while keeping the every-ten-iteration checkpoint cadence.

## Predictions

1. Sighted arms drive `action_awareness_ratio` further below one over training,
   and drive `fine_mse_ratio` down; the blind arm cannot and stays above one.
2. Task success tracks the `kappa`-specific myopic ceiling, not the passive
   `0.69`, except at `kappa` in `{0, 1}` where the two coincide.
3. At `kappa = 1`, `factor_subspace_overlap` falls over training as the passive
   MESS3 factor and the guess-driven register separate — the Factored World
   Hypothesis holding for a factor the policy itself drives.
4. For `0 < kappa < 1`, no factored representation is available at all, so the
   `joint` target should become decodable while the factor targets stay
   degenerate.
5. `marginal_belief_mse` stays well above zero wherever `kappa > 0`, matching
   the network-free finding that the loop is not one stacked HMM.
