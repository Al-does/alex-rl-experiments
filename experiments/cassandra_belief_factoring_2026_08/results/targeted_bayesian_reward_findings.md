# Targeted Cassandra: exact-belief reward analysis

## Scope

This analysis uses the primary study recipe:

- targeted component actions;
- a uniform initial distribution over all \(4^4=256\) states;
- an undiscounted 1,000-step episode return;
- either the next symbol and previous action only, or those features plus the
  previous reward as a `float32` scalar.

The result JSON was produced with 256 independent tuning episodes per
inspection interval and 2,048 held-out evaluation episodes per observation
variant.

## Exact model and Bayesian recursion

For component conditions \(c_i\in\{0,1,2,3\}\), operating has immediate reward

\[
R_{\mathrm{op}}(s)=\prod_{i=1}^4q(c_i),\qquad
q=(0,\ 0.7275,\ 0.9440,\ 0.9985).
\]

Targeted inspect, repair, and replacement cost \(-1\), \(-0.75\), and
\(-3.75\), respectively. Repair changes bad to fair or fair to good with
probability 0.8, cannot fix broken, and leaves good unchanged. Replacement
deterministically sets the selected component to good.

Without reward in the policy observation, the exact update after action \(a\)
and symbol \(o\) is

\[
b_{t+1}(s')=
\eta\,O(o\mid s',a)\sum_s T(s'\mid s,a)b_t(s).
\]

With previous reward observed, reward is generated from the pre-transition
state, so the correct update is

\[
b_{t+1}(s')=
\eta\,O(o\mid s',a)
\sum_s \mathbf{1}\{R(s,a)=r\}T(s'\mid s,a)b_t(s).
\]

The exact finite-horizon solution is the belief-state Bellman recursion

\[
V_h(b)=\max_a\left[
b^\top R_a+\sum_zP(z\mid b,a)V_{h-1}(\tau(b,a,z))
\right],
\qquad V_0=0.
\]

Here \(z=o\) without reward and \(z=(r,o)\) with reward. This is a closed-form
recursive characterization, but not a tractable elementary expression at
horizon 1,000: exact alpha-vector enumeration grows exponentially over the
continuous 255-dimensional belief simplex.

## Reward ceilings

Operating an all-good machine pays

\[
0.9985^4=0.9940134865
\]

per step, giving the naive no-degradation ceiling
\(994.0134865\) over 1,000 steps. It is unattainable: operating degrades each
non-broken component with probability 0.03, while maintenance consumes a step
and has negative reward.

Exact finite-horizon dynamic programming with the true state visible gives a
stronger attainable ceiling of **759.4289**. Even this clairvoyant controller
operates for only 868.56 expected steps, repairs for 130.43, and replaces for
1.00.

Upper bounds that preserve the first one or two partially observed decisions
and then reveal the state are:

| Information available to the real decisions | Upper bound |
|---|---:|
| State revealed after the first action | 758.7254 |
| No previous reward; state revealed after the next action | 758.0190 |
| Previous reward visible; state revealed after the next action | 758.0506 |

Therefore a Bayesian POMDP controller cannot reach either 994.01 or the
fully-observed 759.43 maximum.

## Feasible exact-belief policy

For a reproducible lower bound, the analysis maintains the exact 256-state
posterior, inspects periodically, and chooses other actions from posterior
expectations of the finite-horizon MDP action values. The inspection interval
was selected on separate random streams; both variants selected 16 steps.

| Policy observation | Mean return | 95% Monte Carlo CI | Mean reward/step |
|---|---:|---:|---:|
| No previous reward | **444.8812** | [443.2390, 446.5235] | 0.4449 |
| Previous reward visible | **505.0350** | [503.5535, 506.5165] | 0.5050 |

Previous reward improves this feasible return by **60.1538** points
(13.5% of the no-reward return) and lowers mean posterior entropy from 3.456
to 1.832 nats.

The reward-aware policy does not become omniscient. A zero operating reward
reveals that at least one component is broken but not which component.
Nonzero rewards identify one of 15 permutation-invariant quality multisets,
not the component labels. Targeted actions still require labeled information,
so both selected policies inspect on 5.9% of steps.

## Is uncertainty too high to justify targeted maintenance?

No. The exact-belief policies actively pay for maintenance:

| Previous reward? | Operate | Inspect | Repair | Replace |
|---|---:|---:|---:|---:|
| No | 82.35% | 5.90% | 10.58% | 1.18% |
| Yes | 82.55% | 5.90% | 10.77% | 0.78% |

At repair decisions, the posterior probability that the selected component is
bad or fair is 79.2% without reward and 90.4% with reward. At replacement
decisions, the selected component is believed broken with probability 76.0%
and 82.9%, respectively. The reward signal makes targeting more certain and
reduces unnecessary expensive replacements.

In the optimistic case where the other three components are good, maintenance
cost breaks even after approximately:

- 3.77 future operations for replacing broken;
- 13.90 for replacing bad;
- 69.12 for replacing fair;
- 4.35 for repairing bad;
- 17.28 for repairing fair.

Those horizons are short relative to a 1,000-step episode. By contrast, the
exact expected return from never maintaining (always operate) is only
**3.8973** from the uniform start, where the initial probability of at least
one broken component is \(1-(3/4)^4=68.36\%\).

## Interpretation

The exact Bayes-optimal returns remain between the feasible means above and
the corresponding 758.02/758.05 upper bounds; this analysis does not pretend
the periodic-inspection policy closes that gap. It does establish the two
claims needed for the reward question:

1. stochastic degradation and maintenance costs make the nominal maximum
   unattainable even with full state information;
2. uncertainty is not high enough to make targeted repair or replacement
   irrational. Exact posterior information supports frequent repair and
   selective replacement, and exposing previous reward makes those decisions
   substantially more profitable.
