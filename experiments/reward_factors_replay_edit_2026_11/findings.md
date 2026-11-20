# Counterfactual replay edits in Joint Reward and Marginal Reward

This study tests seeds 42–44 of the paper ε=0.02 Joint Reward agents and
Marginal Reward agents, plus the prior effective v2 no-ε Joint Reward
agents for XOR. The previous ladder replay prompt specified Reward-State
and Full-State Quotient agents, not reward-factor agents; the earlier
reward-factor XOR experiment used twin writes, not the replay-fidelity
comparison below.

## A. The policy is not strictly factored

Means ± sample SD across the three checkpoints. TV for local edits is
at the query; mirror TV is averaged after warmup and reflects output
action alignment. The exact filter always preserved the other factor;
the factor-2 mirror also preserved its state-3 probability and exchanged
its state-1/state-2 probabilities.

| Factor-2 replay | Factor 1 policy TV | Factor 2 policy TV | Block-4 relative displacement |
|---|---:|---:|---:|
| Lag 0 | .1055 ± .0096 | .3904 ± .0803 | .7211 ± .0986 |
| Lag 1 | .0489 ± .0059 | .2154 ± .0085 | .4305 ± .0077 |
| Lag 32 | .0113 ± .0008 | .0230 ± .0031 | .0938 ± .0075 |
| Lag 128 | <1e-6 | <1e-6 | <1e-6 |
| Global mirror, aligned | .0419 ± .0118 | .0791 ± .0058 | .5508 ± .0204 |

The factor-1 marginal shifted despite an exactly invariant factor-1
posterior. The factor-2 mirror was approximate, not exact. Rank-eight
mirror subspace overlap increased from .390 ± .033 at block 1 to
.533 ± .015 at block 4 and .548 ± .015 after the final normalisation.
Complete positional profiles and high-drop strata are in the seed reports.

## B. Late writes need only the latest position

Selected factor-2 marginal fidelity comparisons:

| Replay | Patched positions | Block 1 | Block 4 |
|---|---|---:|---:|
| Lag 1 edit | Latest | .574 ± .097 | 1.000 |
| Lag 1 edit | Changed past | .265 ± .156 | .000 |
| Lag 32 edit | Latest | .417 ± .158 | 1.000 |
| Factor-2 mirror | Latest | .641 ± .082 | 1.000 |
| Factor-2 mirror | All changed inputs | .939 ± .010 | .942 ± .018 |
| Factor-2 mirror | Changed past only | .240 ± .076 | .000 |

An unchanged latest token/action pair can still carry the replay effect
in its residual, so patching *only changed inputs* can miss that effect.
At lag 128 the replay policy change was too small for fidelity to be
meaningful. The [expanded tables](tables.md) include both factor marginals.

## C. Belief and activation context predict displacement

Factor-2 displacement, representative block-4 held-out (R^2).

| Regressor or subset | Score |
|---|---:|
| Probabilities, no conditioning | .673 ± .025 |
| Quadratic log odds, no conditioning | .696 ± .018 |
| Increment from factoring in the other belief, matched feature family | +.089 ± .014 |
| Increment from activation PCs, matched feature family | +.125 ± .015 |
| Selected model (quadratic log odds with both interactions, all three seeds) | .832 ± .013 |
| Selected, low likelihood drop | .844 ± .006 |
| Selected, high likelihood drop | .766 ± .039 |
| Selected, same latest inputs | .708 ± .022 |

This is held-out evidence that context helps predict the displacement.
It does not by itself establish that the other belief causally determines
the write's success or explains the original joint-vs-factored reward gap.

## D. Reachable targets reduced spillover without eliminating it

| Write | Moving reward | Gap closed | Factor 1 reward | Factor 1 actions changed |
|---|---:|---:|---:|---:|
| Intact | .896 ± .011 | — | .402 ± .011 | .000 |
| Exact-belief map | 1.092 | — | .414 | .081 ± .007 |
| Eq. (4), factored point | .995 ± .008 | .506 ± .032 | .344 ± .019 | .135 ± .036 |
| Eq. (4), joint point | 1.035 ± .003 | .708 ± .023 | .368 ± .004 | .116 ± .011 |
| Eq. (4), joint reachable | **1.043 ± .012** | **.754 ± .048** | .391 ± .006 | .059 ± .010 |
| Encoder, joint point | 1.020 ± .022 | .635 ± .105 | .356 ± .023 | .127 ± .029 |
| Encoder, joint reachable | 1.005 ± .004 | .555 ± .004 | .393 ± .014 | .074 ± .032 |
| Reference patch, joint | .902 ± .020 | .035 ± .061 | .398 ± .007 | .002 ± .003 |
| Output-aligned mirror input | .895 ± .011 | −.005 ± .015 | .398 ± .011 | .022 ± .009 |

The encoder recovered most of the factor-1 reward loss when given the
reachable target, but it was not factor-1-invariant. Reachable targets
also helped the old joint write, which earned the most total reward and
changed factor 1's actions less than the reachable encoder. Its factor-2
reward was .367 in normal phases and .938 in reversed phases; the joint
reachable encoder achieved .305 and .919, respectively. The point
encoder's stronger factor-2 exact-map agreement did not come with reduced
factor-1 spillover.

The moving-reward reference patch is **not** a transfer ceiling. On fit
histories the modal factor-2 action was Noop for every MAP-state bin for
seeds 42 and 44, so under this definition there were no opposite-tilt
matches in either seed. Seed 43 had a Tilt Y mode in the state-2 bin;
its joint reference patch had 10.6% decision coverage. On its supported
histories the joint reachable encoder reproduced .631 of the replay
effect on factor 2; joint Eq. (4) reachable reproduced .660. Both had
negative factor-1 fidelity (farther from the replay than the intact
policy), and the oracle mirrored posterior was .259 TV from the
encoder's decoded factor-2 target on its supported steps. The
output-aligned mirror input earned
near-intact reward; it is an equivariance control rather than a substitute
for the raw patch or the exact-belief action map. See the expanded
tables for every write/target combination, per-phase rewards,
map agreement, and both factor fidelities.

## E. Marginal Reward agents are more selective, not invariant

| Checkpoint | Factor-2 edit | Factor 1 policy TV | Block-4 relative displacement |
|---|---|---:|---:|
| Initialization | Lag 0 | .0156 ± .0035 | .3147 ± .0568 |
| Final | Lag 0 | .0327 ± .0071 | .1330 ± .0118 |
| Initialization | Lag 8 | .0013 ± .0005 | .0165 ± .0029 |
| Final | Lag 8 | .0052 ± .0015 | .0144 ± .0009 |
| Initialization | Lag 32 | .0021 ± .0010 | .0282 ± .0101 |
| Final | Lag 32 | .0013 ± .0010 | .0059 ± .0009 |

Lag-0 factor-1 policy TV was lower than in Joint Reward (.033 vs .106)
but greater than at initialization, even as residual displacement fell.
The rare high-drop lag-0 stratum had factor-1 TV .325 ± .101 (two
seeds with support). This falsifies strict invariance; lower overall
sensitivity does not imply that the nuisance factor is excluded from
the policy.

## XOR: the preferred target depends on the cohort

| Write | Paper ε=0.02 | Effective v2 no-ε |
|---|---:|---:|
| Intact | .4824 ± .0050 | .4949 ± .0029 |
| Exact-belief map | .5626 | .6189 |
| Eq. (4), factored point | .4899 ± .0152 | .5851 ± .0041 |
| Eq. (4), joint point | .4908 ± .0118 | .5626 ± .0132 |
| Eq. (4), joint reachable | .5037 ± .0064 | .5683 ± .0178 |
| Encoder, factored point | **.5102 ± .0278** | .5639 ± .0139 |
| Encoder, factored reachable | .4895 ± .0172 | **.5901 ± .0083** |
| Encoder, joint reachable | .4957 ± .0071 | .5688 ± .0171 |
| Reference patch, factored decoder | .4847 ± .0045 | .5381 ± .0049 |

The effective v2 reachable factored encoder beat the archived factored
point write on all three seeds, by .0050 reward per step on average.
It reproduced .727 and .738 of the available replay effect for factors
1 and 2, versus .474 and .397 for the old factored point write, although
each method generated its own executed history. On the paper agents,
the factored **point** encoder beat the old factored write on two seeds
and tied on the third, but its replay fidelity was not consistently
better, and the reachable encoder was not an improvement. The
paper joint reachable Eq. (4) write was stronger than the corresponding
encoder. Patch support was much lower on paper agents (4.4% average
on the factored patch vs 25.3% for v2), making fidelity a limited
conditional comparison. Held-out block-4 factor-2 (R^2) was .832 ± .013
on the paper cohort and .857 ± .018 on v2; high predictive fit did
not make the same write/target win both cohorts.

The [expanded tables](tables.md) include per-seed archive pairings and
the replay/patch diagnostics for both cohorts. Across both XOR cohorts
and moving reward, the rerun point Eq. (4), intact, random, and
exact-belief-map rewards matched the archived values exactly.

## Protocol

- Replay fits, model selection and held-out analysis use separate greedy
  episode sets (seeds 42/43/44; 24/12/12 episodes; 1,024 steps; 64-step
  warmup). Factor-specific edits switch C to A or B or A/B to C; the
  other token component and both recorded previous action components
  are unchanged. Two-edit cases change the latest token as well as lag
  8 or 32. An independent per-factor filter is checked against the
  recorded joint filter; edit and mirror invariance assertions run for
  every replay. Global mirrors change A/B and Tilt X/Tilt Y together.
- Reports contain position-wise factor marginal total variation (TV),
  residual displacement divided by original activation norm, and
  changed-factor posterior TV. Factor-1 and factor-2 mirror subspace
  overlap is mean squared canonical correlation between the leading eight
  centered displacement axes, not overlap of the decoders. Mirrors are
  also scored with output Tilt X/Tilt Y aligned. Log-likelihood drop is
  averaged from first edit through query for recorded actions, using
  training-temperature policy probabilities (1.5); high drop is at
  least 0.5 nats. Analogous mirror scores compare frame-aligned actions.
- Part B patches the reference residual at the original input's changed
  positions, the latest position, or only the changed past positions.
  It scores each factor as \(1 - \sum \mathrm{TV}(\mathrm{write},
  \mathrm{replay})/\sum \mathrm{TV}(\mathrm{original},\mathrm{replay})\).
  A mean replay TV below 1e-6 has no fidelity reported. At
  earlier blocks a write applied at every step persists in *later*
  layers' cached keys and values. That is an ongoing sequence of current
  writes, such that historical positions can also be written; it is not
  equivalent to a latest-only patch into an intact cache, nor exactly
  to a replay's changed-input mask. At block 4 and post-final norm
  there is no downstream attention cache.
- Part C uses displacement least squares (rcond 1e-6, highest JAX
  matmul precision). The probability family uses two independent
  coordinates of the three-state simplex; the omitted probability and
  its quadratic terms are affine combinations of the included basis.
  Log odds are versus state 3 with probabilities floored at 1e-4.
  Latest-token and previous-action input differences are nuisance
  regressors. The conditional models interact feature changes with
  factor 1's current belief and/or four leading activation PCs, computed
  on fit rows only. The closed-loop encoder holds the nuisance inputs
  fixed. It uses the decoded current belief projected by clipping and
  renormalization, so this comparison does not give the encoder an
  oracle readout.
- Part D uses the archived discounted QMDP map and point anchors,
  60,000 historical-protocol decoder rows, the archived fit/tune/confirm
  random streams, 64 tuning episodes, 256 confirmation episodes, greedy
  actions and the full 1,024-step schedule. The legacy Eq. (4) maps retain
  their original solver and point strengths. Other strengths use the
  historical grid and smallest maximizing strength. Block 4 is the
  manuscript comparator; the latest residual there already suffices to
  transmit the replay effect (Part B). Action change is versus the
  intact policy on the *same executed history*, not versus an
  independently unfolding intact episode.
- Reachable target: use a mirror for an opposite tilt (the current tilt
  is the modal fitted training action for the decoded MAP state).
  Otherwise choose the nearest belief in Euclidean distance from the
  observed 25th/50th/75th state-3-probability quantile examples among
  histories that took the requested action. If no such example exists,
  keep the decoded belief. Moving reward preserves factor 1's target.
  XOR edits both factors, preserves a factor when the requested action
  already agrees, and symmetrizes the order of the two encoder edits
  when both are applied.
- A reference patch is available on mirror-supported decisions only.
  A persistent auxiliary cache evaluates the mirrored inputs, sharing
  executed actions relabeled into each frame. The raw replay policy
  measures what a perfect block-4 patch can reproduce. The separate
  mirror-input reference maps the output back to the environment frame;
  for an equivariant policy this can cancel the input mirror rather than
  reverse behavior. Moving reward selects that frame in reversed
  phases; it preserves state-3 probability and cannot implement the
  state-3-to-tilt switch. XOR has no such phases; we use the requested
  opposite-tilt frame as an equivariance control, not a theoretical
  reward ceiling. No exact-match replay patch/fidelity is assigned to
  empirical prototype targets. The point write's fidelity is against
  the available mirror replay on the same rollout, **not** an
  exact point-mass replay. Coverage and distance from the exact mirrored
  posterior to the decoded reachable target are reported separately.

## Reproduction

```sh
python -m experiments.reward_factors_replay_edit_2026_11.run --pilot --seed 42
python -m experiments.reward_factors_replay_edit_2026_11.run --workers 3
python -m experiments.reward_factors_replay_edit_2026_11.summarize
```

No retraining, existing study edits, 15-seed run, moving transfer on
Marginal Reward or v2 agents, exact point-mass reference, or myopic-map
transfer was run. The manuscript's primary long-run map was used.
