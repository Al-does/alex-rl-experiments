# Drivers of within-class belief resolution in the v2 Belief-Quotient Ladder

Reviewer 1 (W2, Q1) asks how much of the belief is affinely decodable inside one greedy-action
class, compared with initialization. W3(i) asks what a slight break of the state-1/state-2
symmetry does. This study asks what determines the within-class answer in the
`persistent_nuisance_v2` environment of PR #219.

All numbers are descriptive affine accessibility (held-out, class-centred R2). They are not
evidence that the policy uses a coordinate. Within a class the two free belief coordinates are
`b3` (reward state) and `b1 - b2` (which non-reward state).

## Design

- **Family.** Tilt X = (+1, lam, -1) and Tilt Y = (-1, -lam, -1) in log-odds direction for states
  (1, 2, 3). `lam = 1` is exactly the Reward-State Quotient and `lam = -1` exactly the Full-State
  Quotient (tested against the PR #219 kernels). `kappa` shifts the probability of token C by
  `-kappa` in state 1 and `+kappa` in state 2 at `lam = 1`; this is the reviewer's emission perturbation.
- **Runs.** PR #219 recipe (transformer, width 64, 4 layers, context 32, temperature 1.5, 2B steps,
  epsilon 0, entropy 0). `lam` in {1, -1}: seeds 43-48. `lam` in {.75, .5, 0, -.5} and `kappa` in
  {.05, .1}: seeds 43-45. 30 new runs, all completed. The PR #219 seed-42 Trivial, Reward-State and
  Full-State runs are probed with the same code and pooled with the endpoints (7 seeds each).
- **Offline reference** (`theory.py`, no training). Belief-MDP value iteration gives the optimal
  reward and the reward of a filter that never separates states 1 and 2. Their difference is the
  reward value of `b1 - b2`.
- **Probe** (`probe.py`). PR #219 within-class protocol: 60,000 fit and 80,000 test rows from the final
  greedy policy, matched initialization replayed on the same histories, per-class affine ridge.
  Trajectories replay 14 saved checkpoints on the final policy's histories and classes.

## Results

Full table: `results/dose_table.md`. Figure: `results/fig_within_class_symmetry_dose.pdf`.
Ranges are over seeds.

| arm | reward lost without `b1 - b2` | `b1 - b2` at init | peak in training | final | final - peak | gain over init |
|---|---|---|---|---|---|---|
| Trivial (1 seed, one class) | 0 | .841 | .873 | .855 | -.018 | +.014 |
| `lam=1` Reward-State (7) | 0 | .876 | .916 | .826 | -.088 [-.201, -.037] | -.050 [-.143, +.026] |
| `lam=.75` (3) | .1% | .869 | .905 | .781 | -.125 [-.243, -.058] | -.088 [-.189, -.018] |
| `lam=.5` (3) | .3% | .859 | .889 | .799 | -.089 [-.130, -.068] | -.060 [-.119, -.025] |
| `kappa=.05` (3) | .6% | .873 | .913 | .818 | -.092 [-.204, -.027] | -.055 [-.161, +.003] |
| `lam=0` (3) | .8% | .861 | .903 | .887 | -.017 [-.037, -.001] | +.026 [+.022, +.029] |
| `kappa=.1` (3) | 3.7% | .883 | .924 | .906 | -.018 [-.035, -.005] | +.023 [+.018, +.026] |
| `lam=-.5` (3) | 26.5% | .845 | .909 | .909 | -.001 [-.002, .000] | +.064 [+.049, +.082] |
| `lam=-1` Full-State (7) | 49.6% | .789 | .909 | .906 | -.003 [-.010, .000] | +.117 [+.097, +.145] |

1. **Every run builds `b1 - b2` resolution early, whatever its reward value.** In all 33 runs it rises
   from initialization to a peak of .87-.93 (all 32 two- and three-class runs: .89-.93).
   In the arms where it later falls, the peak is at 100-300M steps.
2. **Whether it is kept follows the reward value of `b1 - b2`.** Where ignoring it costs at most
   .6% of reward, every one of 16 runs ends below its peak by at least .027 (arm means -.09 to
   -.13) and 13 of 16 end below their own initialization. Where it costs 26-50%, all 10 runs hold
   the peak. The two arms at .8% and 3.7% sit in between (-.02). The grow-then-compress pattern
   from the parent session therefore replicates for the Reward-State Quotient (7/7 seeds) and does
   not occur in the Full-State Quotient (0/7).
3. **`b3` is added and kept** in all 29 two- and three-class runs outside `kappa=.1` (+.01 to +.06
   over init). `kappa=.1` is mixed (+.015, -.019, +.016). `b3` carries reward in every condition.
4. **Location.** The `b1 - b2` gains and losses are the same with the two policy-head directions
   removed (head-blind column). The logits alone carry it only where it is control-relevant
   (trained logits R2 .75-.83 for `lam <= -.5`; .00-.64 elsewhere, highest at `lam=0` and `kappa=.1`).
5. **Slight symmetry breaking (W3(i)).** At `lam=.75`, `lam=.5` and `kappa=.05` the trained network
   behaves like the exact Reward-State Quotient: `b1 - b2` is compressed, `b3` is kept. It tracks the
   approximate quotient, not the full belief. Retention starts between `kappa=.05` and `kappa=.1`,
   and between `lam=.5` and `lam=0`.
6. **Controls.** History-shuffle, random-input and label-permutation within-class R2 for `b1 - b2`
   lie in [-.03, +.04] in all runs.

## What this says about the three rungs

- **Reward-State Quotient.** The within-class number is a mix of a reward coordinate that training
  sharpens (`b3`, to .99) and a nuisance coordinate that is built early and then let go
  (`b1 - b2`). Full-belief gain over init is therefore near zero or negative (-.020 mean) and varies
  by seed and by when training stops. PR #219 seed 42 (-.012) was the mildest of the first three
  seeds; the 7-seed range is -.143 to +.026.
- **Full-State Quotient.** Both coordinates are kept, so the full-belief gain is positive in 7/7 seeds
  (+.061). Part of the larger `b1 - b2` gain is a lower initialization baseline on Full-State
  histories (.79 against .88); the trained level (.91) equals the Reward-State peak.
- **Trivial Quotient.** One action class, one seed. `b3` rises from .913 to .996 by 300M steps and ends
  at .981 although reward is saturated by 20M steps and no coordinate has reward value. This
  resolution is not explained by reward value.

## Hypotheses

- **Grows during learning, then decays:** supported for control-irrelevant coordinates only (result 2).
- **Exploration pressure:** not tested here; all runs use epsilon 0 and entropy 0. The compression
  occurs without any exploration term. PR #229 remains the evidence on exploration.
- **Why richer than the action map:** two separate effects. (a) An early, reward-independent rise
  in both coordinates (result 1, Trivial). (b) Long-run retention of coordinates that change
  achievable reward, which includes `b3` inside a class because the policy boundary is a `b3` threshold.
- **Excess entropy / predictive sufficiency:** not supported as the driver of retention.
  `b1 - b2` predicts future tokens in every arm, and offline it improves prediction of the next
  optimal action beyond `b3` at `lam=.75` and `.5` (14-19% of residual variance), yet it is compressed there.
  Across the 8 arms, rank correlation with the change in `b1 - b2` is .76 (gain) and .69
  (final - peak) for reward value, against -.29 and -.19 for next-action predictability.

## Early rise: diagnostics

`early_rise.py` replays ten checkpoints of nine runs (Trivial, Reward-State and Full-State seed 42;
`lam=1` and `lam=-1` seeds 43-44; `lam=.75` and `kappa=.05` seed 43) on the final policy's fixed
histories and writes `results/early_rise.json`. Inputs, classes and targets are identical across
checkpoints, so a change in R2 is a change in the weights. Exploratory: nine runs, no intervention.

Timing, from `results/dose_trajectory.json` (all 33 runs): `b1 - b2` peaks above its
initialization value in 33/33 runs (mean +.058, minimum +.015). In 17 of the 19 runs with
`lam >= .5` or `kappa > 0`, the peak falls while the archived policy evaluation still shows one
action and flat reward; `b3` resolution rises over the same checkpoints (median within-run correlation of
checkpoint-to-checkpoint changes .64).

| Measurement (within class, `b1 - b2`) | Reward-State-like (5 runs) | Full-State (3 runs) | Trivial (1 run) |
|---|---|---|---|
| Affine probe on raw one-hot inputs, last 32 / 64 steps | .83-.87 / .86-.91 | .83-.84 / .85 | .80 / .82 |
| Initialized network, affine | .86-.88 | .77-.81 | .84 |
| Initialized network, degree-2 probe | .90-.91 | .85-.88 | .87 |
| Peak, affine | .91-.92 | .90-.92 | .87 |
| Peak, degree-2 probe | .93-.95 | .94-.95 | .93 |
| Conditional skew `(b1-b2)/(b1+b2)`, init to peak | +.03 to +.05 | +.07 to +.08 | +.01 |
| A-B input-embedding distance, init to final | -9% to -13% | -2% to +1% | -1% |

- **Not a probe artifact.** Fixed histories; ridge-free probes agree to .002; the
  policy-head-blind stream equals the full stream to .004.
- **Not only linearization.** A degree-2 probe on the initialized network already reaches the
  affine peak level in Reward-State-like runs, but the degree-2 score rises by a similar amount.
- **Reward-State-like runs: real `b1`-versus-`b2` evidence integration.** The conditional skew,
  which removes the `1 - b3` factor from `b1 - b2`, rises with `b1 - b2`. Linear readability of the
  previous token (A versus B, and non-C versus C, alike) rises at the checkpoints where `b3`
  becomes resolved. Logit `b1 - b2` R2 does not rise consistently (two of five runs).
- **Trivial: mostly the `b3` factor.** The skew moves +.01 (3,472 rows with `b1 + b2 > .1`) while
  `b3` goes .913 to .996 and `b1 - b2` .841 to .873. Logit `b3` R2 rises .74 to .89 although the
  greedy action never changes; offline, 59% of within-class action-value-gap variance needs `b3`.
- **Full-State:** the rise coincides with reward take-off and with logit `b1 - b2` R2 (.23-.35 to
  .78-.82); it is reward-driven.
- **Compression signature:** the A and B input embeddings converge only in Reward-State-like runs.

Reading: the rise in Reward-State-like arms is consistent with a by-product of learning `b3`
(the shared events are non-C tokens, whose identity carries the `b1`-versus-`b2` evidence). This is
not established; the direct test is to train the same initialization on `b3` regression alone and
check whether `b1 - b2` rises by the same amount.

## Is `b1 - b2` a short-window token statistic?

`window_filter.py` rebuilds the Bayes filter from only the last `k` (token, previous action)
inputs, starting `k` steps back from the mean belief on the fit histories (a uniform start changes
the numbers below by at most .005). `k = 32` is one attention band; the four-layer network can
reach back 128. On the same nine runs and fixed histories it splits each coordinate into a
**near** part (the 32-input filter) and a **far** part (full filter minus the 32-input filter),
and probes the residual stream of ten checkpoints for both. Results: `results/window_filter.json`.

The filter itself, no network (share of within-class variance the 32-input filter misses):

| | Reward-State-like (5 runs) | Full-State (3 runs) | Trivial (1 run) |
|---|---|---|---|
| `b3` | .001-.004 (`kappa=.05`: .023) | .011-.014 | .000 |
| `b1 - b2` | .05-.08 | .32 | .09 |
| `b1 - b2` at 64 / 128 inputs | <.01 / .000 | .04 / .000 | .006 / .000 |

Network probes, within class, `b1 - b2` (initialization, peak of the full target, final):

| | Reward-State-like (5 runs) | Full-State (3 runs) | Trivial (1 run) |
|---|---|---|---|
| Near part | .87-.91, .92-.93, .68-.90 | .79-.82, .85-.87, .85-.87 | .87, .91, .89 |
| Far part | .45-.50, .41-.50, .06-.19 | .60-.61, .54-.56, .54-.56 | .35, .30, .27 |
| Far part from the 32 raw inputs and the near belief | -.01 to .04 | .64-.66 | .04 |
| Full minus near score, initialization to final | -.01 to -.02, -.00 to -.03 | -.01 to -.02, +.04 to +.06 | -.03, -.04 |

- **The early rise is a gain in the 32-input statistic.** In Reward-State-like and Trivial runs
  the near part rises by +.03 to +.05 to the peak while the far part does not rise (-.06 to +.01).
  The rise is therefore not the network learning to reach further back than one band.
- **The untrained network already carries the far part** (.45-.50 where the last 32 inputs carry
  none of it), so the stacked bands pass older evidence through at initialization.
- **Compression removes the far part first and most.** It falls to .06-.19 in all five
  Reward-State-like runs, starting at the peak. The near part also falls by about .2 in three
  of the five (`lam=1` seed 43, `lam=.75`, `kappa=.05`), so the final Reward-State `b1 - b2` is
  not simply "the 32-input statistic, intact".
- **Full-State matches the full filter better than the 32-input one after training** (full minus
  near +.04 to +.06; negative at initialization and in every other run at every checkpoint).
- **Full-State far part is confounded.** It does not rise with training (.60 to .54-.56), and
  about two thirds of it is already linear in the last 32 inputs, because the previous-action
  inputs record what the policy believed earlier (previous actions alone: .32 in Full-State
  seed 42, -.00 in Reward-State seed 42). "Full-State integrates over all four bands" is therefore
  not shown by the far-part probe; the full-versus-near comparison is the usable evidence.

Exploratory: nine runs, fixed final-policy histories, no intervention.

## Limits

- Reward values of .1-.8% (`lam=.75`, `.5`, `0`, `kappa=.05`) are smaller than their Monte Carlo
  standard error (about .005 absolute); they are not distinguishable from zero or from each other.
  The clear separation is zero-ish against 3.7% and above. `lam=0` retains despite this.
- Three seeds for intermediate arms. Compression size is strongly seed-dependent (two clusters,
  near 0 and -.08 to -.19 relative to init).
- The peak is the maximum of 14 checkpoints chosen after the fact, so final - peak is biased
  downward; the bias is at most .01 judging by the arms that hold (-.001, -.003).
- Five runs fall below 95% of the belief-MDP reward on the 80,000 CPU test rows (open markers
  in the figure; SE about .013). Without them every arm-mean gain keeps its sign, but `kappa=.05`
  moves to -.002 (2 seeds) and `lam=.75` has one seed left (-.056).
- The action-value contrast target has the same within-class R2 as `b3` to three decimals in
  every run, because `b3` dominates its within-class variance. It is not separate evidence.
- The early rise (result 1) has diagnostics but no intervention; see the next section.
- 20 of the 30 run manifests record a dirty tree at commit `5fcd526a`. These are the second and
  third run on each box; the first run on every box is clean. The likely cause is the previous
  run's untracked `results/` directory; this was not checked on the boxes.

## Provenance

- Training: `experiments/belief_quotient_v2_symmetry_dose_2026_11/<arm>/results/<run-id>/`,
  experiment commit `5fcd526a0ff7cff04e44ab56667e8a8419a6e0db`, harness
  `a267508fdeb523e95b3ef3fc42119102540bbe38`, 10 on-demand Vast RTX 4090 boxes (all destroyed).
- Checkpoints: B2 prefixes in each `run_manifest.json`; the probe verifies byte count and SHA-256
  before use. Per-run hashes and source revisions are in `results/dose_probe.json`.
- Regenerate: `python -m experiments.within_class_drivers_2026_11.theory`, then `.probe --trajectory`,
  then `.report` (with `uv run --group jax`).
