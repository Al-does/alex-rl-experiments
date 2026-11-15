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
- The early rise (result 1) has no tested mechanism. The context window of 32 tokens may set the
  common .91 level; this was not measured.
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
