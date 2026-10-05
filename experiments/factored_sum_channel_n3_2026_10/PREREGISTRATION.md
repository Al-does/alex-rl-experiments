# Pre-registration: does the Kelly auxiliary loss help on a factored env?

Written before any full run (2026-10-05). Applies to both studies:
`factored_sum_channel_n3_2026_10` (N=3, main) and
`factored_sum_channel_n4_2026_10` (N=4, replication).

## Hypothesis

Adding the decoupled Kelly wager loss to next-token CE (weight 1) helps a
transformer learn the cross-factor correlations of the sum channel, i.e.
move beyond the factored representation that transformers prefer
(Shai et al. 2026, arXiv:2602.02385), and so generalize better to held-out
sequences.

## Design

- Process: N sharp Mess3 factors (alpha=0.85, x=0.05); each step w.p. 1-eps
  the sub-token tuple is emitted, w.p. eps only the sum mod 3. Main
  condition eps=0.5. Control eps=0: exactly factored, same vocab.
- Leaves: h20/h40/h80 x {CE, CE+Kelly} at eps=0.5, plus eps0_h20 x {CE,
  CE+Kelly}. Seeds 0, 1, 2 for every leaf. The CE and Kelly runs of a seed
  share the sequence pool, data order and trunk init (paired comparison).
- Fixed protocol: 400M env steps (6,152 updates of 512 x 127), Adam 1e-3, no
  weight decay, d128 x 4 layers. No tuning after seeing results; any change
  is a new, separately labelled study.
- Reference gap: the factored assumed-density filter loses about 0.007-0.009
  nats/token vs Bayes at N=3 and about 0.005-0.006 at N=4 (exact value is in
  each run's `references.heldout_factored_gap_nats`). This is the room for
  Kelly to help.

## Primary outcome

Final-checkpoint held-out excess CE over the Bayes floor (equivalently
`heldout_factored_gap_closed`, and perplexity ratio exp(excess CE)).
Delta = excess CE(CE) - excess CE(CE+Kelly), paired by seed.

**Kelly helps (supported)** at a given N if all of:
1. At eps=0.5, Delta > 0 for all 3 seeds in at least 2 of the 3 rungs, and
   the mean Delta over those rungs is at least 10% of the factored gap.
2. The effect is not transient: Delta > 0 in the mean over the last 5
   checkpoints too, not just at the final one.
3. The eps=0 control is neutral: |mean Delta| at eps0_h20 is less than half
   the mean eps=0.5 h20 Delta. If Kelly helps equally at eps=0, the result is
   reported as a generic optimization/regularization effect, not evidence of
   learning cross-factor structure.
4. Kelly stays healthy: held-out guess hit rate within 0.01 of the CE arm and
   expected wager not collapsed to 0 (Kelly arm's mean wager > 0.01).

**Kelly does not help (null)**: criterion 1 fails at both N=3 and N=4. That is
a real negative result provided CE itself trains (held-out excess CE well
below the factored gap at h20, or above it but not diverging).

**Kelly hurts**: Delta < 0 for all 3 seeds in at least 2 rungs.

## Secondary outcomes (mechanism; reported, not required for "supported")

At final and across checkpoints, held-out, Kelly vs CE:
- Joint-belief probe 1-R^2 and correlation-residual probe 1-R^2 lower for
  Kelly (it represents the coupling), with the factor-marginals probe
  similar in both arms.
- `factored_gap_closed` higher; Kelly crosses 0 (beats the factored filter)
  earlier in env steps.
- Effective dimension (95% variance) above the factored 2N, nearer the exact
  belief dimension, earlier or more strongly than CE.
- Effect larger at higher held-out fractions (less unique data) would point
  to a regularization benefit; larger at h20 would point to better use of
  data. Either is reported; neither is predicted.

## Replication

N=4 is the replication. "Supported" overall requires the primary criterion
at N=3 and the same sign of mean Delta at N=4 (its gap is smaller, so the
N=4 magnitude criterion is not required).
