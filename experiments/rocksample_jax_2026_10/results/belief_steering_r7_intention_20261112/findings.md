# Is there an intention between belief and action? (RockSample r7, 2026-11-12)

Follow-up to `../belief_steering_r7_twin_20261112/`, prompted by Bush et al. (ICLR 2025),
who find linearly decodable, causally steerable plans in a model-free Sokoban agent. The
question here is narrower: do these policies carry a commitment to "sample rock i next"
that outlives a change of belief, or does each action read the current belief?

Same three upper-plateau agents and two checkpoints (30M, 330M) as the confirmation, and
the same frozen steering protocol (block 1, `pca_all` rank 4, strength 1.25, fit seed
77003). Probe test seed 79031; rollout seed 90217; 512 fit episodes, 1,024 episodes per
onset cell, 512 traced episodes per readout cell. Code revision `90b4c1af`. All numbers
are in `tables.md`; per-agent JSONs carry checkpoint hashes.

## Result

**The policies act on the current belief; we find no evidence of a stored intention.**

1. **Nothing beyond belief and observables is decodable.** A linear readout of the next
   rock to be sampled (eight classes, held-out episodes, at least two steps ahead) reaches
   0.816 / 0.801 accuracy (30M / 330M) after block 3. A linear readout from the
   observation, sampled flags and the *exact* Bayesian belief reaches 0.813 / 0.796;
   observables alone 0.450 / 0.489; majority class 0.18 / 0.21. Accuracy rises with depth
   (0.62, 0.75, 0.82 at 30M).
2. **The readout follows a belief edit.** Steering a Good rock towards Bad from the first
   step drops the share of steps on which the final-norm readout names that rock from
   0.40 to 0.03 (30M; 0.39 to 0.04 at 330M); steering a Bad rock towards Good raises it
   from 0.03 to 0.44 (0.04 to 0.40). Readout accuracy against what the steered rollout
   actually does is unchanged (about 0.80).
3. **A belief change on the Sample step itself cancels the Sample.** Rollouts share every
   random draw with the intact rollout, so they are identical until the onset. Replacing
   the final activation with that of a twin whose evidence about the target was always
   flipped, starting on the very step where the intact agent samples the Good rock, leaves
   the rock sampled in 0.007 (30M; 153/21,049) and 0.003 (330M; 61/21,214) of episodes,
   the intact Bad-rock rate. The action differs on the onset step in 99.5% / 99.8% of
   episodes. There is no lag to explain.
4. **The readout does not lead the action.** With frozen steering switched on four steps
   before the intact Sample, the block-3 readout departs from the intact readout before
   the action does in 0.38 / 0.34 of episodes, on the same step in 0.30 / 0.42, and after
   it in 0.33 / 0.24.

## Mid-episode steering is slower than a belief change, for a different reason

The frozen block-1 edit only touches tokens written after the onset. Tokens already in
the cache keep the original evidence, and blocks 2 and 3 still attend to them.

| Good target sampled; onset `lead` steps before the intact Sample | lead 0 | lead 2 | lead 4 | lead 8 |
|---|---|---|---|---|
| 30M, frozen steering | 0.162 | 0.066 | 0.035 | 0.016 |
| 30M, twin patch at block 1 on new tokens | 0.059 | 0.026 | 0.016 | 0.008 |
| 30M, twin patch of final activation | 0.007 | 0.005 | 0.004 | 0.004 |
| 330M, frozen steering | 0.481 | 0.267 | 0.206 | 0.110 |
| 330M, twin patch at block 1 on new tokens | 0.325 | 0.196 | 0.153 | 0.083 |
| 330M, twin patch of final activation | 0.003 | 0.002 | 0.002 | 0.002 |

- The block-1 twin patch is the ceiling for any edit confined to new tokens at block 1;
  frozen steering sits a little above it. The gap to the final-activation patch is
  evidence held in older tokens, not a plan.
- The 330M agents depend more on those older tokens: a late block-1 edit is much weaker
  there, although steering from the first step is stronger at 330M (0.011 / 0.954).
- At a fixed onset step, Good→Bad under frozen steering stays at 0.02–0.04 at 30M for
  every onset, and rises from 0.02 to 0.23 with later onsets at 330M.
- Bad→Good falls with later onsets for every intervention, including the final-activation
  patch (0.94 → 0.68 at 30M, 0.91 → 0.56 at 330M, steps 2 → 24; intact ≤ 0.013). By then
  the rover has often passed the rock, so this is lost opportunity rather than commitment.

## Which rocks are harder to steer?

Descriptive: seven rocks, one layout, 42 agent-checkpoint-rock rows.

- **Poor belief resolution is not the signal for the hardest rock.** Rock (0,3) has the
  weakest Bad→Good steering (0.760 / 0.788 pooled; 0.497 for one agent at 330M) but its
  exact belief is as resolved as any (final certainty |2m−1| = 0.97).
- **The worst-resolved rock is only mildly harder.** Rock (1,2) ends at certainty 0.83
  (30M) and 0.62 (330M), with final belief 0.17 / 0.38 when it is Bad; its Bad→Good rate
  is 0.886 / 0.968. Final certainty correlates with Bad→Good steering at ρ = +0.37 over
  all rows (+0.58 within agent-checkpoint).
- **What tracks difficulty is how the flipped evidence shows up at the steering site.**
  Rocks whose twin displacement after block 1 is small (ρ = +0.67; (0,3) at 30M: 7% of
  the activation norm against 15–33% for the easy rocks) and poorly predicted by the
  affine map (ρ = +0.59; −0.68 with the shortfall relative to rank-4 twin interchange)
  are the ones steering under-delivers on. Interchange with the true twin in the same
  four directions still reaches 0.965 / 0.975 on (0,3), so the subspace is adequate and
  the loss is in predicting the displacement.
- **Final-norm belief-probe R² is the strongest single correlate** (ρ = +0.85; +0.84
  within agent-checkpoint), while block-1 probe R² is not (−0.13; −0.40). This refines
  the earlier statement that probe R² does not predict steerability: that held across
  sites and for the steering site's own probe.
- Good→Bad sampling is at the intact Bad-rock rate for every rock (0.003–0.028), so it
  has little variation to explain.

## Caveats

- One layout and three agents; intervals cover episodes, not training seeds.
- "No intention" means none that is linearly decodable beyond belief and observables or
  that resists a belief change. A fixed layout does not require planning.
- The final-activation patch is a behavioural switch to the twin policy, not a
  representation-level edit; it bounds commitment, it does not localise belief.
- The next-rock readout is least squares on one-hot labels, not a tuned classifier.
- The readout-versus-action timing uses the block-1 steering site, so the block-1 and
  block-2 rows of that table lead the action by construction.
