# Findings: exploration pressure vs within-class belief resolution

**Hypothesis tested.** More exploration pressure makes the trained encoder resolve belief *within* fixed
greedy-action regions, beyond what the action head needs.

**Design.** v2 persistent-nuisance environment, transformer, T = 1.5, 2B steps, seed 42. Reward-State
(variant 2) and Full-State (variant 3) Quotients were trained at ε ∈ {0, .05, .075, .1} with no entropy bonus,
and at a constant entropy coefficient ∈ {.002, .004, .008} with ε = 0. The ε = 0 and ε = .05 arms are the
PR #219 runs, re-analyzed with the same pipeline. The analysis is the parent within-class protocol
(`within_class_belief_probes_2026_11`), unchanged. Δ is the final encoder minus the initialized encoder,
both scored on the final policy's exact histories, classes and targets. Full tables are in
[`results/README.md`](results/README.md).

## Reward gate (all final policies pass)

A belief-resolution comparison is only meaningful between policies that solve the task, so every policy was
scored against a reward reference (`reward_reference.py`, `reward_gate.py`).
- **Reference:** belief-MDP value iteration on a simplex grid, with the resulting policy simulated using
  the exact filter and the probe protocol (4,096 episodes, 1,024 steps, 64-step warmup excluded). It earns
  .5334 reward/step for Reward-State and .4903 for Full-State. The grid/discount sweep converged, but this
  is near-optimal, not a certified Bayes bound. For scale: best current-token-only policy .348 / .308,
  Noop forever .176.
- **Final policies** (greedy, 4,096 fresh episodes, common random numbers across arms): Reward-State
  .518–.527 (97.0–98.9% of reference), Full-State .471–.480 (96.0–97.9%). All 14 pass a 95% gate; Full-State
  ε = .1 is the closest (lower CI .947 once reference noise is included). Within each quotient every arm is
  within .010 reward/step of the best arm, so the final-checkpoint comparisons below are between competent
  policies.
- **Reward does not explain the Reward-State entropy loss.** The entropy arms have the *highest* reward
  (98.9%) and the lowest Δ (see [`results/reward_vs_gain.png`](results/reward_vs_gain.png)).
- **Checkpoints:** each sampled checkpoint was screened with its 84-episode probe reward; those ≥ 80% of
  reference were re-scored with 1,024 fresh episodes, and the gate is again 95%. Reward-State checkpoints
  through 100M steps (219M for most arms) earn .32–.41 of reference, near Noop level, and fail. Their ≈ .94
  within-class R² is not evidence about a competent policy. The first passing sampled checkpoint is 219M–498M
  for most arms, 1.1B–1.4B for Full-State ε ≥ .05 and Reward-State ε = .075.

## Verdict: not supported, with one partial exception

1. **Full-State: the gain does not depend on exploration.** All seven arms gain the same within-class
   resolution: Δ full h ranges from +.040 to +.055. The ε > 0 arms gain slightly *less* than ε = 0 (+.040 to
   +.042 vs +.052). Entropy has no visible effect (+.051 to +.055). In every arm the whole gain survives
   removal of the policy-head directions (Δ head-blind ≈ Δ full h).
2. **What ε changes is the policy head, not the representation.** For Full-State, Δ logits-only falls
   steadily with ε: +.328 → +.028 → −.011 → −.051. The head-blind representation stays at the same level.
   The gap between final head-blind and logits-only R² grows from .088 (ε = 0) to .383, .404 and .441.
   Under ε, the belief information beyond the action head therefore *does* grow, but only because the head
   carries less of it. The shared representation does not gain. Entropy does not show this pattern
   (gap .051 to .089).
3. **Reward-State: no consistent effect from ε, and entropy degrades resolution.**
   - ε gives a non-monotonic Δ: +.002, +.020, −.013, +.006.
   - Entropy *lowers* within-class R² below matched init: −.046, −.010, −.052.
   - The loss is entirely in the non-reward contrast b1−b2, the nuisance the action does not need
     (Δ −.100, −.036, −.106).
   - The reward-state coordinate b3 still improves (+.022 to +.024) and is decoded at ≥ .999.
4. **Among reward-qualified checkpoints, entropy makes Reward-State lose resolution.** The trajectories
   are scored on the final policy's histories, and only checkpoints that pass the reward gate count.
   - The original "peaks near .94 around 100M steps" were at failing checkpoints and are dropped.
   - Even so, every Reward-State arm is at .932–.944 at its first competent checkpoint (219M–498M; .918 at
     1.4B for ε = .075). Entropy .002 and .008 then fall to .877 and .869, i.e. gated peak − final = +.060
     and +.064, while reward stays at 98%. Entropy .004 falls +.031, ε = 0 +.022 (mostly at the last
     checkpoint), ε = .075 and .1 about +.015, ε = .05 +.002. Pointwise 95% bands are about ±.004.
   - Full-State is flat once competent: gated peak − final ≤ .005 in every arm.
   - So exploration pressure, entropy in particular, removes belief information the action does not need
     while the policy keeps optimal-level reward. That is the opposite of the hypothesis.
5. **b3 gain shrinks with ε for both quotients:** from +.027 at ε = 0 to +.014–.015 at ε = .1.

## Caveats

- **One training seed per arm.** The 95% intervals are paired whole-episode bootstrap intervals over test
  rollouts, not seed-to-seed intervals. The non-monotone Reward-State entropy sequence (−.046, −.010, −.052)
  suggests seed-level variation is about as large as the effects.
- **Exploratory peaks.** Gated peaks are the best of 12 sampled, reward-passing checkpoints, chosen after
  looking at the curves, and have no selection-adjusted interval. Checkpoint rewards for checkpoints
  below the 80% screen rest on 84 episodes; the screen is far from the gate (.32 vs .95) for all but a few
  Full-State checkpoints at .89–.94, which were re-scored with 1,024 episodes.
- **Approximate reference.** The 95% gate is relative to a grid-policy reference, not a certified optimum.
- **Identical greedy streams.** The Reward-State entropy .002 and .004 final policies produce identical
  greedy action streams on the probe rollouts, so their matched-init scores are identical. Their final
  encoders differ.
- **Controls are near zero.** History-shuffle reached up to +.018 (Full-State entropy .004). Random-input
  and label-permutation controls ranged from −.017 to +.010. There are no one-class final policies.
  Archive reproduction differed by at most 7.3e-4 from archived global R², and no arm was annotated.
- **Decodability is not causal use.** Within-class decodability shows linear accessibility, not that the
  policy uses the information.

## Suggested follow-up

Train 3–5 seeds each at ε ∈ {0, .1} and entropy ∈ {0, .008} for both quotients. This would test whether two
patterns replicate: the Reward-State nuisance loss under entropy, and the Full-State shift of belief
information out of the logits under ε.

## Reproduce (CPU only; checkpoints are fetched from B2 and verified against canonical manifests)

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python -m experiments.within_class_exploration_sweep_2026_11.analyze
uv run python -m experiments.within_class_exploration_sweep_2026_11.trajectory
uv run python -m experiments.within_class_exploration_sweep_2026_11.reward_reference
uv run python -m experiments.within_class_exploration_sweep_2026_11.reward_gate
uv run python -m experiments.within_class_exploration_sweep_2026_11.report
```
