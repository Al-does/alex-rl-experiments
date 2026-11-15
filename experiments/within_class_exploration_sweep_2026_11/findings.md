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
4. **The trajectories show forgetting, not lack of learning.** All seven Reward-State runs reach about
   .94 within-class R² by around 100M steps (one peaks at 498M), then decline.
   - The entropy .002 and .008 runs decline most, to .877 and .869. The ε = .075 run ends at .903.
   - Full-State ε > 0 runs peak at 219–498M steps and end .005–.011 lower.
   - Full-State ε = 0 and entropy runs plateau near .97.
   - So, if anything, exploration pressure speeds up forgetting of belief information the action does not
     need, the opposite of the hypothesis.
5. **b3 gain shrinks with ε for both quotients:** from +.027 at ε = 0 to +.014–.015 at ε = .1.

## Caveats

- **One training seed per arm.** The 95% intervals are paired whole-episode bootstrap intervals over test
  rollouts, not seed-to-seed intervals. The non-monotone Reward-State entropy sequence (−.046, −.010, −.052)
  suggests seed-level variation is about as large as the effects.
- **Exploratory peaks.** Trajectory peaks were picked after looking at the curves. The peak − final
  intervals are not adjusted for that selection.
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
uv run python -m experiments.within_class_exploration_sweep_2026_11.report
```
