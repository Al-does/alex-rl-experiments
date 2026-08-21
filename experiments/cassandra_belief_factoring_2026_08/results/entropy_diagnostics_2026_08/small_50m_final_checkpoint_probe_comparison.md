# Small-model 50M final-checkpoint belief probes

## Sources

| Condition | Final checkpoint |
|---|---|
| Targeted | `targeted_ppo_small_continue_30m`, 50,069,504 steps |
| Global alias | `global_alias_ppo_small_continue_30m`, 50,069,504 steps |

Both checkpoints use the matched seed-42 fixed behavior-policy protocol:
60,000 fit steps, 80,000 held-out test steps, eight environments, and affine
ridge readouts of the pre-final-LayerNorm decision token.

## Held-out probe R²

| Target | Global alias | Targeted |
|---|---:|---:|
| Aggregate contrast | **0.811** | 0.614 |
| Broken-count distribution | **0.859** | 0.693 |
| Component contrast | **0.485** | 0.342 |
| Expected action reward | 0.940 | **0.941** |
| Identity deviation | 0.040 | **0.245** |
| Joint belief | **0.653** | 0.178 |
| Labeled expected condition | **0.446** | 0.402 |
| Sorted expected condition | **0.794** | 0.598 |

## Hypothesis diagnostics

| Metric | Global alias | Targeted |
|---|---:|---:|
| Coarse-over-identity R² advantage | 0.771 | 0.369 |
| Mean component-subspace overlap | 0.356 | 0.101 |
| Sorted-over-labeled R² advantage | 0.348 | 0.196 |

## Change from 10M

Targeted identity-deviation R² increased from 0.117 to **0.245**, component
contrast increased from 0.241 to **0.342**, and labeled expected condition
increased from 0.288 to **0.402**. Global-alias identity-deviation R² decreased
from 0.084 to **0.040**, while its coarse and joint targets stayed high.

This is the clearest result so far in the predicted direction: extended
component-targeted control preserves substantially more identity-specific
belief information, while global aliases remain dominated by coarse,
permutation-invariant structure. Cross-condition R² still reflects different
state histories induced by the action semantics, so it is evidence rather than
a causal proof.

Full metrics:
`small_50m_final_checkpoint_probes/results/20260821T032128Z-29ab8942/`.
