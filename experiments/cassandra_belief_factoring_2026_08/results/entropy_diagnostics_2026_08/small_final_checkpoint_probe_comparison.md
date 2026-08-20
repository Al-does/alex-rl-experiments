# Small-model final-checkpoint belief probes

## Sources

| Condition | Final checkpoint |
|---|---|
| Targeted | `targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m`, 10,027,008 steps |
| Global alias | `global_alias_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m`, 10,027,008 steps |

Both checkpoints were evaluated with the same seed-42 fixed behavior-policy
probe protocol: 60,000 fitting steps, 80,000 held-out test steps, eight
environments, and affine ridge readouts of the pre-final-LayerNorm decision
token.

## Held-out probe R²

| Target | Global alias | Targeted |
|---|---:|---:|
| Aggregate contrast | **0.806** | 0.589 |
| Broken-count distribution | **0.863** | 0.718 |
| Component contrast | **0.501** | 0.241 |
| Expected action reward | 0.951 | **0.956** |
| Identity deviation | 0.084 | **0.117** |
| Joint belief | **0.657** | 0.140 |
| Labeled expected condition | **0.477** | 0.288 |
| Next-operate pass probability | 0.951 | **0.956** |
| Sorted expected condition | **0.796** | 0.588 |
| Total correlation | **0.338** | 0.126 |

## Hypothesis diagnostics

| Metric | Global alias | Targeted |
|---|---:|---:|
| Coarse-over-identity R² advantage | 0.722 | 0.472 |
| Mean component-subspace overlap | 0.313 | 0.110 |
| Sorted-over-labeled R² advantage | 0.319 | 0.300 |

The targeted checkpoint has slightly higher clean identity-deviation
decodability and similar action-reward decodability. The global-alias
checkpoint is substantially more decodable on most coarse, joint, and labeled
targets. Raw cross-condition R² should still be interpreted cautiously because
the fixed action distributions induce different state histories under the two
action semantics.

Full metrics:
`small_final_checkpoint_probes/results/20260820T173719Z-696ceb05/`.
