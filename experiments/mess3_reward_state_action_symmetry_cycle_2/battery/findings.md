# Cycle 2 battery — Vast multi-seed findings

Five seeded full-battery runs (`seeds 42–46`) on Vast RTX 4090 boxes
(`delay=0`, `700k` env steps, experiment SHA `e0c9ac5`, harness
`fcccfa9`). Flash/RunPod results from the prior attempt were abandoned.

Compact results live on the `results` branch under
`experiments/mess3_reward_state_action_symmetry_cycle_2/battery/results/mess3-as-c2-seed{42..46}-*/`.
Seed 42 also completed a full B2 durability upload; seeds 43–46 had compact
results recovered before slow CN egress finished hashing/uploading checkpoints.

## Final probe MSE (lower is better)

| seed | variant_1 | variant_2 | variant_3 |
|-----:|----------:|----------:|----------:|
| 42 | 0.002860 | 0.005516 | 0.000648 |
| 43 | 0.003313 | 0.004761 | 0.000662 |
| 44 | 0.000731 | 0.004546 | 0.000437 |
| 45 | 0.002858 | 0.004386 | 0.000861 |
| 46 | 0.003018 | 0.005538 | 0.003780 |
| **mean±std** | **0.00256±0.00104** | **0.00495±0.00054** | **0.00128±0.00141** |

## Takeaways

1. **Variant 3 is best on average** (lowest mean final MSE), but seed 46 is an
   outlier that collapses the gap (`0.00378` vs ~`0.0004–0.0009` for seeds 42–45).
2. **Variant 2 is consistently worst** across all five seeds (tight stdev).
3. **Variant 1 is intermediate** with one strong seed (44) and otherwise similar
   to the mid-`0.003` band.
4. Relative to cycle 1’s delayed setting, this zero-delay / 0.7M-step battery
   still shows a clear action-effect ranking on most seeds: `3 < 1 < 2`.

See `multi_seed_summary.json` for machine-readable aggregates and run IDs.
