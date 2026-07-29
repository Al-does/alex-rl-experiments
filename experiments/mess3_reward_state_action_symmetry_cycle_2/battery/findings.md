# Cycle 2 battery — Vast multi-seed findings

Five seeded full-battery runs (`seeds 42–46`) on Vast RTX 4090 boxes
(`delay=0`, `700k` env steps, experiment SHA `e0c9ac5`, harness
`fcccfa9`). Flash/RunPod results from the prior attempt were abandoned.

Compact results live on the `results` branch under
`experiments/mess3_reward_state_action_symmetry_cycle_2/battery/results/mess3-as-c2-seed{42..46}-*/`.
Seed 42 also completed a full B2 durability upload; seeds 43–46 had compact
results recovered before slow CN egress finished hashing/uploading checkpoints.

## Final probe MSE (all seeds)

| seed | variant_1 | variant_2 | variant_3 |
|-----:|----------:|----------:|----------:|
| 42 | 0.002860 | 0.005516 | 0.000648 |
| 43 | 0.003313 | 0.004761 | 0.000662 |
| 44 | 0.000731 | 0.004546 | 0.000437 |
| 45 | 0.002858 | 0.004386 | 0.000861 |
| 46 | 0.003018 | 0.005538 | 0.003780 |
| **mean±std** | **0.00256±0.00104** | **0.00495±0.00054** | **0.00128±0.00141** |

## Filtered comparison (non-collapsed / non-outlier runs)

Keep policies that did not hard-collapse to a single action (plus all v1):
**v1 all seeds**, **v2 seed 45 only**, **v3 seeds 42–45**.

| arm | n | mean final MSE | range |
|-----|--:|---------------:|------|
| v3 (42–45) | 4 | **0.00065 ± 0.00017** | 0.00044–0.00086 |
| v1 (all) | 5 | **0.00256 ± 0.00104** | 0.00073–0.00331 |
| v2 (45) | 1 | **0.00439** | — |

Roughly: **v3 ≈ 4× better than v1**, and **v1 ≈ 1.7× better than mixed v2**.

## Final-checkpoint greedy action mix

`greedy_action_fractions` = `[noop, pos, neg]` from final probes:

| variant | typical mix | notes |
|---------|-------------|-------|
| 1 | ~0 / 100 / 0 | always hard-argmax **positive** (5/5) |
| 2 | ~6 / 94 / 0 | almost always **positive**; seed 45 ~27/73/0 |
| 3 | ~21 / 42 / 38 | mixed ~26/37/37 on seeds 42–45; seed 46 ~0/61/39 |

So v1/v2 mostly collapse to “always positive”; only v3 keeps a real action mix
(on the retained seeds). Oracle policies differ by variant
(`v1: +++`, `v2: ++noop`, `v3: +−noop`).

## Takeaways

1. On the filtered set, **variant 3 remains clearly best** for belief MSE.
2. **Variant 2 is worst**, and when it does not fully collapse (seed 45) MSE is
   still ~0.0044.
3. **Variant 1** sits in between and is always action-collapsed to positive.
4. Ranking on retained runs: **3 ≪ 1 < 2**.

See `multi_seed_summary.json` for machine-readable aggregates and run IDs.
