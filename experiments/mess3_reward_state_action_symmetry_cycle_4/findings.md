# Cycle 4 — sticky-state action-symmetry (Vast multi-seed)

Cycle 4 changes one environmental quantity relative to cycle 2: the baseline
transition row from reward state 2 becomes `[0.30, 0.30, 0.40]` instead of
`[0.45, 0.45, 0.10]`. Otherwise the protocol matches cycle 2 plain PPO
(`delay=0`, `700k` env steps, three action variants).

Five seeds (`42–46`) per variant, one Vast RTX 4090 box per variant (sequential
seeds via `run_seeds.sh`). Experiment SHA `8cd58c4`, harness `38f5136`.

| variant | vast id | run prefix |
|---------|--------:|------------|
| 1 | 46223593 | `mess3-rsa-c4-v1` |
| 2 | 46223851 | `mess3-rsa-c4-v2` |
| 3 | 46224326 | `mess3-rsa-c4-v3` |

Compact results recovered from B2 into each variant’s
`results/mess3-rsa-c4-v{1,2,3}-seed{42..46}/`. Full checkpoints remain in B2.

## Final probe MSE (all seeds)

| seed | variant_1 | variant_2 | variant_3 |
|-----:|----------:|----------:|----------:|
| 42 | 0.005726 | 0.003081 | 0.001618 |
| 43 | 0.006986 | 0.003408 | 0.001153 |
| 44 | 0.005575 | 0.003143 | 0.001019 |
| 45 | 0.006055 | 0.004253 | 0.001373 |
| 46 | 0.005849 | 0.003591 | 0.000894 |
| **mean±std** | **0.00604±0.00050** | **0.00350±0.00042** | **0.00121±0.00026** |

Ranking: **variant_3 ≪ variant_2 < variant_1**.

## vs cycle-2 plain PPO (same variant)

Cycle-2 means from `mess3_reward_state_action_symmetry_cycle_2/battery/multi_seed_summary.json`:

| comparison | cycle-2 mean MSE | cycle-4 mean MSE | Δ |
|------------|-----------------:|-----------------:|--:|
| v1 | 0.00256 | 0.00604 | +0.00348 |
| v2 | 0.00495 | 0.00350 | −0.00145 |
| v3 | 0.00128 | 0.00121 | −0.00007 |

The stickier reward-state-2 baseline raises v1 occupancy and worsens v1 belief
MSE, improves v2 MSE (variant now mixes noop and positive instead of
collapsing to always-positive), and leaves v3 best with a small further gain.

## Final-checkpoint greedy action mix

`greedy_action_fractions` = `[noop, pos, neg]`.

| variant | mean mix | notes |
|---------|----------|-------|
| 1 | 0.00 / 1.00 / 0.00 | hard collapse to positive (5/5) |
| 2 | 0.33 / 0.67 / 0.00 | stable noop+positive mix on all seeds |
| 3 | 0.32 / 0.34 / 0.34 | balanced three-way mix on all seeds |

## Reward-state occupancy (`reward_state_2_fraction_greedy`)

| variant | mean±std rs2 | cycle-2 reference |
|---------|-------------:|------------------:|
| 1 | 0.571 ± 0.004 | 0.334 ± 0.002 |
| 2 | 0.331 ± 0.002 | 0.256 ± 0.002 |
| 3 | 0.314 ± 0.002 | 0.242 ± 0.002 |

The sticky baseline materially increases v1 reward-state occupancy (~0.57 vs
~0.33 in cycle 2). v2 and v3 occupancy shift modestly upward while preserving
their relative ordering.

## Takeaways

1. **Variant 3 remains clearly best** for belief MSE under the sticky baseline.
2. **Variant 2 improves vs cycle 2** and now maintains a genuine noop/positive
   mix rather than collapsing.
3. **Variant 1 worsens** — higher occupancy in state 2 with the same
   always-positive collapse yields worse belief geometry.
4. Ranking: **3 ≪ 2 < 1** (same order as cycle 2, shifted levels).

See `multi_seed_summary.json` for machine-readable aggregates and run IDs.
