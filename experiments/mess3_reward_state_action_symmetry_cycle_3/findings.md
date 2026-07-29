# Cycle 3 — Kelly / predictive CE arms (Vast multi-seed)

Three arms on the same Mess3 action-symmetry task as cycle 2
(`delay=0`, `700k` env steps), with auxiliaries borrowed from
`mess3_token_guess_cycle_2`:

| arm | module | aux |
|-----|--------|-----|
| v2kelly | `variant_2_decoupled_kelly` | decoupled Kelly (coeff 1.0) |
| v2pred | `variant_2_predictive_loss` | next-token CE (λ 1.0) |
| v3pred | `variant_3_predictive_loss` | next-token CE (λ 1.0) |

Five seeds (`42–46`) per arm, one Vast RTX 4090 box per arm (sequential
seeds via `run_seeds.sh`). Experiment SHA `b18b486`, harness `f420dd8`.

| arm | vast id | run name |
|-----|--------:|----------|
| v2kelly | 46181131 | `mess3-as-c3-v2kelly-20260729T085558Z` |
| v2pred | 46181236 | `mess3-as-c3-v2pred-20260729T085736Z` |
| v3pred | 46181463 | `mess3-as-c3-v3pred-20260729T090121Z` |

Compact results live on the `results` branch under each arm’s
`results/mess3-as-c3-*-seed{42..46}/`.

## Final probe MSE (all seeds)

| seed | v2kelly | v2pred | v3pred |
|-----:|--------:|-------:|-------:|
| 42 | 0.003708 | 0.001604 | 0.000562 |
| 43 | 0.004531 | 0.000719 | 0.000504 |
| 44 | 0.004931 | 0.000986 | 0.000295 |
| 45 | 0.003892 | 0.001169 | 0.000785 |
| 46 | 0.003370 | 0.002710 | 0.000634 |
| **mean±std** | **0.00409±0.00063** | **0.00144±0.00078** | **0.00056±0.00018** |

Ranking: **v3pred ≪ v2pred ≪ v2kelly**.

## vs cycle-2 plain PPO (same variant)

Cycle-2 means from `battery/findings.md` / `multi_seed_summary.json`
(all five seeds unless noted):

| comparison | cycle-2 PPO mean MSE | cycle-3 mean MSE | Δ |
|------------|---------------------:|-----------------:|--:|
| v2kelly vs c2 v2 | 0.00495 | 0.00409 | −0.00086 |
| v2pred vs c2 v2 | 0.00495 | 0.00144 | **−0.00351** |
| v3pred vs c2 v3 (all) | 0.00128 | 0.00056 | −0.00072 |
| v3pred vs c2 v3 (filtered 42–45) | 0.00065 | 0.00056 | −0.00009 |

Takeaway: **predictive CE helps a lot on variant 2** (~3.4× lower MSE than
c2 v2 PPO). Decoupled Kelly on v2 is roughly in-line with plain PPO.
v3pred is best overall and slightly beats the already-strong filtered c2 v3.

## Final-checkpoint greedy action mix

`greedy_action_fractions` = `[noop, pos, neg]`.

### Per seed

| seed | v2kelly | v2pred | v3pred |
|-----:|---------|--------|--------|
| 42 | 0.273 / 0.727 / 0.000 | 0.127 / 0.873 / 0.000 | 0.250 / 0.365 / 0.384 |
| 43 | 0.274 / 0.726 / 0.000 | 0.000 / 1.000 / 0.000 | 0.262 / 0.368 / 0.369 |
| 44 | 0.000 / 1.000 / 0.000 | 0.000 / 1.000 / 0.000 | 0.263 / 0.369 / 0.367 |
| 45 | 0.238 / 0.762 / 0.000 | 0.000 / 1.000 / 0.000 | 0.264 / 0.368 / 0.368 |
| 46 | 0.000 / 1.000 / 0.000 | 0.000 / 1.000 / 0.000 | 0.266 / 0.370 / 0.364 |
| **mean** | **0.157 / 0.843 / 0.000** | **0.025 / 0.975 / 0.000** | **0.261 / 0.368 / 0.371** |

### Notes

- **v2kelly**: partial noop/pos mix on 3/5 seeds; hard always-pos on 44 & 46.
- **v2pred**: almost always collapsed to positive (4/5 hard collapse).
- **v3pred**: stable mixed ~26/37/37 on **all five** seeds (fixes c2 v3 seed-46
  collapse).

## Reward-state occupancy (`reward_state_2_fraction_greedy`)

Mean reward under the greedy probe (= fraction of time in the rewarding
state). Cycle-2 reference included for the same seeds.

### Means

| arm | mean±std rs2 |
|-----|-------------:|
| c2 v1 PPO | 0.3338 ± 0.0021 |
| c2 v2 PPO | 0.2555 ± 0.0019 |
| c2 v3 PPO | 0.2416 ± 0.0024 |
| c3 v2kelly | 0.2569 ± 0.0027 |
| c3 v2pred | 0.2552 ± 0.0022 |
| c3 v3pred | 0.2432 ± 0.0015 |

### Per-seed cycle-3 values

| seed | v2kelly | v2pred | v3pred |
|-----:|--------:|-------:|-------:|
| 42 | 0.2590 | 0.2583 | 0.2440 |
| 43 | 0.2571 | 0.2547 | 0.2425 |
| 44 | 0.2526 | 0.2526 | 0.2412 |
| 45 | 0.2594 | 0.2543 | 0.2432 |
| 46 | 0.2562 | 0.2562 | 0.2450 |

### Δ vs cycle-2 same variant (c3 − c2)

| seed | v2kelly − c2 v2 | v2pred − c2 v2 | v3pred − c2 v3 |
|-----:|----------------:|---------------:|---------------:|
| 42 | +0.0021 | +0.0014 | +0.0002 |
| 43 | +0.0025 | +0.0000 | +0.0001 |
| 44 | +0.0000 | +0.0000 | +0.0000 |
| 45 | +0.0023 | −0.0028 | +0.0000 |
| 46 | +0.0000 | +0.0000 | +0.0073 |
| **mean Δ** | **+0.0014** | **−0.0003** | **+0.0015** |

Occupancy is essentially unchanged by the auxiliaries. The only notable bump
is **v3 seed 46** (c2 collapsed → c3pred stayed mixed). MSE gains from
predictive CE are therefore **not** from a different reward-state visit
distribution; variant still dominates occupancy
(v1 ~0.33 ≫ v2 ~0.26 �-state visit
distribution; variant still dominates occupancy
(v1 ~0.33 ≫ v2 ~0.26 ≫ v3 ~0.24).

## Takeaways

1. **v3pred is best**: lowest MSE, mixed actions on all seeds, rs2 unchanged.
2. **Predictive CE rescues variant 2 on belief MSE** (~0.0014 vs ~0.005 PPO)
   without changing occupancy; policy still mostly always-positive.
3. **Decoupled Kelly on v2** does not beat plain PPO on MSE and does not
   change occupancy; action mix is only partially less collapsed.
4. Ranking: **v3pred ≪ v2pred ≪ v2kelly ≈ c2 v2 PPO**.

See `multi_seed_summary.json` for machine-readable aggregates, run IDs, and
paired cycle-2 comparisons.
