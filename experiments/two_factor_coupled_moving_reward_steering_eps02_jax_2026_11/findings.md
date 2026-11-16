# XOR and moving-reward steering on the PR #234 emission-ε=0.02 agents

The 15 Joint Reward agents trained in PR #234
(`reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11/reward_both_transformer`,
seeds 42–56, final checkpoints at step 2,012,217,344) run through the unchanged
protocol of `two_factor_coupled_moving_reward_steering_v2_jax_2026_11`
(`findings.md` there defines the rewards, action maps, steering update, sites,
strength tuning and nulls). The agents are frozen; nothing is retrained.

Only the process differs from the no-emission-ε study (PR #233). State 3 emits
A/B with probability 0.01 each instead of never (ε refers to this emission
noise, not to exploration):

```text
emissions (A, B, C)   state 1: .1125 .0375 .85   state 2: .0375 .1125 .85   state 3: .01 .01 .98
```

Transitions, actions, temperature 1.5 and context 32 are unchanged, so the
known-state action values and the maps as functions of the belief are those of
v2. Rollouts, exact beliefs, decoder targets and the exact-belief references
all use the ε=0.02 process (`steering.py::source_process`).

## Results (block 4, long-run map, n=15, 256 episodes per agent)

Mean ± sd across agents [min, max] (`results/summary.json` → `by_setup`,
`results/eps02_xor_moving_steering_15_seeds.pdf`). XOR has maximum 1; the moving
reward is the sum of two per-factor rewards and has maximum 2.

| Condition | XOR | Moving total | Moving factor 1 / factor 2 |
|---|---|---|---|
| Unsteered | 0.482 ± 0.005 [0.478, 0.493] | 0.901 ± 0.009 [0.883, 0.919] | 0.403 / 0.498 |
| Uniform random | 0.182 | 0.602 | 0.098 / 0.504 |
| Factored steering | 0.494 ± 0.014 [0.478, 0.515] | 0.998 ± 0.013 [0.986, 1.039] | 0.352 / 0.646 |
| Joint steering | **0.502 ± 0.012** [0.480, 0.516] | **1.042 ± 0.020** [0.996, 1.069] | 0.378 / 0.664 |
| Exact belief, long-run map | 0.563 | 1.092 | 0.414 / 0.678 |
| Exact belief, next-step map | 0.556 | 1.088 | 0.416 / 0.672 |
| Null, factored | 0.478 ± 0.006 | 0.908 ± 0.015 | |
| Null, joint | 0.472 ± 0.014 | 0.899 ± 0.037 | |

| Steered − unsteered, paired over agents | XOR gain (mean ± SE) | XOR gap closed (mean ± sd) | Moving gain | Moving gap closed |
|---|---|---|---|---|
| Factored steering | +0.011 ± 0.003, 10/15 agents up, 5 unchanged | 15% ± 15% | +0.098 ± 0.004, 15/15 up | 51% ± 7% |
| Joint steering | +0.020 ± 0.003, 12/15 up, 3 unchanged | 24% ± 14% | +0.141 ± 0.004, 15/15 up | 74% ± 10% |

Against the 15 no-emission-ε agents of PR #233 (same protocol, same seeds):

| | XOR unsteered | XOR factored | XOR joint | XOR exact | Moving unsteered | Moving factored | Moving joint | Moving exact |
|---|---|---|---|---|---|---|---|---|
| No emission ε | 0.500 | 0.579 | 0.554 | 0.619 | 0.964 | 1.102 | 1.142 | 1.176 |
| Emission ε=0.02 | 0.482 | 0.494 | 0.502 | 0.563 | 0.901 | 0.998 | 1.042 | 1.092 |
| Gap closed, no emission ε | | 66% | 46% | | | 65% | 84% | |
| Gap closed, emission ε=0.02 | | 15% | 24% | | | 51% | 74% | |

- **Moving reward transfers.** Every agent improves under both steering types;
  joint beats factored in 14/15 agents. In the phases paying states 1/2,
  factor-2 reward goes 0.591 → 0.953 under joint steering (exact 0.973).
  Factor-1 cost: joint −0.025, factored −0.050.
- **XOR steering is weak.** The gain is statistically distinguishable from zero
  across agents but closes a quarter of the gap at best. Tuning picked
  strength 0 (no steering) for 5 agents with the factored target and 3 with the
  joint target; seed 55 is unsteered under both. Averaged over agents on the
  64 tuning episodes, no factored strength improves on unsteered, and the best
  joint strength gains +0.013 ± 0.005 (`summary.json` →
  `strength_tuning_mean_over_agents`).
- **The factored/joint order on XOR reverses** relative to the no-emission-ε
  agents (factored won 15/15 there). Here joint wins 8/15 and the paired
  difference is +0.008 ± 0.004 (p = 0.09), so the order is not established.
- **XOR with the next-step map does nothing**: +0.003 ± 0.002 (factored) and
  −0.002 ± 0.002 (joint); 11 of 15 joint selections are strength 0.
- **Agreement with the exact-belief XOR map** (factor 1 / factor 2): unsteered
  0.68 / 0.68, factored 0.70 / 0.68, joint 0.70 / 0.70. The no-emission-ε
  agents start lower (0.58 / 0.57) and are moved further (0.73 / 0.75
  factored). Steering changes 10% (factored) and 19% (joint) of factor actions
  here against 25% and 32% there.
- **Nulls.** XOR nulls sit slightly below unsteered (factored −0.004 ± 0.001,
  joint −0.010 ± 0.004), so the steered-minus-null contrast is a little larger
  than steered-minus-unsteered. The low joint nulls are the large selected
  strengths: XOR seed 56 (0.434, λ=1.5) and seed 52 (0.455, λ=2.0); moving
  seed 55 (0.778, λ=3.0).
- **Sites** (mean total reward, block 1 / 2 / 3 / 4 / post-final-norm): XOR
  factored 0.460 / 0.490 / 0.494 / 0.494 / 0.496, XOR joint 0.439 / 0.476 /
  0.496 / 0.502 / 0.501; moving factored 0.998 / 1.026 / 1.018 / 0.998 / 1.019,
  moving joint 1.000 / 1.037 / 1.044 / 1.042 / 1.043. Strengths are tuned at
  block 4 only.
- **Decoding.** Final-checkpoint probe error (1−R², from the PR #234 training
  reports) is higher than for the no-emission-ε agents: joint 0.131 ± 0.016
  against 0.099 ± 0.014; factor 1 0.078 against 0.065; factor 2 0.082 against
  0.058. Across the 15 agents joint 1−R² correlates with joint-steered XOR
  reward at r = −0.78 (p = 0.0006); the other three rows are weaker (XOR
  factored −0.28, moving factored 0.04, moving joint −0.44).

## Caveats

- Why XOR steering is weak here is not established. The headroom is smaller
  (exact − unsteered is 0.080 against 0.119), the unsteered agents already
  agree with the XOR map more often, and the beliefs decode less well; this
  study does not separate these.
- Strength is tuned per agent on XOR/moving reward from 64 episodes and
  confirmed on 256 separate episodes. Agents whose tuned strength is 0 score
  exactly their unsteered reward and count as "unchanged".
- SEs in the paired rows are across the 15 agents. Per-agent episode SEs are
  about 0.007 (XOR) and 0.012 (moving), comparable to the XOR gain of a single
  agent.
- Exact-belief maps are QMDP (γ=0.99), not the POMDP optimum. The moving-reward
  controller is given the step `t`; the agents are not.
- Only the Joint Reward agents are covered. The Unrewarded Factor 2 and ladder
  agents of PR #234 are not part of this protocol.

## Provenance

- Each checkpoint was downloaded from the run's B2 prefix and its SHA-256
  checked against `metadata/durability_manifest.json` before use
  (`steering.py::fetch_checkpoint`); the hash is in each report.
- All 15 reports record experiment commit `98b73ffe` and harness commit
  `a267508f`, both clean. Fit seed 42, tuning seed 1042, confirmation seed 2042.
- Seeds 42–44 ran on the 8-core session VM; seeds 45–50 and 51–56 on two
  on-demand Vast boxes with `JAX_PLATFORMS=cpu`, the six agents of a box in
  parallel. The boxes' GPUs were not used.

## Reproduce

```bash
uv run --group jax pytest -q tests/test_two_factor_coupled_moving_reward_steering_eps02_jax.py
JAX_PLATFORMS=cpu uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11.steering --agent-seeds 42
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11.summarize
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11.plot
```

About 30 minutes per agent on 8 CPU cores. Omit `--agent-seeds` for all 15;
B2 credentials are needed to fetch checkpoints.
