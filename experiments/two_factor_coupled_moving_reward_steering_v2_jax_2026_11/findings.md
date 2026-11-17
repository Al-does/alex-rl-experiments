# XOR and moving-reward steering on the PR #225 v2 agents

Follow-up to `two_factor_moved_reward_steering_v2_jax_2026_11` (PR #226). The
moved-state reward there had a near-constant best-action map (Tilt Y almost
everywhere on real beliefs). This study uses two rewards whose best factor-2
action is **not** a function of the factor-2 belief alone, on the same frozen
PR #225 Joint Reward agents (`reward_both`, ctx32 no ε and ε=0.05, seeds 42/43,
step 2,012,217,344; checkpoints and SHA-256 reused from PR #226). No agent is
retrained; the Unrewarded Factor 2 agents are not evaluated.

## Rewards (dynamics, emissions and actions unchanged)

- **XOR:** reward 1 per step when exactly one factor is in state 3, else 0
  (single joint component; replaces both factor rewards; max 1).
- **Moving:** factor 1 pays for state 3, as in training. Factor 2 pays for state
  3 during steps [0, 256) and [512, 768), and for states 1 or 2 during
  [256, 512) and [768, 1024) (`PERIOD = 256`, two components, max 2).

The agents see only tokens, as in training. Only the steering controller and
the exact-belief reference know the step `t`.

## Action maps (`results/v2_xor_moving_best_action_maps.pdf`)

Time-indexed joint tables \(Q_t[s, a]\) over the 9 joint states and 9 joint
actions, applied to a joint belief \(b\): \(a^* = \arg\max_a b^\top Q_t[:, a]\).

- *next-step:* \(Q_t = T_a R_{t+1}\) (expected reward of the next state);
- *long-run:* finite-horizon QMDP over the 1,024-step episode, γ=0.99.

What the maps require:
- **XOR:** factor 2's action depends on factor 1. With factor 1 out of state 3,
  factor 2 is pulled into state 3 (Tilt X in state 1, Tilt Y in state 2, Noop in
  state 3). With factor 1 in state 3, factor 2 is pushed out (Tilt Y in state 1,
  Tilt X in state 2, Tilt X/Y in state 3). Factor 1's action depends on factor 2
  symmetrically, so both factors are steered.
- **Moving:** factor 2's map flips between the training map and its mirror,
  and switches a few steps *before* each boundary (bottom strip of the figure).
  Factor 1's map is the training map, so only factor 2 is steered; factor 1
  keeps its decoded belief.

## Steering

Identical to PR #226 except for the target. Affine factored (4-d) and joint
(8-d) decoders/embeddings are fit on 60K greedy unsteered rows (seed 42). At each
step the decoded belief (joint vector, or product of decoded marginals for the
factored map) picks the joint action \(a^*\) from \(Q_t\). Each steered factor
is then moved to a point mass on its anchor state, the state where the frozen
agent most often takes that factor action (Tilt X → state 1, Tilt Y → state 2,
Noop → state 3, for both factors of every agent). Update:
\(h' = h + \lambda(y^* - D(h))E\). λ is tuned at block 4 from the PR #226 grid
for highest total reward on 64 episodes (seed 1042), then confirmed on 256 fresh
1,024-step episodes (seed 2042) at all 5 sites. Controls: unsteered, uniform
random joint actions, each map on the exact filtered belief, and a matched
decoder-orthogonal null (block 4). Greedy policy; reward per step includes reset
steps; ± is SE across episodes (≈0.006–0.013).

## Results (block 4, mean over the 4 agents)

"Agree" is the fraction of steps whose executed factor action equals the
exact-belief long-run map's action (factor 1 / factor 2).

### XOR

| Condition | XOR reward | Agree f1 / f2 |
|---|---:|---|
| Unsteered | 0.496 | 0.58 / 0.57 |
| Uniform random | 0.181 | 0.33 / 0.33 |
| Factored steering, next-step map | 0.538 | 0.70 / 0.68 |
| Joint steering, next-step map | 0.500 | 0.54 / 0.54 |
| **Factored steering, long-run map** | **0.574** | 0.71 / 0.76 |
| Joint steering, long-run map | 0.547 | 0.64 / 0.72 |
| Exact belief, next-step / long-run map | 0.623 / 0.619 | 1 / 1 |
| Null, factored / joint long-run | 0.494 / 0.493 | |

- Steering closes 63% (factored) and 41% (joint) of the unsteered-to-exact gap
  with the long-run map; every agent improves (factored 0.558–0.589, joint
  0.527–0.567). Nulls are flat.
- Factored beats joint here. This is expected: the exact joint belief is the
  product of the marginals, so XOR needs cross-factor *control* but not joint
  *information*. Joint steering to a joint point mass moves 8 coordinates and
  is worse at early sites (blocks 1–2: 0.42–0.53, below unsteered).
- With the next-step map, tuning chose λ=0 for joint steering in 3 of 4 agents.

### Moving reward

| Condition | Total | Factor 1 | Factor 2 | F2, phases paying 3 | F2, phases paying 1/2 | Agree f2 |
|---|---:|---:|---:|---:|---:|---:|
| Unsteered | 0.961 | 0.470 | 0.491 | 0.457 | 0.525 | 0.47 |
| Uniform random | 0.602 | 0.098 | 0.504 | 0.110 | 0.898 | 0.33 |
| Factored steering, next-step map | 1.090 | 0.421 | 0.669 | 0.416 | 0.923 | 0.64 |
| Joint steering, next-step map | 1.132 | 0.453 | 0.679 | 0.413 | 0.945 | 0.69 |
| Factored steering, long-run map | 1.093 | 0.417 | 0.677 | 0.423 | 0.931 | 0.66 |
| **Joint steering, long-run map** | **1.136** | **0.455** | **0.681** | 0.417 | 0.945 | 0.67 |
| Exact belief, next-step map | 1.168 | 0.473 | 0.695 | 0.425 | 0.965 | 1 |
| Exact belief, long-run map | 1.176 | 0.471 | 0.705 | 0.439 | 0.971 | 1 |
| Null, factored / joint long-run | 0.968 / 0.968 | 0.471 / 0.473 | | | | |

- Joint steering closes 81% of the total-reward gap (every agent 1.123–1.155).
  The gain comes from the phases paying states 1/2, where the steered agents
  leave state 3 (0.525 → 0.945, exact 0.971). In the phases paying state 3 the
  steered agents match the exact map (0.42–0.44), which is slightly below
  unsteered because each such phase starts with factor 2 outside state 3.
- Joint steering preserves factor 1 better than factored steering
  (−0.015 vs −0.053 relative to unsteered), the same pattern as PR #226 and the
  paper; factor-2 gains are equal.
- Best sites are blocks 3–4 and post-final-norm; block 1 recovers about half.

## 15 no-ε seeds (PR #231)

PR #231 trained seeds 44–56 of the no-ε Joint Reward arm
(`reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11/reward_both_transformer`).
Their final checkpoints (step 2,012,217,344, SHA-256 matching each run's B2
durability manifest) were run through the unchanged protocol above; seeds 42–43
are the reports already here. n=15; sd is across seeds; block 4, long-run map
(`results/v2_xor_moving_steering_15_seeds.pdf`, `summary.json` → `by_setup`).

| Condition | XOR mean ± sd [min, max] | Moving total mean ± sd [min, max] | Moving factor 1 / factor 2 |
|---|---|---|---|
| Unsteered | 0.499 ± 0.006 [0.492, 0.514] | 0.964 ± 0.008 [0.952, 0.976] | 0.472 / 0.492 |
| Uniform random | 0.181 | 0.602 | 0.098 / 0.504 |
| Factored steering | **0.579 ± 0.009** [0.561, 0.592] | 1.102 ± 0.022 [1.057, 1.131] | 0.431 / 0.671 |
| Joint steering | 0.554 ± 0.013 [0.535, 0.574] | **1.142 ± 0.007** [1.131, 1.155] | 0.460 / 0.682 |
| Exact belief | 0.619 | 1.176 | 0.471 / 0.705 |
| Null, factored | 0.495 ± 0.006 | 0.962 ± 0.008 | |
| Null, joint | 0.488 ± 0.030 | 0.965 ± 0.009 | |

| Gap closed (unsteered → exact), mean ± sd | XOR | Moving |
|---|---|---|
| Factored steering | 66% ± 8% | 65% ± 9% |
| Joint steering | 46% ± 11% | 84% ± 3% |

- Every one of the 15 agents improves under both steering types on both rewards.
  Factored > joint on XOR in 15/15 agents; joint > factored on moving in 15/15.
- Exact-belief-map agreement (factor 1 / factor 2): XOR unsteered 0.58 / 0.57,
  factored 0.73 / 0.75, joint 0.71 / 0.70. Moving factor 2: unsteered 0.47,
  joint 0.69; in phases paying states 1/2 factor-2 reward goes 0.527 → 0.946
  (joint), exact 0.971.
- Moving-reward factor-1 cost: joint −0.012, factored −0.041.
- Nulls are flat except seed 47 XOR joint (0.387): its tuned λ=3.0 is large
  enough that a decoder-orthogonal push of the same norm degrades the policy.
- Selected λ ranges 0.75–3.0 (XOR joint) and 0.75–2.0 (moving joint); most are
  0.75–1.25.
- Decoding vs steerability (Pearson r of final joint-belief 1−R² against steered
  reward, n=15): XOR factored −0.60 (p=0.018), XOR joint −0.50 (p=0.06),
  moving factored −0.48 (p=0.07), moving joint −0.14 (p=0.62). Lower decoding
  error goes with better steering for factored maps and for XOR; moving-reward
  joint steering is nearly saturated and uncorrelated.
- The two ε=0.05 agents fall at or below the bottom of the no-ε distribution in
  7 of 8 steering rows (exception: ε seed 42, XOR joint, 0.548; 11 of 15 no-ε
  seeds score at least as high).
  Their unsteered rewards are within the no-ε range. This is consistent with a
  small ε penalty on steerability, but with two ε seeds it is not established.

## 15 ε=0.02-emission seeds (PR #234)

PR #234 retrained the Joint Reward arm (seeds 42–56) on the same dynamics with
one change: state 3 now emits A/B with probability ε/2 each, ε=0.02
(`reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11/reward_both_transformer`).
Final checkpoints (step 2,012,217,344, SHA-256 matching each run's B2 durability
manifest) went through the unchanged protocol, except that the ε=0.02 process
now drives the fitted belief targets, the exact-belief filter/maps and the
rollouts (`steering.source_process(study)`; checked against each run's
recorded process). Transitions are identical, so the Q tables are unchanged.
n=15; sd across seeds; block 4, long-run map (`results/summary_eps02.json`,
`results/eps02_xor_moving_steering_15_seeds.pdf`, PR #231 no-ε seeds overlaid).

| Condition | XOR mean ± sd [min, max] | Moving total mean ± sd [min, max] | Moving factor 1 / factor 2 |
|---|---|---|---|
| Unsteered | 0.482 ± 0.005 [0.478, 0.493] | 0.901 ± 0.009 [0.883, 0.919] | 0.403 / 0.498 |
| Uniform random | 0.182 | 0.602 | 0.098 / 0.504 |
| Factored steering | 0.494 ± 0.014 [0.478, 0.515] | 0.999 ± 0.013 [0.986, 1.039] | 0.353 / 0.646 |
| Joint steering | **0.502 ± 0.012** [0.480, 0.516] | **1.042 ± 0.020** [0.996, 1.069] | 0.378 / 0.664 |
| Exact belief | 0.563 | 1.092 | 0.414 / 0.678 |
| Null, factored | 0.478 ± 0.006 | 0.908 ± 0.015 | |
| Null, joint | 0.472 ± 0.014 | 0.899 ± 0.037 | |

| Gap closed (unsteered → exact), mean ± sd | XOR | Moving | no-ε (PR #231) XOR / moving |
|---|---|---|---|
| Factored steering | 15% ± 15% | 51% ± 7% | 66% / 65% |
| Joint steering | 24% ± 14% | 74% ± 10% | 46% / 84% |

- Moving reward still works: 15/15 agents improve under both steering types,
  joint > factored in 14/15. In phases paying states 1/2, factor-2 reward goes
  0.591 → 0.953 (joint), exact 0.973. Factor-1 cost: joint −0.025, factored
  −0.050 (no ε: −0.012 / −0.041).
- XOR steering mostly fails. Tuning on 64 episodes found no helpful strength
  (λ=0 selected) for 5/15 agents with factored steering and 3/15 with joint
  steering. Gains elsewhere are 0.001–0.034. Steering barely changes
  exact-belief-map agreement (unsteered 0.68 / 0.68, factored 0.70 / 0.68,
  joint 0.69 / 0.70 for factor 1 / factor 2; no ε went 0.57 → 0.71–0.75).
  The factored > joint ordering of no ε does not hold: joint > factored in 9/15.
- The headroom shrinks: the exact-belief map's XOR edge over unsteered falls
  from 0.120 to 0.081 (and moving from 0.212 to 0.191), because state 3 can no
  longer be confirmed from one observation.
- Decoding is worse: final joint-belief 1−R² is 0.131 on average (no ε: 0.099).
  Pearson r with steered reward: XOR joint −0.78 (p<0.001), XOR factored −0.28
  (p=0.32), moving joint −0.44 (p=0.10), moving factored 0.04 (p=0.89).
- Held-out decoder R² of the steering maps (`results/eps02_probe_r2.json`,
  `probe_r2.py`: same fit rollout, 80,000 test rows at seed 43), block 4:
  factored 0.89–0.93, joint 0.84–0.90 (no-ε four agents: 0.91–0.93 / 0.87–0.91).
- Nulls are flat except joint nulls at large tuned λ: seed 55 moving (λ=3.0,
  0.778) and seeds 52/56 XOR (drops of 0.03–0.05).
- Reports record `experiment.dirty = true` on `569c2183` (merge of `main` into
  the PR #233 branch); `steering.py` is committed unchanged from the run.

## Caveats

- The 4-agent tables above use two training seeds per arm; the 15-seed section
  covers the no-ε arm only (ε=0.05 still has two seeds). SEs reflect
  episode-to-episode variation only.
- The 15-seed reports were run at a clean commit (merge of PR #231 into this
  branch); `steering.py` is unchanged from the 4-agent runs.
- Exact-belief maps are QMDP, not the POMDP optimum; steering can only express
  actions through the agent's anchor states.
- The steering controller is given the step `t`; the agents are not. This tests
  whether the representation can be driven to a time-varying target, not
  whether the agent can track the phase itself.
- Reports record `experiment.dirty = true` on base commit `6059d074`: the runs
  used this study's files before they were first committed. `steering.py` is
  committed unchanged from the run.

## Reproduce

```bash
uv run --group jax pytest -q tests/test_two_factor_coupled_moving_reward_steering_v2_jax.py
# checkpoints: see ../two_factor_moved_reward_steering_v2_jax_2026_11/findings.md
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.steering
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.summarize
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.plot
```

About 40 minutes on 8 CPU cores with the four agents in parallel (`--only`); the
13 PR #231 seeds took 70 minutes with 7 in parallel, e.g.
`--only ctx32_2026_11/reward_both_transformer/seed44`. Checkpoints come from each
run's B2 prefix, `checkpoints/step_2012217344.npz`, renamed to the
`checkpoint_path` pattern in the PR #226 study.

Decoder R² for the PR #234 agents (a few minutes per agent):

```bash
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.probe_r2 \
  --study reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 \
  --checkpoints experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/artifacts/checkpoints \
  --output experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/results/eps02_probe_r2.json
```

PR #234 agents (~90 min, 8 in parallel; checkpoints named
`eps02_ctx32_2026_11__<leaf>__seed<seed>__<run_id>__step_*.npz`):

```bash
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.steering \
  --checkpoints experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/artifacts/checkpoints \
  --only eps02_jax_temp15_ctx32_2026_11/reward_both_transformer/seed42
uv run --group jax python -m experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.summarize \
  --pattern 'eps02_ctx32*__seed*.json' --output summary_eps02.json
```
