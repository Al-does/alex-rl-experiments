# Moved factor-2 reward: best-action map and belief steering

## Question

The paper's Joint Reward steering pushed factor 2 *away* from reward. Here
the frozen Joint Reward agent is moved to a new environment where factor 2's
reward state is **state 1** (code 0) instead of state 3 (code 2). Can
steering its decoded beliefs make it collect the new reward, and how does the
steered agent compare to the unsteered one?

## Fixed vs changed

- **Agent (unchanged)**: 30M Joint Reward seed 42,
  `two_factor_reward_state_REINFORCE_cycle_4-reward_both-seed42-30m`,
  `step_checkpoints/steps_030000000/.../default_policy`. The
  `module_state.pkl` SHA-256 is `2ce92d45…8d76`, which matches the B2
  `durability_manifest.json`.
- **Environment**: `task.py:PerFactorRewardStateTask` with
  `reward_states=(2, 0)`. Factor 1 still pays for state 3. Transitions,
  emissions, actions, observations and the 1,024-step horizon are those of
  `reward_both`.
- **Protocol**: same as `belief_causal_followup.py` (revision `9ecb945`).
  - Fit/test: greedy streams with seeds 42/43, 12,000 retained steps each,
    excluding 8 warmup steps.
  - Maps and sites: the same five sites and marginal/joint affine maps.
  - Update: \(h' = h + \lambda (y^* - D(h)) E\).
  - Tuning: 16 episodes, seed 1042. Confirmation: 32 complete episodes,
    seed 2042, with the same environment seeds in every condition.
- **Deliberate changes**:
  - Factor-2 target (below).
  - The strength grid adds 0.25, 1.25 and 1.5.
  - Selection takes the smallest strength that *maximizes* tuning factor-2
    reward.

## Best-action map (next-step reward)

Reward at step t is \(\mathbf{1}[S_t = r]\), so action \(a_t\) only affects
\(r_{t+1}\). Given the filtered factor belief \(b_t\), the expected next-step
reward of action \(a\) is \(\sum_i b_t(i)\, T_a(i, r)\). Values of
\(T_a(i, r)\):

| From state | Noop | Tilt X | Tilt Y |
|---|---:|---:|---:|
| reward state 3: 1 | 0.100 | **0.332** | 0.024 |
| reward state 3: 2 | 0.100 | 0.024 | **0.332** |
| reward state 3: 3 | **0.400** | 0.130 | 0.130 |
| reward state 1: 1 | 0.750 | 0.556 | **0.813** |
| reward state 1: 2 | 0.150 | **0.163** | 0.111 |
| reward state 1: 3 | 0.300 | **0.435** | **0.435** |

- **Reward state 3 (training task)**: the map is the Full-State Quotient
  map. The unsteered agent follows it exactly; tabulated by exact-belief
  argmax state, it takes Tilt X in state 1, Tilt Y in state 2 and Noop in
  state 3, 100% of the time.
- **Reward state 1 (moved)**: the map takes Tilt Y near state 1, Tilt X near
  state 2, and Tilt X/Tilt Y (tied) at state 3. A small Noop region lies on
  the state-1/state-2 edge. See `results/figures/moved_reward_best_action_map.pdf`.
- **Joint map**: rewards are additive and the factors are independent, so it
  is the pair of factor maps.

## Steering target

At each step:

1. Read the decoded factor-2 belief and pick the moved-reward best action.
2. Set the target to a factor-2 point mass on the state where the source
   policy already takes that action: Tilt X → state 1, Tilt Y → state 2,
   Noop → state 3.

Factor 1 is held at its decoded belief:

- **Joint**: \(b^*(i,j) = \hat b^{(1)}(i)\,\mathbf{1}[j=j^*]\).
- **Factored**: factor-1 coordinates kept, factor-2 coordinates replaced.

## Results (confirmation, 32 episodes, mean reward per step ± episode SE)

| Condition | Factor 1 | Factor 2 (moved) | F2 action = map |
|---|---:|---:|---:|
| Unsteered agent | 0.3130 ± 0.0024 | 0.3434 ± 0.0039 | 3.0% |
| Uniform random joint actions | 0.1633 ± 0.0018 | 0.4240 ± 0.0040 | 33.5% |
| Factored steering, block 4, λ=1.25 | 0.2725 ± 0.0023 | 0.4898 ± 0.0062 | 73.0% |
| **Joint steering, block 4, λ=0.75** | **0.3130 ± 0.0024** | **0.4958 ± 0.0059** | 74.3% |
| Exact-belief best-action map (reference) | 0.3130 ± 0.0024 | 0.4922 ± 0.0058 | 100% |
| Matched-norm decoder-orthogonal null (either map) | 0.3130 | 0.3433 / 0.3434 | — |

By site (strengths fixed from block 4):

| Site | Factored F1 / F2 | Joint F1 / F2 |
|---|---:|---:|
| Block 1 | 0.2674 / 0.4849 | 0.3130 / 0.4848 |
| Block 2 | 0.2712 / 0.4883 | 0.3130 / 0.4888 |
| Block 3 | 0.2772 / 0.4894 | 0.3130 / 0.4904 |
| Block 4 | 0.2725 / 0.4898 | 0.3130 / 0.4958 |
| Post final norm | 0.2737 / 0.4835 | 0.3130 / 0.4933 |

## Interpretation

- **Unsteered**: the agent keeps its training map. Under the moved reward
  that map is worse than random on factor 2 (0.343 vs 0.424).
- **Joint steering**: factor-2 reward rises by 0.152 per step, to the level
  of the myopic best-action map run on the exact Bayesian belief. Factor 1 is
  unchanged at every site; 0.0% of factor-1 actions change.
- **Factored steering**: it reaches similar factor-2 reward but needs a
  larger strength. It costs about 0.04 factor-1 reward, changing 17–21% of
  factor-1 actions. This matches the paper: the joint representation is the
  one that controls actions selectively.
- **Map agreement**: steered factor-2 actions match the exact-belief map on
  about 74% of steps. The expected next-step reward still equals the
  oracle's (0.4954 vs 0.4928). So the disagreements are near-ties, mostly
  Tilt X vs Tilt Y near state 3, decided from the decoded belief rather than
  the exact one.
- **Controls**: decoder-orthogonal shifts of the same norm do nothing.
- **Caveats**:
  - One frozen training seed; uncertainties are episode SEs.
  - The map is myopic, so it is neither long-run optimal nor a
    matched-information ceiling.
  - The point-mass targets are off-manifold extrapolations.

## Reproduction

```bash
# module files verified against the run's durability manifest, placed under artifacts/
uv run python -m experiments.two_factor_moved_reward_steering_2026_11.steering \
  --checkpoint experiments/two_factor_moved_reward_steering_2026_11/artifacts/ckpt/default_policy \
  --output experiments/two_factor_moved_reward_steering_2026_11/results/steering_seed42.json \
  --arrays experiments/two_factor_moved_reward_steering_2026_11/artifacts/heldout_block4_seed42.npz
uv run python -m experiments.two_factor_moved_reward_steering_2026_11.plot \
  --report experiments/two_factor_moved_reward_steering_2026_11/results/steering_seed42.json \
  --arrays experiments/two_factor_moved_reward_steering_2026_11/artifacts/heldout_block4_seed42.npz \
  --output-dir experiments/two_factor_moved_reward_steering_2026_11/results/figures
uv run pytest -q tests/test_two_factor_moved_reward_steering.py
```

The run took about 4 minutes on one CPU thread. The report's
`experiment_revision` field records the base commit (`2fb56dce`): it ran from
the uncommitted worktree, and this PR's commit holds the exact driver.
Harness revision: `27f1bebe`.
