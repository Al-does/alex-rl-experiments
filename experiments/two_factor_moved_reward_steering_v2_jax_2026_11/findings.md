# Moved factor-2 reward steering on the PR #225 v2 agents

Follow-up to `two_factor_moved_reward_steering_2026_11` (PR #207), applied to
the eight ctx32 transformer agents of PR #225
(`reward_factors_persistent_nuisance_v2_jax_temp15_ctx32{,_eps05}_2026_11`,
final checkpoints at 2,012,217,344 steps). No agent is retrained.

## Setup

- **Environment:** the persistent-nuisance v2 process the agents were trained
  on, with factor 2's reward moved from state 3 (code 2) to state 1 (code 0).
  Factor 1 still pays for state 3. Transitions, emissions, and actions are
  unchanged (`moved_process`).
- **Agents:** all eight final checkpoints, downloaded from B2 into ignored
  `artifacts/checkpoints/` and matched to each run's durability-manifest
  SHA-256 (`checkpoint_sha256` in each report). The Joint Reward agents
  (`reward_both`, no ε and ε=0.05, seeds 42/43) are the main result. The
  Unrewarded Factor 2 agents (`reward_factor_1`) are a secondary comparison.
- **Action maps** (factor 2, `[state, action]` tables in each report):
  - *next-step* (requested): \(\arg\max_a b^\top T_a[:, r]\);
  - *long-run*: QMDP \(\arg\max_a b^\top Q^*(\cdot, a)\) with the training
    discount γ=0.99.
  Factor 1 keeps its training reward and is never steered. Joint actions
  combine the two factor actions.
- **Steering:** identical to PR #207. Affine factored (4-d) and joint (8-d)
  decoders and embeddings are fit on 60K greedy unsteered rows (seed 42, 64
  warmup steps dropped). Probe R² is reported on 80K held-out rows (seed 43).
  The update is \(h' = h + \lambda(y^* - D(h))E\). \(y^*\) keeps decoded factor
  1 and puts factor 2 on the state where the agent most often takes the
  map's action. This gives Tilt X → state 1, Tilt Y → state 2, Noop → state 3
  for every Joint Reward agent.
- **Sites:** after each of the 4 blocks and post-final-norm.
- **Strength:** λ is tuned at block 4 from {0, 0.25, …, 1.5, 2, 3} for the
  highest factor-2 reward on 64 episodes (seed 1042). It is then confirmed on
  256 fresh 1,024-step episodes (seed 2042) at every site.
- **Policy:** greedy, as in the PR #225 probes. Reward is per-step state
  occupancy, including reset steps, ± SE across episodes.
- **Controls:** unsteered agent; uniform random joint actions; each map applied
  to the exact filtered belief; and a decoder-orthogonal random direction with
  the same per-step norm as the steering shift (block 4).

## Random occupancy differs in v2

In v2, actions only change transitions into and out of state 3. Under uniform
random actions the stationary factor occupancy is (0.449, 0.449, 0.102)
(`analytic_factor_occupancy`). Measured random factor-2 reward is
0.422 ± 0.011, below the stationary value because episodes start uniform.
Factor-1 random reward is 0.098.

Known-state stationary occupancy of state 1:
- next-step policy (Tilt Y, Tilt X, Tilt X): 0.499;
- long-run policy (Tilt Y, Tilt Y, Tilt X): 0.819.

The next-step map takes Tilt X only when the factor-2 belief is extremely sure
of state 2 (\(b_1 < 0.004\, b_2\)). The noisy v2 emissions almost never produce
such beliefs. On held-out exact beliefs, the next-step map picks Tilt Y 99.4–99.5%
of the time for the Joint Reward agents, and 93.9–99.9% across all eight agents.
Applied to the exact belief, the two maps reach essentially the same reward:
0.791 vs 0.793.

## Results: Joint Reward agents (block 4, 256 episodes)

Each cell is factor 1 / factor 2 mean reward per step, with λ in parentheses.
SE is 0.008–0.012 for every cell.

| Condition | no ε s42 | no ε s43 | ε=0.05 s42 | ε=0.05 s43 | mean |
|---|---|---|---|---|---|
| Unsteered | 0.461 / 0.258 | 0.481 / 0.264 | 0.464 / 0.269 | 0.474 / 0.269 | 0.470 / 0.265 |
| Uniform random | 0.098 / 0.422 | same | same | same | 0.098 / 0.422 |
| Factored, next-step | 0.338 / 0.706 (3) | 0.265 / 0.741 (3) | 0.339 / 0.710 (3) | 0.283 / 0.681 (3) | 0.306 / 0.709 |
| Joint, next-step | 0.444 / 0.724 (2) | 0.461 / 0.721 (1) | 0.459 / 0.717 (1.25) | 0.412 / 0.679 (1.5) | 0.444 / 0.710 |
| Factored, long-run | 0.462 / 0.793 (1.5) | 0.359 / 0.783 (2) | 0.393 / 0.784 (3) | 0.456 / 0.779 (1.25) | 0.418 / 0.785 |
| **Joint, long-run** | 0.461 / 0.792 (3) | 0.464 / 0.788 (1) | 0.454 / 0.794 (1.25) | 0.472 / 0.789 (1.5) | **0.463 / 0.791** |
| Exact belief, next-step map | 0.479 / 0.791 | same | same | same | 0.479 / 0.791 |
| Exact belief, long-run map | 0.480 / 0.793 | same | same | same | 0.480 / 0.793 |
| Null, joint long-run λ | 0.439 / 0.243 | 0.473 / 0.277 | 0.465 / 0.253 | 0.477 / 0.301 | 0.463 / 0.269 |

The random and exact-belief rows do not depend on the agent, and every condition
shares the same environment keys. Per-site results are in each report under
`confirmation.comparisons`; `results/summary.json` holds this table.

- **Steering works.** Every Joint Reward agent goes from 0.26 (below random)
  to the exact-belief reference of about 0.79 on factor 2.
- **Joint steering preserves factor 1.** Factor-1 reward changes by at most
  0.02 in 7 of the 8 map × agent cells. The exception is ε=0.05 seed 43 with
  the next-step map: −0.06.
- **Factored steering costs factor 1.** Factor-1 reward drops by 0.12–0.22
  with the next-step map and by 0.00–0.12 with the long-run map. This is the
  same joint-vs-factored ordering as in the paper and PR #207.
- **Next-step vs long-run map.** Steering with the next-step map plateaus near
  0.71, about 0.08 below the long-run map. The same map reaches 0.791 on the
  exact belief. Likely cause (not verified): affine-decoded beliefs can
  overshoot the state-2 vertex and land in the tiny Tilt X region.
- **Sites.** With the long-run joint map, factor-2 reward rises with depth.
  For example, no ε seed 43: block 1 0.739, block 2 0.764, block 3 0.784,
  block 4 0.788, post-norm 0.787.
- **Null control.** Decoder-orthogonal directions of the same size leave
  factor-2 reward at unsteered levels (0.24–0.30). One exception: the factored
  next-step null at λ=3 on no ε seed 43 disrupts both factors (0.315 / 0.435).
  That is a non-specific effect of a very large perturbation.

## Results: Unrewarded Factor 2 agents

Steering changes factor-2 reward by at most 0.03 for these agents
(`v2_moved_reward_steering_reward_factor_1.pdf`). Their factor-2 actions do
not track the factor-2 belief the way the Joint Reward agents' actions do:

- **Seed 42 (both ε):** factor 2 is poorly encoded (block-4 factored R²
  0.60–0.62), and the factor-2 action barely depends on the belief. Unsteered
  factor-2 reward is 0.29–0.30 and stays there under steering.
- **Seed 43 (both ε):** the agent almost always takes Tilt Y on factor 2,
  which happens to be the moved-reward optimum. Unsteered factor-2 reward is
  already 0.797 (no ε) and 0.766 (ε=0.05). Steering raises the ε=0.05 agent
  to 0.79.

Factor-1 reward (about 0.47) is unchanged by every intervention.

## Caveats

- All reports record experiment revision `c6319e6` (PR #225 head) with
  `dirty: true`, because the analysis ran from the uncommitted working tree
  that this PR commits.
- There are two training seeds per arm, and SEs are across episodes only.
- λ was selected at block 4 and reused at the other sites. Several selected
  λ values are at the top of the grid (3).
- "Long-run" means QMDP over the exact belief, a heuristic rather than the
  POMDP optimum. Its 0.819 value assumes a known state, so it is not a bound
  under partial observability.

## Reproduce

```bash
uv run --group jax pytest -q tests/test_two_factor_moved_reward_steering_v2_jax.py
# needs the eight final checkpoints in artifacts/checkpoints/ (see each report's
# source.remote_prefix and checkpoint_sha256); ~8 CPU-minutes per agent
uv run --group jax python -m experiments.two_factor_moved_reward_steering_v2_jax_2026_11.steering
uv run --group jax python -m experiments.two_factor_moved_reward_steering_v2_jax_2026_11.summarize
uv run --group jax python -m experiments.two_factor_moved_reward_steering_v2_jax_2026_11.plot \
  --results experiments/two_factor_moved_reward_steering_v2_jax_2026_11/results \
  --arrays experiments/two_factor_moved_reward_steering_v2_jax_2026_11/artifacts/ctx32_2026_11__reward_both__seed42_heldout.npz \
  --output-dir experiments/two_factor_moved_reward_steering_v2_jax_2026_11/results/figures
```
