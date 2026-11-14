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

## Caveats

- Two training seeds per arm; SEs reflect episode-to-episode variation only.
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

About 40 minutes on 8 CPU cores with the four agents in parallel (`--only`).
