# RockSample r7 belief steering with counterfactual-twin displacement maps

Held-out confirmation of the protocol frozen after exploration. Full pooled,
per-agent, selectivity, and fit tables are in `tables.md`; raw per-cell
summaries are in `agent{0,1,2}_update{29,316}.json`.

## Method

For rock *i*, a **twin** history replays the same actions and positions with
only the Good/Bad sensor symbol flipped on every Check of rock *i*. Let `h`
and `h_twin` be the post-block-1 latest-position residuals of the real and
twin histories at all decisions after rock *i* was first checked and before
it is sampled.

1. `U_i`: top four right singular vectors of `h_twin - h`.
2. `g_i`: one affine map fitted by least squares on real and twin rows
   together to predict the displacement to the other member of the pair
   (`h -> h_twin - h`, `h_twin -> h - h_twin`).
3. Steering: `h' = h + 1.25 * g_i(h) U_i^T U_i` at every decision until rock
   *i* is sampled. No twin is run during steering.

In the manuscript's notation this is `h' = h + lambda [y* - D(h)] E` with
`D(h) = h U_i^T`, `E = U_i`, `y* = (h + g_i(h)) U_i^T`, and `lambda = 1.25`.
The same map serves both directions: it moves the activation to where it
would be had the evidence about rock *i* been the opposite.

Site (block 1), rank (4), and strength (1.25) were chosen on fit seed 41101 and
rollout seed 61307 (`belief_steering_cf_explore.py`, suites `explore1`–`5`).
This confirmation refits everything on 512 new episodes (seed 77003) and
evaluates 1,024 sampled-action episodes per cell on new environment seeds
(88019), for each of seven target rocks, both directions, three upper-plateau
agents (0: run 2 seed 3; 1: run 1 seed 2; 2: run 2 seed 2), and the 30M
(update 29) and 330M (update 316) checkpoints. The target rock's true quality
is fixed per direction; the other rocks are random.

## Result

Three agents and seven target rocks pooled (21,504 episodes per direction;
Wilson 95% half-widths are at most 0.007).

| Condition | Good target sampled | Bad target sampled | Other Good sampled | Other Bad sampled |
| --- | --- | --- | --- | --- |
| **30M** intact | 0.980 | 0.013 | 0.978 | 0.013 |
| Uniform random actions | 0.187 | 0.187 | 0.191 | 0.184 |
| Twin steering (frozen) | **0.014** | **0.918** | 0.976 | 0.014 |
| Norm-matched decoder-null direction | 0.836 | 0.008 | 0.971 | 0.012 |
| Marginal-probe clamp (PR #205 method) | 0.122 | 0.623 | 0.890 | 0.028 |
| Rank-4 interchange with the true twin | 0.012 | 0.963 | 0.981 | 0.013 |
| **330M** intact | 0.985 | 0.010 | 0.986 | 0.011 |
| Twin steering (frozen) | **0.011** | **0.954** | 0.978 | 0.011 |
| Norm-matched decoder-null direction | 0.816 | 0.011 | 0.961 | 0.012 |
| Marginal-probe clamp (PR #205 method) | 0.240 | 0.675 | 0.924 | 0.014 |
| Rank-4 interchange with the true twin | 0.010 | 0.983 | 0.985 | 0.010 |

- **Good→Bad.** Steered Good rocks are sampled as rarely as the intact policy
  samples Bad rocks (0.011–0.014 versus 0.010–0.013), far below uniform random
  actions (0.187). Per agent: 0.008–0.019.
- **Bad→Good.** Steered Bad rocks are sampled in 0.90–0.95 (30M) and
  0.91–0.98 (330M) of episodes per agent, against 0.01 intact.
- **Selectivity.** Sampling of the other rocks is unchanged to within 0.01.
  In the 7×7 matrices, off-diagonal Good-rock sampling stays at 0.87–0.99 and
  off-diagonal Bad-rock sampling at 0.00–0.04.
- **Decoded belief.** The final-layer affine probe's P(Good) for the target
  moves from 0.71 to 0.12 (Good rocks) and from 0.14 to 0.69 (Bad rocks) at
  30M; decoded beliefs of the other rocks do not move (0.74/0.13–0.14).
- **Behavior otherwise intact.** Episodes still end by exiting (0.989 at 30M,
  0.999 at 330M) and return changes by the value of the one redirected rock
  (48.6→38.9 for Good targets, 38.8→29.3 for Bad targets at 30M).

## Caveats

- The norm-matched control (same per-step displacement norm, one random
  direction orthogonal to the probe) never induces sampling of a Bad rock, but
  it is not behaviorally inert: it lowers Good-target sampling to 0.84/0.82
  pooled and as low as 0.05 for single rocks. Bad→Good and the selectivity
  matrix are therefore the clean specificity evidence; a drop in Good-rock
  sampling alone could be produced by nonspecific disruption.
- At block 1 the displacement must be applied both on the decisions that
  immediately follow a Check of the target rock and on all other decisions.
  Either subset alone is much weaker (30M: 0.49/0.14 and 0.31/0.35). The edit
  therefore rewrites the residual trace of the evidence as well as the
  aggregated belief; it is not a single-token belief overwrite.
- The map reflects the current evidence state rather than setting an absolute
  target, so both directions use the same map and the result does not test
  steering towards an arbitrary intermediate belief.
- Strength 1.25 was selected on the tuning seed; at strength 1 Bad→Good is
  0.81 (30M) and 0.89 (330M).
- Rock (0,3) is the weakest Bad→Good cell (0.76 at 30M, 0.79 at 330M pooled;
  0.50 for agent 1 at 330M). Its affine displacement fit is also the worst.
- Three agents, one environment layout; intervals are over episodes, not
  training seeds. Reruns on GPU differ by at most 0.003 per cell.

## What makes steering easier (secondary)

- **Direction source.** Displacements from counterfactual twins steer far
  better than regression-probe directions at the same site, although the probe
  directions change the decoded belief more (the clamp drives the readout to
  0.04/1.04).
- **Depth.** Blocks 1 and 2 work (block 2: 0.025/0.857 at 30M, 0.011/0.967 at
  330M); block 3 and the final norm give about 0.1/0.5 with collateral loss.
  Probe R² rises with depth, so decodability does not predict steerability.
- **Training length.** At 330M a single direction suffices (rank 1:
  0.011/0.846) and block 2 matches block 1; at 30M rank 1 gives 0.116/0.436 and
  rank ≥ 4 is needed. Ranks 4, 8, and 128 are equivalent.
- **Predictability of the displacement.** Per rock, the affine fit R² of the
  twin displacement (0.49–0.98) tracks Bad→Good success; true-twin interchange
  in the same rank-4 subspace reaches 0.96–0.98, so the gap is the affine
  predictor, not the subspace.
- **Token-typed maps** (separate maps for post-Check and other decisions) add
  about 0.01 to Bad→Good.

## Reproduction

```bash
uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_cf_confirm \
  --agent {0,1,2} --update {29,316} --output NEW_DIRECTORY
uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_cf_report NEW_DIRECTORY
```

Reports were produced at revision `372b3f886ad4b69e1fcdfb15f6d9350ec4a90568`
on RTX 4090 boxes; checkpoint SHA-256 values are recorded in each JSON.
