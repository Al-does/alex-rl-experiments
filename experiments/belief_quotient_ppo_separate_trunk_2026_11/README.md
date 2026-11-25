# Separate-trunk PPO on the eps02 belief-quotient ladder

Run A compares width-64 shared PPO and PPO with separate width-64 actor and
critic transformers for Trivial Quotient (variant 1) and Full-State Quotient
(variant 3). Width-90 shared PPO is a secondary total-parameter control.
Each seed (42, 43, 44) runs all six arms on the same GPU.

| Arm | Transformer(s) | Trainable parameters |
| --- | --- | ---: |
| Shared PPO | width 64 | 200,772 |
| Separate PPO | width 64 each | 401,284 |
| Wide shared PPO | width 90 | 394,654 (1.65% fewer than separate) |

All trunks have four blocks, one attention head, a 32-step per-layer RoPE
lookback, and a final LayerNorm. The separate actor starts at exactly the
width-64 shared policy's initialization for that seed. The critic trunk starts
from an independent RNG subkey; its linear readout starts at the shared value
head's initialization. Neither trunk receives rewards, beliefs, nor hidden
states as input. Only GAE targets mediate actor/critic coupling in the separate
arm: the value loss has no gradient path to actor parameters and the policy
loss has no gradient path to critic parameters. The critic sees the actor's
realized observations and rewards during value training, not its own episodes.

The [archived shared recipe](../belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11/AGENTS.md)
is matched: uniform reset; exact eps02 token/previous-action observation;
pre-transition state-2 occupancy reward; 19,456 simultaneous complete
1,024-step episodes per iteration (19,922,944 env steps), 64 whole-episode
minibatches, four shuffled epochs. The 2B target rounds up to 101 iterations
and **2,012,217,344 realized training env steps per arm**. Adam lr 4.2e-4,
discount .99, GAE lambda .95 with terminal bootstrap zero, ratio clip .2,
standardized advantages, no KL penalty or gradient clipping. Sampling
temperature is 1.5 (likelihoods use the same temperature); entropy coefficient
and exploration epsilon are both zero throughout; emission epsilon is .02.
The value head predicts (1 − gamma) × value; value MSE coefficient .5, no value
clipping. The wide control changes width, parameter count, head dimension and
compute; it is not a FLOP match, but updates and data budgets match.

## Measurements and interpretation

- All 102 checkpoints (initialization and every rollout) record independent
  greedy fit/test streams keyed by training seed. Same keys and deterministic
  episode ordering at each checkpoint and in all arms; histories can differ
  across trained policies. Actor and critic in the split arm are measured on
  the **same** histories, actions and labels. The critic replays the actor's
  exact observations with the same cached transformer and no critic-driven
  actions. These are last-layer post-final-LayerNorm probes, not layer searches.
- Exclude the first 64 decisions per episode and use 60,000/80,000 fit/test
  rows, highest JAX matmul precision and affine ridge 1e-6, as in the
  [global paper checks](../algorithm_generality_probes_eps02_2026_11/README.md).
  Fit full belief, rho (state-3 mass), delta (state-1 minus state-2 mass),
  and numerical V* independently; report raw held-out 1−R². V* is a
  160-division simplex interpolation of optimal discounted finite-horizon
  belief Bellman backups (gamma .99, zero terminal value, remaining decisions
  indexed by the row's episode step). It is an approximation, not the
  trained critic target, an exact value certificate, or a QMDP estimate.
- Performance: retain sampled training reward and held-out greedy reward
  curves. Full-episode greedy reward uses all complete test episodes including
  reset (with episode-level SE); the historical post-warmup 80K-row metric is
  separate. Do not compare a post-warmup return with a complete-episode
  benchmark. Complete-episode benchmark values from
  [the eps02 reference audit](https://github.com/Al-does/alex-rl-experiments/commit/ebc16c672f317a4b652580daf256591bd30ab3ef):
  Trivial exact optimum .941335636; Full-State feasible estimate .420969
  (95% Monte Carlo CI .419385–.422552), numerical Bayes upper .421412617.
  Use reward and action coverage to gate representation contrasts; if an arm
  is materially below its matched baseline, keep its probe results descriptive.
- Primary contrasts: Trivial rho error on separate actor versus separate
  critic and standard shared; Full-State delta error on separate actor versus
  separate critic, including each trunk's change from its own initialization.
  Compare to the wide control before attributing effects to trunk coupling.
  Show paired per-seed deltas and late trajectories, not just the mean of three
  endpoint scores. Decoding does not show causal use. Splitting removes direct
  value gradients while retaining GAE, but it can change value predictions,
  realized advantage variance and visited histories; it does not hold these
  mediators fixed.

Archived global 1−R² (rho / delta), from the
[eps02 generality tables](../algorithm_generality_probes_eps02_2026_11/results/tables.md):

| Reference | Trivial | Full-State |
| --- | ---: | ---: |
| REINFORCE (15) | .1125 / .2510 | .0275 / .0386 |
| REINFORCE (same 5 seeds as historical PPO) | .1247 / .2642 | .0268 / .0348 |
| Shared PPO (5) | .0050 / .1679 | .0030 / .0327 |

Those cohorts are descriptive references, not new paired controls; only 42
and 44 overlap the historical five. Trivial global probes are not evidence
of within-action-region structure.

## Run and outputs

```bash
python -m experiments.belief_quotient_ppo_separate_trunk_2026_11.seed_queue --seed 42 --smoke
python -m experiments.belief_quotient_ppo_separate_trunk_2026_11.seed_queue --seed 42
```

Use the active experiment virtual environment. Smoke retains the historical
4,096 steps (two 2,048-step iterations), 4,096 rows per split, a four-step
warmup and a 16-division grid. Smoke metrics are not scientific results.
The full queue requires a JAX GPU and enables artifact upload per arm. Each
leaf records its resolved recipe, per-iteration training curves, per-checkpoint
held-out reward and trunk/target errors, and harness manifest in `results/`;
the full parameter checkpoints stay in `artifacts/` and upload to B2.
Use one on-demand Vast instance per seed with required durability and a finite
lifetime backstop; verify each manifest and teardown before closing the run.
On the first box, run `python -m experiments.belief_quotient_ppo_separate_trunk_2026_11.preflight ARM --seed 42`
once each for `separate` and `shared_wide` before committing the remaining
boxes. Each pilot uses two real research-size batches and logs timings and
allocator statistics; neither pilot writes results nor alters queue initialization.
