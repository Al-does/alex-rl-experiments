# Reward references: what is and is not certified

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.optimal_reward \
  --output experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/results/NEW_DIRECTORY/report.json \
  --grids 80 320 --episodes 16384 --starts 64
uv run pytest -q tests/test_eps02_optimal_reward.py
```

The CLI covers both ladder and reward-factor studies, v2 sanity checks, and the
4B continuation's saved final results. It refuses to overwrite reports.

## Objective and comparability

The primary benchmark is **expected undiscounted total reward in a complete
1,024-step episode divided by 1,024**, starting from the uniform prior with a
reset emission before the first action. This is the population counterpart of
training_curves.jsonl's reward_per_step, not infinite-horizon stationary
occupancy. Reward is pre-transition S3 occupancy. reward_both sums two factors
(range 0–2), not their average. State and token ordering use the existing process.

Training uses gamma=.99 in the policy-gradient reward-to-go, but the reported
reward metric is NOT discounted. We optimize that reported task metric here,
not the discounted training surrogate. Temperature=1.5, zero entropy penalty,
and zero exploration do not restrict the set of categorical distributions;
deterministic policies are limits of softmax logits.

The checkpoint_report policy is **greedy** and excludes 64 warmup steps per
episode. It is retained separately, not mixed with stochastic training rewards.
The JSON includes per-run/seed/leaf final training and greedy values, checkpoint
steps, and input SHA256s. Saved aggregate means cannot give episode-level CIs.
Continuation results are separate records, not duplicate independent seeds.

## Full-history control

Belief-MDP dynamic programming maximizes the 1,024-step reward using the exact
filtered posterior over the current hidden state (tokens and executed actions,
never observed rewards). Policies may depend on remaining horizon and the full
belief, with no neural receptive-field restriction.

A triangular simplex mesh and nonnegative barycentric interpolation give
**upper bounds**: finite-horizon POMDP values are convex in belief, so the
interpolated value of exact grid nodes is an upper envelope; induction preserves
this property when grid-node values are themselves upper bounds. No discretized
posterior is used in policy simulation. Each candidate action uses the exact
next posterior and the interpolated continuation table.

The policy induced by the table is feasible. Independent episodes estimate its
expected reward with a 95% Student-t CI across episode averages; rewards are
Rao–Blackwellized as P(S3 | history). This is a CI for a **lower-bound policy's
value**, NOT a CI for the optimum. The upper bound is numerical up to floating
point, not interval-arithmetic certified. Refinement is reported explicitly.

Variant 1 is exact: aggregate S1/S2 into a nonreward state; action 1 maximizes
next-S3 probability in both aggregate states, and its next-S3 probability is
increasing in the current reward-state indicator. Backward induction therefore
makes constant action 1 optimal even for a fully informed controller.

Independent factors with Cartesian joint actions and additive rewards have
additive full-history Bayes values. reward_both is twice variant 3;
reward_factor_1 is once variant 3. There is no extra information about one
independent factor in the other factor's observations when own actions/history
are known. Reported joint feasible values can be attained by independent copies
of the per-factor policy. The JSON links to the per-factor interval; it does
not pretend to have independently simulated a joint-policy CI.

## Memoryless control

Memoryless means a **stationary stochastic mapping from current token only**,
matching models.py's feature_count=n_tokens, not the previous-action-augmented
observation seen by the recurrent models. Exact matrix powers evaluate a
candidate's complete-episode expected reward, with analytic gradients.

The search includes every deterministic 3-token policy, then simplex-constrained
multistart optimization allowing stochastic policies. For joint tasks it searches
all 9-token/9-action probability rows, not just independent product policies.
The product of good single-factor policies is included as an initial candidate.
The other factor must not simply be discarded: a joint reactive policy can use
its controlled dynamics as an external memory device.

**Except for variant 1, these are best-known reactive values, not proven global
memoryless optima.** Deterministic enumeration alone does not rule out a better
stochastic policy. Multistart convergence does not prove global optimality.
Thus this computation provides an honest quantitative reference and Bayes
brackets, but does not complete a globally certified solution of every task.

## Prior analytical code

PR #208's design.py is for v1, not v2, and explicitly disclaims Bayes optimality.
We reuse its stationary-distribution solver and reproduce its recorded
current-token .228861871655982 and three-token .251424641014780 references.
On v2, the best deterministic current-token stationary reference is
.345326852727761 (variant 2, actions [1,1,0]); that is not the finite-episode
value and not a Bayes optimum. The new CLI also runs the full belief calculation
and reactive search on v2 without changing historical code or hard-coded
emission-dependent formulas.
