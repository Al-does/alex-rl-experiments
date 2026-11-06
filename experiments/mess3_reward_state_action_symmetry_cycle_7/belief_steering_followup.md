# Full-State Quotient offline activation steering

Run from this repository with the editable sibling harness installed:

```sh
uv run pytest -q tests/test_fsq_belief_steering.py
uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_followup --smoke --seeds 42
uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_followup
```

No training, Algorithm restoration, or paid compute is involved. Final native
modules for the five prespecified training seeds 42–46 are selected from each
run's final `checkpoint_probe_curve.json` record, then downloaded through
`control_campaign._download_module`. All three module files are SHA-256 checked
against the canonical durability manifest. Historical cycle-6 run identifiers
are preserved. Checkpoints and smoke reports live only in ignored `artifacts/`.
Compact per-seed and aggregate reports live under `variant_3_ctx32_ent/results/`.
Existing reports are never overwritten; a resumed full campaign reuses completed
per-seed reports. Smoke reports must be moved aside before repeating a smoke.

The baseline is the immutable Joint Reward driver and seed-42 report at
`9ecb94542a2d0977e1e369040c956ca8d6353cbe`, not the older exploratory
intervention. The five sites are post-MLP residuals after blocks 1–4, plus final
LayerNorm output. Hooks capture the input to `ln2` and add the MLP output.
Shifts occur after that block computes its KV cache; later blocks see the
shifted residual. `encode_step` concatenates/stacks fresh cache tensors, so the
discarded shadow pass cannot mutate the executed cache. Tests exercise these
properties on the actual four-block context-32 architecture.

Fit and held-out test trajectories use fixed seeds 42 and 43 respectively for
every training seed, with 12,000 retained greedy decisions per split and eight
excluded warmup decisions per episode, matching the baseline. Eight steps do
not exhaust this banded encoder's possible 128-observation lookback; this is
an explicit protocol choice, not a claim of full-context warmup. Separate
affine least-squares regressions fit activation→(b1,b2) and (b1,b2)→activation.
The third probability is reconstructed by normalization. Raw decoded beliefs
are not clipped or projected. Both targets preserve decoded reward-state mass;
point ties choose state 2. Mirror swaps states 1 and 2. Each installed shift is
`lambda * (target - decoded) @ embedding` at exactly one site.

Tuning uses 16 complete 1,024-decision episodes, seed 1042. For each intervention
independently select the smallest grid lambda with occupancy at or below the
matched random-action baseline, or the lowest-occupancy lambda if none qualify
(ties choose the smaller lambda). Freeze that lambda across sites and for the
stepwise norm-matched decoder-orthogonal control. Confirmation uses 32 new
complete episodes, seed 2042, with matched environment seeds for every condition.
All reset and warmup decisions count in occupancy. The final pre-LayerNorm site
also receives the entire prespecified lambda grid on confirmation episodes.
Shadow comparisons use the same current observation and executed recurrent
state, discarding shadow state and actions. Random actions use the intact
forward's recurrent state and the actually executed preceding-action features.

Decision-time `belief_current` and `state_current` diagnostics are read before
the action; the environment emits a reset token and has delay zero. Prediction
tests independently reconstruct the exact filter and compare all controlled
transition matrices against the environment. These diagnostics never enter
the actor observation, which contains only the token and preceding action.

`belief_steering_prediction.py` ports `~/scratch/fsq_pred.py`, vectorizing its
closed-loop simulations. It retains N=60, gamma=0.97, 800 value iterations,
nearest-grid projection, uniform reset prior, and the exact token/action filter.
It applies the target to the exact belief, unlike activation steering's decoded
belief target. Its action is evaluated at `(1-lambda)*b + lambda*target(b)`;
filtering then follows the executed action, not the installed counterfactual.
Predictions use 32 complete 1,024-step episodes and seed 2042. RNG vectorization
changes draws from the scratch script, not the transition/filter law. This is a
near-optimal feasible numerical reference, **not a certified Bayes-optimal
policy**; the reported installed-belief agreement is against this reference.
The point target and mirror need not induce the same partial-strength curve.

Per-condition reports contain decision-time occupancy ± episode SE, action
fractions, changes versus the shadow, state/action counts and conditional action
probabilities, Noop probability in reward state 3, and reference-policy
agreement. R² pools the first two belief coordinates. Aggregate SEs are across
the five training-seed means, not across episodes. Conditional probabilities
with no visited state are undefined (`null` in per-seed JSON). The full protocol,
checkpoint hashes, source hashes, both repository revisions, and runtime
versions are recorded in JSON. Off-manifold shifts and action agreement do not
uniquely identify internal Bayesian inference.
