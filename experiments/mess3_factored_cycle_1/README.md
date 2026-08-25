# Factored MESS3 cycle 1

This family implements the four-condition design in
`two_factor_mess3_design_4.md` using the factored-HMM and representation
geometry APIs added in
[RL-Harness PR 35](https://github.com/Al-does/RL-Harness/pull/35).

## Frozen common recipe

- Joint state and symbol indices are `3 * first + second`.
- Both MESS3 factors use
  `[[.75,.15,.10],[.15,.75,.10],[.30,.30,.40]]`.
- The static model is built by `envs.hmm.factored_model`; experiment-local
  tasks supply action-conditioned joint kernels.
- PPO and the `64 × 4` transformer match
  `mess3_reward_state_action_symmetry_cycle_5`.
- `gamma=0.99`, horizon 1,024, uniform initial states, and current-state
  occupancy rewards are explicit.
- Every RL run trains a matched next-joint-symbol predictor on trajectories
  sampled from its restored final stochastic policy. The predictor uses the
  same encoder, Adam at `3e-4`, one smoke epoch or six full epochs, and stores
  its raw trajectory dataset under ignored `artifacts/`. Its causal examples
  are `(history through x_t, executed a_t) -> x_{t+1}`; a separate restored
  policy episode supplies held-out metrics.

## Conditions

- E1: primary diagonal F2 reward, reward-site swap, and product-action control.
- E2: factored-input lambda sweep `0, .5, 1, 1.5, 2`; lambda 1 is the
  operating point. Values 1.5 and 2 are dose-response runs above the registered
  incentive ceiling, not operating conditions.
- E3: additive/product, conjunctive/product, and additive/diagonal.
- E4: primary `alpha1=.50`, reactive-sufficient null `alpha1=.85`.
- Encoding panel: matched factored/joint E1 and E3a asymmetric pairs, plus E2
  and E4 joint-symbol arms.

Each runnable leaf owns an explicit `Condition` value and exports fresh
`build_config(context)` and `run(context)` functions.

## Scientific scope correction for E2

The controlled latent block process is exactly lumpable: F1 and F2's
within-`N` identity cannot change block-to-block transition probabilities.
With the full three-symbol F2 channel, however, fine information slightly
sharpens the block posterior. The positive aware-minus-coarse incentive in the
design document is evidence of that leak.

E2 therefore tests an **approximate, incentive-controlled quotient**: whether
learned F1 signal tracks control incentive rather than predictive visibility.
It must not be reported as exact belief-state bisimulation.

## Audits and analysis

`audits/experiment.py` runs deterministic A1 lumpability, A2 fully observed
value invariance, and E3 function-coupling checks. Smoke runs record the
registered A3-A6 thresholds and reference targets but deliberately do not
mislabel small simulations as the specified 4,096-chain acceptance campaign.
Non-smoke training is blocked until the complete registered campaign is
recorded at `results/reference_audits.json` with passing A1-A6 status and the
specified chain/step/burn-in/standard-error protocol.

`analysis.py` uses PR 35's:

- `factor_marginals` and `product_distribution` for exact targets;
- `variance_geometry` for CEV;
- `regression_factor_geometry` for jointly fitted factor subspaces;
- `representation_dimension_predictions` for the four-dimensional factored
  versus eight-dimensional joint baselines.

Geometry is not interpreted until held-out probes establish decodability.
E4 has no hard-coded four-dimensional quotient prediction: its report includes
the full joint residual and relative-phase posterior because equal marginals
need not imply equal gauge-control actions.

Every completed RL run now writes `final_probe.json` from disjoint policy
rollouts. It reports held-out joint/F1/F2/block/within-N/relative-phase probes,
PR 35 CEV and subspace geometry, and (for E3) nested factor-only versus
factor-plus-interaction readouts of exact QMDP and learned value/logit targets.

## Smoke examples

```bash
uv run rl-harness \
  experiments.mess3_factored_cycle_1.audits.experiment --smoke

uv run rl-harness \
  experiments.mess3_factored_cycle_1.e1_f2_diagonal_factored.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.mess3_factored_cycle_1.e2_lambda_1p0_factored.experiment \
  --smoke --hardware-profile cpu
```
